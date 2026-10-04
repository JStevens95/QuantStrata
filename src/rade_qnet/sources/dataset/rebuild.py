"""
Rebuilding a dataset the way a saved model already saw it.

:meth:`DataModule.build` does five things in order::

    load  ->  split  ->  fit_state  ->  transform  ->  signature

Rebuilding does three of them. The two it skips are the two that would
silently invalidate every comparison a re-loaded model is used for::

    load  ->  [split: READ FROM LINEAGE]  ->  [fit_state: READ FROM BUNDLE]
          ->  transform  ->  signature

Why the split is read rather than re-derived
--------------------------------------------
A split is a property of a particular training run, not a function anyone
should expect to be stable. Re-deriving one means that a change to split
logic silently re-scores an old model against different data, and its new
metrics stop being comparable to its old ones with nothing to indicate it.

The sharper version of the same problem needs no code change at all. A
chronological split re-derived against a *longer* history moves its
boundaries forward, so rows that were training data when the model was fitted
are now its test set. The model scores beautifully, and the number is
worthless.

Why the state is read rather than re-fitted
--------------------------------------------
This is the classic production failure, and it is quiet in exactly the same
way. A scaler re-fitted on live data standardises today's inputs by today's
mean, while the model learned a response to inputs standardised by the
*training* mean. The model is then shown inputs on a scale it has never
encountered and returns confident nonsense -- no exception, no warning, just
numbers that are wrong by an amount nobody can estimate.

Phase 1 made this avoidable by separating :meth:`DataModule.fit_state` from
:meth:`DataModule.transform`. This module is where that separation earns its
keep: it calls the second and never the first.

What is still allowed to change
--------------------------------
The raw inputs. Re-scoring last month's model against this month's data is a
legitimate thing to want, and refusing it would make the module useless for
monitoring. So a changed source is reported rather than rejected:
:class:`RebuiltDataset` carries ``source_changed``, and the caller decides.
What is *not* allowed is for it to happen without anyone being told.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from ... import __version__
from ...core.contract.data import DataLineage, SplitIndices
from ...core.runtime.errors import ContractError
from ...core.runtime.hashing import digest_spec
from ...core.runtime.logging import get_logger
from .cache import PreparedDataset

if TYPE_CHECKING:
    from ...core.contract.state import FittedState
    from ...core.spec.data import SourceSpec
    from .module import DataModule

__all__ = ["RebuiltDataset", "rebuild_dataset"]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RebuiltDataset:
    """
    A dataset rebuilt for a saved model, and what changed since it was saved.

    Parameters
    ----------
    dataset
        The rebuilt dataset: the same transforms the model was trained with,
        applied to whatever the source holds now, split exactly as the
        training run split it.
    source_changed
        Whether the raw inputs differ from the ones the bundle was trained
        on. ``False`` means the rebuild is a faithful reconstruction and any
        metric computed from it is directly comparable to the recorded one.
        ``True`` means it is a re-score against different data, which is
        often the point -- but is never something to discover afterwards.
    original_fingerprint
        The source fingerprint recorded in the bundle's lineage.
    current_fingerprint
        The source fingerprint of the inputs just read.
    """

    dataset: PreparedDataset
    source_changed: bool
    original_fingerprint: str
    current_fingerprint: str

    def describe(self) -> str:
        """
        Return a one-line description, for a log and a report.

        Returns
        -------
        str
            Names the change when there is one, because a silent re-score
            against different data is the thing this type exists to prevent.
        """
        if not self.source_changed:
            return f"rebuilt {self.dataset.lineage.n_scenarios} scenario(s) from the saved source"
        return (
            f"rebuilt {self.dataset.lineage.n_scenarios} scenario(s) from a CHANGED source "
            f"({self.original_fingerprint[:8]} -> {self.current_fingerprint[:8]}); "
            f"metrics are not comparable to the recorded ones"
        )


def rebuild_dataset(
    module: DataModule,
    spec: SourceSpec,
    *,
    state: FittedState,
    lineage: DataLineage,
) -> RebuiltDataset:
    """
    Rebuild a dataset using a bundle's saved state and split.

    Parameters
    ----------
    module
        The model's data module -- the same one that built the dataset
        originally, obtained from the model definition the bundle names.
    spec
        The source specification to read from. Normally the bundle's own,
        but a caller re-scoring against newer inputs may supply a different
        one; that is what ``source_changed`` then reports.
    state
        The fitted state loaded from the bundle. Applied, never re-fitted.
    lineage
        The bundle's data lineage, which supplies the split.

    Returns
    -------
    RebuiltDataset
        The dataset and what changed.

    Raises
    ------
    ContractError
        If the lineage's split indices do not fit the data that was read.
        Raised rather than clipped: an index past the end of the array means
        the source has *shrunk*, and silently scoring on whatever rows
        remain would produce a number for a split that no longer exists.
    """
    raw = module.load(spec)
    fingerprint = module.source_fingerprint(spec)
    changed = fingerprint != lineage.source_fingerprint

    # The state is applied, not fitted. This single line is the difference
    # between a faithful reconstruction and the most common production bug
    # in applied machine learning.
    features, target = module.transform(raw, state)
    signature = module.signature(spec, features=features, state=state)

    splits = _splits_from(lineage, n_rows=int(features.shape[0]))

    if changed:
        _LOGGER.warning(
            "source fingerprint differs from the bundle's (%s -> %s); this is a "
            "re-score against different data, not a reproduction",
            lineage.source_fingerprint[:8],
            fingerprint[:8],
        )

    dataset = PreparedDataset(
        features=features,
        target=target,
        splits=splits,
        signature=signature,
        state=state,
        lineage=DataLineage(
            source_fingerprint=fingerprint,
            spec_digest=digest_spec(spec),
            # The *saved* split, carried forward unchanged, so that a bundle
            # written from this rebuild records the split it actually used
            # rather than one it might have derived.
            split_indices=splits.as_lineage(),
            n_scenarios=int(features.shape[0]),
            n_entities=module.n_entities(raw),
            framework_version=__version__,
            created_at=datetime.now(UTC),
            notes={**dict(lineage.notes), "rebuilt_from": lineage.source_fingerprint[:16]},
        ),
        static_inputs=module.static_inputs(raw, state),
        feature_names=module.feature_names(raw, state),
        entity_ids=module.entity_ids(raw),
    )

    rebuilt = RebuiltDataset(
        dataset=dataset,
        source_changed=changed,
        original_fingerprint=lineage.source_fingerprint,
        current_fingerprint=fingerprint,
    )
    _LOGGER.info("%s", rebuilt.describe())
    return rebuilt


def _splits_from(lineage: DataLineage, *, n_rows: int) -> SplitIndices:
    """
    Rebuild the split indices a training run used.

    Parameters
    ----------
    lineage
        The bundle's lineage, holding one index list per split.
    n_rows
        How many rows were just read, used to check the indices still fit.

    Returns
    -------
    SplitIndices
        The saved split.

    Raises
    ------
    ContractError
        If any saved index is beyond the data that was read.
    """
    indices = {
        name: np.asarray(values, dtype=np.int64) for name, values in lineage.split_indices.items()
    }

    for name, values in indices.items():
        if values.size and int(values.max()) >= n_rows:
            raise ContractError(
                f"the bundle's {name!r} split references scenario "
                f"{int(values.max())} but the source now holds only {n_rows} "
                f"scenario(s). The source has shrunk since the model was "
                f"trained, so the split it was scored on no longer exists"
            )

    return SplitIndices(
        train=indices.get("train", np.empty(0, dtype=np.int64)),
        validation=indices.get("validation", np.empty(0, dtype=np.int64)),
        test=indices.get("test", np.empty(0, dtype=np.int64)),
    )
