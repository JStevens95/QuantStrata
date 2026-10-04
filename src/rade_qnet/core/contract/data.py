"""
The payloads a data module hands to an engine.

:class:`DataBundle` is what a data build produces and what everything
downstream consumes. It is generic over the engine's native payload type, so
one pipeline serves every engine without branching on which.

There is one payload type rather than two. An earlier design carried a second,
``ArrayData``, so that a one-shot engine could take a whole split at once
instead of a stream of batches. It was retired in Phase 6 because the mismatch
it bridged does not exist: this layer imports no training library, so a batch
is already a mapping of NumPy arrays, and a tree engine draining one pays a
single concatenation that its own ``DMatrix`` construction was going to cost
anyway. Making the second path live would have required an array branch in
every consumer of a split -- scoring, evaluation and inference included --
which is how a framework acquires two lifecycles and the claim of having one
becomes false.

Why the engine payload is opaque here
-------------------------------------
``core`` may not import a training library -- not even under
``TYPE_CHECKING``, because then ``core`` could not be read without that
library installed. So :data:`TensorLike` is deliberately opaque: this module
declares the *shape* of the hand-off, and the engine that constructs the
payload knows the real element type. The alternative, moving these types into
``engines``, would mean every pipeline signature mentioning a specific
library.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
from pydantic import Field

from ..lifecycle.errors import ContractError, SpecError
from .base import ContractModel
from .signature import InputSignature
from .state import FittedState

__all__ = [
    "FEATURE_MATRIX_RANK",
    "SPLIT_NAMES",
    "TARGET_KEY",
    "Batch",
    "DataBundle",
    "DataLineage",
    "SplitIndices",
    "TensorBatchData",
    "TensorLike",
]

#: A stand-in for an engine's native tensor type. ``core`` cannot name the real
#: type, so this is opaque by design; see the module docstring.
type TensorLike = object

#: One batch: a mapping from input name to tensor.
type Batch = Mapping[str, TensorLike]

#: The key a batch holds its target under.
#:
#: Here rather than in the data layer or the engine, because it is the one name
#: both ends of :data:`Batch` have to agree on. Defined twice -- once by
#: whoever writes the batch and once by whoever reads it -- a rename in one
#: place produces a ``KeyError`` at the first training step, or worse, a
#: silently ignored target where the reader treats unknown keys as inputs.
TARGET_KEY = "target"

#: The split names the framework recognises. Fixed rather than free-form so a
#: report, a metric table and a bundle all agree on what "validation" is called.
SPLIT_NAMES: tuple[str, ...] = ("train", "validation", "test")

#: A feature matrix is samples by features, and nothing else is accepted.
FEATURE_MATRIX_RANK = 2


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """
    Which scenarios belong to each split.

    Parameters
    ----------
    train, validation, test
        Scenario indices. ``validation`` and ``test`` may be empty; ``train``
        may not.

    Raises
    ------
    SpecError
        If training indices are empty, or any index appears in two splits.
        Validated on construction rather than trusted, because an overlapping
        split is the single most expensive mistake available here: it produces
        an encouraging validation score and a model that fails in production.
    """

    train: NDArray[np.int64]
    validation: NDArray[np.int64]
    test: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate that the splits are usable and disjoint."""
        if self.train.size == 0:
            raise SpecError("a split must contain at least one training scenario")

        as_sets = {name: set(self[name].tolist()) for name in SPLIT_NAMES}
        for index, first in enumerate(SPLIT_NAMES):
            for second in SPLIT_NAMES[index + 1 :]:
                shared = as_sets[first] & as_sets[second]
                if shared:
                    raise SpecError(
                        f"{first} and {second} splits share {len(shared)} scenario "
                        f"index(es), e.g. {sorted(shared)[:5]}; splits must be disjoint"
                    )

    def __getitem__(self, name: str) -> NDArray[np.int64]:
        """
        Return the indices for a named split.

        Parameters
        ----------
        name
            One of ``train``, ``validation`` or ``test``.

        Returns
        -------
        numpy.ndarray
            Scenario indices.

        Raises
        ------
        ContractError
            If the name is not a recognised split.
        """
        if name not in SPLIT_NAMES:
            raise ContractError(f"unknown split {name!r}; expected one of {list(SPLIT_NAMES)}")
        return getattr(self, name)

    @property
    def sizes(self) -> dict[str, int]:
        """Number of scenarios in each split."""
        return {name: int(self[name].size) for name in SPLIT_NAMES}

    def as_lineage(self) -> dict[str, tuple[int, ...]]:
        """
        Render the indices in the JSON-encodable form a lineage record holds.

        Recording the indices themselves, rather than the fractions that
        produced them, is what lets a saved bundle be re-evaluated against
        exactly the data it was trained against -- even after the split logic
        has changed.

        Returns
        -------
        dict
            Mapping of split name to a tuple of indices.
        """
        return {name: tuple(int(value) for value in self[name]) for name in SPLIT_NAMES}


class DataLineage(ContractModel):
    """
    Where a dataset came from, in enough detail to rebuild it.

    Parameters
    ----------
    source_fingerprint
        Digest of the raw input. Changing the input changes this, which is
        what makes a stale cache entry detectable.
    spec_digest
        Digest of the source specification that produced the build.
    split_indices
        The exact indices used, from :meth:`SplitIndices.as_lineage`.
    n_scenarios
        Length of the scenario axis before splitting.
    n_entities
        Number of entities, for models with an entity axis. ``None`` when the
        problem has no such axis.
    framework_version
        Version of ``rade_qnet`` that produced the build.
    created_at
        When the build completed.
    quality
        Data-quality metrics for the build, from
        :func:`~rade_qnet.analysis.metrics.quality.quality_metrics`. Empty when
        the pipeline did not compute them.

        Recorded here rather than left to a report because a report runs
        against whatever data is to hand at the time, and these numbers
        describe the data the model was *trained* on. Once training is over
        that dataset may not be reconstructable, so the numbers are either
        captured in the bundle or lost.

        Float-valued rather than folded into ``notes`` so that a threshold
        comparison stays a numeric comparison. Filled in by the orchestration
        layer: a data source cannot compute them itself, since ``sources`` may
        not import ``analysis``.
    notes
        Free-form annotations, such as which wrappers or compatibility flags
        were active. A reward-shaping change or a parity flag must be visible
        in the bundle, or two runs that are not comparable will look
        comparable.
    """

    source_fingerprint: str
    spec_digest: str
    split_indices: Mapping[str, tuple[int, ...]]
    n_scenarios: int = Field(ge=0)
    n_entities: int | None = Field(default=None, ge=0)
    framework_version: str
    created_at: datetime
    quality: Mapping[str, float] = Field(default_factory=dict)
    notes: Mapping[str, str] = Field(default_factory=dict)

    @property
    def split_sizes(self) -> dict[str, int]:
        """
        Number of scenarios in each split, derived from the recorded indices.

        Returns
        -------
        dict
            Mapping of split name to scenario count.
        """
        return {name: len(indices) for name, indices in self.split_indices.items()}


@dataclass(frozen=True, slots=True)
class TensorBatchData:
    """
    One split's data, as the gradient-based engines consume it.

    Parameters
    ----------
    loader
        An iterable of batches. Re-iterable: a training loop will traverse it
        once per epoch, so a bare generator is not acceptable.
    static
        Inputs constant across every batch, held once rather than collated per
        sample.

        This is the field that fixes a concrete defect. The previous
        implementation merged every static tensor into every sample, then
        compared them across the batch during collation and returned the
        first -- so an adjacency matrix was compared against itself
        ``batch_size`` times per batch, per epoch, to confirm something true
        by construction. Keeping static inputs here delivers the same tensors
        to the same place and deletes the comparison.
    n_samples
        Number of samples in the split, for metric denominators and reports.
    n_batches
        Number of batches per pass, or ``None`` if the source is unbounded.
    """

    loader: Iterable[Batch]
    static: Mapping[str, TensorLike] = field(default_factory=dict)
    n_samples: int = 0
    n_batches: int | None = None


@dataclass(frozen=True, slots=True)
class DataBundle[PayloadT]:
    """
    Everything a data build produces.

    Generic over the engine's payload type, which is what allows one pipeline
    to serve every engine. Every engine in this framework produces
    ``DataBundle[TensorBatchData]``; the parameter remains because an engine
    written outside it may carry something else, and a pipeline that named the
    concrete type would exclude it.

    Parameters
    ----------
    splits
        Payload per split name. ``train`` is required; the others are optional
        so a run with no held-out data -- a final refit on everything -- is
        expressible.
    signature
        The declared interface between this data and a model.
    state
        What was fitted while building the data, and what is needed again at
        inference time.
    lineage
        Where the data came from.
    entity_ids
        The entities the model was built against, in their training order,
        or ``None`` for a model with no entity axis -- which is most of them.

        Carried here rather than left on the prepared dataset because
        inference needs it and nothing else can supply it. Asking a model to
        predict for an instrument absent from training is a request that
        must be refused unless the model declares
        :class:`~rade_qnet.core.authoring.capabilities.Inductive`, and deciding
        whether an instrument was absent means knowing which were present.
        Without this field the only alternatives are to dig the list out of
        whichever fitted sub-state happens to hold it, or to not check --
        and not checking means returning a default embedding as though it
        were a prediction.

    Raises
    ------
    ContractError
        If no training split is present, or a split name is unrecognised.
    """

    splits: Mapping[str, PayloadT]
    signature: InputSignature
    state: FittedState
    lineage: DataLineage
    entity_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        """Validate the split names."""
        unknown = sorted(set(self.splits) - set(SPLIT_NAMES))
        if unknown:
            raise ContractError(
                f"unknown split name(s) {unknown}; expected a subset of {list(SPLIT_NAMES)}"
            )
        if "train" not in self.splits:
            raise ContractError(
                f"a data bundle requires a 'train' split; received {sorted(self.splits)}"
            )

    @property
    def split_names(self) -> tuple[str, ...]:
        """Present split names, in canonical order rather than insertion order."""
        return tuple(name for name in SPLIT_NAMES if name in self.splits)

    def split(self, name: str) -> PayloadT:
        """
        Return one split's payload.

        Parameters
        ----------
        name
            A split name.

        Returns
        -------
        PayloadT
            The engine payload for that split.

        Raises
        ------
        ContractError
            If the split is absent. The message lists what is present, since
            the usual cause is a configuration with no validation fraction
            meeting a callback that monitors validation loss.
        """
        try:
            return self.splits[name]
        except KeyError:
            raise ContractError(
                f"split {name!r} is not present in this bundle; "
                f"available splits are {list(self.split_names)}"
            ) from None

    def has_split(self, name: str) -> bool:
        """
        Return whether a split is present.

        Parameters
        ----------
        name
            A split name.

        Returns
        -------
        bool
            True if present.
        """
        return name in self.splits
