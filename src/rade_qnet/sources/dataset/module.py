"""
``DataModule`` -- the base a model's data build subclasses.

A data module turns raw inputs into a :class:`~.io.PreparedDataset`. It does
so through named stages that :meth:`DataModule.build` drives in a fixed order,
which is the mechanism that makes the leakage rule structural rather than
advisory: :meth:`DataModule.fit_state` is handed training indices and nothing
else, so a subclass cannot fit on everything by accident. It would have to go
out of its way.

Stage order, and a correction to the phase document
---------------------------------------------------
``PHASE_2_TORCH_ENGINE.md`` §2.1 draws the stages as
``load -> fit_state -> transform -> split -> signature -> package``.

That order cannot be implemented. ``fit_state`` must see training indices, and
training indices are what ``split`` produces -- so ``split`` has to run before
``fit_state``, not two stages after it. The order built here is::

    load -> split -> fit_state -> transform -> signature -> package

Splitting first is also free: every strategy needs only the length of the
scenario axis and, for a grouped split, a label per row. Both are known the
moment ``load`` returns, so moving ``split`` earlier costs nothing and makes
the dependency honest. The document's intent -- that fitting is downstream of
deciding what training data *is* -- is preserved exactly; only the drawing was
wrong.

What is a stage and what is a helper
------------------------------------
Each stage is a method a subclass may override independently, and
:meth:`DataModule.build` is deliberately not one of them. A model that needs a
genuinely different sequence should override ``build_data`` on its definition
and not use this base at all, rather than reorder stages underneath code that
assumes the order. The four customisation tiers in ``ARCHITECTURE.md`` exist
so that "I need a different order" has an answer that is not "subtly break the
contract".

Why the standard build is not here
-----------------------------------
``TabularDataModule`` -- the build that reads a file, scales, reduces, splits
and packages, and which most models use unchanged -- lives in
:mod:`~rade_qnet.sources.dataset.tabular`. The two were one module and the
separation is worth the import: this file is the *contract* a data build
satisfies, and the next one is *one implementation of it*. A reader asking
"what must my data build provide?" should not have to scroll past three
hundred lines of CSV handling to find out, and a reader changing how scaling
is applied should not be editing the file that defines the abstraction.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ... import __version__
from ...core.contract.data import DataLineage
from ...core.contract.signature import InputSignature
from ...core.provenance.hashing import digest_spec
from ...core.provenance.logging import get_logger
from ..batching.dataset import DatasetSource, sources_for
from .cache import DatasetCache, PreparedDataset
from .rebuild import RebuiltDataset, rebuild_dataset
from .splits import split_scenarios
from .tables import fingerprint_source
from .transforms.composite import DatasetState

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ...core.contract.data import SplitIndices
    from ...core.contract.state import FittedState
    from ...core.spec.data import SourceSpec

__all__ = ["DataModule"]

_LOGGER = get_logger(__name__)

#: Dtype the engines receive. Public because the standard build in
#: ``tabular.py`` declares it too, and declared once so the signature and
#: the arrays cannot disagree -- a signature saying ``float32`` over a ``float64`` array
#: fails inside a forward pass, a long way from the cause.
ELEMENT_DTYPE = "float32"

#: Name the standard tabular signature gives its single dynamic input.
FEATURE_INPUT_NAME = "features"

#: Stand-in directory for a disabled cache, which neither reads nor writes.
#: A disabled cache still needs a path to construct, and a named placeholder
#: makes an accidental write obvious in a directory listing.
_DISABLED_CACHE_DIRECTORY = Path(".rade_qnet_cache_disabled")


class DataModule[RawT](ABC):
    """
    Base for a model's data build.

    Generic over the raw representation a subclass loads, so a module reading
    CSV and a module assembling a graph both type-check without the base
    naming either. ``RawT`` is whatever :meth:`load` returns, and the
    framework never inspects it -- only the subclass's own stages do.

    Parameters
    ----------
    cache_directory
        Where prepared datasets are cached. ``None`` disables caching
        regardless of what the specification asks for, which is the right
        behaviour when no run root has been established yet.
    """

    #: The fitted-state class :meth:`fit_state` produces. Declared as a class
    #: attribute because the cache must know what to load back, and recording
    #: the class path on disk is the fragility ``FittedState`` avoids.
    state_type: type[FittedState] = DatasetState

    def __init__(self, *, cache_directory: Path | None = None) -> None:
        self.cache_directory = cache_directory

    # -- Stages a subclass implements -------------------------------------

    @abstractmethod
    def load(self, spec: SourceSpec) -> RawT:
        """
        Read the raw inputs.

        The only stage with no default, because it is the only one whose
        answer the framework cannot guess.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        RawT
            Whatever this module's later stages consume.
        """

    @abstractmethod
    def n_scenarios(self, raw: RawT) -> int:
        """
        Return the length of the scenario axis.

        Asked separately from :meth:`load` so that :meth:`split` can run
        before anything is fitted, which is what keeps training indices
        available to :meth:`fit_state`.

        Parameters
        ----------
        raw
            The loaded inputs.

        Returns
        -------
        int
            Number of scenarios.
        """

    @abstractmethod
    def fit_state(
        self, raw: RawT, spec: SourceSpec, *, train_indices: NDArray[np.int64]
    ) -> FittedState:
        """
        Fit every transform that needs fitting.

        Receives ``train_indices`` and is expected to respect them on the
        scenario axis. Fitting along the *entity* axis over the full universe
        is correct and expected -- the two axes differ, for the reason given
        in this package's docstring.

        Parameters
        ----------
        raw
            The loaded inputs.
        spec
            The source specification, carrying the transform settings.
        train_indices
            Scenario indices that may be observed on the time axis.

        Returns
        -------
        FittedState
            An instance of :attr:`state_type`.
        """

    @abstractmethod
    def transform(
        self, raw: RawT, state: FittedState
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Apply the fitted state to the whole scenario axis.

        Applied to every row, not only to training rows. That is not a
        leak: the *parameters* came from training rows alone, and a held-out
        row must be transformed with those parameters or the model would see
        it on a different scale from the one it was trained on.

        Parameters
        ----------
        raw
            The loaded inputs.
        state
            The state from :meth:`fit_state`.

        Returns
        -------
        tuple
            Transformed features and target, aligned by row.
        """

    @abstractmethod
    def signature(
        self,
        spec: SourceSpec,
        *,
        features: NDArray[np.floating],
        state: FittedState,
    ) -> InputSignature:
        """
        Declare the interface between this data and a model.

        Takes the fitted state as well as the transformed features, for the
        same reason :meth:`feature_names` does: fitting changes what the
        interface *is*. A module whose static inputs are themselves fitted --
        an encoded attribute matrix, a nearest-neighbour adjacency -- can only
        declare their shapes by consulting the state. A module whose index
        arrays must be computed *after* a basis selection can only compute
        them from the selected basis, which is also held in the state.

        Without the state here, such a module would have to stash it on
        ``self`` during :meth:`fit_state` and read it back, which makes the
        module stateful across stages: the second build on one instance would
        silently reuse the first build's state, and a cached dataset would
        declare a signature fitted to data it was not built from.

        Parameters
        ----------
        spec
            The source specification, carrying the sequence length.
        features
            The transformed feature matrix, for its column count.
        state
            The state fitted on the training rows, for any shape or index
            array that depends on what was fitted.

        Returns
        -------
        InputSignature
            Static inputs, dynamic inputs and the target.
        """

    # -- Stages with a usable default --------------------------------------

    def split(self, raw: RawT, spec: SourceSpec, *, seed: int = 0) -> SplitIndices:
        """
        Decide which scenarios belong to each split.

        Defaults to :func:`~.splits.split_scenarios`, which is sequence-aware:
        the gap it leaves at each boundary is derived from the sequence length,
        so a window beginning in the training split cannot reach into
        validation.

        Parameters
        ----------
        raw
            The loaded inputs.
        spec
            The source specification.
        seed
            Seed for the strategies that randomise -- grouped assignment.

        Returns
        -------
        SplitIndices
            Disjoint scenario indices per split.
        """
        return split_scenarios(
            spec.split,
            n_scenarios=self.n_scenarios(raw),
            sequence_length=spec.transforms.sequence.length,
            group_labels=self.group_labels(raw, spec),
            seed=seed,
        )

    def group_labels(self, raw: RawT, spec: SourceSpec) -> NDArray[np.int64] | None:
        """
        Return a group label per scenario, for a grouped split.

        ``None`` by default, which is correct for every strategy except
        ``grouped`` -- and :func:`~.splits.split_scenarios` raises a named
        error if a grouped split is requested without them, rather than
        falling back to a random split.

        Parameters
        ----------
        raw
            The loaded inputs.
        spec
            The source specification.

        Returns
        -------
        numpy.ndarray or None
            One label per scenario, or ``None``.
        """
        del raw, spec  # Unused in the default: see the docstring.
        return None

    def n_entities(self, raw: RawT) -> int | None:
        """
        Return the size of the entity universe, if the problem has one.

        Parameters
        ----------
        raw
            The loaded inputs.

        Returns
        -------
        int or None
            Number of entities, or ``None``.
        """
        del raw
        return None

    def entity_ids(self, raw: RawT) -> tuple[str, ...] | None:
        """
        Return an entity identifier per row, if the problem has one.

        Carried into the prepared dataset so a prediction can be attributed
        back to a real instrument, which every downstream report needs and
        none can reconstruct.

        Parameters
        ----------
        raw
            The loaded inputs.

        Returns
        -------
        tuple of str or None
            One identifier per row, or ``None``.
        """
        del raw
        return None

    def source_fingerprint(self, spec: SourceSpec) -> str:
        """
        Return a digest identifying the raw input.

        Override when a build draws on something other than a single file: a
        database query, a universe definition, several upstream artifacts.
        Whatever changing would change the correct output belongs in here, or
        the cache will serve a stale build after it changes.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        str
            Lowercase hexadecimal digest.
        """
        path = getattr(spec, "path", None)
        return fingerprint_source(path if path is not None and path.is_file() else None)

    def lineage_notes(self, spec: SourceSpec) -> dict[str, str]:
        """
        Return annotations to record in the lineage.

        Flags that change what a run *means* belong here, so that two runs
        which are not comparable do not look comparable six months later. The
        default records the one such flag the framework itself owns: whether
        reduction was fitted on held-out rows.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        dict
            Annotations, as strings.
        """
        notes = {"reduction_fit_on": spec.transforms.reduction.fit_on}
        if spec.transforms.reduction.fit_on == "all":
            notes["leakage_warning"] = (
                "reduction was fitted on all rows including held-out data; "
                "metrics from this run are not comparable with a correct run"
            )
        return notes

    # -- The driver --------------------------------------------------------

    def build(self, spec: SourceSpec, *, seed: int = 0) -> PreparedDataset:
        """
        Drive every stage in order and return the prepared dataset.

        Consults the cache first, and writes to it afterwards. The cache is
        always present -- disabled rather than absent when caching is off --
        so there is one code path and it is the cached one.

        Parameters
        ----------
        spec
            The source specification.
        seed
            Seed for the stages that randomise.

        Returns
        -------
        PreparedDataset
            Transformed arrays, splits, fitted state, signature and lineage.
        """
        cache = self._cache(spec)
        spec_digest = digest_spec(spec)
        fingerprint = self.source_fingerprint(spec)
        key = cache.key(
            spec_digest=spec_digest,
            source_fingerprint=fingerprint,
            framework_version=__version__,
        )

        cached = cache.load(key, state_type=self.state_type)
        if cached is not None:
            _LOGGER.info("reusing cached dataset %s", key)
            return cached

        raw = self.load(spec)
        splits = self.split(raw, spec, seed=seed)
        state = self.fit_state(raw, spec, train_indices=splits.train)
        features, target = self.transform(raw, state)
        signature = self.signature(spec, features=features, state=state)

        _LOGGER.info(
            "built dataset: %d scenario(s), %d feature(s), splits %s",
            features.shape[0],
            features.shape[1] if features.ndim > 1 else 1,
            splits.sizes,
        )

        dataset = PreparedDataset(
            features=features,
            target=target,
            splits=splits,
            signature=signature,
            state=state,
            lineage=DataLineage(
                source_fingerprint=fingerprint,
                spec_digest=spec_digest,
                split_indices=splits.as_lineage(),
                n_scenarios=int(features.shape[0]),
                n_entities=self.n_entities(raw),
                framework_version=__version__,
                created_at=datetime.now(UTC),
                notes=self.lineage_notes(spec),
            ),
            static_inputs=self.static_inputs(raw, state),
            feature_names=self.feature_names(raw, state),
            entity_ids=self.entity_ids(raw),
        )
        cache.save(key, dataset)
        return dataset

    def rebuild(
        self, spec: SourceSpec, *, state: FittedState, lineage: DataLineage
    ) -> RebuiltDataset:
        """
        Rebuild a dataset the way a saved model already saw it.

        The counterpart to :meth:`build`, for evaluating or serving a bundle
        that was trained earlier. Where ``build`` splits the data and fits
        the state, this reads both from the bundle -- so a change to split
        logic cannot silently re-score an old model against different rows,
        and a scaler cannot be re-fitted on live data under a model that
        learned a response to training-scaled inputs.

        Deliberately not cached. :meth:`build`'s cache is keyed on the source
        spec and fingerprint, which say nothing about *which bundle's* state
        was applied, so two rebuilds of the same source under two different
        models would collide. The expensive stage, reading the raw inputs, is
        the same one ``build`` caches anyway.

        Parameters
        ----------
        spec
            The source specification to read from. Normally the bundle's
            own; a different one re-scores against other inputs, which the
            result reports rather than hides.
        state
            The fitted state loaded from the bundle. Applied, never fitted.
        lineage
            The bundle's lineage, which supplies the split.

        Returns
        -------
        RebuiltDataset
            The dataset, and whether the source has changed since the model
            was trained.
        """
        return rebuild_dataset(self, spec, state=state, lineage=lineage)

    def batch_sources(
        self, prepared: PreparedDataset, spec: SourceSpec, *, seed: int = 0
    ) -> dict[str, DatasetSource]:
        """
        Build one batch source per non-empty split.

        Lives on the data module rather than on the model definition because
        batching is a property of how the data was built, and because
        ``core`` may not import ``sources`` -- so a model base in ``core``
        could not construct a source even though it needs one. Asking the data
        module, which is already in ``sources``, keeps the layering intact and
        leaves a model with nothing to write.

        Override when a model needs several dynamic inputs or an entity axis.

        Parameters
        ----------
        prepared
            The prepared dataset from :meth:`build`.
        spec
            The source specification, carrying the loader and sequence
            settings.
        seed
            Base seed for the batch order.

        Returns
        -------
        dict
            Split name to source, in canonical split order.
        """
        return sources_for(
            prepared,
            loader=spec.loader,
            sequence=spec.transforms.sequence,
            seed=seed,
        )

    def static_inputs(self, raw: RawT, state: FittedState) -> Mapping[str, NDArray[np.generic]]:
        """
        Return inputs that are identical for every sample.

        A hook rather than something the base can derive, because only the
        module knows which of the things it built are per-sample and which
        are not. A graph model's adjacency is the motivating case: it does
        not vary by scenario, so collating it per sample would multiply its
        memory by the batch size and its host-to-device transfer by the
        number of batches, for no information gained.

        The keys must be ones the signature declares as static. Anything
        else is rejected when the engine uploads them, which is the right
        place for it: the complaint is about a disagreement between the
        module and its own signature.

        Parameters
        ----------
        raw
            The loaded inputs.
        state
            The fitted state.

        Returns
        -------
        Mapping
            Static inputs by name. Empty by default, which is correct for
            every module whose inputs all vary by sample.
        """
        del raw, state
        return {}

    def feature_names(self, raw: RawT, state: FittedState) -> tuple[str, ...] | None:
        """
        Return the column names of the transformed matrix.

        Takes the state as well as the raw inputs because reduction changes
        which columns exist: the names after a basis selection are a subset of
        the names before it, and reporting the pre-reduction names against
        post-reduction columns mislabels every importance figure.

        Parameters
        ----------
        raw
            The loaded inputs.
        state
            The fitted state.

        Returns
        -------
        tuple of str or None
            Column names, or ``None`` when the module does not track them.
        """
        del raw, state
        return None

    def _cache(self, spec: SourceSpec) -> DatasetCache:
        """
        Return the cache this build should use.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        DatasetCache
            Enabled only when the spec asks for it *and* a directory is
            available. A spec requesting a cache with nowhere to put it is
            warned about rather than failing the run, because an uncached
            build is slower and still correct.
        """
        directory = spec.cache.directory or self.cache_directory
        if spec.cache.enabled and directory is None:
            _LOGGER.warning(
                "cache.enabled is set but no cache directory is configured; "
                "building without a cache"
            )
        enabled = spec.cache.enabled and directory is not None
        return DatasetCache(directory or _DISABLED_CACHE_DIRECTORY, enabled=enabled)
