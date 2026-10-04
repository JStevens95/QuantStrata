# `src/rade_qnet/sources/dataset`

7 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 59 | 2621 | `bf7f3fb0f0814d67` |
| 2 | `cache.py` | 355 | 13232 | `a0eab4ed54f687d8` |
| 3 | `module.py` | 630 | 22367 | `1336d3633e002262` |
| 4 | `rebuild.py` | 250 | 9229 | `aa447d0e624c00f6` |
| 5 | `splits.py` | 553 | 21260 | `bcb1ada5e215aa5b` |
| 6 | `tables.py` | 473 | 16069 | `4aecedb17e4a109a` |
| 7 | `tabular.py` | 373 | 12287 | `329f5b278aac36a3` |

---

## 1. `src/rade_qnet/sources/dataset/__init__.py`

2621 bytes · SHA-256 `bf7f3fb0f0814d67`

```python
"""
Fixed, finite training data.

A data module is responsible for turning raw inputs into a ``DataBundle``: the
split arrays, the state that was fitted while preparing them, the input
signature, and the lineage needed to reproduce the result.

Two rules are enforced here because getting them wrong is the most common
source of silently optimistic backtests:

1. **Fitting along the scenario (time) axis may only observe training rows.**
   Scalers, basis selection and any other time-axis estimator are fitted on
   train indices alone.
2. **Fitting along the entity axis may observe the full universe.**  An
   attribute encoder or a nearest-neighbour graph over instruments is not
   leakage -- instrument identity is known at decision time.  Conflating these
   two axes would wrongly reject legitimate models.

Modules
-------
``module.py``
    ``DataModule`` -- the base a model's data build subclasses.  Declares the
    stages (load, fit state, transform, split, signature, package) that the
    train pipeline drives.  The *contract*, with no implementation of it.
    [Phase 2]
``tabular.py``
    ``TabularDataModule`` -- one implementation of that contract, and the one
    most models use unchanged: read a file, scale, reduce, split, package.
    Separate from ``module.py`` so the abstraction can be read without the
    file handling, and the file handling changed without touching the
    abstraction.  [Phase 2]
``tables.py``
    Reading raw tabular data from disk, and ``fingerprint_source``, which
    decides what counts as "the input".  [Phase 2]
``cache.py``
    The content-addressed cache for expensive builds, and
    ``PreparedDataset`` -- the cacheable intermediate that sits between the
    expensive work and the engine-specific packaging.  Keyed on the source
    fingerprint from ``tables.py``, among other things; the two modules were
    one until that coupling was made an import rather than an adjacency.
    [Phase 2]
``splits.py``
    Split strategies: ``chronological``, ``purged_kfold``, ``grouped`` and
    ``explicit``.  Sequence-aware, so a window may not straddle a split
    boundary.  [Phase 2]
``rebuild.py``
    Reconstructing a dataset from a bundle's recorded lineage, which is what
    lets evaluation and inference run without the original build.  [Phase 5]
``transforms/``
    Fitted transforms with explicit inverses.  [Phase 2]

Planned modules
---------------
``transitions.py``
    Reads a stored transition table into a dataset source, which is how
    *offline* reinforcement learning reuses this package unchanged.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/sources/dataset/cache.py`

13232 bytes · SHA-256 `a0eab4ed54f687d8`

```python
"""
Caching the expensive result of preparing a dataset.

What the cache key covers, and why
----------------------------------
:class:`DatasetCache` is content-addressed on three things: the digest of the
source specification, the fingerprint of the raw input -- from
:func:`~rade_qnet.sources.dataset.tables.fingerprint_source` -- and the
framework version. Any of them changing invalidates the entry.

The framework version is in the key for a reason that is easy to leave out.
A cached build is the *output of this code*, not just of the spec and the
input. Fixing a leakage defect in the splitter changes what a correct build
looks like without changing either the spec or the data, so a cache keyed only
on those two would keep serving the leaky arrays after the fix shipped. That
failure is invisible: the run succeeds, the metrics look like yesterday's, and
nothing reports that the fix did not take effect.

What is cached is the *prepared* dataset -- transformed arrays, splits, fitted
state -- and not a :class:`~rade_qnet.core.contract.data.DataBundle`. A bundle
holds live loaders, which are not serialisable, and packaging prepared arrays
into loaders is cheap. Caching the stage before the cheap stage is the whole
point.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import DataLineage, SplitIndices
from ...core.contract.signature import InputSignature
from ...core.lifecycle.errors import BundleError, ContractError
from ...core.provenance.hashing import abbreviate_digest, digest_payload
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.state import FittedState

__all__ = ["DatasetCache", "PreparedDataset"]

_LOGGER = get_logger(__name__)

#: Filenames inside a cache entry. Part of the on-disk format.
_ARRAYS_FILENAME = "arrays.npz"
_METADATA_FILENAME = "metadata.json"
_STATE_DIRECTORY = "state"

#: Where a build's static inputs are cached, separate from the arrays file
#: so that a model's signature keys cannot collide with the fixed ones.
_STATIC_FILENAME = "static.npz"

#: Length of the cache subdirectory name. Long enough that a collision across
#: every build a person will ever inspect is not a practical concern.
_KEY_LENGTH = 16


@dataclass(frozen=True, slots=True)
class PreparedDataset:
    """
    A data build after the expensive work and before the engine-specific work.

    This is the cacheable unit: transformed arrays, the splits, the fitted
    state, the signature and the lineage. Turning it into a
    :class:`~rade_qnet.core.contract.data.DataBundle` means constructing loaders
    or array payloads, which is cheap and engine-specific -- so it happens on
    every run rather than being stored.

    Parameters
    ----------
    features
        Transformed feature matrix over the whole scenario axis, not split.
        Held whole because a sequence window near a split boundary needs rows
        from before that split's first index, and slicing per split first
        would make those rows unreachable.
    target
        Transformed target, aligned with ``features``.
    splits
        Which scenario indices belong to each split.
    signature
        The declared interface between this data and a model.
    state
        What was fitted while preparing the data.
    lineage
        Where the data came from.
    feature_names
        Column names after reduction, in column order.
    entity_ids
        Identifier per row, where the problem has an entity axis.
    static_inputs
        Inputs that are identical for every sample, keyed as the signature
        declares them. A graph is the motivating case: the adjacency does
        not vary by scenario, so collating a copy of it into every sample
        multiplies its memory by the batch size and its transfer cost by
        the number of batches -- defect 4. Empty for a model without any,
        which is most of them.

    Raises
    ------
    ContractError
        If the arrays disagree on row count, or a split index falls outside
        the scenario axis -- which would otherwise surface as a confusing
        out-of-bounds error during windowing.
    """

    features: NDArray[np.floating]
    target: NDArray[np.floating]
    splits: SplitIndices
    signature: InputSignature
    state: FittedState
    lineage: DataLineage
    feature_names: tuple[str, ...] | None = None
    entity_ids: tuple[str, ...] | None = None
    static_inputs: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate row alignment and index bounds."""
        n_rows = self.features.shape[0]
        if self.target.shape[0] != n_rows:
            raise ContractError(
                f"features have {n_rows} row(s) but target has {self.target.shape[0]}"
            )
        for name in ("train", "validation", "test"):
            indices = self.splits[name]
            if indices.size and (indices.min() < 0 or indices.max() >= n_rows):
                raise ContractError(
                    f"{name} split indices span [{indices.min()}, {indices.max()}] "
                    f"but the scenario axis has {n_rows} row(s)"
                )

    @property
    def n_scenarios(self) -> int:
        """Length of the scenario axis before splitting."""
        return int(self.features.shape[0])


class DatasetCache:
    """
    A content-addressed store of prepared datasets.

    Parameters
    ----------
    directory
        Root of the store. One subdirectory per entry, named for the key.
    enabled
        When false, every lookup misses and every write is skipped. A disabled
        cache rather than an absent one, so a caller never branches on whether
        caching is configured -- there is one code path and it is the cached
        one.
    """

    def __init__(self, directory: Path, *, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled

    def key(self, *, spec_digest: str, source_fingerprint: str, framework_version: str) -> str:
        """
        Return the cache key for a build.

        Parameters
        ----------
        spec_digest
            Digest of the source specification.
        source_fingerprint
            Digest of the raw input, from :func:`fingerprint_source`.
        framework_version
            Version of ``rade_qnet``. In the key for the reason given in the
            module docstring: a cached build is the output of this code, so a
            fix to the splitter must invalidate it.

        Returns
        -------
        str
            A short hexadecimal key, usable as a directory name.
        """
        return abbreviate_digest(
            digest_payload(
                {
                    "spec": spec_digest,
                    "source": source_fingerprint,
                    "version": framework_version,
                }
            ),
            length=_KEY_LENGTH,
        )

    def path_for(self, key: str) -> Path:
        """
        Return the directory an entry occupies.

        Parameters
        ----------
        key
            A cache key.

        Returns
        -------
        pathlib.Path
            The entry directory, which may not exist.
        """
        return self.directory / key

    def load(self, key: str, *, state_type: type[FittedState]) -> PreparedDataset | None:
        """
        Return a cached dataset, or ``None`` on a miss.

        A miss is a return value rather than an exception: missing is the
        normal first-run condition, and a caller that has to catch an
        exception for the normal case writes worse code.

        A *corrupt* entry is different, and is treated as a miss with a
        warning rather than as a failure. A half-written entry -- an
        interrupted run, a full disk -- should cost a rebuild, not a crash
        hours into a job set. The warning is what keeps it from being silent.

        Parameters
        ----------
        key
            A cache key, from :meth:`key`.
        state_type
            The concrete fitted-state class to load. Passed in rather than
            recorded on disk, for the reason given in
            :class:`~rade_qnet.core.contract.state.FittedState`.

        Returns
        -------
        PreparedDataset or None
            The cached dataset, or ``None``.
        """
        if not self.enabled:
            return None

        entry = self.path_for(key)
        if not (entry / _METADATA_FILENAME).is_file():
            _LOGGER.debug("dataset cache miss for key %s", key)
            return None

        try:
            return self._read_entry(entry, state_type=state_type)
        except (OSError, ValueError, KeyError, BundleError, ContractError) as error:
            _LOGGER.warning(
                "ignoring unreadable dataset cache entry %s (%s: %s); rebuilding",
                key,
                type(error).__name__,
                error,
            )
            return None

    def _read_entry(self, entry: Path, *, state_type: type[FittedState]) -> PreparedDataset:
        """
        Read one entry from disk.

        Separated from :meth:`load` so that the error handling there wraps a
        single call, and every failure mode inside it is treated the same way.

        Parameters
        ----------
        entry
            The entry directory.
        state_type
            The concrete fitted-state class to load.

        Returns
        -------
        PreparedDataset
            The stored dataset.
        """
        metadata = json.loads((entry / _METADATA_FILENAME).read_text(encoding="utf-8"))
        with np.load(entry / _ARRAYS_FILENAME, allow_pickle=False) as arrays:
            features = arrays["features"]
            target = arrays["target"]
            splits = SplitIndices(
                train=arrays["train"],
                validation=arrays["validation"],
                test=arrays["test"],
            )

        with np.load(entry / _STATIC_FILENAME, allow_pickle=False) as stored:
            static_inputs = {name: stored[name] for name in stored.files}

        names = metadata["feature_names"]
        entities = metadata["entity_ids"]
        return PreparedDataset(
            features=features,
            target=target,
            splits=splits,
            signature=InputSignature.model_validate(metadata["signature"]),
            state=state_type.load(entry / _STATE_DIRECTORY),
            lineage=DataLineage.model_validate(metadata["lineage"]),
            feature_names=tuple(names) if names is not None else None,
            entity_ids=tuple(entities) if entities is not None else None,
            static_inputs=static_inputs,
        )

    def save(self, key: str, dataset: PreparedDataset) -> None:
        """
        Write a dataset into the store.

        Writes with ``allow_pickle=False``, which is not a detail. A cache
        directory is an ordinary writable path, and loading a pickle from one
        executes whatever it contains -- so an entry that an attacker, or a
        stale incompatible version, could place there would run as code. The
        arrays here are plain numeric data and need nothing more.

        Parameters
        ----------
        key
            A cache key, from :meth:`key`.
        dataset
            The dataset to store.
        """
        if not self.enabled:
            return

        entry = self.path_for(key)
        entry.mkdir(parents=True, exist_ok=True)
        (entry / _STATE_DIRECTORY).mkdir(parents=True, exist_ok=True)

        np.savez(
            entry / _ARRAYS_FILENAME,
            features=dataset.features,
            target=dataset.target,
            train=dataset.splits.train,
            validation=dataset.splits.validation,
            test=dataset.splits.test,
        )
        # Written to their own file rather than into the arrays file,
        # because their names come from the model's signature and could
        # otherwise collide with `features`, `target` or a split name.
        np.savez(entry / _STATIC_FILENAME, **dict(dataset.static_inputs))
        dataset.state.save(entry / _STATE_DIRECTORY)

        metadata = {
            "signature": dataset.signature.model_dump(mode="json"),
            "lineage": dataset.lineage.model_dump(mode="json"),
            "feature_names": list(dataset.feature_names)
            if dataset.feature_names is not None
            else None,
            "entity_ids": list(dataset.entity_ids) if dataset.entity_ids is not None else None,
        }
        # Metadata is written last, and `load` keys presence off it.  That
        # ordering is what makes an interrupted write read as a miss rather
        # than as an entry with arrays and no description of them.
        (entry / _METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        _LOGGER.info("cached prepared dataset under key %s", key)
```

---

## 3. `src/rade_qnet/sources/dataset/module.py`

22367 bytes · SHA-256 `1336d3633e002262`

```python
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
```

---

## 4. `src/rade_qnet/sources/dataset/rebuild.py`

9229 bytes · SHA-256 `aa447d0e624c00f6`

```python
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
from ...core.lifecycle.errors import ContractError
from ...core.provenance.hashing import digest_spec
from ...core.provenance.logging import get_logger
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
```

---

## 5. `src/rade_qnet/sources/dataset/splits.py`

21260 bytes · SHA-256 `bcb1ada5e215aa5b`

```python
"""
Split strategies for the scenario axis -- every one of them sequence aware.

This is the most consequential module in the framework, and the reason is that
its failure mode is encouraging rather than alarming. A leaky split does not
raise; it produces a validation score better than the model deserves, and the
model then fails in production for reasons nobody can reconstruct.

Three properties every strategy here guarantees
-----------------------------------------------
**Disjoint.** No scenario index appears in two splits. Enforced again by
:class:`~rade_qnet.core.contract.data.SplitIndices`, so a strategy that got it
wrong cannot return.

**Chronologically ordered, where order means anything.** For the
chronological strategy, every training index precedes every validation index,
which precedes every test index.

**Window safe.** This is the property the implementation being replaced did
not have. A model consuming a window of ``sequence_length`` scenarios reads
indices ``[i - length + 1, i]`` to predict at ``i``. If ``i`` is the first
validation index, that window reaches back into training -- so the validation
score is partly a memory of what the model was fitted on.

The fix is a gap of ``sequence_length - 1`` scenarios discarded at each
boundary, plus any additional ``gap_scenarios`` the spec asks for. The two are
additive and both are needed:

- The *sequence* gap handles a window that looks **backwards** from its label.
- The *explicit* gap handles a target computed **forwards** from its index --
  a five-day forward return at scenario ``i`` is a function of ``i + 5``, so
  without a gap the last five training labels encode validation scenarios.

Why the gap is discarded rather than reassigned
-----------------------------------------------
Giving the boundary scenarios to the earlier split would leave the window
problem exactly where it was; giving them to the later one makes its first
windows depend on training data. The only correct answer is that those
scenarios belong to neither split, which costs ``sequence_length - 1`` rows
per boundary and buys a validation score that means what it says.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import SplitIndices
from ...core.lifecycle.errors import SpecError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.spec.data import (
        ChronologicalSplitSpec,
        ExplicitSplitSpec,
        GroupedSplitSpec,
        PurgedKFoldSplitSpec,
        SplitSpec,
    )

__all__ = [
    "boundary_gap",
    "split_by_group",
    "split_chronologically",
    "split_explicitly",
    "split_purged_kfold",
    "split_scenarios",
]

_LOGGER = get_logger(__name__)


def boundary_gap(sequence_length: int, gap_scenarios: int = 0) -> int:
    """
    Return how many scenarios must be discarded at each split boundary.

    Parameters
    ----------
    sequence_length
        Window length the model consumes. ``1`` for a non-sequential model,
        which needs no sequence gap at all.
    gap_scenarios
        Additional scenarios the specification asks to discard, for a target
        computed over a forward window.

    Returns
    -------
    int
        Total gap, the sum of the two contributions.

    Raises
    ------
    SpecError
        If either argument is negative, or ``sequence_length`` is zero. A
        window of zero scenarios is not a shorter window; it is a model with
        no input.
    """
    if sequence_length < 1:
        raise SpecError(
            f"sequence_length must be at least 1, received {sequence_length}; "
            f"use 1 for a non-sequential model"
        )
    if gap_scenarios < 0:
        raise SpecError(f"gap_scenarios cannot be negative, received {gap_scenarios}")
    # A window of length 1 reads only its own scenario, so it cannot straddle
    # anything and contributes no gap.
    return (sequence_length - 1) + gap_scenarios


def split_scenarios(
    spec: SplitSpec,
    *,
    n_scenarios: int,
    sequence_length: int = 1,
    group_labels: NDArray[np.int64] | None = None,
    seed: int = 0,
) -> SplitIndices:
    """
    Split a scenario axis using whichever strategy the specification names.

    The single entry point a data module calls. Dispatching here rather than
    in the data module means a new strategy is a new function and one branch,
    not a change to every data module in existence.

    Parameters
    ----------
    spec
        The split strategy, from the source specification.
    n_scenarios
        Length of the scenario axis.
    sequence_length
        Window length the model consumes, used to size the boundary gap.
    group_labels
        Group identifier per scenario, required by the grouped strategy and
        ignored by the others.
    seed
        Seed for the grouped strategy's group assignment. The other
        strategies are deterministic and ignore it.

    Returns
    -------
    SplitIndices
        Disjoint index arrays.

    Raises
    ------
    SpecError
        If the axis is too short for the requested split, or the strategy
        needs information that was not supplied.
    """
    if n_scenarios < 1:
        raise SpecError(f"cannot split an axis of {n_scenarios} scenarios; at least 1 is required")

    match spec.kind:
        case "chronological":
            return split_chronologically(
                spec, n_scenarios=n_scenarios, sequence_length=sequence_length
            )
        case "purged_kfold":
            return split_purged_kfold(spec, n_scenarios=n_scenarios)
        case "grouped":
            return split_by_group(
                spec, n_scenarios=n_scenarios, group_labels=group_labels, seed=seed
            )
        case "explicit":
            return split_explicitly(spec, n_scenarios=n_scenarios)
        case _:  # pragma: no cover - the discriminated union makes this unreachable
            raise SpecError(f"unknown split kind {spec.kind!r}")


def split_chronologically(
    spec: ChronologicalSplitSpec,
    *,
    n_scenarios: int,
    sequence_length: int = 1,
) -> SplitIndices:
    """
    Split by time order, with a gap at each boundary.

    Test takes the last ``test_fraction`` of the axis and validation the
    ``validation_fraction`` before it. Boundaries are cut on the full axis
    first, then **the gap is removed from the tail of the earlier split** at
    each boundary.

    Which side loses the rows
    -------------------------
    Taking the gap from the earlier split is the established treatment --
    purging training observations whose windows or labels reach into the
    held-out period -- and it is the right way round for two reasons.

    The held-out sets keep the size the configuration asked for, so a test
    score does not quietly become a score over fewer rows when the sequence
    length changes. And the alternative, trimming the *later* split's head,
    would discard the oldest held-out rows while leaving training rows that
    overlap them: it costs the same rows and removes none of the leakage.

    The consequence is that ``train`` and ``validation`` are each one gap
    narrower than their nominal fractions, and ``test`` is exactly its
    fraction. A validation fraction narrower than the gap is rejected rather
    than silently emptied.

    Parameters
    ----------
    spec
        Fractions and any additional gap.
    n_scenarios
        Length of the scenario axis.
    sequence_length
        Window length the model consumes.

    Returns
    -------
    SplitIndices
        Contiguous, ordered, disjoint index arrays.

    Raises
    ------
    SpecError
        If the gap leaves no training scenarios, or is wider than the
        requested validation split. The message reports the arithmetic,
        because the usual cause is a long sequence length on a short history
        and the fix depends on which of the two the user can change.
    """
    gap = boundary_gap(sequence_length, spec.gap_scenarios)

    n_test = round(n_scenarios * spec.test_fraction)
    n_validation = round(n_scenarios * spec.validation_fraction)

    # A non-zero fraction that rounds down to no scenarios is refused rather
    # than honoured.  Requesting no validation data is legitimate and is
    # expressed by setting the fraction to zero; asking for 15% of a
    # three-scenario axis and silently receiving none is not, because the run
    # then reports no held-out score while appearing to have succeeded.
    for name, fraction, count in (
        ("validation_fraction", spec.validation_fraction, n_validation),
        ("test_fraction", spec.test_fraction, n_test),
    ):
        if fraction > 0.0 and count == 0:
            raise SpecError(
                f"{name}={fraction} of {n_scenarios} scenario(s) rounds to zero, so "
                f"that split would be empty; use a longer history, or set {name}=0.0 "
                f"if an empty split is intended"
            )

    test_start = n_scenarios - n_test
    validation_start = test_start - n_validation

    test = np.arange(test_start, n_scenarios, dtype=np.int64)

    # A gap is only needed where two non-empty splits actually meet. With no
    # test split there is no validation/test boundary, so validation runs to
    # the end of the axis and loses nothing.
    validation_end = (test_start - gap) if n_test else n_scenarios
    if n_validation and validation_end <= validation_start:
        raise SpecError(
            f"validation_fraction={spec.validation_fraction} gives "
            f"{n_validation} scenario(s) out of {n_scenarios}, which is not wider "
            f"than the boundary gap of {gap} (sequence_length={sequence_length} + "
            f"gap_scenarios={spec.gap_scenarios}); the whole validation split would "
            f"be discarded. Increase validation_fraction, shorten the sequence, or "
            f"use a longer history"
        )
    validation = (
        np.arange(validation_start, validation_end, dtype=np.int64)
        if n_validation
        else np.empty(0, dtype=np.int64)
    )

    # Training ends one gap before whichever split follows it. With neither
    # held-out split present there is no boundary at all.
    if n_validation:
        train_end = validation_start - gap
    elif n_test:
        train_end = test_start - gap
    else:
        train_end = n_scenarios
    train = np.arange(0, max(0, train_end), dtype=np.int64)

    if train.size == 0:
        raise SpecError(
            f"a chronological split of {n_scenarios} scenarios with "
            f"validation_fraction={spec.validation_fraction}, "
            f"test_fraction={spec.test_fraction} and a boundary gap of {gap} "
            f"(sequence_length={sequence_length} + gap_scenarios="
            f"{spec.gap_scenarios}) leaves no training scenarios; "
            f"reduce the fractions, shorten the sequence, or use a longer history"
        )

    if gap:
        _LOGGER.debug(
            "chronological split discarded %d scenario(s) at each boundary to keep "
            "windows of length %d inside one split",
            gap,
            sequence_length,
        )
    return SplitIndices(train=train, validation=validation, test=test)


def split_purged_kfold(
    spec: PurgedKFoldSplitSpec,
    *,
    n_scenarios: int,
    fold: int = 0,
) -> SplitIndices:
    """
    Hold out one contiguous fold, with an embargo either side of it.

    Cross-validation on serially correlated data leaks in both directions
    across a fold boundary, which plain k-fold does nothing about. The embargo
    discards scenarios either side of the held-out fold so that neither the
    fold's first rows nor the training rows adjacent to it carry the other's
    information.

    One fold at a time
    ------------------
    This returns the split for a *single* fold rather than a list of folds,
    because the framework's unit of work is one run: a pipeline trains one
    model against one train/validation/test division. A cross-validated sweep
    is a job set over folds, which is Phase 4's concern and composes from this
    without change.

    Parameters
    ----------
    spec
        Fold count, embargo width and the final test fraction.
    n_scenarios
        Length of the scenario axis.
    fold
        Which fold to hold out as validation, zero based.

    Returns
    -------
    SplitIndices
        Training indices either side of the embargoed fold, the fold itself as
        validation, and the axis tail as test.

    Raises
    ------
    SpecError
        If ``fold`` is out of range, or the embargo consumes the training set.
    """
    if not 0 <= fold < spec.n_folds:
        raise SpecError(
            f"fold must be in [0, {spec.n_folds}), received {fold}; "
            f"the specification declares n_folds={spec.n_folds}"
        )

    # The test tail is removed from cross-validation entirely, so folds are cut
    # over the remainder. A test set that moved with the fold would make the
    # folds' scores incomparable.
    n_test = round(n_scenarios * spec.test_fraction)
    n_cross_validated = n_scenarios - n_test
    if n_cross_validated < spec.n_folds:
        raise SpecError(
            f"{n_cross_validated} scenarios remain after reserving "
            f"test_fraction={spec.test_fraction} of {n_scenarios}, which cannot be "
            f"divided into {spec.n_folds} folds; reduce n_folds or test_fraction"
        )

    fold_edges = np.linspace(0, n_cross_validated, spec.n_folds + 1).astype(np.int64)
    fold_start, fold_end = int(fold_edges[fold]), int(fold_edges[fold + 1])

    embargo = spec.embargo_scenarios
    validation = np.arange(fold_start, fold_end, dtype=np.int64)
    # Training is everything outside the fold *and* outside its embargo, which
    # is why this is two ranges rather than a slice.
    before = np.arange(0, max(0, fold_start - embargo), dtype=np.int64)
    after = np.arange(min(n_cross_validated, fold_end + embargo), n_cross_validated, dtype=np.int64)
    train = np.concatenate((before, after))
    test = np.arange(n_cross_validated, n_scenarios, dtype=np.int64)

    if train.size == 0:
        raise SpecError(
            f"fold {fold} of {spec.n_folds} with embargo_scenarios={embargo} leaves no "
            f"training scenarios out of {n_cross_validated}; reduce the embargo or "
            f"use more folds so each is smaller"
        )
    return SplitIndices(train=train, validation=validation, test=test)


def split_by_group(
    spec: GroupedSplitSpec,
    *,
    n_scenarios: int,
    group_labels: NDArray[np.int64] | None,
    seed: int = 0,
) -> SplitIndices:
    """
    Assign whole groups to splits, so no group spans two of them.

    The strategy for data where rows within a group are near-duplicates: the
    same trade observed at several tenors, or several scenarios sharing one
    event. Splitting such rows independently puts near-copies on both sides of
    the boundary, which is leakage that no amount of time ordering prevents.

    Groups are assigned by a seeded permutation rather than by label order,
    because label order is frequently meaningful -- group identifiers are
    often allocated chronologically -- and taking the last groups as test
    would silently turn this into a chronological split on a different axis.

    Parameters
    ----------
    spec
        Group key and the fractions, which are fractions *of groups*.
    n_scenarios
        Length of the scenario axis.
    group_labels
        Group identifier per scenario. Required.
    seed
        Seed for the group permutation.

    Returns
    -------
    SplitIndices
        Index arrays whose groups do not overlap.

    Raises
    ------
    SpecError
        If labels are absent, the wrong length, or there are too few groups
        for the requested fractions.
    """
    if group_labels is None:
        raise SpecError(
            f"a grouped split needs one label per scenario for group_key="
            f"{spec.group_key!r}, but none were supplied; the data module must "
            f"pass group_labels"
        )
    if group_labels.shape[0] != n_scenarios:
        raise SpecError(
            f"group_labels has {group_labels.shape[0]} entries but the scenario axis "
            f"has {n_scenarios}; they must correspond one to one"
        )

    groups = np.unique(group_labels)
    n_groups = groups.size
    n_test = round(n_groups * spec.test_fraction)
    n_validation = round(n_groups * spec.validation_fraction)

    # A fraction that rounds down to zero groups is refused rather than
    # honoured.  Asking for no validation data is legitimate and expressed by
    # setting the fraction to zero; *asking* for 15% of two groups and
    # receiving none is a different thing, and accepting it produces a run
    # whose validation split is empty, whose early stopping therefore never
    # fires, and which reports no held-out score while looking successful.
    for name, fraction, count in (
        ("validation_fraction", spec.validation_fraction, n_validation),
        ("test_fraction", spec.test_fraction, n_test),
    ):
        if fraction > 0.0 and count == 0:
            raise SpecError(
                f"{name}={fraction} of {n_groups} group(s) rounds to zero groups, so "
                f"that split would be empty; group more finely, or set {name}=0.0 if "
                f"an empty split is intended"
            )

    if n_groups - n_test - n_validation < 1:
        raise SpecError(
            f"{n_groups} group(s) cannot be divided with validation_fraction="
            f"{spec.validation_fraction} and test_fraction={spec.test_fraction}: "
            f"no training group would remain; reduce the fractions or group more coarsely"
        )

    # Seeded rather than global: a split must be reproducible from the run's
    # seed alone, and must not depend on NumPy's global random state, which
    # something else in the process may have reseeded.
    permuted = np.random.default_rng(seed).permutation(groups)
    test_groups = set(permuted[:n_test].tolist())
    validation_groups = set(permuted[n_test : n_test + n_validation].tolist())

    def indices_for(selected: set[int]) -> NDArray[np.int64]:
        """
        Return the scenario indices whose group is in a selected set.

        Parameters
        ----------
        selected
            Group labels assigned to one split.

        Returns
        -------
        numpy.ndarray
            Scenario indices, ascending.
        """
        if not selected:
            return np.empty(0, dtype=np.int64)
        membership = np.isin(group_labels, np.array(sorted(selected)))
        return np.flatnonzero(membership).astype(np.int64)

    test = indices_for(test_groups)
    validation = indices_for(validation_groups)
    held_out = test_groups | validation_groups
    train = np.flatnonzero(~np.isin(group_labels, np.array(sorted(held_out)))).astype(np.int64)
    if not held_out:
        train = np.arange(n_scenarios, dtype=np.int64)

    return SplitIndices(train=train, validation=validation, test=test)


def split_explicitly(spec: ExplicitSplitSpec, *, n_scenarios: int) -> SplitIndices:
    """
    Use the indices the specification supplies, verbatim.

    The strategy used for refactor parity: a golden fixture records the exact
    indices the previous implementation produced, and reproducing them removes
    the split from the list of things that could explain a difference.

    No gap is inserted. The caller supplied the indices, so the caller owns
    window safety -- which is why
    :class:`~rade_qnet.core.spec.run.SupervisedRunSpec` rejects an explicit
    split combined with a sequence length above one, rather than silently
    leaking here.

    Parameters
    ----------
    spec
        The supplied indices.
    n_scenarios
        Length of the scenario axis, used only to check the indices are in
        range.

    Returns
    -------
    SplitIndices
        The supplied indices, sorted.

    Raises
    ------
    SpecError
        If any index is outside the axis. Checked here because an out-of-range
        index from a stale fixture would otherwise surface as a confusing
        indexing error inside a transform.
    """
    supplied = {"train": spec.train, "validation": spec.validation, "test": spec.test}
    for name, values in supplied.items():
        out_of_range = sorted(value for value in values if not 0 <= value < n_scenarios)
        if out_of_range:
            raise SpecError(
                f"explicit {name} indices {out_of_range[:5]} are outside the scenario "
                f"axis [0, {n_scenarios}); the indices may have been captured from a "
                f"different dataset"
            )

    return SplitIndices(
        # Sorted so that downstream code may rely on ascending order, which the
        # other strategies produce naturally and a hand-written list may not.
        train=np.array(sorted(spec.train), dtype=np.int64),
        validation=np.array(sorted(spec.validation), dtype=np.int64),
        test=np.array(sorted(spec.test), dtype=np.int64),
    )
```

---

## 6. `src/rade_qnet/sources/dataset/tables.py`

16069 bytes · SHA-256 `4aecedb17e4a109a`

```python
"""
Reading raw tabular data from disk, and saying what was read.

Why CSV is parsed here instead of by pandas
-------------------------------------------
``rade_qnet`` does not depend on pandas. That is a deliberate cost: parsing CSV
with the standard library is more code than one ``read_csv`` call.

The reason is the framework's stated promise that a host which only reads a
bundle need not install a training stack -- and pandas is a large dependency to
impose on every consumer for the benefit of the one source kind that reads
files. A deployment that already has pandas loses nothing, because
:func:`read_table` dispatches Parquet through it when a Parquet path is
supplied, and says precisely what to install when it is absent.

The alternative -- depending on pandas and using it everywhere -- was rejected
rather than overlooked. It would be less code here and a heavier install for
everyone.

Why the fingerprint lives here
-------------------------------
:func:`fingerprint_source` is what :mod:`~rade_qnet.sources.dataset.cache`
keys its entries on, so the two modules are coupled -- and the coupling is
deliberately an import rather than an adjacency. What counts as "the input" is
a question about reading, answered once, here; the cache is one of its
callers. While both halves shared a module that relationship was true and
invisible, which is the kind of thing that survives a refactor by accident
rather than on purpose.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np

from ...core.lifecycle.errors import BundleError, ContractError, SpecError
from ...core.provenance.hashing import digest_file, digest_payload
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ["TableData", "fingerprint_source", "read_table"]

_LOGGER = get_logger(__name__)

#: File suffixes handled natively, by the standard library CSV reader.
_DELIMITED_SUFFIXES: Mapping[str, str] = {".csv": ",", ".tsv": "\t", ".txt": ","}

#: File suffixes handled through pandas, when it is installed.
_PARQUET_SUFFIXES = frozenset({".parquet", ".pq"})


@dataclass(frozen=True, slots=True)
class TableData:
    """
    One tabular file, read into arrays and still in its original units.

    Deliberately not a data frame. Everything downstream of here wants a float
    matrix, a target vector and a handful of string columns, and naming those
    four things in a type means a transform cannot quietly depend on a column
    that was never declared.

    Parameters
    ----------
    features
        Samples by features, in ``feature_names`` order.
    target
        One value per sample.
    feature_names
        Column names in column order. Carried rather than reconstructed,
        because a feature importance table reported by index is nearly
        useless to the person reading it.
    attributes
        Non-numeric columns kept aside, by name: an entity identifier, a group
        key for a grouped split. Strings rather than codes, because encoding
        them is a fitted transform's job and doing it during reading would put
        the fit in the wrong place.

    Raises
    ------
    ContractError
        If the row counts disagree. Checked here rather than left to the first
        transform, because a feature matrix misaligned with its target trains
        without complaint and learns a permutation.
    """

    features: NDArray[np.float64]
    target: NDArray[np.float64]
    feature_names: tuple[str, ...]
    attributes: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that every column has one entry per sample."""
        n_rows = self.features.shape[0]
        if self.target.shape[0] != n_rows:
            raise ContractError(
                f"features have {n_rows} row(s) but target has "
                f"{self.target.shape[0]}; they must correspond one to one"
            )
        if len(self.feature_names) != self.features.shape[1]:
            raise ContractError(
                f"feature_names has {len(self.feature_names)} entry(ies) but "
                f"features has {self.features.shape[1]} column(s)"
            )
        for name, values in self.attributes.items():
            if len(values) != n_rows:
                raise ContractError(
                    f"attribute column {name!r} has {len(values)} value(s) but "
                    f"there are {n_rows} row(s)"
                )

    @property
    def n_rows(self) -> int:
        """Number of samples."""
        return int(self.features.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns."""
        return int(self.features.shape[1])

    def attribute(self, name: str) -> tuple[str, ...]:
        """
        Return one attribute column.

        Parameters
        ----------
        name
            Column name.

        Returns
        -------
        tuple of str
            One value per row.

        Raises
        ------
        ContractError
            If the column was not read. The message lists what was read,
            because the usual cause is a grouped split naming a ``group_key``
            that the reader was never told to keep.
        """
        try:
            return self.attributes[name]
        except KeyError:
            raise ContractError(
                f"attribute column {name!r} was not read; columns kept aside are "
                f"{sorted(self.attributes)}. Name it in attribute_columns when "
                f"reading if a split or transform needs it"
            ) from None


def read_table(
    path: Path,
    *,
    target_column: str = "target",
    feature_columns: Sequence[str] | None = None,
    attribute_columns: Sequence[str] = (),
) -> TableData:
    """
    Read a tabular file into a :class:`TableData`.

    Parameters
    ----------
    path
        File to read. The suffix selects the reader.
    target_column
        Column holding the target.
    feature_columns
        Columns to use as features, in the order given. ``None`` means every
        remaining numeric column, sorted by name -- sorted rather than
        file order, so two files with the same columns in a different order
        produce the same matrix and therefore the same cache key.
    attribute_columns
        Columns to keep aside as strings.

    Returns
    -------
    TableData
        The parsed table.

    Raises
    ------
    SpecError
        If the suffix is not supported, or a named column is absent.
    BundleError
        If the file does not exist.
    """
    if not path.is_file():
        raise BundleError(f"no such file: {path}")

    suffix = path.suffix.lower()
    if suffix in _DELIMITED_SUFFIXES:
        columns = _read_delimited(path, delimiter=_DELIMITED_SUFFIXES[suffix])
    elif suffix in _PARQUET_SUFFIXES:
        columns = _read_parquet(path)
    else:
        supported = sorted({*_DELIMITED_SUFFIXES, *_PARQUET_SUFFIXES})
        raise SpecError(
            f"cannot read {path.name}: suffix {suffix!r} is not supported (supported: {supported})"
        )

    return _assemble(
        columns,
        source=path.name,
        target_column=target_column,
        feature_columns=feature_columns,
        attribute_columns=attribute_columns,
    )


def _read_delimited(path: Path, *, delimiter: str) -> dict[str, list[str]]:
    """
    Read a delimited text file into a column-per-key mapping of raw strings.

    Values are left as strings so that one place -- :func:`_assemble` --
    decides what is numeric and what is an attribute. Converting during the
    read would mean deciding before the caller's column choices are known.

    Parameters
    ----------
    path
        File to read.
    delimiter
        Field separator.

    Returns
    -------
    dict
        Column name to raw values.

    Raises
    ------
    SpecError
        If the file has no header row, or a row's field count disagrees with
        the header. A short row usually means an unquoted delimiter inside a
        value, and silently padding it would shift every subsequent column.
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise SpecError(f"{path.name} is empty; expected a header row") from None

        columns: dict[str, list[str]] = {name.strip(): [] for name in header}
        if len(columns) != len(header):
            duplicated = sorted({name for name in header if header.count(name) > 1})
            raise SpecError(
                f"{path.name} has duplicate column name(s) {duplicated}; "
                f"column names must be unique"
            )

        names = list(columns)
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue  # A trailing blank line is not an error.
            if len(row) != len(names):
                raise SpecError(
                    f"{path.name} line {line_number} has {len(row)} field(s) but "
                    f"the header declares {len(names)}; check for an unquoted "
                    f"{delimiter!r} inside a value"
                )
            for name, value in zip(names, row, strict=True):
                columns[name].append(value.strip())

    return columns


def _read_parquet(path: Path) -> dict[str, list[str]]:
    """
    Read a Parquet file through pandas, if pandas is installed.

    Parameters
    ----------
    path
        File to read.

    Returns
    -------
    dict
        Column name to values as strings, matching the delimited reader so
        that :func:`_assemble` has one input shape to handle.

    Raises
    ------
    SpecError
        If pandas is not installed, naming what to install. An import error
        from six frames down would not tell the reader that the fix is a pip
        install and that CSV would work without one.
    """
    if find_spec("pandas") is None:
        raise SpecError(
            f"reading {path.name} needs pandas, which rade_qnet does not require "
            f"(see the note in sources.dataset.tables). Install it with "
            f"'pip install pandas pyarrow', or supply the data as CSV"
        )

    import pandas  # noqa: PLC0415  Imported lazily: see the check above.

    frame = pandas.read_parquet(path)
    return {str(name): [str(value) for value in frame[name]] for name in frame.columns}


def _assemble(
    columns: Mapping[str, Sequence[str]],
    *,
    source: str,
    target_column: str,
    feature_columns: Sequence[str] | None,
    attribute_columns: Sequence[str],
) -> TableData:
    """
    Turn raw string columns into a :class:`TableData`.

    Parameters
    ----------
    columns
        Column name to raw values.
    source
        Filename, for error messages.
    target_column
        Column holding the target.
    feature_columns
        Chosen feature columns, or ``None`` to infer.
    attribute_columns
        Columns to keep aside as strings.

    Returns
    -------
    TableData
        The assembled table.

    Raises
    ------
    SpecError
        If a named column is absent, if a chosen feature column is not
        numeric, or if no feature column remains.
    """
    available = sorted(columns)
    _require_columns([target_column, *attribute_columns], available=available, source=source)

    if feature_columns is None:
        # Inferred: everything that is neither the target nor an attribute and
        # that parses as numbers.  Sorted for a stable cache key.
        candidates = sorted(set(columns) - {target_column, *attribute_columns})
        chosen = tuple(name for name in candidates if _parse_floats(columns[name]) is not None)
        if not chosen:
            raise SpecError(
                f"{source}: no numeric feature column found among {candidates}; "
                f"name the feature columns explicitly if they need conversion first"
            )
        _LOGGER.debug("inferred %d feature column(s) from %s: %s", len(chosen), source, chosen)
    else:
        _require_columns(feature_columns, available=available, source=source)
        chosen = tuple(feature_columns)

    feature_arrays = []
    for name in chosen:
        values = _parse_floats(columns[name])
        if values is None:
            raise SpecError(
                f"{source}: feature column {name!r} is not numeric; drop it, or "
                f"keep it aside as an attribute column and encode it"
            )
        feature_arrays.append(values)

    target = _parse_floats(columns[target_column])
    if target is None:
        raise SpecError(f"{source}: target column {target_column!r} is not numeric")

    # `column_stack` on an empty list raises; the no-column case is already
    # refused above, so reaching here guarantees at least one column.
    return TableData(
        features=np.column_stack(feature_arrays),
        target=target,
        feature_names=chosen,
        attributes={name: tuple(columns[name]) for name in attribute_columns},
    )


def _require_columns(requested: Sequence[str], *, available: Sequence[str], source: str) -> None:
    """
    Raise if any requested column is absent.

    Parameters
    ----------
    requested
        Column names the caller asked for.
    available
        Column names present in the file.
    source
        Filename, for the error message.

    Raises
    ------
    SpecError
        Naming the absent columns and listing what is present, because the
        usual cause is a typo or a renamed upstream column and the fix is
        obvious once the real names are visible.
    """
    missing = sorted(set(requested) - set(available))
    if missing:
        raise SpecError(
            f"{source}: column(s) {missing} are not present; the file has {list(available)}"
        )


def _parse_floats(values: Sequence[str]) -> NDArray[np.float64] | None:
    """
    Parse a column as floats, or return ``None`` if it is not numeric.

    An empty field becomes ``nan`` rather than failing, because a missing
    value is a data condition the quality report is designed to surface,
    whereas a non-numeric *column* is a configuration mistake.

    Parameters
    ----------
    values
        Raw strings.

    Returns
    -------
    numpy.ndarray or None
        The parsed column, or ``None`` if any non-empty value is not a number.
    """
    parsed = np.empty(len(values), dtype=np.float64)
    for index, raw in enumerate(values):
        if raw == "":
            parsed[index] = np.nan
            continue
        try:
            parsed[index] = float(raw)
        except ValueError:
            return None
    return parsed


def fingerprint_source(path: Path | None, *, extra: Mapping[str, object] | None = None) -> str:
    """
    Return a digest identifying the raw input of a build.

    Hashes file *contents* rather than a path and modification time. A path is
    not the data -- two runs on two machines resolve the same path to different
    files -- and a modification time changes when nothing did, so a cache keyed
    on it misses after a checkout.

    Parameters
    ----------
    path
        The input file, or ``None`` for a build with no single file input --
        a model data module assembling several sources, which supplies
        ``extra`` instead.
    extra
        Additional material to fold in: query parameters, a list of upstream
        digests, a universe definition.

    Returns
    -------
    str
        Lowercase hexadecimal digest.
    """
    material: dict[str, object] = {"extra": dict(extra or {})}
    material["file"] = digest_file(path) if path is not None else None
    return digest_payload(material)
```

---

## 7. `src/rade_qnet/sources/dataset/tabular.py`

12287 bytes · SHA-256 `329f5b278aac36a3`

```python
"""
The standard data build: read a file, scale, reduce, split, package.

Most models need no data code at all, and this module is why. It supplies
every stage :class:`~rade_qnet.sources.dataset.module.DataModule` declares,
so a model's ``data.py`` can be three lines that hand back one of these --
which is what makes the short model definition in ``ARCHITECTURE.md`` section
1 possible, and what Phase 6's baselines rely on to keep the framework
honest.

The split from ``module.py``
-----------------------------
That module defines what a data build *must* do; this one is a single
implementation of it. They were one file, and separating them means the
abstraction can be read without the CSV handling, and the CSV handling can be
changed without touching the abstraction. The name is honest about its scope:
it reads tables. A build whose raw input is a graph or a stream writes its own
subclass and shares nothing here but the base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.signature import InputSignature, TensorSpec
from ...core.lifecycle.errors import SpecError
from ...core.provenance.logging import get_logger
from .module import ELEMENT_DTYPE, FEATURE_INPUT_NAME, DataModule
from .tables import TableData, read_table
from .transforms.composite import DatasetState
from .transforms.reduction import ReductionState
from .transforms.scaling import ScalingState

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.contract.state import FittedState
    from ...core.spec.data import SourceSpec, TabularSourceSpec

__all__ = ["DataModule", "TabularDataModule"]

_LOGGER = get_logger(__name__)


__all__ = ["TabularDataModule"]


class TabularDataModule(DataModule[TableData]):
    """
    The standard data build: read a file, scale, reduce, split, package.

    Supplies every stage, so a straightforward model needs no data code at
    all -- which is what makes the short model definition in
    ``ARCHITECTURE.md`` §1 possible, and what Phase 6's baselines rely on to
    keep the framework honest.
    """

    def load(self, spec: SourceSpec) -> TableData:
        """
        Read the tabular file named by the specification.

        Parameters
        ----------
        spec
            A :class:`~rade_qnet.core.spec.data.TabularSourceSpec`.

        Returns
        -------
        TableData
            The parsed table.

        Raises
        ------
        SpecError
            If the spec is not a tabular source, or names no path. The path is
            optional on the spec so that it is constructible with defaults for
            testing; this is the stage that notices, and it can name itself in
            the message.
        """
        tabular = self._tabular(spec)
        if tabular.path is None:
            raise SpecError(
                "a tabular source needs 'path' to be set; the specification was "
                "constructed with defaults and has no file to read"
            )
        return read_table(
            tabular.path,
            target_column=tabular.target_column,
            feature_columns=tabular.feature_columns,
            attribute_columns=self._attribute_columns(tabular),
        )

    def n_scenarios(self, raw: TableData) -> int:
        """
        Return the number of rows read.

        Parameters
        ----------
        raw
            The parsed table.

        Returns
        -------
        int
            Row count.
        """
        return raw.n_rows

    def group_labels(self, raw: TableData, spec: SourceSpec) -> NDArray[np.int64] | None:
        """
        Return integer group labels when a grouped split asks for them.

        Parameters
        ----------
        raw
            The parsed table.
        spec
            The source specification.

        Returns
        -------
        numpy.ndarray or None
            One label per row for a grouped split, otherwise ``None``.
        """
        if spec.split.kind != "grouped":
            return None
        values = raw.attribute(spec.split.group_key)
        # Sorted so the label a group receives depends on the set of groups and
        # not on which row happened to be read first.
        codes = {name: code for code, name in enumerate(sorted(set(values)))}
        return np.array([codes[value] for value in values], dtype=np.int64)

    def fit_state(
        self, raw: TableData, spec: SourceSpec, *, train_indices: NDArray[np.int64]
    ) -> DatasetState:
        """
        Fit scaling and reduction on training rows.

        Both receive the full matrix *and* the training indices, rather than a
        pre-sliced matrix. That signature is what makes the leakage rule
        unavoidable: there is no way to pass held-out rows in as though they
        were training rows.

        Parameters
        ----------
        raw
            The parsed table.
        spec
            The source specification.
        train_indices
            Scenario indices that may be observed.

        Returns
        -------
        DatasetState
            The composed state, with scaling owning the target inverse.
        """
        transforms = spec.transforms
        scaling = (
            ScalingState.fit(
                raw.features, raw.target, spec=transforms.scaling, train_indices=train_indices
            )
            if transforms.scaling.method != "none"
            else None
        )

        # Reduction is fitted on scaled features, because both of its methods
        # compare columns against each other and an unscaled comparison is
        # dominated by whichever column happens to have the largest units.
        scaled = scaling.transform_features(raw.features) if scaling else raw.features
        reduction = (
            ReductionState.fit(
                scaled, raw.target, spec=transforms.reduction, train_indices=train_indices
            )
            if transforms.reduction.method != "none"
            else None
        )
        return DatasetState.of(scaling=scaling, reduction=reduction)

    def transform(
        self, raw: TableData, state: FittedState
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Apply scaling then reduction to every row.

        Parameters
        ----------
        raw
            The parsed table.
        state
            The state from :meth:`fit_state`.

        Returns
        -------
        tuple
            Transformed features and target.

        Raises
        ------
        SpecError
            If the state is not the composed state this module fits, which
            would mean a subclass overrode one stage and not the other.
        """
        composed = self._composed(state)
        features, target = raw.features, raw.target
        if composed.has_part("scaling"):
            scaling = composed.part("scaling")
            features = scaling.transform_features(features)
            target = scaling.transform_targets(target)
        if composed.has_part("reduction"):
            features = composed.part("reduction").transform_features(features)
        return features, target

    def signature(
        self,
        spec: SourceSpec,
        *,
        features: NDArray[np.floating],
        state: FittedState,
    ) -> InputSignature:
        """
        Declare one dynamic feature input and a scalar target.

        The batch dimension is wildcarded and the feature dimension concrete,
        which is what lets the engine synthesise a dummy batch of any size to
        materialise a lazily shaped model.

        Parameters
        ----------
        spec
            The source specification, carrying the sequence length.
        features
            The transformed feature matrix.
        state
            Unused here: the feature count is read from the transformed
            matrix, which reduction has already narrowed, so the state would
            only restate what the matrix already shows. Accepted because the
            hook is shared with modules whose static inputs are fitted.

        Returns
        -------
        InputSignature
            The declared interface.
        """
        del state
        n_features = int(features.shape[1])
        length = spec.transforms.sequence.length
        # A length of one means a non-sequential model, and a window axis of
        # size one would make every such model carry a pointless dimension.
        shape: tuple[int | None, ...] = (
            (None, n_features) if length == 1 else (None, length, n_features)
        )
        return InputSignature(
            dynamic={
                FEATURE_INPUT_NAME: TensorSpec(
                    shape=shape,
                    dtype=ELEMENT_DTYPE,
                    description=f"{n_features} feature(s)"
                    + ("" if length == 1 else f" over a {length}-scenario window"),
                )
            },
            target=TensorSpec(shape=(None,), dtype=ELEMENT_DTYPE),
        )

    def feature_names(self, raw: TableData, state: FittedState) -> tuple[str, ...] | None:
        """
        Return column names, narrowed to whatever survived reduction.

        Parameters
        ----------
        raw
            The parsed table.
        state
            The fitted state.

        Returns
        -------
        tuple of str or None
            Column names in column order. ``None`` after a projection, where
            a column is a combination of inputs and no original name
            describes it.
        """
        composed = self._composed(state)
        if not composed.has_part("reduction"):
            return raw.feature_names

        reduction = composed.part("reduction")
        if reduction.method == "basis_selection":
            return tuple(raw.feature_names[index] for index in reduction.selected)
        # A principal component is a mixture of every input column, so naming
        # it after one of them would be worse than admitting there is no name.
        return None

    @staticmethod
    def _tabular(spec: SourceSpec) -> TabularSourceSpec:
        """
        Narrow a source spec to a tabular one.

        Parameters
        ----------
        spec
            The source specification.

        Returns
        -------
        TabularSourceSpec
            The same spec, narrowed.

        Raises
        ------
        SpecError
            If the spec names a different source kind.
        """
        if spec.kind != "tabular":
            raise SpecError(
                f"TabularDataModule requires a tabular source, received "
                f"kind={spec.kind!r}; a model source needs its own data module"
            )
        return spec

    @staticmethod
    def _composed(state: FittedState) -> DatasetState:
        """
        Narrow a fitted state to the composed state this module fits.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        DatasetState
            The same state, narrowed.

        Raises
        ------
        SpecError
            If it is some other state, which means ``fit_state`` was
            overridden without overriding the stages that read its result.
        """
        if not isinstance(state, DatasetState):
            raise SpecError(
                f"expected a DatasetState from fit_state, received "
                f"{type(state).__name__}; override transform and feature_names "
                f"too if fit_state produces a different state"
            )
        return state

    @staticmethod
    def _attribute_columns(spec: TabularSourceSpec) -> tuple[str, ...]:
        """
        Return the non-feature columns this build needs kept aside.

        Derived from the split strategy rather than configured separately, so
        a grouped split cannot be requested against a file whose group column
        was never read.

        Parameters
        ----------
        spec
            The tabular source specification.

        Returns
        -------
        tuple of str
            Column names to keep as strings.
        """
        if spec.split.kind == "grouped":
            return (spec.split.group_key,)
        return ()
```

