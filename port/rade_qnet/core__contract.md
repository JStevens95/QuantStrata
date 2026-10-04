# `src/rade_qnet/core/contract`

9 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 58 | 2574 | `03e5e8f82e62f796` |
| 2 | `base.py` | 65 | 2674 | `81ce5965d3e537f2` |
| 3 | `bundle.py` | 311 | 10031 | `f89d48966cd37a1d` |
| 4 | `data.py` | 376 | 13517 | `7683689102e6d630` |
| 5 | `requirement.py` | 442 | 16835 | `b0fd78b19bc479cd` |
| 6 | `result.py` | 641 | 21578 | `42cbfb1d3debd019` |
| 7 | `signature.py` | 386 | 12855 | `fcb7ccefc6bb9043` |
| 8 | `source.py` | 176 | 6275 | `0f58535c7104056f` |
| 9 | `state.py` | 204 | 6774 | `c9795030254a3196` |

---

## 1. `src/rade_qnet/core/contract/__init__.py`

2574 bytes · SHA-256 `03e5e8f82e62f796`

```python
"""
Typed payloads passed between pipeline stages.

A contract is the promise one stage makes to the next.  Because the promise is
a declared type rather than a convention, a stage can be overridden, cached,
replayed or executed in another process without the neighbouring stages
needing to change.

Contracts are generic over the engine where it matters: a PyTorch run carries
``TensorBatchData``, as every other engine's does, and all of them travel
inside the same ``DataBundle`` and through the same pipeline.

Which modules are pydantic and which are dataclasses is governed by one rule,
stated and justified in ``base.py``: pydantic where the contract must
round-trip through JSON, a frozen dataclass where it holds live data.

Modules
-------
``base.py``
    ``ContractModel``, the strict immutable pydantic base for contract
    metadata, and the documented policy on how validation failures surface.
    [Phase 1, delivered]
``signature.py``
    ``TensorSpec``, ``InputSignature`` (static inputs, dynamic inputs and the
    target) and ``PolicySignature`` (observation and action spaces).  A
    signature is what lets a model be rebuilt from a saved bundle without
    re-running the data build.  [Phase 1, delivered]
``state.py``
    ``FittedState`` -- the abstract base for everything a model fits during
    training and needs again at inference time.  Replaces ad-hoc sidecar files
    with one typed, self-describing object.  [Phase 1, delivered]
``data.py``
    ``DataBundle`` (splits, fitted state, signature, lineage), the engine-side
    payload ``TensorBatchData``, ``SplitIndices`` and
    ``DataLineage`` (source fingerprint, split indices, spec hash).
    [Phase 1, delivered]
``source.py``
    ``BatchSource``, the single protocol every training loop consumes.  This is
    the unification point: fixed datasets, on-policy rollouts, replay buffers
    and differentiable simulators all satisfy it.  [Phase 1, delivered]
``result.py``
    ``EpochRecord``, ``FitOutcome``, ``EvalResult``, ``TrainingResult`` and
    ``Predictions``.  [Phase 1, delivered]
``bundle.py``
    ``ModelBundle`` (what a finished run produces), ``SavedBundle`` (what it
    looks like on disk), ``Manifest`` and ``ManifestEntry``.
    [Phase 1, delivered]

Planned modules
---------------
``experience.py``
    ``Transition``, ``Trajectory`` and ``EpisodeStats`` -- the interactive
    learning payloads.  ``ActionResult`` lands here too, rather than in
    ``result.py``, because it is meaningless outside an interactive run.
    [Phase 7]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/core/contract/base.py`

2674 bytes · SHA-256 `81ce5965d3e537f2`

```python
"""
The base class for contract metadata.

Contracts come in two shapes, and the split is principled rather than
historical:

**Metadata contracts** are pure description -- shapes, dtypes, digests,
metrics, file lists. They are written to disk as JSON and read back months
later, so they need validation, exact round-tripping and immutability. Those
derive from :class:`ContractModel`.

**Payload contracts** carry live data -- arrays, loaders, a built model. They
are never serialised as a whole, they are constructed and consumed within one
process, and routing a loader through a validation layer would be pointless
overhead. Those are frozen dataclasses.

The rule is therefore: *pydantic where it must round-trip through JSON, a
dataclass where it holds live data.*

How validation failures surface
-------------------------------
The two shapes report failures differently, and the ``Raises`` sections in
this package reflect that rather than papering over it.

A **dataclass** validates in ``__post_init__`` and its exception propagates
unchanged, so it raises :class:`~rade_qnet.core.runtime.errors.ContractError`
(or ``SpecError`` for :class:`~rade_qnet.core.contract.data.SplitIndices`, where
the fault is a split specification rather than a payload) exactly as
documented.

A **contract model** validates through pydantic, which *collects* a validator
exception instead of letting it propagate -- so the caller sees
``ValidationError`` carrying the ``SpecError`` message and the dotted path of
the offending field. That is the right outcome here: unlike a run spec, a
contract model is constructed by framework code rather than parsed from user
input, so an invalid one is a bug and pydantic's report locates it precisely.
See :mod:`rade_qnet.core.spec.base` for the matching policy on specifications.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["ContractModel"]


class ContractModel(BaseModel):
    """
    Base class for contract metadata that is persisted as JSON.

    Carries the same immutability and strictness as a specification, for
    similar reasons: a stage must not be able to mutate a payload another
    stage is holding, and a manifest read from a six-month-old bundle must
    either validate or fail loudly rather than load with fields missing.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        # Contract metadata is description, never a live object. Keeping this
        # false makes "a signature cannot smuggle a tensor" structural rather
        # than a convention.
        arbitrary_types_allowed=False,
    )
```

---

## 3. `src/rade_qnet/core/contract/bundle.py`

10031 bytes · SHA-256 `f89d48966cd37a1d`

```python
"""
What a finished run produces, in memory and on disk.

A bundle is **self-describing**. Given only a bundle directory, the framework
can state which model produced it, from which spec, against which data
fingerprint, at which code version -- and can rebuild the model and invert its
target transforms without consulting the original run or the script that
launched it.

That property is what makes a six-month-old model usable rather than
archaeological, and it is why :class:`Manifest` carries a content hash for
every file. Corruption and silent drift then fail at load time with a message
naming the file, instead of surfacing as predictions that are subtly wrong.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import Field, model_validator

from ..runtime.errors import BundleError, SpecError
from ..spec.run import ReinforcementRunSpec, SupervisedRunSpec
from .base import ContractModel
from .data import DataLineage
from .result import TrainingResult
from .signature import InputSignature, PolicySignature
from .state import FittedState

__all__ = [
    "BUNDLE_SCHEMA_VERSION",
    "Manifest",
    "ManifestEntry",
    "ModelBundle",
    "SavedBundle",
]

#: Schema version of the on-disk bundle layout. Incremented when the layout
#: changes incompatibly, so a reader can refuse a bundle it cannot understand
#: rather than loading it partially.
BUNDLE_SCHEMA_VERSION = 1

#: Canonical file and directory names inside a bundle. Named constants rather
#: than literals scattered across the writer and the reader, because the two
#: disagreeing is a failure that only shows up at load time.
MANIFEST_FILENAME = "manifest.json"
SPEC_FILENAME = "spec.json"
SIGNATURE_FILENAME = "signature.json"
LINEAGE_FILENAME = "lineage.json"
RESULT_FILENAME = "result.json"
WEIGHTS_FILENAME = "weights.bin"
FITTED_STATE_DIRNAME = "fitted_state"


class ManifestEntry(ContractModel):
    """
    One file in a bundle, with its content hash.

    Parameters
    ----------
    relative_path
        Path relative to the bundle directory, in POSIX form so a bundle
        written on one platform verifies on another.
    sha256
        Hexadecimal digest of the file's contents.
    size_bytes
        File size, recorded so a truncated file is detectable without reading
        it.
    """

    relative_path: str
    sha256: str = Field(min_length=64, max_length=64)
    size_bytes: int = Field(ge=0)


class Manifest(ContractModel):
    """
    The bundle's own description of itself.

    Parameters
    ----------
    schema_version
        Layout version. A reader refuses a version it does not know.
    model_name
        Registered name of the model, which is how it is rebuilt.
    engine
        Registered name of the engine that trained it.
    version
        Monotonic version within the model and job, assigned by the catalog.
    created_at
        When the bundle was written.
    framework_version
        Version of ``rade_qnet`` that wrote it.
    spec_digest
        Digest of the run spec, so two bundles claiming the same
        configuration can be shown to have had it.
    job_id
        Job identifier within a job set, or ``None`` for a single run.
    files
        Every file in the bundle, with hashes.
    metrics
        Headline metrics, duplicated here so a catalog listing need not open
        the result file for every bundle it displays.
    tags
        Free-form labels for querying.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a file is listed more than once.
    """

    schema_version: int = Field(default=BUNDLE_SCHEMA_VERSION, ge=1)
    model_name: str
    engine: str
    version: int = Field(ge=1)
    created_at: datetime
    framework_version: str
    spec_digest: str
    job_id: str | None = None
    files: tuple[ManifestEntry, ...] = ()
    metrics: Mapping[str, float] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _check_files_are_unique(self) -> Manifest:
        """
        Reject a duplicated file path.

        Returns
        -------
        Manifest
            The validated manifest.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any path appears twice, which would make verification
            ambiguous about which hash is authoritative.
        """
        counts = Counter(entry.relative_path for entry in self.files)
        duplicates = sorted(path for path, count in counts.items() if count > 1)
        if duplicates:
            raise SpecError(f"manifest lists file(s) more than once: {duplicates}")
        return self

    @property
    def file_paths(self) -> tuple[str, ...]:
        """Every listed path, sorted."""
        return tuple(sorted(entry.relative_path for entry in self.files))

    def entry_for(self, relative_path: str) -> ManifestEntry:
        """
        Return the entry for one path.

        Parameters
        ----------
        relative_path
            Path relative to the bundle directory.

        Returns
        -------
        ManifestEntry
            The entry.

        Raises
        ------
        BundleError
            If the path is not listed.
        """
        for entry in self.files:
            if entry.relative_path == relative_path:
                return entry
        raise BundleError(
            f"{relative_path!r} is not listed in the manifest; "
            f"listed files are {list(self.file_paths)}"
        )

    @property
    def identifier(self) -> str:
        """
        A human-readable identifier, as used on the command line.

        Returns
        -------
        str
            ``model/job/vN``, or ``model/vN`` for a single run.
        """
        parts = [self.model_name]
        if self.job_id is not None:
            parts.append(self.job_id)
        parts.append(f"v{self.version}")
        return "/".join(parts)


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """
    A finished run, in memory, before or after being written.

    A dataclass rather than a contract model because it holds a live model
    object and a live fitted state. Its serialisable parts are written as
    separate files, each hashed in the manifest.

    Parameters
    ----------
    model
        The trained model. Typed as ``object`` because ``core`` cannot name an
        engine's model type; the engine that produced it knows what it is.
    state
        What was fitted beyond the model's parameters. A policy fits nothing
        beyond its parameters, so an interactive run supplies
        :class:`~.state.IdentityFittedState`.
    signature
        The declared interface of the thing that was trained, sufficient to
        rebuild it.

        An :class:`~.signature.InputSignature` for a supervised model --
        inputs and a target. A :class:`~.signature.PolicySignature` for a
        policy -- an observation space and an action space, and no target,
        because a policy has none.

        A union rather than one type, because this field means "what does the
        saved object expect", and for a policy the honest answer is the
        second. The alternative was to record the experience stream's input
        signature instead, which would have type-checked and been wrong in a
        way that only shows up much later: a tensor description loses the
        number of discrete actions and the bounds of a continuous space, so
        the policy could no longer be rebuilt from its own bundle -- the one
        thing the field exists for.

        Readers that only make sense for one paradigm check which they have
        and refuse the other by name; see ``orchestration.pipelines.reload``.
    spec
        The run specification that produced this bundle.
    lineage
        Where the experience came from. For a supervised run that is the
        dataset and its split; for an interactive one it is the environment
        and how much was collected, recorded in the same shape so two runs
        remain comparable.
    result
        Metrics and training history.
    manifest
        Set once the bundle has been written. ``None`` beforehand, which is
        how "has this been persisted?" is answered without touching the
        filesystem.
    """

    model: object
    state: FittedState
    signature: InputSignature | PolicySignature
    spec: SupervisedRunSpec | ReinforcementRunSpec
    lineage: DataLineage
    result: TrainingResult
    manifest: Manifest | None = None

    @property
    def is_persisted(self) -> bool:
        """Whether this bundle has been written to disk."""
        return self.manifest is not None


@dataclass(frozen=True, slots=True)
class SavedBundle:
    """
    A bundle on disk, located and verified.

    Parameters
    ----------
    directory
        The bundle directory.
    manifest
        The manifest read from it.
    """

    directory: Path
    manifest: Manifest

    @property
    def spec_path(self) -> Path:
        """Path to the saved run specification."""
        return self.directory / SPEC_FILENAME

    @property
    def signature_path(self) -> Path:
        """Path to the saved input signature."""
        return self.directory / SIGNATURE_FILENAME

    @property
    def lineage_path(self) -> Path:
        """Path to the saved data lineage."""
        return self.directory / LINEAGE_FILENAME

    @property
    def result_path(self) -> Path:
        """Path to the saved training result."""
        return self.directory / RESULT_FILENAME

    @property
    def weights_path(self) -> Path:
        """Path to the saved model weights."""
        return self.directory / WEIGHTS_FILENAME

    @property
    def fitted_state_directory(self) -> Path:
        """Directory holding the saved fitted state."""
        return self.directory / FITTED_STATE_DIRNAME
```

---

## 4. `src/rade_qnet/core/contract/data.py`

13517 bytes · SHA-256 `7683689102e6d630`

```python
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

from ..runtime.errors import ContractError, SpecError
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
        :class:`~rade_qnet.core.capability.protocols.Inductive`, and deciding
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
```

---

## 5. `src/rade_qnet/core/contract/requirement.py`

16835 bytes · SHA-256 `b0fd78b19bc479cd`

```python
"""
What a model consumes, declared by the model rather than inferred from data.

The gap this closes
-------------------
:class:`~rade_qnet.core.contract.signature.InputSignature` describes what a
data build *produced*. Until this module existed there was nothing
describing what a model *requires*, so the information only ever flowed one
way -- data to model -- and the model had to cope at runtime with whatever
arrived.

Coping looks reasonable and fails quietly. The recurrent baseline used to
read its window like this::

    for value in inputs.values():
        if isinstance(value, Tensor):
            return value

Handed two feature blocks, that trains on whichever one the data build's
dictionary happened to yield first. Reordering the build changes the model
and nothing reports it: no exception, no warning, a plausible loss curve
and a different model. The same shape of defect produced the
``torch.atleast_3d`` bug, where a wrongly placed time axis gave a tensor
the recurrence accepted happily and learned nonsense from.

Both are the same root cause. The model knew what it needed and had no way
to say so, so the framework could not check it.

What a requirement constrains
------------------------------
Names and ranks, which are the two properties a data build can get wrong
while still producing something that runs. Dtype is optional, and exact
shapes are deliberately not expressible: a model that works at any feature
width should not have to restate its width, and pinning one here would
duplicate a number the signature already carries and make the two able to
disagree.

Unnamed requirements
---------------------
A model like ``ridge`` or ``lstm_tabular`` is generic over what its input
is *called* -- the name is the source's choice. Such a model declares an
unnamed requirement, which matches any one input that no named requirement
claimed. One per group, because two anonymous requirements would be
indistinguishable and the matching would depend on iteration order, which
is the defect this module exists to prevent.

Examples
--------
A model that accepts whatever it is given::

    REQUIRES = InputRequirement.unconstrained()

A model needing exactly one dynamic input of rank 2 or 3, under any name::

    REQUIRES = InputRequirement(
        dynamic=(RequiredInput(rank=(2, 3), description="the feature window"),),
    )

A model needing specific inputs by name::

    REQUIRES = InputRequirement(
        dynamic=(RequiredInput(name="pnl_history", rank=3),),
        static=(RequiredInput(name="adjacency_values", rank=1),),
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import field_validator, model_validator

from ..runtime.errors import ContractError, SpecError
from .base import ContractModel
from .signature import TensorSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .signature import InputSignature

__all__ = ["InputRequirement", "RequiredInput"]


class RequiredInput(ContractModel):
    """
    One input a model consumes.

    Parameters
    ----------
    rank
        Acceptable numbers of axes, counting the batch axis. A tuple rather
        than a single integer because accepting more than one rank is a
        real and reasonable position -- the recurrent baseline treats a
        rank-2 input as a window of length one -- and because writing that
        down is better than discovering it from a reshape buried in a
        forward pass. A bare integer is accepted and widened.
    name
        The batch key. ``None`` means the model does not care what the
        input is called, which is the right declaration for a model that
        should run against whatever a user already has.
    dtype
        Required element type, as a library-agnostic string. ``None``
        leaves it unconstrained, which is the common case: an engine that
        casts on the way to the device makes the data build's choice
        irrelevant.
    shape
        Optional refinement of ``rank``, with ``None`` for any axis whose
        size the model does not care about. ``(None, None, 2)`` says "rank
        3, and the last axis is a pair" -- which is the useful case: a
        model that reads edge endpoints needs exactly two of them, and that
        is a genuine requirement rather than a restatement of the data.

        Most models should leave this unset. Pinning a feature width here
        duplicates a number the signature already carries, and two copies
        of a number can disagree.
    description
        What the model does with this input. Surfaced in the error message
        when the requirement is not met, where it is usually the thing that
        tells a user which of their columns belongs here.
    """

    rank: tuple[int, ...]
    name: str | None = None
    dtype: str | None = None
    shape: tuple[int | None, ...] | None = None
    description: str | None = None

    @field_validator("rank", mode="before")
    @classmethod
    def _widen_a_bare_rank(cls, value: object) -> object:
        """
        Accept ``rank=3`` as shorthand for ``rank=(3,)``.

        Parameters
        ----------
        value
            Whatever was supplied.

        Returns
        -------
        object
            A tuple if the input was a bare integer, otherwise unchanged.
        """
        return (value,) if isinstance(value, int) else value

    @model_validator(mode="after")
    def _check_ranks_are_usable(self) -> RequiredInput:
        """
        Reject an empty or non-positive rank set.

        Returns
        -------
        RequiredInput
            The validated requirement.

        Raises
        ------
        ValidationError
            Wrapping a :class:`SpecError`. An empty tuple would be a
            requirement nothing can satisfy, and a rank of zero would
            describe a scalar with no batch axis -- both are far more
            likely to be a typo than an intention.
        """
        if not self.rank:
            raise SpecError("a required input must accept at least one rank")
        for value in self.rank:
            if value < 1:
                raise SpecError(
                    f"rank {value} is not usable; every input has at least a "
                    f"batch axis, so the minimum rank is 1"
                )
        # A shape whose length is not an acceptable rank is a requirement
        # nothing can satisfy. Caught here rather than at check time, so the
        # author of the model hears about it instead of their user.
        if self.shape is not None and len(self.shape) not in self.rank:
            raise SpecError(
                f"shape {self.shape} has rank {len(self.shape)}, which is not "
                f"among the accepted ranks {self.rank}"
            )
        return self

    def describe(self) -> str:
        """
        Return a compact one-line description for error messages.

        Returns
        -------
        str
            For example ``'pnl_history' rank 3, float32 -- the window``.
        """
        name = f"{self.name!r}" if self.name is not None else "<any name>"
        ranks = " or ".join(str(value) for value in self.rank)
        parts = [f"{name} rank {ranks}"]
        if self.dtype is not None:
            parts.append(self.dtype)
        if self.description is not None:
            parts.append(f"-- {self.description}")
        return " ".join(parts)

    def accepts(self, spec: TensorSpec) -> str | None:
        """
        Return why this requirement rejects a tensor spec, or ``None``.

        A reason rather than a boolean, because the caller's job is to
        produce an error a user can act on, and "rank 2, expected 3" is
        actionable in a way that "incompatible" is not.

        Parameters
        ----------
        spec
            What the data build declared for this input.

        Returns
        -------
        str or None
            ``None`` if the spec satisfies the requirement.
        """
        if spec.rank not in self.rank:
            ranks = " or ".join(str(value) for value in self.rank)
            return f"has rank {spec.rank}, but rank {ranks} is required"
        if self.dtype is not None and spec.dtype != self.dtype:
            return f"has dtype {spec.dtype}, but {self.dtype} is required"
        if self.shape is not None:
            # Delegated to TensorSpec, which already implements exactly this
            # comparison: equal dtypes, equal ranks, and every dimension pair
            # either equal or wildcarded on at least one side. The dtype is
            # borrowed from the actual spec when the requirement does not
            # constrain it, so that the shape check stays a shape check.
            declared = TensorSpec(shape=self.shape, dtype=self.dtype or spec.dtype)
            if not declared.is_compatible_with(spec):
                return f"has shape {spec.describe()}, but {declared.describe()} is required"
        return None


class InputRequirement(ContractModel):
    """
    The complete set of inputs a model consumes.

    The dual of :class:`~rade_qnet.core.contract.signature.InputSignature`:
    one says what the data produced, this says what the model needs, and
    :meth:`check` is where the two meet.

    Parameters
    ----------
    dynamic
        Requirements on inputs that vary per sample.
    static
        Requirements on inputs constant across every batch -- a graph, an
        entity table.
    exact
        Whether an input the model did not ask for is an error. Default
        ``True``, which is the position worth defending: a data build
        producing something the model never reads is either wasted work or
        a sign that the user believes the model is using information it
        cannot see. The second is the dangerous one, because it is
        invisible in every metric.

        Set ``False`` for a model that is genuinely generic over its
        inputs, such as one that flattens everything it is handed.
    """

    dynamic: tuple[RequiredInput, ...] = ()
    static: tuple[RequiredInput, ...] = ()
    exact: bool = True

    @model_validator(mode="after")
    def _check_the_declaration_is_unambiguous(self) -> InputRequirement:
        """
        Reject duplicate names, or more than one unnamed entry per group.

        Returns
        -------
        InputRequirement
            The validated requirement.

        Raises
        ------
        ValidationError
            Wrapping a :class:`SpecError`. Two unnamed requirements in one
            group cannot be told apart, so which actual input satisfied
            which would depend on iteration order -- reintroducing exactly
            the order-dependence this type exists to remove.
        """
        for group, entries in (("dynamic", self.dynamic), ("static", self.static)):
            names = [entry.name for entry in entries if entry.name is not None]
            if len(names) != len(set(names)):
                raise SpecError(f"the {group} requirements name an input twice: {sorted(names)}")
            if sum(1 for entry in entries if entry.name is None) > 1:
                raise SpecError(
                    f"the {group} requirements contain more than one unnamed "
                    f"entry; unnamed requirements are indistinguishable, so "
                    f"which input satisfied which would depend on ordering"
                )
        return self

    @classmethod
    def unconstrained(cls) -> InputRequirement:
        """
        Return a requirement that accepts any signature.

        The honest declaration for a model that reads whatever it is given
        -- a linear model or a tree over a flattened matrix genuinely does
        not care how many blocks the data build emitted or what they are
        called.

        Preferred over leaving the requirement unset, because "this model
        has no constraints" and "nobody has written the constraints down"
        look identical from the outside and mean very different things.

        Returns
        -------
        InputRequirement
            Empty, and not exact.
        """
        return cls(exact=False)

    def check(self, signature: InputSignature, *, model: str) -> None:
        """
        Verify a data build produced what this model consumes.

        Called by the pipeline between building the data and building the
        model, which is the last moment at which a mismatch can be reported
        before it becomes a tensor of the wrong shape inside a forward pass.

        Parameters
        ----------
        signature
            What the data build declared.
        model
            The registered model name, for the error message.

        Raises
        ------
        ContractError
            If any requirement is unmet, listing every problem rather than
            the first. A user fixing a data build wants the whole list:
            reporting one mismatch per run turns a five-minute correction
            into five runs.
        """
        problems = [
            *self._problems(self.dynamic, signature.dynamic, group="dynamic"),
            *self._problems(self.static, signature.static, group="static"),
        ]
        if not problems:
            return
        raise ContractError(
            f"the data build does not match what {model!r} consumes:\n  "
            + "\n  ".join(problems)
            + "\n\nthe model declares:\n  "
            + "\n  ".join(self.describe())
        )

    def describe(self) -> tuple[str, ...]:
        """
        Return one line per declared input, for error messages and reports.

        Returns
        -------
        tuple of str
            Empty if the requirement is unconstrained.
        """
        return tuple(
            f"{group}: {entry.describe()}"
            for group, entries in (("dynamic", self.dynamic), ("static", self.static))
            for entry in entries
        )

    def _problems(
        self,
        required: tuple[RequiredInput, ...],
        actual: Mapping[str, TensorSpec],
        *,
        group: str,
    ) -> list[str]:
        """
        Return every way one group of inputs fails its requirements.

        Named requirements are matched first and by name, so that an
        unnamed requirement never consumes an input that something else
        asked for. Whatever is left over is then matched against the single
        unnamed requirement, if there is one.

        Parameters
        ----------
        required
            The declared requirements for this group.
        actual
            What the data build produced for this group.
        group
            ``"dynamic"`` or ``"static"``, for the message.

        Returns
        -------
        list of str
            One entry per problem, in a stable order.
        """
        problems: list[str] = []
        unclaimed = dict(actual)

        for entry in required:
            if entry.name is None:
                continue
            spec = unclaimed.pop(entry.name, None)
            if spec is None:
                problems.append(
                    f"{group} input {entry.name!r} is required but was not "
                    f"produced; the build produced {sorted(actual)}"
                )
                continue
            reason = entry.accepts(spec)
            if reason is not None:
                problems.append(f"{group} input {entry.name!r} {reason}")

        anonymous = next((e for e in required if e.name is None), None)
        if anonymous is not None:
            if len(unclaimed) != 1:
                problems.append(
                    f"exactly one unnamed {group} input is required "
                    f"({anonymous.describe()}), but {len(unclaimed)} were "
                    f"available: {sorted(unclaimed)}"
                )
                # Cleared so the exactness check below stays quiet: the
                # leftovers have already been reported, and saying the same
                # thing twice in different words makes a user wonder which
                # of the two problems they have.
                unclaimed.clear()
            else:
                name, spec = next(iter(unclaimed.items()))
                reason = anonymous.accepts(spec)
                if reason is not None:
                    problems.append(f"{group} input {name!r} {reason}")
                unclaimed.pop(name)

        if self.exact and unclaimed:
            problems.append(
                f"the build produced {group} input(s) {sorted(unclaimed)} that "
                f"the model does not consume; either the model is not reading "
                f"data the user believes it is reading, or the build is doing "
                f"work nothing uses"
            )
        return problems
```

---

## 6. `src/rade_qnet/core/contract/result.py`

21578 bytes · SHA-256 `42cbfb1d3debd019`

```python
"""
What the fit, evaluate and predict stages return.

One field recurs and is the most important thing in this module:
``in_original_units``. Every result that carries numbers a user will read
carries a flag asserting those numbers are in the units of the original target
rather than in whatever space the model happened to train in.

It is a field rather than a convention because it is checked. The conformance
suite asserts it is true on anything a pipeline reports, which turns "we
remember to invert the target scaling" from a habit into a property. A mean
absolute error of 0.03 is either excellent or meaningless depending on this
one boolean, and the number alone does not say which.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import Field, model_validator

from ..runtime.errors import ContractError, SpecError
from .base import ContractModel

__all__ = [
    "EpochRecord",
    "EvalResult",
    "EvaluationResult",
    "FitOutcome",
    "Predictions",
    "TrainingResult",
    "TrialRecord",
    "TuningResult",
]


class EpochRecord(ContractModel):
    """
    One epoch's outcome.

    Parameters
    ----------
    epoch
        Zero-based epoch index.
    train_loss
        Mean training loss over the epoch.
    val_loss
        Mean validation loss, or ``None`` when there is no validation split.
    metrics
        Any additional metrics computed for the epoch.
    learning_rate
        Rate in effect, recorded so a schedule's behaviour is visible in the
        saved history rather than having to be inferred from the loss curve.
    seconds
        Wall time for the epoch.
    """

    epoch: int = Field(ge=0)
    train_loss: float
    val_loss: float | None = None
    metrics: Mapping[str, float] = Field(default_factory=dict)
    learning_rate: float | None = Field(default=None, gt=0.0)
    seconds: float = Field(default=0.0, ge=0.0)


class FitOutcome(ContractModel):
    """
    What a fit produced, whatever engine performed it.

    A gradient loop over two hundred epochs and a single boosted-tree call both
    report through this type. The tree engine emits one record per boosting
    round, so a "learning curve" means the same thing for both and the same
    report renders either.

    Parameters
    ----------
    history
        One record per epoch or boosting round, in order.
    monitor
        Metric used to identify the best epoch.
    best_epoch
        Index of the best epoch, or ``None`` if nothing was monitored.
    best_monitor_value
        Value of the monitored metric at the best epoch.
    stopped_early
        Whether training ended before the epoch budget was exhausted.
    restored_best
        Whether the best epoch's parameters were restored at the end. When
        false, the reported metrics and the saved weights describe different
        models, so this is recorded rather than assumed.
    total_seconds
        Wall time for the whole fit.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if ``best_epoch`` does not identify a record
        in ``history``.
    """

    history: tuple[EpochRecord, ...] = ()
    monitor: str = "val_loss"
    best_epoch: int | None = Field(default=None, ge=0)
    best_monitor_value: float | None = None
    stopped_early: bool = False
    restored_best: bool = False
    total_seconds: float = Field(default=0.0, ge=0.0)

    @model_validator(mode="after")
    def _check_best_epoch_exists(self) -> FitOutcome:
        """
        Reject a best epoch that is not present in the history.

        Returns
        -------
        FitOutcome
            The validated outcome.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if ``best_epoch`` indexes no record. This catches an off-by-one
            between a callback's epoch counter and the history it recorded,
            which would otherwise surface as the wrong checkpoint being
            restored.
        """
        if self.best_epoch is None:
            return self
        if not any(record.epoch == self.best_epoch for record in self.history):
            recorded = [record.epoch for record in self.history]
            raise SpecError(
                f"best_epoch={self.best_epoch} is not present in the history "
                f"(recorded epochs: {recorded})"
            )
        return self

    @property
    def n_epochs(self) -> int:
        """Number of records in the history."""
        return len(self.history)

    @property
    def best_record(self) -> EpochRecord | None:
        """The best epoch's record, or ``None`` if there is no best epoch."""
        if self.best_epoch is None:
            return None
        return next(record for record in self.history if record.epoch == self.best_epoch)

    def curve(self, name: Literal["train_loss", "val_loss"]) -> tuple[float | None, ...]:
        """
        Return one series from the history, for plotting.

        Parameters
        ----------
        name
            Which series to extract.

        Returns
        -------
        tuple
            One value per record. ``val_loss`` may contain ``None`` entries,
            which a plotting function must handle rather than treating as
            zero -- a gap in a curve and a loss of zero look very different
            and mean very different things.
        """
        return tuple(getattr(record, name) for record in self.history)


class EvalResult(ContractModel):
    """
    Metrics for one split.

    Parameters
    ----------
    split
        Which split was scored.
    metrics
        Metric name to value.
    n_samples
        Number of samples scored, so a metric can be weighted or pooled
        correctly across splits.
    in_original_units
        Whether the metrics are in the original target units. See the module
        docstring: this is checked, not assumed.
    baseline_metrics
        The same metrics for a naive reference model on the same split.
        Carried alongside rather than reported separately, because a headline
        metric without a reference point is not interpretable.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a metric value is not finite.
    """

    split: str
    metrics: Mapping[str, float]
    n_samples: int = Field(ge=0)
    in_original_units: bool = True
    baseline_metrics: Mapping[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_metrics_are_finite(self) -> EvalResult:
        """
        Reject non-finite metric values.

        Returns
        -------
        EvalResult
            The validated result.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any metric is ``NaN`` or infinite. Such a value almost always
            means a degenerate split or a divide-by-zero inside a metric, and
            it is far cheaper to diagnose here than after it has been written
            to a bundle and compared against other runs.
        """
        offenders = sorted(name for name, value in self.metrics.items() if not np.isfinite(value))
        if offenders:
            raise SpecError(
                f"metric(s) {offenders} for split {self.split!r} are not finite; "
                f"this usually means a degenerate split or a division by zero "
                f"inside a metric"
            )
        return self


class TrainingResult(ContractModel):
    """
    The complete outcome of a training run.

    Parameters
    ----------
    fit
        What the fit itself produced.
    evaluations
        Metrics per split, keyed by split name.
    seed
        The seed actually applied, recorded so a run can be repeated without
        re-deriving it.
    notes
        Free-form annotations, such as which compatibility flags were active.
    bundle_directory
        Where the trained model was saved, or ``None`` if nothing was
        written. Recorded because a result that reports a model's metrics
        but not the model is awkward in exactly the case that matters:
        training and then evaluating, predicting or promoting it. Without
        it the caller has to reconstruct a path from the run id, the model
        name and the version, which is three things to get wrong and a
        private layout to depend on.
    """

    fit: FitOutcome
    evaluations: Mapping[str, EvalResult] = Field(default_factory=dict)
    seed: int = Field(default=0, ge=0)
    notes: Mapping[str, str] = Field(default_factory=dict)
    bundle_directory: str | None = None

    def metric(self, split: str, name: str) -> float:
        """
        Return one metric for one split.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float
            The metric value.

        Raises
        ------
        ContractError
            If the split or the metric is absent, with the available names
            listed.
        """
        if split not in self.evaluations:
            raise ContractError(
                f"no evaluation for split {split!r}; "
                f"evaluated splits are {sorted(self.evaluations)}"
            )
        metrics = self.evaluations[split].metrics
        if name not in metrics:
            raise ContractError(
                f"no metric {name!r} for split {split!r}; available: {sorted(metrics)}"
            )
        return metrics[name]

    def headline(self, *, split: str = "test") -> Mapping[str, float]:
        """
        Return the metrics most runs are judged on.

        Parameters
        ----------
        split
            Split to report. Falls back to validation, then train, so a run
            configured without a test split still has a headline.

        Returns
        -------
        Mapping
            Metrics for the first available split, or empty if none were
            evaluated.
        """
        for candidate in (split, "validation", "train"):
            if candidate in self.evaluations:
                return self.evaluations[candidate].metrics
        return {}


class EvaluationResult(ContractModel):
    """
    The outcome of scoring a *saved* model, with the provenance to read it by.

    Distinct from :class:`TrainingResult` because the two answer different
    questions. A training result says what a run produced; an evaluation
    result says what was scored, from which bundle, against which data. The
    second half is not decoration. A metric detached from the source it was
    computed over cannot be compared to another metric with any confidence,
    and the comparison people most want to make -- "is this model still as
    good as it was?" -- is exactly the one that goes wrong when the data
    changed underneath without anybody recording it.

    Which is why :attr:`source_changed` is a field rather than something a
    reader is expected to work out. Re-scoring on new data is a legitimate
    and common thing to do; doing it while believing you reproduced an old
    number is not.

    Parameters
    ----------
    evaluations
        Metrics per split, keyed by split name.
    model_name
        The name the model is registered under, from the bundle's manifest.
    bundle_version
        Which version of the bundle was scored.
    spec_digest
        The digest of the specification the model was trained from.
    source_fingerprint
        Fingerprint of the data actually scored against.
    trained_on_fingerprint
        Fingerprint of the data the model was trained on.
    source_changed
        Whether the two fingerprints differ. Redundant by construction, and
        kept anyway: a reader scanning a result should not have to compare
        two hashes to learn whether the headline number means what they
        assume it means.
    notes
        Free-form annotations.
    """

    evaluations: Mapping[str, EvalResult] = Field(default_factory=dict)
    model_name: str = ""
    bundle_version: int = Field(default=1, ge=1)
    spec_digest: str = ""
    source_fingerprint: str = ""
    trained_on_fingerprint: str = ""
    source_changed: bool = False
    notes: Mapping[str, str] = Field(default_factory=dict)

    def metric(self, split: str, name: str) -> float:
        """
        Return one metric for one split.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float
            The metric value.

        Raises
        ------
        ContractError
            If the split or the metric is absent, with the available names
            listed.
        """
        if split not in self.evaluations:
            raise ContractError(
                f"no evaluation for split {split!r}; "
                f"evaluated splits are {sorted(self.evaluations)}"
            )
        metrics = self.evaluations[split].metrics
        if name not in metrics:
            raise ContractError(
                f"no metric {name!r} for split {split!r}; available: {sorted(metrics)}"
            )
        return metrics[name]

    def describe(self) -> str:
        """
        Return a one-line summary, for a log and a report header.

        Returns
        -------
        str
            Names the source change when there is one, in the same words
            every layer of this phase uses for it, so that the warning is
            recognisable wherever it surfaces.
        """
        scored = ", ".join(
            f"{split}={self.evaluations[split].metrics.get('mae', float('nan')):.6g}"
            for split in sorted(self.evaluations)
        )
        suffix = (
            ""
            if not self.source_changed
            else (
                f" [CHANGED SOURCE {self.trained_on_fingerprint[:8]} -> "
                f"{self.source_fingerprint[:8]}; not a reproduction]"
            )
        )
        return f"evaluated {self.model_name} v{self.bundle_version}: mae {scored}{suffix}"


class TrialRecord(ContractModel):
    """
    One trial of a search: what was proposed, and what came of it.

    A failed trial is a record rather than an absence, for the same reason a
    failed job is. A search that dropped its failures and reported the best
    of the rest would look exactly like a search in which everything
    succeeded -- and if eleven of forty failed, the conclusion drawn from
    the other twenty-nine is probably wrong, with nothing in the report to
    suggest it.

    Parameters
    ----------
    trial
        The trial number, which is also its position in the search.
    overrides
        The flat proposal, dotted path to value. Flat rather than nested
        because this is what a reader compares across trials, and a nested
        fragment has to be mentally flattened before two of them can be read
        side by side.
    status
        Whether it finished.
    objective
        The metric the search optimises, on the objective split. ``None``
        for a failed trial, and also for one that produced no such metric --
        which is a distinct situation from scoring badly, and must not be
        collapsed into a score of zero.
    metrics
        Every metric on the objective split, so the winner can be examined
        on more than the one number it was chosen by.
    bundle_directory
        Where the trial's model was written, relative to nothing -- an
        absolute path, because a search's trials live under the search and a
        caller holding only this record has no base to resolve against.
        Recorded so the winner can be *used* rather than retrained, which is
        the difference between acting on a search and repeating it.
    wall_seconds
        How long it took, successes and failures alike.
    failure_kind
        Exception class name, for a failed trial.
    failure_message
        Exception message, for a failed trial.
    """

    trial: int = Field(ge=0)
    overrides: Mapping[str, Any] = Field(default_factory=dict)
    status: Literal["succeeded", "failed"] = "succeeded"
    objective: float | None = None
    metrics: Mapping[str, float] = Field(default_factory=dict)
    bundle_directory: str | None = None
    wall_seconds: float = Field(default=0.0, ge=0.0)
    failure_kind: str | None = None
    failure_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """
        Return whether the trial finished.

        Returns
        -------
        bool
            True if it did.
        """
        return self.status == "succeeded"


class TuningResult(ContractModel):
    """
    Every trial of a search, and which one won.

    Parameters
    ----------
    trials
        The trials, in the order they ran.
    best_trial
        The winning trial number.
    objective
        The metric that was optimised.
    direction
        Whether it was minimised or maximised. Recorded because a column of
        objective values cannot be read without it -- and a reader who
        assumes the wrong direction concludes the opposite of the truth.
    objective_split
        Which split the objective was read from. Recorded for the same
        reason: a winner selected on test means something different from one
        selected on validation, and only one of them is a held-out estimate.
    name
        A label for the search.
    """

    trials: tuple[TrialRecord, ...] = ()
    best_trial: int = Field(default=0, ge=0)
    objective: str = "mae"
    direction: Literal["minimise", "maximise"] = "minimise"
    objective_split: str = "validation"
    name: str | None = None

    @property
    def best(self) -> TrialRecord:
        """
        Return the winning trial.

        Returns
        -------
        TrialRecord
            The winner.

        Raises
        ------
        ContractError
            If the search recorded no trials.
        """
        for record in self.trials:
            if record.trial == self.best_trial:
                return record
        raise ContractError(
            f"the search names trial {self.best_trial} as its best, but holds "
            f"{len(self.trials)} trial(s) and none with that number"
        )

    @property
    def failed(self) -> tuple[TrialRecord, ...]:
        """
        Return the trials that did not finish.

        Returns
        -------
        tuple of TrialRecord
            The failures, in trial order.
        """
        return tuple(record for record in self.trials if not record.succeeded)

    def describe(self) -> str:
        """
        Return a one-line summary, naming any failures.

        Returns
        -------
        str
            The winner, its objective, and how many trials failed -- which
            is said even when none did, because "0 failed" is information
            and a missing clause is not.
        """
        best = self.best
        return (
            f"best of {len(self.trials)} trial(s): #{best.trial} with "
            f"{self.objective}={best.objective:.6g} on {self.objective_split} "
            f"({self.direction}d); {len(self.failed)} failed"
        )


@dataclass(frozen=True, slots=True)
class Predictions:
    """
    Model output, with enough provenance to be acted on.

    A dataclass rather than a contract model because it carries arrays. Its
    metadata is small enough to be written beside it when predictions are
    persisted.

    Parameters
    ----------
    values
        Predicted values.
    in_original_units
        Whether the values are in the original target units. See the module
        docstring.
    entity_ids
        Identifier per prediction, where one exists. Without this a prediction
        cannot be attributed to a real instrument, which makes it unusable
        downstream however accurate it is.
    scenario_indices
        Scenario index per prediction.
    bundle_version
        Which bundle produced these predictions.
    provenance
        Additional annotations, such as the spec digest and the time of
        inference.

    Raises
    ------
    ContractError
        If the identifier or index lengths disagree with the values.
    """

    values: NDArray[np.floating]
    in_original_units: bool = True
    entity_ids: tuple[str, ...] | None = None
    scenario_indices: NDArray[np.int64] | None = None
    bundle_version: str | None = None
    provenance: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate that any per-prediction metadata aligns with the values."""
        n_predictions = self.values.shape[0] if self.values.ndim else 0
        if self.entity_ids is not None and len(self.entity_ids) != n_predictions:
            raise ContractError(
                f"entity_ids has {len(self.entity_ids)} entries but there are "
                f"{n_predictions} predictions"
            )
        if self.scenario_indices is not None and self.scenario_indices.shape[0] != n_predictions:
            raise ContractError(
                f"scenario_indices has {self.scenario_indices.shape[0]} entries but "
                f"there are {n_predictions} predictions"
            )

    @property
    def n_predictions(self) -> int:
        """Number of predictions."""
        return int(self.values.shape[0]) if self.values.ndim else 0
```

---

## 7. `src/rade_qnet/core/contract/signature.py`

12855 bytes · SHA-256 `fcb7ccefc6bb9043`

```python
"""
Declared shapes and dtypes of a model's inputs and target.

:class:`InputSignature` is the most load-bearing contract in the framework,
because three unrelated mechanisms depend on it:

**Static input handling.** The signature says which inputs are constant across
batches, so the engine can upload them to the device once instead of collating
them into every sample of every batch.

**Lazy parameter materialisation.** A model whose shapes are known only after
the data build has no parameters until it has seen one batch. The signature
carries exactly enough information to synthesise that batch, so the model can
be materialised before an optimiser, a checkpoint or a distributed wrapper
touches it.

**Rebuilding from a bundle.** Given a saved signature and saved weights, a
model can be reconstructed months later without re-running the data build.

The static/dynamic split
------------------------
A *static* input is the same for every sample: an adjacency matrix, an entity
feature table, an index array. A *dynamic* input varies per sample: a history
window, a feature row. Keeping the two apart in the type is what allows the
engine to treat them differently without inspecting the data.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from ..runtime.errors import ContractError, SpecError
from .base import ContractModel

__all__ = ["InputSignature", "PolicySignature", "SpaceSpec", "TensorSpec"]


class TensorSpec(ContractModel):
    """
    The shape and dtype of one tensor, named independently of any library.

    Parameters
    ----------
    shape
        Dimensions, with ``None`` marking a dimension that varies -- normally
        the batch dimension. A fully concrete shape is permitted and is what
        a static input usually has.
    dtype
        Element type, as a library-agnostic string such as ``float32`` or
        ``int64``. A string rather than a library's dtype object, because
        ``core`` may not import a training library and because a signature
        written by the Torch engine must be readable by the XGBoost one.
    description
        Optional human-readable note, surfaced in error messages and reports.
    """

    shape: tuple[int | None, ...]
    dtype: str
    description: str | None = None

    @model_validator(mode="after")
    def _check_dimensions_are_positive(self) -> TensorSpec:
        """
        Reject a shape with a non-positive concrete dimension.

        Returns
        -------
        TensorSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any concrete dimension is zero or
            negative. A zero dimension
            produces an empty tensor that trains without error and learns
            nothing, which is far harder to diagnose later than a rejection
            here.
        """
        for position, dimension in enumerate(self.shape):
            if dimension is not None and dimension <= 0:
                raise SpecError(
                    f"dimension {position} of shape {self.shape} is {dimension}; "
                    f"concrete dimensions must be positive"
                )
        return self

    @property
    def rank(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    def is_compatible_with(self, other: TensorSpec) -> bool:
        """
        Return whether another spec could describe the same tensor.

        Compatibility is deliberately weaker than equality: a ``None``
        dimension matches any size. That is what lets a signature declared
        with an unknown batch size be checked against a concrete batch.

        Parameters
        ----------
        other
            The spec to compare against.

        Returns
        -------
        bool
            True if the dtypes match and every dimension pair is either equal
            or wildcarded on at least one side.
        """
        if self.dtype != other.dtype or self.rank != other.rank:
            return False
        return all(
            mine is None or theirs is None or mine == theirs
            for mine, theirs in zip(self.shape, other.shape, strict=True)
        )

    def concrete_shape(self, batch_size: int) -> tuple[int, ...]:
        """
        Resolve wildcard dimensions to a concrete shape.

        Used to synthesise the dummy batch that materialises a model's lazy
        parameters.

        Parameters
        ----------
        batch_size
            Size to substitute for every ``None`` dimension.

        Returns
        -------
        tuple of int
            A fully concrete shape.

        Raises
        ------
        SpecError
            If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise SpecError(f"batch_size must be positive, received {batch_size}")
        return tuple(batch_size if dimension is None else dimension for dimension in self.shape)

    def describe(self) -> str:
        """
        Return a compact one-line description, for error messages and reports.

        Returns
        -------
        str
            For example ``float32[?, 20, 64]``.
        """
        dimensions = ", ".join("?" if size is None else str(size) for size in self.shape)
        return f"{self.dtype}[{dimensions}]"


class InputSignature(ContractModel):
    """
    The complete declared interface between a data source and a model.

    Parameters
    ----------
    dynamic
        Inputs that vary per sample. At least one is required -- a model with
        no varying input cannot learn from data.
    static
        Inputs constant across every batch. Usually empty; non-empty for
        models that consume a graph or an entity table.
    target
        The quantity being predicted.
    """

    dynamic: Mapping[str, TensorSpec]
    static: Mapping[str, TensorSpec] = Field(default_factory=dict)
    target: TensorSpec

    @model_validator(mode="after")
    def _check_names_are_usable(self) -> InputSignature:
        """
        Reject an empty dynamic set, or a name used in both groups.

        Returns
        -------
        InputSignature
            The validated signature.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if ``dynamic`` is empty, or a name appears
            in both ``dynamic`` and ``static``. A duplicated name is unresolvable: the engine would
            have to guess whether to collate it or upload it once, and either
            choice is silently wrong half the time.
        """
        if not self.dynamic:
            raise SpecError(
                "an input signature needs at least one dynamic input; "
                "a model with only static inputs cannot vary with the data"
            )
        shared = sorted(set(self.dynamic) & set(self.static))
        if shared:
            raise SpecError(
                f"input name(s) {shared} appear in both dynamic and static; "
                f"each input must be one or the other"
            )
        return self

    @property
    def input_names(self) -> tuple[str, ...]:
        """Every input name, static and dynamic, sorted."""
        return tuple(sorted({*self.dynamic, *self.static}))

    def spec_for(self, name: str) -> TensorSpec:
        """
        Return the spec for one input, from either group.

        Parameters
        ----------
        name
            An input name.

        Returns
        -------
        TensorSpec
            The input's spec.

        Raises
        ------
        ContractError
            If the name is not declared. Listing the declared names makes the
            usual cause -- a renamed key in a data module -- immediately
            obvious.
        """
        if name in self.dynamic:
            return self.dynamic[name]
        if name in self.static:
            return self.static[name]
        raise ContractError(
            f"input {name!r} is not declared in the signature; "
            f"declared inputs are {list(self.input_names)}"
        )

    def validate_batch_keys(
        self,
        batch: Mapping[str, object],
        *,
        where: str,
        target_key: str = "target",
    ) -> None:
        """
        Check that a batch carries every dynamic input and no unknown key.

        Only keys are checked, not shapes: ``core`` cannot inspect an engine's
        tensors. Shape checking belongs to the engine, which knows what a
        tensor is. Key checking here still catches the most common failure --
        a data module and a model disagreeing on a name -- and catches it with
        a message naming both sides rather than as a dimension mismatch six
        frames into a forward pass.

        Parameters
        ----------
        batch
            A batch produced by a source.
        where
            Description of the caller, used in the error message so the
            failure names the stage that produced the bad batch.
        target_key
            Key the target is carried under, permitted but not required -- an
            inference batch legitimately has no target.

        Raises
        ------
        ContractError
            If a dynamic input is absent, or the batch carries a key that is
            neither a declared input nor the target.
        """
        keys = set(batch)
        missing = sorted(set(self.dynamic) - keys)
        if missing:
            raise ContractError(
                f"{where}: batch is missing declared dynamic input(s) {missing}; "
                f"batch provided {sorted(keys)}"
            )
        # Static inputs are permitted in a batch but not required: a source may
        # deliver them separately, which is the path that avoids collating them
        # per sample.
        permitted = {*self.dynamic, *self.static, target_key}
        unexpected = sorted(keys - permitted)
        if unexpected:
            raise ContractError(
                f"{where}: batch carries undeclared key(s) {unexpected}; "
                f"the signature declares {list(self.input_names)} and target "
                f"{target_key!r}"
            )

    def describe(self) -> str:
        """
        Return a multi-line description of the signature.

        Returns
        -------
        str
            One line per input, plus the target.
        """
        lines = [f"target: {self.target.describe()}"]
        for group, specs in (("dynamic", self.dynamic), ("static", self.static)):
            for name in sorted(specs):
                lines.append(f"{group}: {name} = {specs[name].describe()}")
        return "\n".join(lines)


class SpaceSpec(ContractModel):
    """
    An observation or action space.

    Minimal in Phase 1 and expanded by the reinforcement-learning phase. The
    shape is settled now because :class:`PolicySignature` is referenced by the
    run spec, and a contract that changes shape later invalidates every bundle
    written against it.

    Parameters
    ----------
    kind
        ``box`` for a continuous space, ``discrete`` for a finite one.
    shape
        Dimensions, for a ``box`` space.
    dtype
        Element type, for a ``box`` space.
    low, high
        Inclusive bounds, for a ``box`` space. ``None`` means unbounded.
    n
        Number of actions, for a ``discrete`` space.
    """

    kind: Literal["box", "discrete"]
    shape: tuple[int, ...] = ()
    dtype: str = "float32"
    low: float | None = None
    high: float | None = None
    n: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_fields_match_kind(self) -> SpaceSpec:
        """
        Reject fields that do not apply to the chosen kind.

        Returns
        -------
        SpaceSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if a ``discrete`` space has no ``n``, or a
            ``box`` space has one.
        """
        if self.kind == "discrete" and self.n is None:
            raise SpecError("a discrete space requires 'n', the number of actions")
        if self.kind == "box" and self.n is not None:
            raise SpecError("'n' applies to a discrete space, not a box space")
        return self


class PolicySignature(ContractModel):
    """
    The declared interface between an environment and a policy.

    The interactive counterpart of :class:`InputSignature`, serving the same
    three purposes: constructing a policy, materialising it, and rebuilding it
    from a saved bundle.

    Parameters
    ----------
    observation
        What the policy sees.
    action
        What the policy produces.
    """

    observation: SpaceSpec
    action: SpaceSpec
```

---

## 8. `src/rade_qnet/core/contract/source.py`

6275 bytes · SHA-256 `0f58535c7104056f`

```python
"""
``BatchSource`` -- the one protocol every training loop consumes.

This is the framework's unification point, and the reason there is one training
loop rather than two frameworks sharing a repository.

Supervised learning and reinforcement learning differ far less than their
tooling suggests. Both consume a stream of batches. They disagree only about
where the batches come from: a fixed dataset, a freshly collected rollout, a
replay buffer, a stored transition table, or a differentiable simulator. Wrap
each of those as a ``BatchSource`` and the loop stops needing to know.

Why a protocol rather than a base class
---------------------------------------
A source satisfies this by having the right members. It need not inherit from
anything of ours and need not import this module at all, which is what lets a
source live outside this repository. It also means the five framework sources
are checked against the same definition a user's source is.

Bounded and unbounded sources
-----------------------------
:attr:`BatchSource.steps_per_epoch` returning ``None`` means "unbounded", and
that single value is what selects the loop driver. A fixed dataset has a
meaningful notion of a pass over the data, so it is driven by ``fit_epochs``.
An environment does not, so it is driven by ``fit_steps``. The source declares
which it is; the engine does not guess.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Protocol, runtime_checkable

from .data import Batch, TensorLike
from .signature import InputSignature

__all__ = ["BatchSource", "OrderedSource"]


@runtime_checkable
class BatchSource(Protocol):
    """
    A re-iterable stream of training batches.

    Implementations must satisfy two properties that are easy to get wrong and
    both tested by the shared contract suite:

    **Re-iterable.** :meth:`batches` may be called more than once, and a
    bounded source must yield the same batches each time under a fixed seed. A
    training loop traverses a dataset once per epoch; a source backed by a bare
    generator silently yields nothing on the second epoch.

    **Honest about size.** :attr:`steps_per_epoch` is either the true number of
    batches per pass, or ``None`` for an unbounded source. It is not an
    estimate: callbacks, progress reporting and learning-rate schedules are
    computed from it.

    Note that "the same batches" constrains the *contents* of a pass, not its
    order. A training source is expected to reshuffle between epochs, which is
    the whole purpose of shuffling. Any consumer that needs two passes to line
    up row for row must say so -- see :class:`OrderedSource`.
    """

    @property
    def signature(self) -> InputSignature:
        """
        The declared interface of the batches this source yields.

        Returns
        -------
        InputSignature
            Static inputs, dynamic inputs and the target.
        """
        ...

    @property
    def static(self) -> Mapping[str, TensorLike]:
        """
        Inputs constant across every batch.

        Delivered once, outside the batch stream, so the engine can upload
        them to the device a single time. Empty for most sources.

        Returns
        -------
        Mapping
            Static inputs, keyed as declared in the signature.
        """
        ...

    @property
    def steps_per_epoch(self) -> int | None:
        """
        Batches per pass, or ``None`` if the source is unbounded.

        Returns
        -------
        int or None
            The batch count, or ``None``.
        """
        ...

    @property
    def n_samples(self) -> int | None:
        """
        Samples per pass, or ``None`` if unbounded.

        Used for metric denominators and for reports. Distinct from
        :attr:`steps_per_epoch` because a final short batch means the two are
        not related by a constant.

        Returns
        -------
        int or None
            The sample count, or ``None``.
        """
        ...

    def batches(self) -> Iterator[Batch]:
        """
        Yield batches for one pass.

        For an unbounded source this yields indefinitely, and the loop driver
        decides when to stop.

        Yields
        ------
        Batch
            A mapping from input name to tensor, carrying every dynamic input
            declared in the signature.
        """
        ...


@runtime_checkable
class OrderedSource(Protocol):
    """
    A source that can produce a stable-order view of itself.

    Why this is a separate capability
    ---------------------------------
    Evaluation needs two passes over the same split: one to run the forward
    pass and one to collect the targets to score it against. Over a training
    source those two passes are in *different orders*, because a training
    source reshuffles between epochs by design. Pairing them row for row then
    compares every prediction against some other row's target.

    The failure is specific and nasty. The sample count matches, so no shape
    check fires. Every metric computes cleanly. The numbers are simply those of
    a model predicting at random, so a correctly trained model reports a
    negative r-squared on its training split and a good one on its held-out
    splits -- a pattern that reads as a bizarre inversion of overfitting rather
    than as an alignment bug.

    Keeping this separate from :class:`BatchSource` rather than adding
    ``ordered`` to it is what stops the addition being a breaking change: a
    source satisfies ``BatchSource`` structurally, so a new required member
    would silently drop every existing implementation out of
    ``isinstance``. Declared as its own capability, ``isinstance`` *routes* --
    a source that shuffles implements it, and one that never shuffles does not
    need to.
    """

    def ordered(self) -> BatchSource:
        """
        Return a view of this source whose pass order is stable.

        Same samples, same batch boundaries, repeatable order. May return
        ``self`` when the source never shuffles.

        Returns
        -------
        BatchSource
            A source safe to traverse more than once with aligned results.
        """
        ...
```

---

## 9. `src/rade_qnet/core/contract/state.py`

6774 bytes · SHA-256 `c9795030254a3196`

```python
"""
Everything a model fits at training time and needs again at inference.

Model weights are not the whole of what training produces. A scaler's mean and
scale, the ordered subset of features a basis selection chose, an entity
encoder's category table, a nearest-neighbour graph -- all of these are fitted
from training data and all are required to make a single prediction later.

The implementation this framework replaces wrote roughly twenty loose sidecar
files for exactly this, with the relationship between them encoded only in the
code that loaded them. The result was predictable: any change to that code
silently invalidated every previously saved model, and nothing detected it.

:class:`FittedState` replaces that with one typed object that knows how to save
itself, load itself, and -- crucially -- put a prediction back into the units
the user cares about.

Why the inverse is abstract
---------------------------
:meth:`FittedState.inverse_transform_targets` has no default implementation.
A model that scales its target and cannot invert the scaling reports a mean
absolute error in standardised space, which is not a quantity anyone can act
on. Making the method abstract means that failure cannot be shipped by
omission -- it has to be chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["FittedState", "IdentityFittedState"]


class FittedState(ABC):
    """
    Abstract base for a model's fitted, non-parameter state.

    Implementations are free to hold whatever they need -- arrays, mappings,
    sparse structures -- provided the three methods below are honest about it.

    Notes
    -----
    Saving is directory-based rather than single-file. A scaler is naturally a
    pair of arrays, a graph is three, and forcing them through one pickle
    reintroduces the fragility this class exists to remove. A directory of
    named ``.npy`` and ``.json`` files can be inspected with standard tools
    and loaded by code that has never imported this framework.
    """

    @abstractmethod
    def save(self, directory: Path) -> None:
        """
        Write the state beneath a directory.

        The directory already exists when this is called. Implementations
        should write named files rather than one opaque blob, and must not
        write outside the directory given.

        Parameters
        ----------
        directory
            Destination directory, created by the caller.
        """

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> Self:
        """
        Read the state back from a directory written by :meth:`save`.

        Parameters
        ----------
        directory
            Directory previously written by :meth:`save`.

        Returns
        -------
        Self
            A state equal to the one that was saved.

        Raises
        ------
        BundleError
            If the directory is missing a required file.
        """

    @abstractmethod
    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return predictions to the original target units.

        Called by the evaluate and infer pipelines *before* any metric is
        computed, so every number a user reads is in the units they think it
        is in.

        A model that does not transform its target should return the input
        unchanged -- see :class:`IdentityFittedState` -- but it must say so
        explicitly rather than inheriting a default.

        Parameters
        ----------
        predictions
            Model output, in whatever space the model produces.

        Returns
        -------
        numpy.ndarray
            Predictions in original target units, with the same shape.
        """

    def describe(self) -> dict[str, object]:
        """
        Return a summary of what was fitted, for reports and logs.

        The default reports only the class name. Implementations should
        override it with the things a reader would want to check at a glance:
        how many features a basis selected, how many categories an encoder
        saw, how many edges a graph has. This ends up in the run summary, so
        it is often the first place an anomaly becomes visible.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {"type": type(self).__name__}


class IdentityFittedState(FittedState):
    """
    Fitted state for a model that fits nothing beyond its parameters.

    Appropriate for a model whose target is untransformed: a tree model on raw
    features, or a network trained on an unscaled target. Using this when the
    target *is* transformed produces metrics in the wrong units, which is why
    it is a named, deliberate choice rather than the base class default.

    Also used by ``rade_qnet.testkit.fixtures`` so that a pipeline test needs no
    bespoke state implementation.
    """

    def save(self, directory: Path) -> None:
        """
        Write a marker file recording that there was nothing to save.

        An empty directory is indistinguishable from a failed write, so a
        marker is written instead. Bundle verification can then confirm the
        state was saved deliberately.

        Parameters
        ----------
        directory
            Destination directory.
        """
        (directory / "identity.marker").write_text(
            "This model fits no state beyond its parameters.\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Return a fresh instance, ignoring the directory's contents.

        Parameters
        ----------
        directory
            Unused; accepted to satisfy the interface.

        Returns
        -------
        Self
            A new instance.
        """
        del directory  # Nothing to read: the state is, by definition, empty.
        return cls()

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return the predictions unchanged.

        Parameters
        ----------
        predictions
            Model output.

        Returns
        -------
        numpy.ndarray
            The same array.
        """
        return predictions

    def __eq__(self, other: object) -> bool:
        """Return whether the other object is also an identity state."""
        return isinstance(other, IdentityFittedState)

    def __hash__(self) -> int:
        """Return a constant hash, since all instances are equivalent."""
        return hash(IdentityFittedState)
```

