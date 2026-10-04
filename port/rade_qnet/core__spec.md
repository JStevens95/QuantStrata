# `src/rade_qnet/core/spec`

10 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 53 | 2597 | `8d034c0a484bd379` |
| 2 | `base.py` | 76 | 3189 | `0a27b41675af387a` |
| 3 | `data.py` | 460 | 15051 | `460a756d41e5f3a6` |
| 4 | `hardware.py` | 111 | 4569 | `82afd37e1cf52534` |
| 5 | `jobs.py` | 533 | 19900 | `52fd20676baea8dc` |
| 6 | `merge.py` | 182 | 7103 | `89eeb600e0a48ae8` |
| 7 | `reports.py` | 76 | 2488 | `090156f7b9a0c8a7` |
| 8 | `run.py` | 383 | 12864 | `1b82a04a916ca8f0` |
| 9 | `training.py` | 336 | 12006 | `70e3ec454e04fd70` |
| 10 | `tune.py` | 522 | 17481 | `04de021fa2194e1d` |

---

## 1. `src/rade_qnet/core/spec/__init__.py`

2597 bytes · SHA-256 `8d034c0a484bd379`

```python
"""
Configuration schemas -- the declarative description of a run.

Every specification is a ``pydantic`` model with ``extra="forbid"``, so a typo
in a YAML key is a load-time error rather than a silently ignored setting.
Specs round-trip exactly: ``Spec.model_validate(spec.model_dump())`` returns an
equal object, which is what makes a persisted spec a faithful record of how a
model was produced.

Specs describe *intent* only.  They never hold fitted state, file handles,
tensors or live objects, which keeps them cheap to hash, log and send to a
worker process.

Modules
-------
``run.py``
    ``RunSpec``, discriminated on ``task`` into a supervised run and a
    reinforcement-learning run.  The single object a pipeline is built from.
    [Phase 1, delivered]
``jobs.py``
    ``JobSpec`` (one training job) and ``JobSetSpec`` (shared defaults, a list
    of jobs with per-job overrides, and placement).  [Phase 4, delivered]
``data.py``
    ``SourceSpec`` union, ``SplitSpec`` and ``LoaderSpec``.  [Phase 1, delivered]
``environment.py``
    ``EnvSpec``, ``VectorSpec`` and ``WrapperSpec`` for interactive sources.
    [Phase 7]
``batching.py``
    ``DatasetBatchingSpec``, ``RolloutSpec``, ``ReplaySpec`` and
    ``SimulationSpec``, discriminated on ``kind``.  [Phase 1 / Phase 7, delivered]
``training.py``
    Engine-discriminated training specs: ``TorchTrainSpec``, ``XGBTrainSpec``,
    ``SklearnTrainSpec`` and ``RlTrainSpec``.  [Phase 1, delivered]
``hardware.py``
    ``HardwareSpec`` -- how a *single* job uses its machine (device,
    precision, compilation, distribution, determinism).  Deliberately separate
    from placement, which decides *where* jobs run -- with the exception of
    ``threads_per_worker``, which lives here precisely because it must *not*
    be a placement decision: the thread count fixes the order a reduction
    accumulates in, so a budget travelling with the executor would let
    placement change results.  [Phase 1, delivered; see defect 13]
``reports.py``
    ``ReportsSpec`` -- which report writers are enabled for a run.  [Phase 1, delivered]
``merge.py``
    Deep-merge of job-set defaults with per-job overrides, with the merge rules
    stated explicitly so an override never silently drops a nested default.
    Merges *raw mappings*, before validation: once a mapping has been through
    the schema, a field the user never set is indistinguishable from one they
    did, so merging validated specs would have every job overwrite the shared
    defaults with its own copy of them.  [Phase 4, delivered]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/core/spec/base.py`

3189 bytes · SHA-256 `0a27b41675af387a`

```python
"""
The base class every specification derives from.

Three settings are applied to every spec in the framework, and each one exists
because of a specific way configuration goes wrong.

``extra="forbid"``
    A misspelled key is a load-time error. The alternative -- silently
    ignoring it -- means a run that was configured with ``seq_len: 20``
    trains with the default of ``1``, reports plausible numbers, and nobody
    finds out.

``frozen=True``
    A spec cannot be mutated after validation. A pipeline that received a spec
    and a pipeline that recorded it in a bundle must have received the same
    object, or the bundle is a record of something that did not happen.

``str_strip_whitespace=True``
    A trailing space in a YAML string is invisible in a text editor and fatal
    to a registry lookup.

Specs describe *intent only*. They hold no fitted state, no file handles, no
tensors and no live objects, which is what keeps them cheap to hash, log and
send across a process boundary.

How validation failures surface
-------------------------------
Validators in this package raise :class:`~rade_qnet.core.runtime.errors.SpecError`.
Pydantic *collects* that rather than letting it propagate, because
``SpecError`` subclasses ``ValueError`` -- which is the entire reason for that
dual inheritance. The practical consequence is a deliberate two-level policy:

- **Direct construction** of a spec in Python raises ``ValidationError``,
  carrying the ``SpecError`` message and the dotted path of the offending
  field. This is the right outcome: constructing an invalid spec in code is a
  programming error, and pydantic's report locates it precisely.
- **Parsing** untrusted input goes through
  :func:`~rade_qnet.core.spec.run.parse_run_spec`, which reformats the
  ``ValidationError`` into a single ``SpecError``. A user who mistyped a YAML
  key sees one framework error, not a pydantic traceback.

So the ``Raises`` sections below name ``ValidationError``, and the ``SpecError``
mentioned alongside it is the message the user ends up reading.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

__all__ = ["Spec"]


class Spec(BaseModel):
    """
    Base class for every rade_qnet specification.

    Subclasses add fields and validation; they should not override
    :attr:`model_config`. If a spec appears to need ``extra="allow"``, the
    open-ended part of it belongs in an explicit ``params`` mapping whose
    contents are validated by whoever consumes them -- usually a model's own
    spec class.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        # Validate on construction as well as on parse, so a spec built in
        # Python is held to the same rules as one loaded from YAML. Without
        # this, a test that constructs a spec directly would not exercise the
        # validation that production configuration relies on.
        validate_default=True,
        # Nothing in a spec may be an arbitrary live object; see the module
        # docstring. Leaving this false makes that structural.
        arbitrary_types_allowed=False,
    )
```

---

## 3. `src/rade_qnet/core/spec/data.py`

15051 bytes · SHA-256 `460a756d41e5f3a6`

```python
"""
Where data comes from, how it is split, and how it is batched.

The module's central decision is visible in its type list: **splitting and
batching are separate specs**.

:class:`SplitSpec` decides which scenarios are training data. :class:`LoaderSpec`
decides the order batches are visited in. They are unrelated questions, and the
implementation this framework replaces drove both from a single ``shuffle``
flag -- so ``shuffle=True``, the natural choice for training throughput, turned
a chronological split of a time series into a random one. Nothing raised, and
the validation score improved.

A second decision worth naming: :class:`ReductionSpec` carries an explicit
``fit_on``. Basis selection in the previous implementation ran over the full
scaled history, validation and test rows included, which leaks the held-out
distribution into the choice of features. The flag makes the old behaviour
reachable -- for refactor parity -- and the correct behaviour the default.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec

__all__ = [
    "CacheSpec",
    "ChronologicalSplitSpec",
    "ExplicitSplitSpec",
    "GroupedSplitSpec",
    "LoaderSpec",
    "ModelSourceSpec",
    "PurgedKFoldSplitSpec",
    "ReductionSpec",
    "ScalingSpec",
    "SequenceSpec",
    "SourceSpec",
    "SplitSpec",
    "TabularSourceSpec",
    "TransformsSpec",
]


class ChronologicalSplitSpec(Spec):
    """
    Split the scenario axis by time order.

    The default strategy, and the only defensible one for a time series
    without a specific reason to do otherwise.

    Parameters
    ----------
    kind
        Discriminator.
    validation_fraction, test_fraction
        Fractions of the scenario axis taken from the end, test last. Must sum
        to less than one.
    gap_scenarios
        Scenarios discarded at each boundary, in addition to the automatic gap
        a sequence length implies. Use this when a target is computed over a
        forward window, since such a target carries information from
        scenarios after its own index.
    """

    kind: Literal["chronological"] = "chronological"
    validation_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    gap_scenarios: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _check_fractions_leave_training_data(self) -> ChronologicalSplitSpec:
        """
        Reject fractions that leave no training scenarios.

        Returns
        -------
        ChronologicalSplitSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if the fractions sum to one or more.
        """
        total = self.validation_fraction + self.test_fraction
        if total >= 1.0:
            raise SpecError(
                f"split fractions must sum to less than 1.0, received "
                f"validation_fraction={self.validation_fraction} + "
                f"test_fraction={self.test_fraction} = {total}; reduce one of them"
            )
        return self


class PurgedKFoldSplitSpec(Spec):
    """
    Cross-validation with an embargo either side of each held-out fold.

    Parameters
    ----------
    kind
        Discriminator.
    n_folds
        Number of folds.
    embargo_scenarios
        Scenarios purged either side of a held-out fold. Without an embargo,
        a fold boundary leaks in both directions, which is the specific
        failure plain k-fold has on serially correlated data.
    test_fraction
        Fraction held back from cross-validation entirely, as a final test
        set.
    """

    kind: Literal["purged_kfold"] = "purged_kfold"
    n_folds: int = Field(default=5, ge=2)
    embargo_scenarios: int = Field(default=0, ge=0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)


class GroupedSplitSpec(Spec):
    """
    Split so that every row of a group lands in exactly one split.

    Parameters
    ----------
    kind
        Discriminator.
    group_key
        Column or attribute identifying the group. Required: a grouped split
        with no group is just a random split, and silently becoming one is
        the failure this spec exists to prevent.
    validation_fraction, test_fraction
        Fractions of *groups*, not of rows.
    """

    kind: Literal["grouped"] = "grouped"
    group_key: str
    validation_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)


class ExplicitSplitSpec(Spec):
    """
    Use caller-supplied scenario indices verbatim.

    The strategy used for refactor parity: a golden fixture records the exact
    indices the previous implementation produced, and reproducing them removes
    the split from the list of things that could explain a difference.

    Parameters
    ----------
    kind
        Discriminator.
    train, validation, test
        Scenario indices. All three are required, because an explicit split
        with an inferred part is not explicit.
    """

    kind: Literal["explicit"] = "explicit"
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]

    @model_validator(mode="after")
    def _check_disjoint_and_non_empty(self) -> ExplicitSplitSpec:
        """
        Reject overlapping splits or an empty training set.

        Returns
        -------
        ExplicitSplitSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any index appears in more than one
            split, or train is empty.
        """
        if not self.train:
            raise SpecError("an explicit split requires at least one training index")

        splits = {
            "train": set(self.train),
            "validation": set(self.validation),
            "test": set(self.test),
        }
        names = sorted(splits)
        for index, first in enumerate(names):
            for second in names[index + 1 :]:
                shared = splits[first] & splits[second]
                if shared:
                    sample = sorted(shared)[:5]
                    raise SpecError(
                        f"explicit split indices overlap between {first} and {second}: "
                        f"{len(shared)} shared index(es), e.g. {sample}"
                    )
        return self


#: A split strategy, discriminated on ``kind``. Using a discriminated union
#: means an invalid ``kind`` reports that one field, rather than reporting four
#: failed alternatives and leaving the reader to work out which was intended.
SplitSpec = Annotated[
    ChronologicalSplitSpec | PurgedKFoldSplitSpec | GroupedSplitSpec | ExplicitSplitSpec,
    Field(discriminator="kind"),
]


class LoaderSpec(Spec):
    """
    How batches are formed and visited. Not how data is split.

    Parameters
    ----------
    batch_size
        Samples per batch.
    shuffle
        Whether to randomise batch order within the training split. This
        affects *order only*. It cannot affect which scenarios are in which
        split -- that is :class:`SplitSpec`, and a test asserts the
        independence directly.
    drop_last
        Discard a final short batch. Useful when a model requires a fixed
        batch size.
    num_workers
        Worker processes for data loading. Zero means load in the main
        process, which is the right default under a job-set process pool --
        nested pools oversubscribe the machine.
    pin_memory, persistent_workers
        Loader performance settings, meaningful only when workers are in use.
    """

    batch_size: int = Field(default=32, ge=1)
    shuffle: bool = True
    drop_last: bool = False
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = False
    persistent_workers: bool = False

    @model_validator(mode="after")
    def _check_worker_settings(self) -> LoaderSpec:
        """
        Reject worker settings that cannot take effect.

        Returns
        -------
        LoaderSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if ``persistent_workers`` is requested
            with no workers.
        """
        if self.persistent_workers and self.num_workers == 0:
            raise SpecError(
                "persistent_workers=True requires num_workers >= 1; "
                "with no workers there is nothing to keep alive"
            )
        return self


class ScalingSpec(Spec):
    """
    Feature and target scaling, fitted on the scenario axis.

    Fitted on training rows only. That is not configurable, because there is
    no legitimate reason to fit a scaler on held-out data.

    Parameters
    ----------
    method
        ``standard`` subtracts the mean and divides by the standard deviation.
        ``robust`` uses the median and interquartile range, which is the
        better choice for P&L series with heavy tails.
    scale_target
        Whether to scale the target as well as the features. When true, the
        fitted state must be able to invert it -- enforced by
        :class:`~rade_qnet.core.contract.state.FittedState`.
    """

    method: Literal["none", "standard", "robust"] = "standard"
    scale_target: bool = True


class ReductionSpec(Spec):
    """
    Dimensionality reduction, fitted on the scenario axis.

    Parameters
    ----------
    method
        ``basis_selection`` chooses a subset of the input series;
        ``pca`` projects onto components.
    n_components
        Target dimensionality. ``None`` lets the method choose.
    fit_on
        Which rows the reduction may observe.

        ``train``
            The default and the correct setting. Selection sees training rows
            only.
        ``all``
            Selection sees every row, including validation and test. This
            **leaks** the held-out distribution into the choice of features.
            It exists for one reason: reproducing the previous
            implementation's behaviour when verifying a refactor. It should
            never be set in a production configuration.
    """

    method: Literal["none", "basis_selection", "pca"] = "none"
    n_components: int | None = Field(default=None, ge=1)
    fit_on: Literal["train", "all"] = "train"


class SequenceSpec(Spec):
    """
    Rolling-window construction over the scenario axis.

    Parameters
    ----------
    length
        Window length. ``1`` means a non-sequential model.
    stride
        Step between consecutive windows.
    confine_to_split
        Whether a window must lie entirely within its own split.

        False -- the default -- keeps every label and relies on the
        boundary gap below to stop a window reaching into a neighbouring
        split. That is this framework's design, and
        :func:`~rade_qnet.sources.dataset.transforms.sequence.windows_stay_within`
        is the check that the gap is wide enough.

        True drops each split's first ``length - 1`` labels instead of
        requiring a gap. The two cost the same number of scenarios; the
        difference is whether they come from between the splits or from
        the front of each. Set it when the splits are given explicitly
        and inserting a gap is not an option.

    Notes
    -----
    A ``length`` above one makes the split boundary gap mandatory: a window
    starting in the training split would otherwise extend into validation, so
    the validation score would be partly a memory of training. The splitter
    derives that gap from this value, which is why the two specs are consumed
    together.
    """

    length: int = Field(default=1, ge=1)
    stride: int = Field(default=1, ge=1)
    confine_to_split: bool = False


class TransformsSpec(Spec):
    """
    The fitted transforms applied while building a dataset.

    Parameters
    ----------
    scaling, reduction, sequence
        Each transform's settings. All three default to a usable
        configuration, so a source spec need not mention them.
    """

    scaling: ScalingSpec = Field(default_factory=ScalingSpec)
    reduction: ReductionSpec = Field(default_factory=ReductionSpec)
    sequence: SequenceSpec = Field(default_factory=SequenceSpec)


class CacheSpec(Spec):
    """
    Caching of an expensive data build.

    Parameters
    ----------
    enabled
        Whether to reuse a previous build with the same fingerprint.
    directory
        Where cached builds live. ``None`` places them under the run root.
    """

    enabled: bool = False
    directory: Path | None = None


class _SourceSpecBase(Spec):
    """
    Settings shared by every source kind.

    Not part of the public union -- see :data:`SourceSpec`.
    """

    split: SplitSpec = Field(default_factory=ChronologicalSplitSpec)
    loader: LoaderSpec = Field(default_factory=LoaderSpec)
    transforms: TransformsSpec = Field(default_factory=TransformsSpec)
    cache: CacheSpec = Field(default_factory=CacheSpec)


class TabularSourceSpec(_SourceSpecBase):
    """
    A dataset read from a tabular file.

    The path a simple model takes: point at a file, name the target column,
    and the framework's standard data module does the rest.

    Parameters
    ----------
    kind
        Discriminator.
    path
        File to read. ``None`` is permitted so the spec is constructible with
        defaults for testing; a run with no path fails when the source is
        built, where the error can name the stage.
    target_column
        Column holding the target.
    feature_columns
        Columns to use as features. ``None`` means every column except the
        target, which is convenient and also a common source of surprise --
        naming them explicitly is recommended for anything that matters.
    """

    kind: Literal["tabular"] = "tabular"
    path: Path | None = None
    target_column: str = "target"
    feature_columns: tuple[str, ...] | None = None


class ModelSourceSpec(_SourceSpecBase):
    """
    A dataset built by the model's own data module.

    The path a complex model takes. The framework still owns splitting,
    transforms, loading and caching; the model owns how raw inputs become
    arrays.

    Parameters
    ----------
    kind
        Discriminator.
    params
        Settings passed to the model's data module, validated by that
        module's own spec class rather than here. This is the one place a
        specification is deliberately open-ended, and it is narrow and named.
    """

    kind: Literal["model"] = "model"
    params: Mapping[str, object] = Field(default_factory=dict)


#: A data source, discriminated on ``kind``.
SourceSpec = Annotated[TabularSourceSpec | ModelSourceSpec, Field(discriminator="kind")]
```

---

## 4. `src/rade_qnet/core/spec/hardware.py`

4569 bytes · SHA-256 `82afd37e1cf52534`

```python
"""
How a single job uses its machine.

:class:`HardwareSpec` answers *how does this one job use the hardware it has* --
device, precision, compilation, distribution, determinism. It deliberately
does **not** answer *where do jobs run*, which is placement and belongs to the
job-set spec.

Keeping them apart matters because they are configured by different people for
different reasons. "Use bfloat16" is a modelling decision made once. "Run
forty of these across four GPUs" is an operational decision made per run.
Conflating them is why "use the GPU" and "run forty in parallel" so often end
up as the same overloaded flag.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from ..runtime.seeding import Determinism
from .base import Spec

__all__ = ["HardwareSpec"]


class HardwareSpec(Spec):
    """
    Device, precision and determinism settings for one job.

    Parameters
    ----------
    device
        ``auto`` selects the best available accelerator and falls back to the
        CPU. Naming a device explicitly is a request, not a guarantee: an
        engine that cannot honour it warns and degrades rather than failing a
        run that would otherwise have succeeded.
    device_index
        Which device of the chosen type to use. ``None`` means the engine
        chooses, which is the correct setting under a job set -- the executor
        has already pinned device visibility per worker, so a job that also
        picks an index would fight it.
    precision
        Compute precision. Reduced precision is a throughput choice with
        numerical consequences, so it is explicit rather than inferred from
        the device.
    compile_model
        Whether to apply the engine's graph compiler. Off by default because
        compilation perturbs floating-point results, and a refactor being
        verified against a golden fixture needs that off.
    distributed
        Distribution strategy for a single job across several devices.
    determinism
        How hard to work for bit-for-bit reproducibility. See
        :mod:`rade_qnet.core.runtime.seeding` for what each level costs.
    threads_per_worker
        Intra-op thread budget. ``None`` leaves the library's default, which
        is correct for a single job and wrong under a process pool -- there
        the executor sets it, because N workers each claiming every core
        collapses throughput.
    """

    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    device_index: int | None = Field(default=None, ge=0)
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    compile_model: bool = False
    distributed: Literal["none", "ddp"] = "none"
    determinism: Determinism = "off"
    threads_per_worker: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _reject_impossible_combinations(self) -> HardwareSpec:
        """
        Reject settings that cannot be honoured on the requested device.

        Checked here rather than in the engine so the failure arrives before
        any data is read. A four-hour job that dies on an unsupported
        precision setting in its first optimiser step has wasted the data
        build for nothing.

        Returns
        -------
        HardwareSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if the combination cannot work.
        """
        # float16 on a CPU is not merely slow: most CPU kernels have no fp16
        # path, so the result is either an error deep in the engine or a
        # silent upcast that delivers none of the expected speed-up. bfloat16
        # is genuinely supported on modern CPUs and so is permitted.
        if self.precision == "fp16" and self.device == "cpu":
            raise SpecError(
                "precision='fp16' is not supported on device='cpu'; "
                "use 'bf16' for reduced precision on a CPU, or 'fp32'"
            )

        # Pinning an index while also asking for automatic device selection is
        # contradictory, and the two settings would be resolved by different
        # layers -- which is exactly how a job ends up on an unexpected device.
        if self.device == "auto" and self.device_index is not None:
            raise SpecError(
                "device_index cannot be set when device='auto'; "
                "name the device explicitly, or leave device_index unset"
            )
        return self
```

---

## 5. `src/rade_qnet/core/spec/jobs.py`

19900 bytes · SHA-256 `52fd20676baea8dc`

```python
"""
The specification for running one model across many jobs.

A job set is a list of jobs, each a full independent training run of the same
model, differing in its data slice and -- optionally -- in its architecture
complexity. This module describes that in the same declarative, validated,
hashable style as everything else in the package.

One schema, not two
-------------------
The job set introduces **no new field names**. A job set's ``defaults`` and
each job's overrides are *fragments of a* :data:`~.run.RunSpec`, written with
the same keys a single-run file uses, and :meth:`JobSetSpec.run_spec_for`
merges them and validates the result through the same
:func:`~.run.parse_run_spec` a single run goes through.

That is worth more than it first appears. A separate job-set vocabulary would
need its own translation layer, its own error messages and its own tests, and
would give every field name a second place to drift. Reusing the run schema
also means every rule Phase 1 wrote -- the sequence-versus-split compatibility
check, the hardware combination checks -- applies per job, for free, with no
further code.

Why the overrides stay raw until the merge
------------------------------------------
:attr:`JobSpec.overrides` is an unvalidated mapping, which is unusual in this
package and deliberate. The reasoning is in
:mod:`rade_qnet.core.spec.merge`: a validated fragment cannot distinguish a
field the user set from a field that defaulted, so merging validated specs
silently overwrites shared defaults with per-job defaults.

The openness is bounded. An override is validated the moment it is merged, by
the full run schema with ``extra="forbid"``, so a misspelled key is still a
load-time error -- just one raised against the merged specification, where the
message can name the real field. :meth:`JobSetSpec.validate_jobs` performs
that check for every job up front, so a typo in the fortieth job is reported
before the first one starts training.

Placement is not hardware
-------------------------
:class:`PlacementSpec` answers *where do jobs run*;
:class:`~.hardware.HardwareSpec` answers *how does one job use the machine it
has*. They are kept apart because they are set by different people for
different reasons: "use bfloat16" is a modelling decision made once, and "run
forty of these across four GPUs" is an operational decision made per run.
Conflating them is why "use the GPU" and "run forty in parallel" so often end
up as the same overloaded flag.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec
from .merge import deep_merge
from .run import RunSpec, parse_run_spec

__all__ = [
    "ExecutorName",
    "JobSetSpec",
    "JobSpec",
    "PlacementSpec",
    "dump_job_set_spec",
    "load_job_set_spec",
    "parse_job_set_spec",
]

#: Where jobs run. ``auto`` defers to
#: :func:`rade_qnet.orchestration.compute.policy.choose_placement`.
ExecutorName = Literal["auto", "local", "processes", "gpus"]

#: Top-level keys of a job entry that are the job's own metadata rather than
#: part of its run-specification override. Everything else in a job entry is
#: an override, which is what makes the shorthand in :class:`JobSpec` work.
_JOB_METADATA_KEYS = frozenset({"id", "overrides", "description"})

#: Fields of a job-set file that are the set's own, as opposed to the shared
#: run-specification defaults. Used to reject the common mistake of writing a
#: run-spec field at the top level instead of under ``defaults``.
_JOB_SET_KEYS = frozenset({"defaults", "jobs", "placement", "name", "output_root", "tags"})


class JobSpec(Spec):
    """
    One job: an identifier, and what it changes about the shared defaults.

    Two spellings are accepted, because both are natural in YAML::

        - id: EURUSD
          source: {params: {directory: clusters/EURUSD}}
          model: {units: 64}

        - id: EURUSD
          overrides:
            source: {params: {directory: clusters/EURUSD}}

    The first keeps the overrides at the top level, which is how a user would
    write them and how the architecture document presents them. The ``before``
    validator normalises it into the second. Both are supported rather than
    one, because the explicit form is clearer when a job's overrides are
    long, and the shorthand is clearer when they are two lines.

    Parameters
    ----------
    id
        Identifier for this job, unique within the set. It names the job's
        output directory, its catalog entries and its row in the manifest, and
        -- importantly -- it is what the job's seed is derived from, so
        changing it changes the model. See
        :meth:`rade_qnet.core.runtime.context.RunContext.for_job`.
    overrides
        A fragment of a run specification, merged over the set's defaults.
        Unvalidated here and validated on merge; see this module's docstring.
    description
        Free text for reports. Carries no behaviour.
    """

    id: str = Field(min_length=1)
    overrides: Mapping[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, value: object) -> object:
        """
        Fold top-level override keys into ``overrides``.

        Parameters
        ----------
        value
            The raw job entry from the specification file.

        Returns
        -------
        object
            Either the value unchanged, or a normalised mapping.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if both spellings are used at once. That
            combination has no obvious meaning -- it is unclear whether the
            top-level keys should merge into ``overrides`` or replace it --
            and silently picking one would be a configuration mistake that
            trains a model rather than reporting itself.
        """
        if not isinstance(value, Mapping):
            return value

        inline = {key: item for key, item in value.items() if key not in _JOB_METADATA_KEYS}
        if not inline:
            return value

        if "overrides" in value:
            raise SpecError(
                f"job {value.get('id', '<unnamed>')!r} mixes the two override spellings: "
                f"it sets 'overrides' and also names {sorted(inline)} at the top level. "
                f"Use one or the other"
            )
        return {
            **{key: item for key, item in value.items() if key in _JOB_METADATA_KEYS},
            "overrides": inline,
        }


class PlacementSpec(Spec):
    """
    Where the jobs of a set run, and how many at a time.

    Parameters
    ----------
    executor
        ``auto`` lets the placement policy decide from the hardware present
        and the size of the set, which is the right default: a sensible
        choice unaided is worth more than a hand-tuned one that is copied
        between machines and stops being sensible.
    workers
        How many jobs run concurrently. ``None`` lets the policy decide.
        Ignored by the sequential executor, which is always one.
    threads_per_worker
        Intra-op thread budget for each worker, set in the worker before the
        training library is imported. ``None`` lets the policy divide the
        machine's cores between the workers.

        This is the setting that decides whether a parallel run is faster or
        slower than a sequential one. Eight workers each defaulting to every
        core produces eight times the core count in threads, and throughput
        collapses below sequential -- with nothing in any log to say why.
    start_method
        How worker processes are created. ``spawn`` is the default and the
        only one tested: ``fork`` is unsafe in a process that has already
        initialised a threaded numerical library or a GPU context, and the
        resulting hangs are intermittent and extremely hard to attribute.
    memory_per_job_gb
        An estimate of a single job's peak resident memory, used by the
        policy to cap the worker count. ``None`` means do not cap, which is
        correct when jobs are small and wrong when they are not -- an
        over-subscribed machine kills workers with a signal that carries no
        explanation.

        Taken as an input rather than measured: estimating a job's memory
        before running it is not something the framework can do, and sniffing
        total system memory portably is not worth a dependency. A user who
        knows the figure can supply it; one who does not is no worse off.
    """

    executor: ExecutorName = "auto"
    workers: int | None = Field(default=None, ge=1)
    threads_per_worker: int | None = Field(default=None, ge=1)
    start_method: Literal["spawn", "forkserver"] = "spawn"
    memory_per_job_gb: float | None = Field(default=None, gt=0.0)


class JobSetSpec(Spec):
    """
    Shared defaults, a list of jobs, and where they run.

    Parameters
    ----------
    defaults
        A run-specification fragment applied to every job. Unvalidated on its
        own, because defaults alone are rarely a complete run -- a job set
        whose defaults name no data source is perfectly legitimate, since
        each job supplies its own.
    jobs
        The jobs. At least one, and identifiers must be unique.
    placement
        Where the jobs run.
    name
        A label for the set, used in the manifest and in log lines.
    output_root
        Root for everything the set writes. Each job writes beneath it.
    tags
        Free-form labels recorded against every bundle the set produces, so a
        whole set can be found later by one query.
    """

    defaults: Mapping[str, Any] = Field(default_factory=dict)
    jobs: tuple[JobSpec, ...] = Field(min_length=1)
    placement: PlacementSpec = Field(default_factory=PlacementSpec)
    name: str | None = None
    output_root: Path = Path("artifacts/rade_qnet")
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_duplicate_job_ids(self) -> JobSetSpec:
        """
        Reject a set in which two jobs share an identifier.

        Returns
        -------
        JobSetSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any identifier repeats. Left
            unchecked, the second job would write into the first one's
            directory, overwrite its bundle and replace its row in the
            manifest -- producing a set that reports forty successes and
            holds thirty-nine models.
        """
        counts = Counter(job.id for job in self.jobs)
        duplicated = sorted(name for name, count in counts.items() if count > 1)
        if duplicated:
            raise SpecError(
                f"job identifiers must be unique within a set; repeated: {duplicated}. "
                f"Each identifier names an output directory and derives a seed, so two "
                f"jobs sharing one would overwrite each other"
            )
        return self

    @property
    def job_ids(self) -> tuple[str, ...]:
        """
        Return every job identifier, in the order the set declares them.

        Returns
        -------
        tuple of str
            The identifiers.
        """
        return tuple(job.id for job in self.jobs)

    def run_spec_for(self, job: JobSpec | str) -> RunSpec:
        """
        Merge the defaults with one job's overrides and validate the result.

        This is where a job becomes a real run specification, and it is the
        only place that conversion happens -- so the merge rules, the
        validation and the error message all live in one spot.

        Parameters
        ----------
        job
            A job, or the identifier of one in this set.

        Returns
        -------
        RunSpec
            The validated specification for that job.

        Raises
        ------
        SpecError
            If the identifier is not in this set, or if the merged
            specification is invalid. The message names the job, because a
            validation error against a merged document is otherwise very hard
            to trace back to the override that caused it.
        """
        resolved = self.job(job) if isinstance(job, str) else job
        merged = deep_merge(self.defaults, resolved.overrides)
        # Tags are a property of the set, so they are unioned rather than
        # merged: a job may add its own, and dropping the set's would make a
        # set unfindable by the label it was launched under.
        if self.tags:
            merged["tags"] = tuple(dict.fromkeys((*self.tags, *merged.get("tags", ()))))
        merged.setdefault("output_root", str(self.output_root))
        return parse_run_spec(merged, origin=f"job {resolved.id!r}")

    def job(self, job_id: str) -> JobSpec:
        """
        Return the job with an identifier.

        Parameters
        ----------
        job_id
            The identifier.

        Returns
        -------
        JobSpec
            The job.

        Raises
        ------
        SpecError
            If no job has that identifier.
        """
        for candidate in self.jobs:
            if candidate.id == job_id:
                return candidate
        raise SpecError(f"no job named {job_id!r} in this set; available: {list(self.job_ids)}")

    def validate_jobs(self) -> dict[str, RunSpec]:
        """
        Merge and validate every job, reporting all failures at once.

        Called before a set starts, so a typo in the fortieth job's overrides
        is reported in the first second rather than three hours in. Reporting
        every failure together rather than the first matters just as much: a
        user who mistyped two keys should learn both now, not discover the
        second after fixing the first and waiting again.

        Returns
        -------
        dict
            Job identifier to validated run specification, in declaration
            order.

        Raises
        ------
        SpecError
            If any job fails to validate, with one entry per failing job.
        """
        specs: dict[str, RunSpec] = {}
        failures: list[str] = []
        for job in self.jobs:
            try:
                specs[job.id] = self.run_spec_for(job)
            except SpecError as error:
                failures.append(f"  {job.id}: {error}")

        if failures:
            raise SpecError(
                f"{len(failures)} of {len(self.jobs)} job(s) in this set are invalid:\n"
                + "\n".join(failures)
            )
        return specs


def parse_job_set_spec(payload: Mapping[str, Any], *, origin: str = "<mapping>") -> JobSetSpec:
    """
    Validate a mapping into a job-set specification.

    Accepts one convenience the schema does not: ``model`` at the top level,
    which is folded into ``defaults``. Naming the model once, at the top, is
    how the architecture document presents a job set and how a user would
    write one -- a job set is "this model, over these slices", and burying
    the model inside ``defaults`` reads as though it were an incidental
    default rather than the subject.

    Parameters
    ----------
    payload
        The raw mapping, typically parsed from YAML or JSON.
    origin
        Where the payload came from, for error messages.

    Returns
    -------
    JobSetSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the payload is not a mapping, carries run-specification fields at
        the top level, or fails validation.
    """
    if not isinstance(payload, Mapping):
        raise SpecError(
            f"{origin}: a job set must be a mapping of fields, received {type(payload).__name__}"
        )

    remaining = dict(payload)
    model = remaining.pop("model", None)
    if model is not None:
        defaults = dict(remaining.get("defaults", {}))
        if "model" in defaults:
            raise SpecError(
                f"{origin}: 'model' is set both at the top level and under 'defaults'. "
                f"Use one or the other"
            )
        # Normalised to a mapping rather than stored as the bare string the
        # user wrote. Left as a string it would not deep-merge: a job that
        # overrides `model.params` would be merging a mapping over a string,
        # which replaces rather than recurses, and the model's *name* would
        # vanish. The failure reads "model.name: Field required" from a file
        # whose only mention of a name is at the top level and plainly there.
        defaults["model"] = {"name": model} if isinstance(model, str) else model
        remaining["defaults"] = defaults

    # Caught here rather than by `extra="forbid"` so the message can say where
    # the field belongs. A user writing `training:` at the top level has made
    # a reasonable mistake -- it is where it goes in a single-run file -- and
    # "extra inputs are not permitted" would not tell them what to do.
    misplaced = sorted(set(remaining) - _JOB_SET_KEYS)
    if misplaced:
        raise SpecError(
            f"{origin}: {misplaced} are run-specification fields and belong under "
            f"'defaults' (where they apply to every job) or inside a job (where they "
            f"apply to one). A job set's own fields are {sorted(_JOB_SET_KEYS)}"
        )

    try:
        return JobSetSpec.model_validate(remaining)
    except ValueError as exc:
        raise SpecError(f"{origin}: invalid job set: {exc}") from exc


def load_job_set_spec(path: Path | str) -> JobSetSpec:
    """
    Load and validate a job-set specification from a YAML or JSON file.

    Parameters
    ----------
    path
        Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns
    -------
    JobSetSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the file is missing, unparseable, not a mapping, or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"job set specification file not found: {path}")

    try:
        # YAML is a superset of JSON, so one parser handles both suffixes and
        # a mislabelled file still loads.
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse {path} as YAML or JSON: {exc}") from exc

    return parse_job_set_spec(payload, origin=str(path))


def dump_job_set_spec(spec: JobSetSpec, path: Path | str) -> Path:
    """
    Write a job-set specification to a YAML or JSON file.

    The output round-trips exactly, which is what lets a set that was
    assembled in Python -- by expanding a portfolio, say -- be saved as the
    record of what was run.

    Parameters
    ----------
    spec
        The specification to write.
    path
        Destination. The suffix selects the format.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    SpecError
        If the suffix is not a recognised format.
    """
    path = Path(path)
    payload = spec.model_dump(mode="json")

    if path.suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    elif path.suffix == ".json":
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        raise SpecError(
            f"unsupported specification format {path.suffix!r}; use .yaml, .yml or .json"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
```

---

## 6. `src/rade_qnet/core/spec/merge.py`

7103 bytes · SHA-256 `89eeb600e0a48ae8`

```python
"""
Combining shared defaults with per-job overrides.

A job set states most of its configuration once and overrides a little of it
per job. Doing that correctly is a surprisingly narrow target, and the
failure mode is silent, so the rules are written down here and tested at
every nesting depth rather than inferred from the implementation.

Why this merges raw mappings rather than validated specs
---------------------------------------------------------
This module deliberately operates on plain mappings, *before* anything is
validated, and the merged result is validated once at the end.

Merging validated specs instead cannot work, and fails in the worst possible
way -- plausibly. A validated spec has no notion of which of its fields the
user set and which came from a default; both are simply fields with values.
So given shared defaults of ``model: {name: hybrid_gnn_rnn, units: 256}`` and
a job override of ``model: {gnn_layers: 1}``, validating the override on its
own produces a spec carrying ``units`` at *its default*, which then overwrites
the shared ``256``. The job trains at the wrong width, nothing raises, and
the only symptom is a model that underperforms for no visible reason.

Merging raw mappings has no such failure mode. A key the override does not
mention is simply not present in the override, so the default survives.

The rules
---------
========================== ============================================
Case                       Rule
========================== ============================================
Both sides are mappings    Recurse, key by key
Override is not a mapping  Replaces, wholesale
Either side is a sequence  Replaces, wholesale
Override value is ``None`` Replaces, with ``None``
Key only in the override   Added
Key only in the base       Kept
========================== ============================================

Two of those deserve their reasoning stated.

**Sequences replace rather than concatenate.** There is no identity to merge
list elements on -- two lists of reports, or of tags, or of column names have
no key by which an element of one corresponds to an element of the other. And
the case that matters most argues the same way: a job naming
``reports: [summary]`` means *those* reports, not those plus whatever the
defaults asked for.

**``None`` is a value, not an absence.** Many optional settings use ``None``
to mean "let the framework decide" -- ``device_index``, ``threads_per_worker``,
``workers``. A job that explicitly sets one back to ``None`` is asking for
that behaviour, and a merge that treated ``None`` as "no opinion" would make
it impossible to ask. Absence is expressed by omitting the key, which is
exactly what a configuration file makes easy.

"Deep merge" means at least three different things in common usage, which is
why none of this is left implicit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["deep_merge", "merge_all"]


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge ``override`` onto ``base``, recursing into nested mappings.

    Neither argument is modified, and the result shares no mutable mapping
    with either of them -- a nested dictionary in the output is a fresh one.
    That matters here more than it usually would: the base is a job set's
    shared defaults, merged once per job, and a result that aliased it would
    let one job's later mutation reach every other job's configuration.

    Parameters
    ----------
    base
        The shared defaults. Values survive unless the override mentions
        their key.
    override
        The per-job overrides. Wins at every depth.

    Returns
    -------
    dict
        A new mapping. Nested mappings are new too.

    Examples
    --------
    A nested override keeps its siblings, which is the entire point::

        >>> deep_merge(
        ...     {"model": {"units": 256, "layers": 3}},
        ...     {"model": {"layers": 1}},
        ... )
        {'model': {'units': 256, 'layers': 1}}

    A sequence replaces rather than extends::

        >>> deep_merge({"reports": ["summary", "curves"]}, {"reports": ["summary"]})
        {'reports': ['summary']}
    """
    merged: dict[str, Any] = {key: _copied(value) for key, value in base.items()}

    for key, value in override.items():
        existing = merged.get(key)
        # Both sides mappings is the only case that recurses. Everything else
        # replaces, including the case where one side is a mapping and the
        # other is not -- there is no sensible way to merge a mapping with a
        # scalar, and guessing one would hide a configuration mistake.
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(existing, value)
        else:
            merged[key] = _copied(value)

    return merged


def merge_all(*layers: Mapping[str, Any]) -> dict[str, Any]:
    """
    Merge a sequence of layers left to right, each overriding the last.

    Provided so a caller with three layers -- framework defaults, job-set
    defaults, per-job overrides -- expresses that as one call rather than as
    a fold that a reader has to evaluate to understand the precedence.

    Parameters
    ----------
    *layers
        Mappings in increasing order of precedence. The rightmost wins.

    Returns
    -------
    dict
        A new mapping. Empty if no layers were given.

    Examples
    --------
    Precedence reads left to right, like the argument list::

        >>> merge_all({"a": 1, "b": 1}, {"b": 2, "c": 2}, {"c": 3})
        {'a': 1, 'b': 2, 'c': 3}
    """
    merged: dict[str, Any] = {}
    for layer in layers:
        merged = deep_merge(merged, layer)
    return merged


def _copied(value: Any) -> Any:  # noqa: ANN401 -- merges arbitrary YAML payloads
    """
    Return a value detached from the mapping it came from.

    Only containers are copied, and only one level at a time: a nested
    mapping is copied recursively through this function, a sequence is
    rebuilt as a list, and everything else is returned as it is.

    Scalars are returned unchanged because they are immutable, and the
    arbitrary objects a ``params`` mapping may legitimately hold are
    returned unchanged because copying them is not this module's business --
    a deep copy here would silently duplicate whatever a user put in a
    specification, which could be large and need not be copyable at all.

    Parameters
    ----------
    value
        Any value from a specification mapping.

    Returns
    -------
    object
        The value, detached if it is a container.
    """
    if isinstance(value, Mapping):
        return {key: _copied(item) for key, item in value.items()}
    # Strings and bytes are sequences, and rebuilding them as lists would turn
    # a model name into a list of characters.
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_copied(item) for item in value]
    return value
```

---

## 7. `src/rade_qnet/core/spec/reports.py`

2488 bytes · SHA-256 `090156f7b9a0c8a7`

```python
"""
Which report writers a run produces.

Reports are declarative: enabling one is a specification change, resolved by
name through the component registry, not a pipeline change.

The default for ``fail_fast`` is the important setting in this module. A report
that raises produces a warning and the run completes. Discarding four hours of
training because a figure failed to render is not a trade anyone would make
deliberately, so it is not the default -- but it is available, because during
development a silently skipped report is worse than a loud one.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec

__all__ = ["ReportsSpec"]


class ReportsSpec(Spec):
    """
    Report selection and output settings.

    Parameters
    ----------
    enabled
        Registered report names, in the order they will run. A tuple rather
        than a set because order is occasionally meaningful -- a summary
        report that lists the figures produced must run after them.
    directory_name
        Subdirectory of the run directory that reports write beneath.
    fail_fast
        Whether a failing report should fail the run. See the module
        docstring.
    figure_format
        Image format for saved figures.
    figure_dpi
        Resolution for raster formats. Ignored for ``svg`` and ``pdf``.
    """

    enabled: tuple[str, ...] = ("summary",)
    directory_name: str = "reports"
    fail_fast: bool = False
    figure_format: Literal["png", "svg", "pdf"] = "png"
    figure_dpi: int = Field(default=150, ge=50, le=600)

    @model_validator(mode="after")
    def _reject_duplicate_reports(self) -> ReportsSpec:
        """
        Reject a report listed more than once.

        A duplicate would run twice and overwrite its own output, so it is
        always a mistake -- usually a merge artefact from combining job-set
        defaults with a per-job override.

        Returns
        -------
        ReportsSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any name appears more than once.
        """
        seen: set[str] = set()
        duplicates = sorted({name for name in self.enabled if name in seen or seen.add(name)})
        if duplicates:
            raise SpecError(f"reports listed more than once: {', '.join(duplicates)}")
        return self
```

---

## 8. `src/rade_qnet/core/spec/run.py`

12864 bytes · SHA-256 `1b82a04a916ca8f0`

```python
"""
The top-level specification: everything one run is.

:data:`RunSpec` is the single object a pipeline is constructed from, and it is
discriminated on ``task`` into a supervised run and an interactive one. The two
differ in exactly one respect -- where the training signal comes from -- which
is why they share a base holding everything else.

This module also owns loading. A specification file is the user's interface to
the framework, so parsing it, validating it and reporting a useful error when
it is wrong is a core concern rather than something each entry point
reimplements.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import Field, TypeAdapter, ValidationError, model_validator

from ..runtime.errors import SpecError
from .base import Spec
from .data import ModelSourceSpec, SourceSpec
from .hardware import HardwareSpec
from .reports import ReportsSpec
from .training import RlTrainingSpec, TorchTrainingSpec, TrainingSpec

__all__ = [
    "ComponentRef",
    "ReinforcementRunSpec",
    "RunSpec",
    "SupervisedRunSpec",
    "dump_run_spec",
    "load_run_spec",
    "parse_run_spec",
]

#: Upper bound on a seed, matching the numerical libraries the framework wraps.
_SEED_MODULUS = 2**32


class ComponentRef(Spec):
    """
    A reference to a registered component, plus its parameters.

    Two spellings are accepted, because both are natural in YAML::

        model: ridge  # no parameters
        model: {name: ridge, alpha: 1.0}  # with parameters

    The second form keeps parameters at the top level rather than nesting them
    under ``params``, which is how a user would expect to write them. The
    ``before`` validator below normalises both spellings into this type.

    Parameters
    ----------
    name
        Registered component name, resolved through the component registry.
    params
        Parameters for the component, validated by that component's own spec
        class rather than here. This framework cannot know a model's
        hyper-parameters, so the open-endedness is confined to this one named
        field instead of being granted to the whole spec.
    """

    name: str
    params: Mapping[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, value: object) -> object:
        """
        Normalise the accepted spellings into ``{name, params}``.

        Parameters
        ----------
        value
            The raw value from the specification file.

        Returns
        -------
        object
            Either the value unchanged, or a normalised mapping.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if a mapping is given with no ``name``.
        """
        if isinstance(value, str):
            return {"name": value, "params": {}}
        if isinstance(value, Mapping) and "params" not in value:
            remaining = dict(value)
            name = remaining.pop("name", None)
            if name is None:
                raise SpecError(
                    f"a component reference needs a 'name'; received keys {sorted(remaining)}"
                )
            return {"name": name, "params": remaining}
        return value

    def describe(self) -> str:
        """
        Return a compact one-line form, for logs and reports.

        Provided so a caller never has to fall back on the pydantic repr,
        which renders as ``name='ridge' params={}`` and reads as a debugging
        artefact when it lands in a report a human is meant to read.

        Returns
        -------
        str
            For example ``ridge`` or ``ridge(alpha=1.0)``.
        """
        if not self.params:
            return self.name
        rendered = ", ".join(f"{key}={self.params[key]!r}" for key in sorted(self.params))
        return f"{self.name}({rendered})"


class _RunSpecBase(Spec):
    """
    Everything a run needs regardless of where its training signal comes from.

    Not part of the public union -- see :data:`RunSpec`.
    """

    model: ComponentRef
    name: str | None = None
    seed: int = Field(default=0, ge=0, lt=_SEED_MODULUS)
    output_root: Path = Path("artifacts/rade_qnet")
    hardware: HardwareSpec = Field(default_factory=HardwareSpec)
    reports: ReportsSpec = Field(default_factory=ReportsSpec)
    tags: tuple[str, ...] = ()


class SupervisedRunSpec(_RunSpecBase):
    """
    A run that learns from a fixed dataset.

    Parameters
    ----------
    task
        Discriminator.
    source
        Where the data comes from. Defaults to the model's own data module,
        which is the case for any model complex enough to need one.
    training
        Engine-specific training settings.
    """

    task: Literal["supervised"] = "supervised"
    source: SourceSpec = Field(default_factory=ModelSourceSpec)
    training: TrainingSpec = Field(default_factory=TorchTrainingSpec)

    @model_validator(mode="after")
    def _check_sequence_split_compatibility(self) -> SupervisedRunSpec:
        """
        Reject a sequential model with a split that cannot respect windows.

        Returns
        -------
        SupervisedRunSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if a sequence length above one is
            combined with an explicit split, where the framework cannot
            insert the boundary gap that
            stops a window from straddling two splits. An explicit split is
            caller-supplied, so the caller must supply indices that already
            account for the window.
        """
        sequence_length = self.source.transforms.sequence.length
        if sequence_length > 1 and self.source.split.kind == "explicit":
            raise SpecError(
                f"sequence length {sequence_length} with an explicit split: the "
                f"framework cannot insert a boundary gap into caller-supplied "
                f"indices, so a window could straddle two splits. Either supply "
                f"indices that already leave a gap of {sequence_length - 1} "
                f"scenarios, or use a chronological split"
            )
        return self


class ReinforcementRunSpec(_RunSpecBase):
    """
    A run that learns by interacting with an environment.

    Parameters
    ----------
    task
        Discriminator.
    environment
        The environment to learn in, referenced by registered name. Phase 7
        replaces this with the richer ``EnvSpec`` in
        ``rade_qnet.core.spec.environment``; the shape of the reference does not
        change, so specifications written now remain valid.
    training
        Interactive training settings.
    """

    task: Literal["reinforcement"] = "reinforcement"
    environment: ComponentRef
    training: RlTrainingSpec = Field(default_factory=RlTrainingSpec)


#: A complete run, discriminated on ``task``.
RunSpec = Annotated[
    SupervisedRunSpec | ReinforcementRunSpec,
    Field(discriminator="task"),
]

#: Built once at import. Constructing a ``TypeAdapter`` compiles a validator,
#: which is expensive enough that doing it per call would be visible in a
#: forty-job run.
_RUN_SPEC_ADAPTER: TypeAdapter[SupervisedRunSpec | ReinforcementRunSpec] = TypeAdapter(RunSpec)


def _format_validation_error(error: ValidationError, *, origin: str) -> str:
    """
    Render a pydantic validation error as an actionable message.

    Pydantic's default rendering is accurate but hard to act on. This version
    leads with the dotted path to the offending field, because that is the
    thing the user has to go and edit.

    Parameters
    ----------
    error
        The validation error.
    origin
        Where the specification came from, for the message header.

    Returns
    -------
    str
        A multi-line message, one line per problem.
    """
    lines = [f"invalid specification from {origin}:"]
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"]) or "<root>"
        lines.append(f"  {location}: {detail['msg']}")
    return "\n".join(lines)


def parse_run_spec(payload: Mapping[str, Any], *, origin: str = "<mapping>") -> RunSpec:
    """
    Validate a mapping into a run specification.

    ``task`` may be omitted, in which case ``supervised`` is assumed. The
    default has to be applied here rather than declared on
    :class:`SupervisedRunSpec`, because a discriminated union reads its tag
    from the *input* before any member's defaults exist -- so a field default
    cannot serve as the union's default tag.

    Defaulting at this single entry point, rather than inside the schema, keeps
    the rule in one documented place. An unrecognised ``task`` still fails with
    pydantic's message listing the permitted values.

    Parameters
    ----------
    payload
        The raw mapping, typically parsed from YAML or JSON.
    origin
        Description of where the payload came from, used in error messages.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the payload is not a mapping, or if validation fails. The pydantic
        error is chained, so the full detail remains available, but the
        message is the actionable summary.
    """
    # Checked before anything else, because a YAML file whose top level is a
    # list is a common mistake and indexing it would raise a bare TypeError
    # from inside the defaulting below -- which says nothing a user can act on.
    if not isinstance(payload, Mapping):
        raise SpecError(
            f"{origin}: a run specification must be a mapping of fields, "
            f"received {type(payload).__name__}. A file whose top level is a "
            f"list is the usual cause -- remove the leading '- '."
        )
    if "task" not in payload:
        payload = {**payload, "task": "supervised"}
    try:
        return _RUN_SPEC_ADAPTER.validate_python(payload)
    except ValidationError as exc:
        raise SpecError(_format_validation_error(exc, origin=origin)) from exc


def load_run_spec(path: Path | str) -> RunSpec:
    """
    Load and validate a run specification from a YAML or JSON file.

    Parameters
    ----------
    path
        Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the file is missing, unreadable, not a mapping, or invalid. Every
        failure mode of "the user pointed us at the wrong file" is reported
        as a spec error, because in every case the fix is the user's.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"specification file not found: {path}")

    text = path.read_text(encoding="utf-8")
    try:
        # YAML is a superset of JSON, so one parser handles both suffixes and
        # a mislabelled file still loads.
        payload = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse {path} as YAML or JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        received = type(payload).__name__
        raise SpecError(f"{path} must contain a mapping at the top level, found {received}")
    return parse_run_spec(payload, origin=str(path))


def dump_run_spec(spec: RunSpec, path: Path | str) -> Path:
    """
    Write a run specification to a YAML or JSON file.

    The output round-trips exactly: loading it returns a specification equal to
    the one written. That is what makes a persisted spec a faithful record of
    how a model was produced, and it is asserted by test rather than assumed.

    Parameters
    ----------
    spec
        The specification to write.
    path
        Destination. The suffix selects the format.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    SpecError
        If the suffix is not a recognised format.
    """
    path = Path(path)
    # `mode="json"` reduces Path, Enum and similar fields to their serialised
    # form, which is what makes the round trip exact rather than approximate.
    payload = spec.model_dump(mode="json")

    if path.suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    elif path.suffix == ".json":
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        raise SpecError(
            f"unsupported specification format {path.suffix!r}; use .yaml, .yml or .json"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
```

---

## 9. `src/rade_qnet/core/spec/training.py`

12006 bytes · SHA-256 `70e3ec454e04fd70`

```python
"""
Training settings, discriminated by engine.

Each engine's settings are a separate type rather than a bag of optional
fields shared between them. The reason is that a shared bag cannot tell the
difference between "not set" and "not applicable": ``max_depth: null`` on a
neural network is meaningless, and ``gradient_clip_norm: 1.0`` on a tree model
is a setting that will be silently ignored. Both are configuration errors, and
with a discriminated union both are rejected at load time.

Note what is *absent* from every spec here: ``batch_size``. Batching belongs
to :class:`~rade_qnet.core.spec.data.LoaderSpec`, because it describes how data
is presented rather than how parameters are updated. Keeping it there means a
tuning sweep over batch size does not have to know which engine it is
configuring.

Reinforcement learning
----------------------
:class:`RlTrainingSpec` is not a member of the :data:`TrainingSpec` union. It
runs on the Torch engine, so it cannot be discriminated from
:class:`TorchTrainingSpec` by the ``engine`` field; instead it is selected one
level up, by ``task`` on the run spec. That keeps both discriminators
unambiguous.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec

__all__ = [
    "CheckpointSpec",
    "EarlyStoppingSpec",
    "RlTrainingSpec",
    "SchedulerSpec",
    "SklearnTrainingSpec",
    "TorchTrainingSpec",
    "TrainingSpec",
    "XGBoostTrainingSpec",
]


class EarlyStoppingSpec(Spec):
    """
    When to stop training before the epoch budget is exhausted.

    Parameters
    ----------
    enabled
        Whether early stopping is active. Defaults to ``False``, for two
        reasons. A run should train for the epoch budget it was configured
        with -- silently training for fewer, and keeping a different epoch's
        weights, is surprising behaviour to get without asking. And the
        default ``monitor`` is a validation metric, so defaulting to enabled
        would make a bare spec invalid for any run configured without a
        validation split.
    monitor
        Metric to watch. Must be a metric the run actually produces; an
        unknown name is caught when the callback is constructed, where the
        error can list what is available.
    patience
        Epochs without improvement before stopping.
    min_delta
        Improvement below this is treated as no improvement, which stops a
        run from continuing for fifty epochs on fourth-decimal noise.
    mode
        Whether a lower or higher value is better.
    """

    enabled: bool = False
    monitor: str = "val_loss"
    patience: int = Field(default=20, ge=1)
    min_delta: float = Field(default=0.0, ge=0.0)
    mode: Literal["min", "max"] = "min"


class CheckpointSpec(Spec):
    """
    Which parameters are kept, and which are restored at the end.

    Parameters
    ----------
    enabled
        Whether to checkpoint at all.
    monitor, mode
        Metric that identifies the best epoch.
    restore_best
        Whether to restore the best epoch's parameters when training ends.
        Default true: without it, the model that gets saved is the one from
        the last epoch while the metrics reported are from the best, so the
        bundle's weights and its metrics describe different models.
    keep_last
        How many checkpoints to retain on disk.
    """

    enabled: bool = True
    monitor: str = "val_loss"
    mode: Literal["min", "max"] = "min"
    restore_best: bool = True
    keep_last: int = Field(default=1, ge=1)


class SchedulerSpec(Spec):
    """
    Learning-rate schedule.

    Parameters
    ----------
    kind
        ``none`` holds the rate constant. ``plateau`` reduces it when a
        monitored metric stops improving; the others are time-based.
    factor
        Multiplicative reduction, for ``step`` and ``plateau``.
    patience
        Epochs without improvement before reducing, for ``plateau``.
    step_epochs
        Period between reductions, for ``step``.
    min_learning_rate
        Floor below which the rate will not be reduced.
    """

    kind: Literal["none", "step", "cosine", "plateau"] = "none"
    factor: float = Field(default=0.1, gt=0.0, lt=1.0)
    patience: int = Field(default=10, ge=1)
    step_epochs: int = Field(default=30, ge=1)
    min_learning_rate: float = Field(default=0.0, ge=0.0)


class TorchTrainingSpec(Spec):
    """
    Gradient-based training on the PyTorch engine.

    Parameters
    ----------
    engine
        Discriminator.
    learner
        Registered name of the update rule. ``supervised`` is forward, loss,
        backward. Naming it rather than implying it from the task is what
        lets a reinforcement-learning learner reuse this same engine.
    epochs
        Maximum passes over the training split. Early stopping usually ends
        the run sooner.
    learning_rate, optimiser, weight_decay
        Optimiser settings.
    loss
        Registered loss name.
    gradient_clip_norm
        Global gradient-norm clip. ``None`` disables clipping.
    early_stopping, checkpoint, scheduler
        Callback settings, each with usable defaults.
    """

    engine: Literal["torch"] = "torch"
    learner: str = "supervised"
    epochs: int = Field(default=100, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    optimiser: Literal["adam", "adamw", "sgd"] = "adam"
    weight_decay: float = Field(default=0.0, ge=0.0)
    loss: str = "mse"
    gradient_clip_norm: float | None = Field(default=None, gt=0.0)
    early_stopping: EarlyStoppingSpec = Field(default_factory=EarlyStoppingSpec)
    checkpoint: CheckpointSpec = Field(default_factory=CheckpointSpec)
    scheduler: SchedulerSpec = Field(default_factory=SchedulerSpec)

    @model_validator(mode="after")
    def _check_monitored_metrics_agree(self) -> TorchTrainingSpec:
        """
        Reject early stopping and checkpointing that disagree.

        Returns
        -------
        TorchTrainingSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if the two callbacks monitor the same
            metric in opposite directions, which would make them select
            different epochs as
            best -- so the saved weights and the stopping decision would
            describe different models.
        """
        if (
            self.early_stopping.enabled
            and self.checkpoint.enabled
            and self.early_stopping.monitor == self.checkpoint.monitor
            and self.early_stopping.mode != self.checkpoint.mode
        ):
            raise SpecError(
                f"early_stopping and checkpoint both monitor "
                f"{self.early_stopping.monitor!r} but disagree on direction "
                f"({self.early_stopping.mode!r} vs {self.checkpoint.mode!r}); "
                f"they would select different epochs as best"
            )
        return self


class XGBoostTrainingSpec(Spec):
    """
    One-shot gradient-boosted tree training.

    No epoch loop: the engine translates this into a single ``train`` call and
    reports the boosting history through the same result contract a neural
    network produces.

    Parameters
    ----------
    engine
        Discriminator.
    n_estimators, max_depth, learning_rate, subsample, colsample_bytree,
    min_child_weight, reg_lambda
        Boosting settings, named as XGBoost names them so that a user who
        knows the library is not made to learn a translation layer.
    objective
        Training objective.
    early_stopping_rounds
        Boosting rounds without improvement before stopping. ``None``
        disables it. The library's own implementation is used rather than a
        framework reimplementation, because it operates per round inside the
        booster.
    """

    engine: Literal["xgboost"] = "xgboost"
    n_estimators: int = Field(default=500, ge=1)
    max_depth: int = Field(default=6, ge=1)
    learning_rate: float = Field(default=0.05, gt=0.0)
    subsample: float = Field(default=1.0, gt=0.0, le=1.0)
    colsample_bytree: float = Field(default=1.0, gt=0.0, le=1.0)
    min_child_weight: float = Field(default=1.0, ge=0.0)
    reg_lambda: float = Field(default=1.0, ge=0.0)
    objective: str = "reg:squarederror"
    early_stopping_rounds: int | None = Field(default=50, ge=1)


class SklearnTrainingSpec(Spec):
    """
    One-shot scikit-learn estimator training.

    Estimator hyper-parameters belong to the model's own spec, not here -- a
    ridge regression's ``alpha`` is part of what the model *is*. This spec
    carries only what the framework needs to drive the fit.

    Parameters
    ----------
    engine
        Discriminator.
    fit_params
        Extra keyword arguments forwarded to the estimator's ``fit``, such as
        sample weights.
    """

    engine: Literal["sklearn"] = "sklearn"
    fit_params: Mapping[str, object] = Field(default_factory=dict)


#: Supervised training settings, discriminated on ``engine``.
TrainingSpec = Annotated[
    TorchTrainingSpec | XGBoostTrainingSpec | SklearnTrainingSpec,
    Field(discriminator="engine"),
]


class RlTrainingSpec(Spec):
    """
    Interactive training on the PyTorch engine.

    Driven by update count rather than epochs, because the source is
    unbounded: there is no meaningful notion of a pass over an environment.

    Parameters
    ----------
    engine
        Fixed to ``torch``. Present for symmetry with
        :data:`TrainingSpec` and because a future engine would need it.
    learner
        Which update rule. ``pathwise`` requires a differentiable
        environment and back-propagates a risk measure through the simulated
        dynamics; the others are transition-based.
    total_steps
        Total optimisation steps.
    steps_per_update, batch_size
        How much experience is gathered per update, and how much of it each
        update consumes.
    discount
        Reward discount factor. Ignored by ``pathwise``, which optimises a
        risk measure of the terminal outcome directly.
    gradient_clip_norm, learning_rate, optimiser
        Optimiser settings, as for supervised training.
    evaluate_every_steps, evaluation_episodes
        How often to run evaluation episodes, and how many.
    """

    engine: Literal["torch"] = "torch"
    learner: Literal["dqn", "ppo", "sac", "pathwise"] = "ppo"
    total_steps: int = Field(default=100_000, ge=1)
    steps_per_update: int = Field(default=2048, ge=1)
    batch_size: int = Field(default=256, ge=1)
    discount: float = Field(default=0.99, ge=0.0, le=1.0)
    learning_rate: float = Field(default=3e-4, gt=0.0)
    optimiser: Literal["adam", "adamw", "sgd"] = "adam"
    gradient_clip_norm: float | None = Field(default=0.5, gt=0.0)
    evaluate_every_steps: int = Field(default=10_000, ge=1)
    evaluation_episodes: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _check_batch_fits_the_update(self) -> RlTrainingSpec:
        """
        Reject a batch larger than the experience gathered per update.

        Returns
        -------
        RlTrainingSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` for an on-policy learner whose batch
            exceeds the experience collected between updates, which would
            require reusing samples the algorithm assumes are fresh.
        """
        on_policy = {"ppo"}
        if self.learner in on_policy and self.batch_size > self.steps_per_update:
            raise SpecError(
                f"learner={self.learner!r} is on-policy, so batch_size "
                f"({self.batch_size}) cannot exceed steps_per_update "
                f"({self.steps_per_update})"
            )
        return self
```

---

## 10. `src/rade_qnet/core/spec/tune.py`

17481 bytes · SHA-256 `04de021fa2194e1d`

```python
"""
Declaring a hyper-parameter search.

A search is a base run specification plus a space to vary, a budget, and a
rule for deciding which trial won. None of those fit in a
:data:`~.run.RunSpec`, which describes *one* run -- so this is a separate
document, and a trial is the result of folding one point in the space into
the base.

Why the overrides stay raw until the merge
-------------------------------------------
Exactly as in :mod:`~.jobs`: a search dimension names a dotted path into a
run specification, and the value it proposes is merged in before anything is
validated. Validating a fragment on its own is not possible -- half a
specification is not a specification -- and the merged document is validated
in full, so nothing escapes.

The difference from a job set is who writes the values. A job set's overrides
are typed by a person; a search's are generated. That makes the validation
*more* important rather than less, because a generated value that lands on a
path nobody spelled correctly would otherwise be explored, measured, and
reported as making no difference.

Defect 7
--------
``rade_ml_pt``'s ``TunePipeline._resolve_trial_training_config`` passed
``learning_rate`` and ``batch_size`` straight into
``dataclasses.replace(TrainingConfig, ...)``, which raises ``TypeError``
because neither is a field of that class. It never fired in production only
because the hybrid model overrode the method -- so the framework's own tuning
path was broken for every model that did not.

The cure is structural rather than a fix: a trial's overrides go through the
same validated merge a job's do, so an unknown path fails at trial
construction, before a single epoch runs, with a message naming the path.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..runtime.errors import SpecError
from .base import Spec
from .merge import deep_merge
from .run import RunSpec, parse_run_spec

__all__ = [
    "Dimension",
    "SearchSpace",
    "TuneSpec",
    "load_tune_spec",
    "parse_tune_spec",
]

#: Directions an objective can be optimised in. Spelled out rather than
#: inferred from the metric's name: ``r2`` is maximised and ``mae`` minimised,
#: and a framework that guessed from a substring would eventually guess wrong
#: on a custom metric and silently select the worst trial in the search.
Direction = Literal["minimise", "maximise"]


class Dimension(Spec):
    """
    One axis of a search, and the values it may take.

    Parameters
    ----------
    path
        A dotted path into a run specification, such as
        ``training.learning_rate`` or ``model.params.hidden_size``. Dotted
        rather than nested so that a space reads as a flat list of knobs,
        which is how anyone thinks about one.
    values
        The values to choose among, for a categorical or discrete axis.
    low
        Lower bound, for a continuous axis.
    high
        Upper bound, for a continuous axis.
    log
        Whether to sample a continuous axis on a logarithmic scale. Almost
        always correct for a learning rate or a regularisation strength,
        where the interesting range spans orders of magnitude and a uniform
        sample would spend nine tenths of its budget in the top decade.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if the axis is neither a clean enumeration
        nor a clean range.
    """

    path: str = Field(min_length=1)
    values: tuple[Any, ...] = ()
    low: float | None = None
    high: float | None = None
    log: bool = False

    @model_validator(mode="after")
    def _reject_an_ambiguous_axis(self) -> Dimension:
        """
        Require exactly one of an enumeration or a range, and a sane range.

        Returns
        -------
        Dimension
            The validated dimension.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError``. An axis with both would leave the
            sampler to decide which to honour, and an axis with neither has
            nothing to propose -- in both cases the search would run to
            completion and report a result, which is why this is rejected
            at parse time rather than handled at sample time.
        """
        low, high = self.low, self.high
        bounded = low is not None and high is not None

        if bool(self.values) == bounded:
            described = "both an enumeration and a range" if self.values else "neither"
            raise SpecError(
                f"dimension {self.path!r} gives {described}; it must give either "
                f"'values' or both 'low' and 'high'"
            )

        if not bounded:
            if self.log:
                raise SpecError(
                    f"dimension {self.path!r} sets log=True but enumerates its "
                    f"values; a scale only applies to a range"
                )
            return self

        # Narrowed by `bounded`, and re-read from the locals so the narrowing
        # is visible to a reader and to a type checker alike.
        assert low is not None and high is not None
        if low >= high:
            raise SpecError(
                f"dimension {self.path!r} has low={low} and high={high}; the "
                f"range is empty, so every trial would propose the same value"
            )
        if self.log and low <= 0:
            raise SpecError(
                f"dimension {self.path!r} is sampled logarithmically but has "
                f"low={low}; a log scale is undefined at or below zero"
            )
        return self

    @property
    def is_continuous(self) -> bool:
        """
        Return whether this axis is a range rather than an enumeration.

        Returns
        -------
        bool
            True for a range.
        """
        return self.low is not None


class SearchSpace(Spec):
    """
    The axes a search varies, and nothing else.

    A type of its own rather than a bare list, so that the checks which
    belong to a space as a whole -- no path proposed twice -- have somewhere
    to live.

    Parameters
    ----------
    dimensions
        The axes. At least one: a search over nothing is a single run, and
        should be written as one.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a path appears more than once.
    """

    dimensions: tuple[Dimension, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _reject_duplicate_paths(self) -> SearchSpace:
        """
        Reject a space that varies the same path twice.

        Returns
        -------
        SearchSpace
            The validated space.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError``. Two axes over one path means the
            second silently wins every merge, so half the search budget is
            spent exploring a dimension that has no effect -- and the
            report would show it having none, which reads as a finding
            rather than a bug.
        """
        counts = Counter(dimension.path for dimension in self.dimensions)
        repeated = sorted(path for path, count in counts.items() if count > 1)
        if repeated:
            raise SpecError(
                f"a search space must not vary the same path twice; repeated: "
                f"{repeated}. The later axis would win every merge, so the "
                f"earlier one would be explored and have no effect"
            )
        return self

    @property
    def paths(self) -> tuple[str, ...]:
        """
        Return every path this space varies, in declaration order.

        Returns
        -------
        tuple of str
            The dotted paths.
        """
        return tuple(dimension.path for dimension in self.dimensions)


class TuneSpec(Spec):
    """
    A base run, a space to search over it, and how the winner is chosen.

    Parameters
    ----------
    base
        A run-specification fragment every trial starts from. Unvalidated on
        its own, for the reason in the module docstring: a base that omits
        whatever the space supplies is a legitimate base, and validating it
        alone would reject it.
    space
        The axes to vary.
    trials
        How many to run.
    objective
        The metric to optimise.
    direction
        Whether that metric is better high or low.
    objective_split
        Which split the objective is read from. Validation by default, which
        is the only defensible choice: selecting on test turns the held-out
        split into part of the training procedure, and the reported test
        metric then overstates the model.
    sampler
        How points are proposed. ``random`` samples independently; ``grid``
        enumerates the product of the axes, which requires every axis to be
        an enumeration.
    seed
        Seed for the proposal sequence, so a search is reproducible.
    refit
        Whether to retrain the winner on train and validation combined once
        the search is over. Off by default, because it produces a model
        whose reported objective was measured on data it has now seen, and
        that should be an explicit choice.
    name
        A label for the search.

    Raises
    ------
    ValidationError
        Wrapping a ``SpecError`` if a grid search is asked to enumerate a
        continuous axis.
    """

    base: Mapping[str, Any] = Field(default_factory=dict)
    space: SearchSpace
    trials: int = Field(default=20, ge=1)
    objective: str = "mae"
    direction: Direction = "minimise"
    objective_split: str = "validation"
    sampler: Literal["random", "grid"] = "random"
    seed: int = Field(default=0, ge=0)
    refit: bool = False
    name: str | None = None

    @model_validator(mode="after")
    def _reject_a_grid_over_a_range(self) -> TuneSpec:
        """
        Reject a grid over an axis that is a range rather than a list.

        Returns
        -------
        TuneSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError``. The alternative -- discretising a
            range into some number of steps the user did not choose --
            would make the search silently coarser than it reads, and the
            number of steps would be a framework decision affecting the
            result.
        """
        if self.sampler != "grid":
            return self
        continuous = sorted(
            dimension.path for dimension in self.space.dimensions if dimension.is_continuous
        )
        if continuous:
            raise SpecError(
                f"a grid search cannot enumerate the continuous axis(es) {continuous}. "
                f"Either list their values explicitly or use the random sampler; "
                f"discretising a range into steps you did not choose would make "
                f"the search coarser than it reads"
            )
        return self

    @property
    def grid_size(self) -> int:
        """
        Return how many distinct points a grid over this space contains.

        Returns
        -------
        int
            The product of the axis lengths, or zero for a space with a
            continuous axis, which has no finite grid.
        """
        if any(dimension.is_continuous for dimension in self.space.dimensions):
            return 0
        size = 1
        for dimension in self.space.dimensions:
            size *= len(dimension.values)
        return size

    def run_spec_for(self, overrides: Mapping[str, Any], *, trial: int) -> RunSpec:
        """
        Fold one proposal into the base and validate the result.

        The cure for defect 7, and the only place a trial becomes a real run
        specification -- so the merge rules, the validation and the error
        message live in one spot, as they do for a job set.

        Parameters
        ----------
        overrides
            A nested mapping, already expanded from the dotted paths.
        trial
            The trial number, named in errors. A validation failure against
            a merged document is otherwise very hard to trace back to the
            proposal that caused it, and in a search nobody typed the
            proposal.

        Returns
        -------
        RunSpec
            The validated specification for that trial.

        Raises
        ------
        SpecError
            If the merged specification is invalid.
        """
        merged = deep_merge(self.base, overrides)
        return parse_run_spec(merged, origin=f"trial {trial}")

    def is_better(self, candidate: float, incumbent: float) -> bool:
        """
        Return whether one objective value beats another.

        One implementation of the comparison, so that the direction is
        honoured identically wherever a winner is picked. A search that
        compared with ``<`` in one place and ``>`` in another would select
        correctly for half its configurations.

        Parameters
        ----------
        candidate
            The new value.
        incumbent
            The best so far.

        Returns
        -------
        bool
            True if the candidate is better. Ties go to the incumbent, which
            makes the winner the *earliest* best trial and therefore stable
            under a re-run.
        """
        if self.direction == "maximise":
            return candidate > incumbent
        return candidate < incumbent


def parse_tune_spec(payload: Mapping[str, Any], *, origin: str = "<mapping>") -> TuneSpec:
    """
    Validate a mapping into a tuning specification.

    Accepts two conveniences the schema does not, both because they are how
    a person writes a search rather than how it is structured:

    - ``model`` at the top level, folded into ``base``. A search is "this
      model, over these knobs", and burying the model inside ``base`` reads
      as though it were incidental.
    - ``space`` as a plain mapping of path to axis, so a space can be
      written as a dictionary of knobs rather than a list of objects with a
      repeated ``path`` key.

    Parameters
    ----------
    payload
        The raw mapping, normally from YAML.
    origin
        Where it came from, named in errors.

    Returns
    -------
    TuneSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the mapping is not a valid tuning specification.
    """
    fields = dict(payload)

    model = fields.pop("model", None)
    if model is not None:
        base = dict(fields.get("base") or {})
        base["model"] = {"name": model} if isinstance(model, str) else model
        fields["base"] = base

    space = fields.get("space")
    if isinstance(space, Mapping) and "dimensions" not in space:
        fields["space"] = {
            "dimensions": [
                {"path": path, **_axis(path, axis)} for path, axis in space.items()
            ]
        }

    try:
        return TuneSpec.model_validate(fields)
    except SpecError:
        raise
    except ValueError as error:
        raise SpecError(f"{origin} is not a valid tuning specification: {error}") from error


def load_tune_spec(path: Path | str) -> TuneSpec:
    """
    Load and validate a tuning specification from a YAML or JSON file.

    Parameters
    ----------
    path
        Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns
    -------
    TuneSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the file is missing, unparseable, not a mapping, or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"tuning specification file not found: {path}")

    try:
        # YAML is a superset of JSON, so one parser handles both suffixes and
        # a mislabelled file still loads.
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse {path} as YAML or JSON: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise SpecError(
            f"{path} holds a {type(payload).__name__} at the top level; a tuning "
            f"specification must be a mapping"
        )
    return parse_tune_spec(payload, origin=str(path))


def _axis(path: str, axis: object) -> Mapping[str, Any]:
    """
    Normalise one entry of the shorthand space mapping.

    Parameters
    ----------
    path
        The dotted path, named in errors.
    axis
        Either a mapping of axis fields, or a bare sequence of values --
        which is the shortest honest way to write a categorical knob and is
        what a user reaches for first.

    Returns
    -------
    Mapping
        Axis fields, without the path.

    Raises
    ------
    SpecError
        If the entry is neither.
    """
    if isinstance(axis, Mapping):
        return {key: value for key, value in axis.items() if key != "path"}
    if isinstance(axis, Sequence) and not isinstance(axis, str | bytes):
        return {"values": tuple(axis)}
    raise SpecError(
        f"the search axis for {path!r} is a {type(axis).__name__}; it must be "
        f"a list of values or a mapping with 'values' or 'low' and 'high'"
    )
```

