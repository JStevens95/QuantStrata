# `src/rade_qnet/engines/torch`

12 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 90 | 4121 | `a386a3c89b5a653f` |
| 2 | `callbacks.py` | 689 | 21954 | `dd2f519795a353cc` |
| 3 | `checkpoint.py` | 300 | 10634 | `ede6db2968503f60` |
| 4 | `distributed.py` | 276 | 9479 | `0c0492e6dc473abb` |
| 5 | `engine.py` | 578 | 19124 | `2d7faa241a18fb2b` |
| 6 | `hardware.py` | 369 | 12848 | `1e6d10f5dc810e55` |
| 7 | `loaders.py` | 288 | 10182 | `856ddcfb1c2ca4e9` |
| 8 | `loops.py` | 462 | 14963 | `f43d783b05f0da87` |
| 9 | `losses.py` | 348 | 9939 | `43b81664fc961647` |
| 10 | `materialise.py` | 329 | 11265 | `b19a9c1b5031cb03` |
| 11 | `predictor.py` | 179 | 6452 | `87c7e1bff8d8aba1` |
| 12 | `seeding.py` | 124 | 5679 | `34f9f4cfe1207350` |

---

## 1. `src/rade_qnet/engines/torch/__init__.py`

4121 bytes · SHA-256 `a386a3c89b5a653f`

```python
"""
The PyTorch engine.

This is the framework's primary engine and the one the flagship model uses.
Its central idea is a split of concerns that most training code leaves tangled:

*The loop* decides when to step, validate, checkpoint and stop.
*The learner* decides what a single update means.

A supervised regression, a deep Q-network and a pathwise hedging objective are
then three learners sharing one loop, rather than three training scripts.

Modules
-------
``engine.py``
    The ``Engine`` implementation: build, materialise, fit, checkpoint,
    predict.  [Phase 2]
``loops.py``
    ``fit_epochs`` -- passes over a finite dataset -- and the ``Learner``
    protocol it drives.  ``fit_steps``, for a fixed number of updates against
    an unbounded source, arrives with reinforcement learning.  Supervised
    learning naturally wants the first, reinforcement learning the second.
    [Phase 2 / Phase 7]
``learners/``
    Update rules, one per algorithm.
``callbacks.py``
    Early stopping, checkpointing, learning-rate scheduling, gradient-norm
    tracking.  [Phase 2]
``losses.py``
    The loss registry, including the asymmetric and quantile objectives.  A
    loss is engine code because the *learner* consumes it; a loss only one
    model needs is registered from that model's package instead.  [Phase 2]
``hardware.py``
    Device selection, autocast and precision policy, and ``torch.compile``
    application -- resolved from ``HardwareSpec``.  [Phase 2]
``distributed.py``
    Distributed data-parallel setup and teardown.  Wrapping happens *after*
    lazy parameters are materialised, which is the ordering the previous
    implementation got wrong.  [Phase 2]
``materialise.py``
    Runs one dummy forward pass from the input signature so lazy modules
    acquire concrete shapes before any optimiser, checkpoint or distributed
    wrapper touches them.  [Phase 2]
``checkpoint.py``
    Checkpoints as ``state_dict`` payloads rather than pickled modules, so a
    saved model survives a refactor and can be loaded without executing
    arbitrary code.  [Phase 2]
``loaders.py``
    Conversion of a ``BatchSource``'s NumPy batches into device-resident
    tensors.  Static inputs are kept out of per-sample collation and uploaded
    to the device once.  No ``DataLoader`` is constructed: a ``BatchSource``
    already yields whole batches, so prefetching across processes would mean
    pickling the source for no gain on in-memory arrays.  [Phase 2]

Planned modules
---------------
``risk.py``
    Differentiable risk measures (mean-variance, conditional value at risk,
    entropic) used as objectives by the pathwise learner.  [Phase 7]
``predictor.py``
    Batched inference, including the precompute path for models that can cache
    an encoding of their static inputs.  [Phase 5]

Importing this package registers its components
-----------------------------------------------
Importing ``rade_qnet.engines.torch`` registers :class:`TorchEngine` under the
name ``"torch"`` and :class:`SupervisedLearner` under ``"supervised"``, which
is what lets a specification name them as strings.  It also registers
:func:`~rade_qnet.engines.torch.seeding.seed_torch`, without which
``seed_everything`` would leave Torch unseeded -- so two runs of one
configuration would differ in every weight initialisation while both reported
the same seed.

The import is eager rather than lazy on purpose.  A registry populated only
once someone happens to have imported the right module is the classic source
of "no engine named 'torch'" from a run whose configuration is perfectly
correct, and the only reliable cure is for the registration to be a
consequence of importing the package that owns it.  Importing this package
already implies that ``torch`` is installed, so nothing is paid by a host that
does not use it.
"""

from .engine import TorchEngine
from .learners.supervised import SupervisedLearner

# Importing the name is what registers the seeder, since registration happens
# at that module's import.
from .seeding import seed_torch

__all__ = ["SupervisedLearner", "TorchEngine", "seed_torch"]
```

---

## 2. `src/rade_qnet/engines/torch/callbacks.py`

21954 bytes · SHA-256 `dd2f519795a353cc`

```python
"""
What the loop consults at the end of each epoch.

The loop decides *when* things happen; a callback decides *what* happens. Early
stopping, best-weight tracking and the learning-rate schedule are therefore
three small objects consulted at one point, rather than three sets of
conditionals threaded through the training loop.

One monitor implementation, shared
----------------------------------
:class:`Monitor` holds the "has this improved?" logic, and both
:class:`EarlyStopping` and :class:`BestCheckpoint` use it rather than each
implementing the comparison.

That is not only to avoid duplication. The comparison has three pieces of
state -- the direction, the minimum delta, and whether this is the first epoch
-- and two independent implementations of it will eventually disagree about
some epoch. When they do, training stops at one epoch while the weights kept
are another's, and the run reports the metrics of a model it did not save.
``TorchTrainingSpec`` already refuses a spec where the two monitor the same
metric in opposite *directions*; sharing the implementation closes the rest.

When an unknown monitor name is caught
--------------------------------------
``EarlyStoppingSpec.monitor``'s docstring says an unknown name is caught when
the callback is constructed. That is true for the case that matters and worth
stating precisely:

* Monitoring ``val_loss`` on a run with no validation split is caught at
  construction, because that is knowable then. It is also the most common and
  most damaging version -- early stopping that silently never fires.
* A misspelled *custom* metric cannot be caught at construction, because the
  set of metric names does not exist until an epoch has produced one. It is
  caught at the end of the first epoch, which is as early as it is knowable,
  and the message lists the names that were actually produced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

import torch

from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from .checkpoint import restore_state_dict, snapshot_state_dict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ...core.spec.training import CheckpointSpec, EarlyStoppingSpec, SchedulerSpec

__all__ = [
    "BestCheckpoint",
    "Callback",
    "EarlyStopping",
    "EpochContext",
    "GradientNorms",
    "LearningRateSchedule",
    "Monitor",
]

_LOGGER = get_logger(__name__)

#: Metric names every epoch produces, so they are always valid monitors.
_ALWAYS_AVAILABLE = ("train_loss", "val_loss")


@dataclass(frozen=True, slots=True)
class EpochContext:
    """
    What a callback is told at the end of an epoch.

    Parameters
    ----------
    epoch
        Zero-based epoch index.
    train_loss
        Mean training loss over the epoch.
    val_loss
        Mean validation loss, or ``None`` when there is no validation split.
    metrics
        Any additional metrics for the epoch.
    learning_rate
        The rate that was in effect during the epoch.
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    metrics: Mapping[str, float] = field(default_factory=dict)
    learning_rate: float | None = None

    def value_of(self, monitor: str) -> float | None:
        """
        Return the monitored metric's value for this epoch.

        Parameters
        ----------
        monitor
            Metric name.

        Returns
        -------
        float or None
            The value, or ``None`` when the metric exists but has no value
            this epoch -- ``val_loss`` on a run with no validation split.

        Raises
        ------
        EngineError
            If the name is not a metric this epoch produced, listing what it
            did produce. See the module docstring for why this is checked
            here rather than at construction.
        """
        if monitor == "train_loss":
            return self.train_loss
        if monitor == "val_loss":
            return self.val_loss
        if monitor in self.metrics:
            return self.metrics[monitor]
        raise EngineError(
            f"nothing monitors {monitor!r}: epoch {self.epoch} produced "
            f"{[*_ALWAYS_AVAILABLE, *sorted(self.metrics)]}. Check the spelling "
            f"in early_stopping.monitor or checkpoint.monitor"
        )

    def available(self) -> tuple[str, ...]:
        """
        Return every metric name this epoch produced.

        Returns
        -------
        tuple of str
            Metric names, sorted within each group.
        """
        return (*_ALWAYS_AVAILABLE, *sorted(self.metrics))


class Callback(Protocol):
    """
    Something the loop consults at the end of each epoch.

    A protocol rather than a base class, for the same reason ``BatchSource``
    is one: a model package can supply a callback without importing the
    engine, and the framework's own callbacks are held to the same definition
    a user's is.
    """

    def on_epoch_end(self, context: EpochContext) -> None:
        """
        React to a finished epoch.

        Parameters
        ----------
        context
            The epoch's outcome.
        """
        ...

    @property
    def should_stop(self) -> bool:
        """
        Whether training should end now.

        Returns
        -------
        bool
            True to stop. A callback with no opinion returns ``False``.
        """
        ...


class Monitor:
    """
    Tracks whether a watched metric has improved, and by enough.

    Parameters
    ----------
    monitor
        Metric name.
    mode
        ``min`` if lower is better, ``max`` if higher is.
    min_delta
        Improvement at or below this counts as no improvement. Guards against
        a run continuing for fifty epochs on fourth-decimal noise.
    """

    def __init__(
        self,
        *,
        monitor: str,
        mode: Literal["min", "max"] = "min",
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.min_delta = min_delta
        self.best_value: float | None = None
        self.best_epoch: int | None = None

    def is_improvement(self, value: float) -> bool:
        """
        Return whether a value beats the best seen by more than ``min_delta``.

        Parameters
        ----------
        value
            The candidate value.

        Returns
        -------
        bool
            True if it is an improvement. The first finite value always is.
        """
        # A NaN loss is never an improvement.  Without this, `nan < best` is
        # False and `best < nan` is also False, so a diverged run would keep
        # whichever epoch happened to be compared first -- and on `max` mode
        # with a naive comparison a NaN can win outright.
        if not math.isfinite(value):
            return False
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value - self.min_delta
        return value > self.best_value + self.min_delta

    def update(self, context: EpochContext) -> bool:
        """
        Record an epoch and return whether it was the new best.

        Parameters
        ----------
        context
            The epoch's outcome.

        Returns
        -------
        bool
            True if this epoch improved on every previous one.

        Raises
        ------
        EngineError
            If the monitored metric has no value this epoch -- ``val_loss``
            with no validation split. Raised rather than treated as "no
            improvement", because the latter makes the callback a silent
            no-op for the whole run.
        """
        value = context.value_of(self.monitor)
        if value is None:
            raise EngineError(
                f"monitor={self.monitor!r} has no value at epoch {context.epoch}; "
                f"a run with no validation split cannot monitor a validation "
                f"metric. Monitor {'train_loss'!r}, or configure a validation "
                f"fraction"
            )

        if not self.is_improvement(value):
            return False
        self.best_value = value
        self.best_epoch = context.epoch
        return True

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "best_epoch": self.best_epoch,
            "best_value": self.best_value,
        }


class EarlyStopping:
    """
    Stops training once the monitored metric has not improved for a while.

    Parameters
    ----------
    spec
        Monitor, patience, direction and minimum delta.
    has_validation
        Whether the run has a validation split. Checked at construction so
        that monitoring a validation metric without one fails before any data
        is read, rather than becoming a callback that never fires.

    Raises
    ------
    EngineError
        If a validation metric is monitored with no validation split.
    """

    def __init__(self, spec: EarlyStoppingSpec, *, has_validation: bool) -> None:
        if spec.enabled and spec.monitor.startswith("val") and not has_validation:
            raise EngineError(
                f"early_stopping monitors {spec.monitor!r} but this run has no "
                f"validation split, so the metric will never have a value and "
                f"training would always run the full epoch budget. Monitor "
                f"'train_loss', configure a validation fraction, or set "
                f"early_stopping.enabled=False"
            )

        self.spec = spec
        self.monitor = Monitor(monitor=spec.monitor, mode=spec.mode, min_delta=spec.min_delta)
        self.epochs_without_improvement = 0
        self._should_stop = False

    def on_epoch_end(self, context: EpochContext) -> None:
        """
        Update the patience counter and decide whether to stop.

        Parameters
        ----------
        context
            The epoch's outcome.
        """
        if not self.spec.enabled:
            return

        if self.monitor.update(context):
            self.epochs_without_improvement = 0
            return

        self.epochs_without_improvement += 1
        if self.epochs_without_improvement >= self.spec.patience:
            self._should_stop = True
            _LOGGER.info(
                "early stopping at epoch %d: %s has not improved for %d epoch(s) "
                "(best %.6g at epoch %s)",
                context.epoch,
                self.spec.monitor,
                self.epochs_without_improvement,
                self.monitor.best_value,
                self.monitor.best_epoch,
            )

    @property
    def should_stop(self) -> bool:
        """Whether patience has run out."""
        return self._should_stop


class BestCheckpoint:
    """
    Keeps the best epoch's weights, and restores them when training ends.

    Parameters
    ----------
    spec
        Monitor, direction, whether to restore, and how many files to keep.
    model
        The *unwrapped* model, so the snapshot has no wrapper prefixes.
    has_validation
        Whether the run has a validation split.
    directory
        Where to write checkpoint files. ``None`` keeps snapshots in memory
        only, which is correct for a short run and for a tuning sweep where
        hundreds of trials would otherwise each leave files behind.

    Raises
    ------
    EngineError
        If a validation metric is monitored with no validation split.
    """

    def __init__(
        self,
        spec: CheckpointSpec,
        *,
        model: torch.nn.Module,
        has_validation: bool,
        directory: Path | None = None,
    ) -> None:
        if spec.enabled and spec.monitor.startswith("val") and not has_validation:
            raise EngineError(
                f"checkpoint monitors {spec.monitor!r} but this run has no "
                f"validation split. Monitor 'train_loss', configure a validation "
                f"fraction, or set checkpoint.enabled=False"
            )

        self.spec = spec
        self.model = model
        self.directory = directory
        self.monitor = Monitor(monitor=spec.monitor, mode=spec.mode)
        self._best_state: dict[str, torch.Tensor] | None = None
        self._written: list[Path] = []

    def on_epoch_end(self, context: EpochContext) -> None:
        """
        Snapshot the weights if this epoch is the new best.

        Parameters
        ----------
        context
            The epoch's outcome.
        """
        if not self.spec.enabled or not self.monitor.update(context):
            return

        # A detached copy, not a view: see `snapshot_state_dict`.  A view would
        # make "best weights" mean "last weights" and nothing would say so.
        self._best_state = snapshot_state_dict(self.model)
        _LOGGER.debug(
            "epoch %d is the new best (%s=%.6g)",
            context.epoch,
            self.spec.monitor,
            self.monitor.best_value,
        )

        if self.directory is not None:
            self._write(self._best_state, directory=self.directory, epoch=context.epoch)

    def _write(self, state: Mapping[str, torch.Tensor], *, directory: Path, epoch: int) -> None:
        """
        Write a snapshot to disk and prune the files it displaces.

        Parameters
        ----------
        state
            The snapshot to write.
        directory
            Where to write it.
        epoch
            The epoch being written, used in the filename.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"epoch_{epoch:04d}.pt"
        torch.save(dict(state), path)
        self._written.append(path)

        # Pruned oldest-first.  `keep_last` counts files retained, so a value
        # of one leaves exactly the best epoch on disk.
        while len(self._written) > self.spec.keep_last:
            stale = self._written.pop(0)
            stale.unlink(missing_ok=True)

    def restore_best(self) -> bool:
        """
        Load the best epoch's weights back into the model.

        Parameters
        ----------
        None

        Returns
        -------
        bool
            Whether weights were restored. False when restoring was not
            requested, or when no epoch was ever recorded as best -- which
            happens if every epoch produced a non-finite loss. The return
            value is recorded in ``FitOutcome.restored_best`` so that a
            bundle whose weights and metrics describe different models says
            so rather than being assumed consistent.
        """
        if not self.spec.enabled or not self.spec.restore_best:
            return False
        if self._best_state is None:
            _LOGGER.warning(
                "restore_best was requested but no epoch was ever recorded as "
                "best; keeping the final epoch's weights. This happens when "
                "every epoch produced a non-finite monitored value"
            )
            return False

        restore_state_dict(self.model, self._best_state)
        _LOGGER.info(
            "restored weights from epoch %s (%s=%.6g)",
            self.monitor.best_epoch,
            self.spec.monitor,
            self.monitor.best_value,
        )
        return True

    @property
    def should_stop(self) -> bool:
        """Always false: checkpointing never ends a run."""
        return False


class LearningRateSchedule:
    """
    Adjusts the learning rate between epochs.

    Parameters
    ----------
    spec
        Which schedule, and its settings.
    optimiser
        The optimiser whose rate is adjusted.
    epochs
        The epoch budget, needed by the cosine schedule to know its period.
    has_validation
        Whether the run has a validation split.

    Raises
    ------
    EngineError
        If a plateau schedule is requested on a run with no validation split,
        since it has nothing to detect a plateau in.
    """

    def __init__(
        self,
        spec: SchedulerSpec,
        *,
        optimiser: torch.optim.Optimizer,
        epochs: int,
        has_validation: bool = True,
    ) -> None:
        if spec.kind == "plateau" and not has_validation:
            raise EngineError(
                "scheduler.kind='plateau' reduces the learning rate when the "
                "validation loss stops improving, but this run has no validation "
                "split. Use 'cosine' or 'step', or configure a validation fraction"
            )

        self.spec = spec
        self.optimiser = optimiser
        self._scheduler = self._build(spec, optimiser, epochs)

    @staticmethod
    def _build(
        spec: SchedulerSpec, optimiser: torch.optim.Optimizer, epochs: int
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """
        Construct the Torch scheduler a spec names.

        Parameters
        ----------
        spec
            The schedule settings.
        optimiser
            The optimiser to adjust.
        epochs
            The epoch budget.

        Returns
        -------
        torch.optim.lr_scheduler.LRScheduler or None
            The scheduler, or ``None`` for a constant rate.
        """
        if spec.kind == "none":
            return None
        if spec.kind == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimiser, step_size=spec.step_epochs, gamma=spec.factor
            )
        if spec.kind == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimiser, T_max=epochs, eta_min=spec.min_learning_rate
            )
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            factor=spec.factor,
            patience=spec.patience,
            min_lr=spec.min_learning_rate,
        )

    def on_epoch_end(self, context: EpochContext) -> None:
        """
        Advance the schedule.

        Parameters
        ----------
        context
            The epoch's outcome, supplying the metric a plateau schedule
            watches.
        """
        if self._scheduler is None:
            return

        before = self.current_rate
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # The only schedule that is data-dependent, so the only one that
            # needs a value rather than just a tick.  Guarded by the
            # constructor, so `val_loss` is present here.
            self._scheduler.step(context.val_loss)
        else:
            self._scheduler.step()

        after = self.current_rate
        if after != before:
            _LOGGER.info(
                "learning rate changed after epoch %d: %.3g -> %.3g",
                context.epoch,
                before,
                after,
            )

    @property
    def current_rate(self) -> float:
        """The rate currently in effect, taken from the first parameter group."""
        return float(self.optimiser.param_groups[0]["lr"])

    @property
    def should_stop(self) -> bool:
        """Always false: a schedule never ends a run."""
        return False


class GradientNorms:
    """
    Records the gradient norm of every step, and summarises it per epoch.

    Cheap and worth having by default, because the two most common training
    pathologies are both visible here and in no other recorded number. A norm
    growing without bound is a diverging run that the loss curve only reveals
    once it has already diverged; a norm collapsing to zero is a dead network
    whose loss curve simply goes flat and looks like convergence.

    Parameters
    ----------
    clip_norm
        The clip threshold in effect, recorded so that a report can show how
        often clipping actually bound. ``None`` when clipping is off.
    """

    def __init__(self, *, clip_norm: float | None = None) -> None:
        self.clip_norm = clip_norm
        self._norms: list[float] = []
        self._clipped = 0

    def record(self, norm: float) -> None:
        """
        Record one step's gradient norm.

        Parameters
        ----------
        norm
            The norm before clipping.
        """
        self._norms.append(norm)
        if self.clip_norm is not None and norm > self.clip_norm:
            self._clipped += 1

    def summarise(self) -> dict[str, float]:
        """
        Return this epoch's summary and reset for the next.

        Returns
        -------
        dict
            Mean and maximum norm, and the fraction of steps that were
            clipped. Empty when no step was recorded.
        """
        if not self._norms:
            return {}

        summary = {
            "grad_norm_mean": sum(self._norms) / len(self._norms),
            "grad_norm_max": max(self._norms),
        }
        if self.clip_norm is not None:
            summary["grad_clipped_fraction"] = self._clipped / len(self._norms)

        self._norms.clear()
        self._clipped = 0
        return summary

    def on_epoch_end(self, context: EpochContext) -> None:
        """
        Do nothing: the loop reads :meth:`summarise` before building the context.

        Present so this satisfies :class:`Callback` and can sit in the same
        list as the others.

        Parameters
        ----------
        context
            The epoch's outcome.
        """
        del context

    @property
    def should_stop(self) -> bool:
        """Always false: tracking never ends a run."""
        return False
```

---

## 3. `src/rade_qnet/engines/torch/checkpoint.py`

10634 bytes · SHA-256 `ede6db2968503f60`

```python
"""
Checkpoints as tensors, never as pickled modules.

**Defect 10.** The implementation this framework replaces saved whole modules
and loaded them with ``weights_only=False``. Two separate problems come with
that, and only one of them is a security problem.

**It executes arbitrary code.** A pickle names the classes to construct and
``torch.load`` constructs them. A checkpoint is therefore executable, and a
checkpoint directory is an ordinary writable path -- a shared scratch mount, an
artifact bucket, a CI cache. Loading one is running whatever it contains with
the privileges of whoever loaded it.

**It welds the weights to the code.** A pickled module records its class by
import path, so renaming the class, moving the module or reorganising the
package breaks every checkpoint written before the change. The failure arrives
as an ``AttributeError`` during unpickling, months later, from a model that
trained fine. There is no migration path: the numbers are there and
unreachable.

A ``state_dict`` has neither problem. It is a flat mapping of names to
tensors, it loads under ``weights_only=True``, and it is reattached to a model
the caller constructs from the spec and the signature. That the caller must
supply a model is the feature: it is what guarantees the model can be rebuilt
from its specification rather than resurrected from a blob.

Saving the unwrapped model
--------------------------
Distributed and compiled wrappers prefix every parameter name --
``module.layer.weight``, ``_orig_mod.layer.weight``. A checkpoint written
through the wrapper therefore loads only into an identically wrapped model, so
a single-GPU evaluation of a model trained on four GPUs fails on every key.

:func:`save_weights` takes the handle rather than the model and writes
:attr:`~rade_qnet.engines.base.ModelHandle.unwrapped`, which is why that field
exists.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from ...core.runtime.errors import BundleError, EngineError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ..base import ModelHandle

__all__ = [
    "WEIGHTS_FILENAME",
    "load_weights",
    "restore_state_dict",
    "save_weights",
    "snapshot_state_dict",
]

_LOGGER = get_logger(__name__)

#: On-disk filename for the tensors. Part of the format.
WEIGHTS_FILENAME = "weights.pt"

#: Sidecar describing what the weights belong to. Written for the person
#: holding an unlabelled checkpoint in six months, not for the loader -- which
#: is why a missing or stale sidecar is never fatal.
_MANIFEST_FILENAME = "weights.json"

#: Prefixes a wrapper adds to every parameter name.
_WRAPPER_PREFIXES = ("module.", "_orig_mod.")


def _strip_wrapper_prefixes(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Remove a distributed or compile wrapper's prefix from every key.

    A belt-and-braces measure. :func:`save_weights` writes the unwrapped
    model, so these prefixes should never reach disk -- but a checkpoint
    written by an older version, or by a caller that bypassed the handle,
    would otherwise fail to load with an unhelpful list of several hundred
    missing keys.

    Parameters
    ----------
    state
        A state dictionary, possibly prefixed.

    Returns
    -------
    dict
        The same tensors under unprefixed names.
    """
    stripped: dict[str, torch.Tensor] = {}
    for key, tensor in state.items():
        name = key
        # Looped rather than applied once: a compiled model inside a
        # distributed wrapper carries both prefixes.
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if name.startswith(prefix):
                    name = name[len(prefix) :]
                    changed = True
        stripped[name] = tensor
    return stripped


def snapshot_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    Return a detached CPU copy of a model's tensors.

    Copied rather than referenced, and that is the whole point of the
    function. ``model.state_dict()`` returns views of the live parameters, so
    a "best epoch" snapshot taken that way tracks the model and ends up
    holding the *last* epoch's weights -- which is how a run reports the best
    epoch's metrics beside the final epoch's weights and nothing notices.

    Moved to the CPU so that a snapshot held across an epoch does not occupy
    accelerator memory that the next epoch needs.

    Parameters
    ----------
    model
        The model to snapshot. Pass the unwrapped model, for the reason in the
        module docstring.

    Returns
    -------
    dict
        Parameter name to a detached CPU tensor.
    """
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


def restore_state_dict(model: torch.nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """
    Load a snapshot back into a model, in place.

    Parameters
    ----------
    model
        The model to restore into.
    state
        A snapshot from :func:`snapshot_state_dict`.

    Raises
    ------
    EngineError
        If any key is missing or unexpected. Strict by design: Torch's
        non-strict mode would leave some layers at their initial values and
        report success, so a model would evaluate as though it had trained
        while part of it never did.
    """
    stripped = _strip_wrapper_prefixes(state)
    try:
        model.load_state_dict(stripped, strict=True)
    except RuntimeError as error:
        raise EngineError(
            f"could not restore weights into {type(model).__name__}: {error}. "
            f"The checkpoint and the model disagree on their parameters, which "
            f"usually means the model was constructed from a different "
            f"specification or signature than the one that trained it"
        ) from error


def save_weights(
    handle: ModelHandle,
    path: Path,
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """
    Write a handle's weights to a file.

    Writes :attr:`~rade_qnet.engines.base.ModelHandle.unwrapped`, so the
    checkpoint loads into a plain model regardless of how training was
    parallelised.

    Parameters
    ----------
    handle
        The prepared model.
    path
        Destination file. Its parent is created if absent.
    metadata
        Extra annotations for the sidecar: the epoch, the monitored value.

    Raises
    ------
    EngineError
        If the unwrapped model is not a Torch module, which would mean a
        handle from another engine reached this one.
    """
    model = handle.unwrapped
    if not isinstance(model, torch.nn.Module):
        raise EngineError(
            f"the Torch engine can only checkpoint a torch.nn.Module, received "
            f"{type(model).__name__}; this handle was prepared by a different engine"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    state = snapshot_state_dict(model)
    torch.save(state, path)

    sidecar = {
        "model_class": type(model).__name__,
        "n_tensors": len(state),
        "n_elements": sum(tensor.numel() for tensor in state.values()),
        "device_trained_on": handle.device,
        "precision": handle.precision,
        "was_distributed": handle.is_distributed,
        **dict(metadata or {}),
    }
    (path.parent / _MANIFEST_FILENAME).write_text(
        json.dumps(sidecar, indent=2, sort_keys=True), encoding="utf-8"
    )
    _LOGGER.info(
        "saved %d tensor(s), %d element(s) to %s",
        sidecar["n_tensors"],
        sidecar["n_elements"],
        path,
    )


def load_weights(model: object, path: Path) -> torch.nn.Module:
    """
    Load weights from a file into a model the caller constructed.

    The model is a parameter rather than something this function builds, and
    that asymmetry is the design: a checkpoint holds numbers, a specification
    holds the architecture, and keeping them apart is what makes a six-month-
    old bundle loadable after the code has moved on.

    Parameters
    ----------
    model
        A model built from the same spec and signature that trained it.
        Annotated ``object`` to match the engine protocol, and narrowed here.
    path
        The checkpoint file.

    Returns
    -------
    torch.nn.Module
        The same model, with the saved weights loaded.

    Raises
    ------
    BundleError
        If the file does not exist.
    EngineError
        If the model is not a Torch module, or the weights do not fit it.
    """
    if not isinstance(model, torch.nn.Module):
        raise EngineError(
            f"the Torch engine can only load into a torch.nn.Module, received "
            f"{type(model).__name__}"
        )
    if not path.is_file():
        raise BundleError(
            f"no checkpoint at {path}; the bundle may be incomplete, or the run "
            f"may have been configured with checkpoint.enabled=False"
        )

    # `weights_only=True` is the whole point of this module: it restricts
    # unpickling to tensors and plain containers, so a checkpoint cannot
    # execute code.  See the module docstring for the two reasons.
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        # Torch's own message for a rejected pickle explains how to disable
        # the safety check, which is the opposite of the advice a reader
        # should act on.  Replaced rather than chained through, so the
        # suggested fix is to regenerate the checkpoint.
        raise EngineError(
            f"the checkpoint at {path} could not be loaded as tensors "
            f"({type(error).__name__}). It appears to be a pickled object rather "
            f"than a state dictionary, which this engine will not load because "
            f"unpickling it would execute whatever the file contains. Regenerate "
            f"it by re-running the fit, or convert it on a trusted machine by "
            f"loading the module once and saving model.state_dict()"
        ) from error

    if not isinstance(state, dict):
        raise EngineError(
            f"the checkpoint at {path} holds a {type(state).__name__} rather than "
            f"a state dictionary; it must be regenerated"
        )

    restore_state_dict(model, state)
    _LOGGER.info("loaded %d tensor(s) from %s", len(state), path)
    return model
```

---

## 4. `src/rade_qnet/engines/torch/distributed.py`

9479 bytes · SHA-256 `0c0492e6dc473abb`

```python
"""
Distributed data-parallel setup and teardown.

Small, because the hard part of distributed training in this framework is an
*ordering* guarantee rather than a configuration one, and that guarantee is
enforced in :mod:`.materialise` and in the pipeline's stage order.

The ordering, and what goes wrong without it
--------------------------------------------
**Defect 6, the other half.** ``DistributedDataParallel`` records the
parameters it must synchronise when it wraps a model. A model with lazily
shaped parameters has none yet, so wrapping before materialisation hands the
wrapper an empty set.

The two outcomes are a crash inside the distributed library, or -- the reason
this is a defect rather than an inconvenience -- a wrapper that synchronises
nothing. Each rank then trains an independent copy on its own shard, the loss
curves look entirely normal, and the "distributed" run produces N unrelated
models of which one is saved. Nothing in the logs distinguishes that from a
correct run.

:func:`distribute` therefore refuses to wrap an unmaterialised model. It is
the last line of defence rather than the mechanism: the pipeline already runs
``materialise`` as its own stage before ``prepare_hardware``, so reaching this
check means something was reordered.

Single-process runs are the normal case
---------------------------------------
``distributed='none'`` returns the model untouched, and that is what almost
every run does. The framework's answer to "train forty models" is a job set
across processes, not one model across devices -- so distribution here is for
the single model too large for one device, which is a different and rarer
problem.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as distributed_backend

from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from .materialise import has_lazy_parameters

if TYPE_CHECKING:
    from ...core.spec.hardware import HardwareSpec

__all__ = [
    "distribute",
    "is_distributed_run",
    "local_rank",
    "shutdown_distributed",
    "undistribute",
    "world_size",
]

_LOGGER = get_logger(__name__)

#: Environment variables a launcher sets. Read rather than configured, because
#: the launcher is the authority on how many processes exist -- a spec field
#: duplicating it would be a second source of truth that could disagree.
_WORLD_SIZE_VARIABLE = "WORLD_SIZE"
_LOCAL_RANK_VARIABLE = "LOCAL_RANK"


def _read_count(variable: str, *, default: int) -> int:
    """
    Read a non-negative integer from the environment, tolerating rubbish.

    These variables are set by an external launcher, so their contents are not
    this framework's to guarantee. An unparseable value is treated as absent
    and warned about, because the alternative is an ``int()`` traceback at
    import-adjacent code with no indication that the cause is a stray shell
    export rather than the model.

    Parameters
    ----------
    variable
        Environment variable name.
    default
        Value to use when the variable is absent or unreadable.

    Returns
    -------
    int
        The parsed value, or ``default``.
    """
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _LOGGER.warning(
            "%s=%r is not an integer; treating it as %d. This variable is set "
            "by the launcher, so an unreadable value usually means a stray "
            "shell export rather than a problem with the job",
            variable,
            raw,
            default,
        )
        return default


def world_size() -> int:
    """
    Return how many processes are taking part.

    Returns
    -------
    int
        The process count, or one when this is not a distributed launch.
    """
    if distributed_backend.is_available() and distributed_backend.is_initialized():
        return int(distributed_backend.get_world_size())
    return _read_count(_WORLD_SIZE_VARIABLE, default=1)


def local_rank() -> int:
    """
    Return this process's index on its own machine.

    Returns
    -------
    int
        The local rank, or zero when this is not a distributed launch.
    """
    return _read_count(_LOCAL_RANK_VARIABLE, default=0)


def is_distributed_run() -> bool:
    """
    Return whether more than one process is taking part.

    Returns
    -------
    bool
        True when a launcher has started several processes.
    """
    return world_size() > 1


def distribute(
    model: torch.nn.Module, *, spec: HardwareSpec, device: torch.device
) -> tuple[torch.nn.Module, bool]:
    """
    Wrap a model for distributed training, if that was asked for and is possible.

    Parameters
    ----------
    model
        The **materialised** model.
    spec
        The hardware request, naming the strategy.
    device
        The device this process owns.

    Returns
    -------
    tuple
        The model -- wrapped or not -- and whether it was wrapped. The flag
        travels into :class:`~rade_qnet.engines.base.ModelHandle` so a report
        can state whether a run that asked to be distributed actually was.

    Raises
    ------
    EngineError
        If the model still has lazy parameters. See the module docstring: this
        is the one condition here that must not degrade, because degrading
        would mean training N unsynchronised models that look like one.
    """
    if spec.distributed == "none":
        return model, False

    if has_lazy_parameters(model):
        raise EngineError(
            "cannot wrap a model with uninitialised parameters for distributed "
            "training: the wrapper would record an empty parameter set and "
            "synchronise nothing, so each rank would train its own independent "
            "copy while reporting a normal loss curve. Run materialise() first -- "
            "the train pipeline does this as its own stage, so reaching this "
            "error means the stage order was changed"
        )

    if not is_distributed_run():
        # Warn and degrade: a single-process run of a distributed
        # configuration is correct, just not parallel.  This is the same
        # policy as an absent accelerator, and for the same reason.
        _LOGGER.warning(
            "distributed=%r was requested but %s reports a world size of 1; "
            "training in a single process. Launch with torchrun to distribute",
            spec.distributed,
            _WORLD_SIZE_VARIABLE,
        )
        return model, False

    if not distributed_backend.is_available():
        _LOGGER.warning(
            "distributed=%r was requested but this Torch build has no "
            "distributed support; training in a single process",
            spec.distributed,
        )
        return model, False

    if not distributed_backend.is_initialized():
        # NCCL for CUDA, Gloo for everything else.  Chosen here rather than
        # configured, because picking the wrong one is never what anybody
        # wanted and the right one follows from the device.
        backend = "nccl" if device.type == "cuda" else "gloo"
        _LOGGER.info(
            "initialising the %s process group: rank %d of %d",
            backend,
            local_rank(),
            world_size(),
        )
        distributed_backend.init_process_group(backend=backend)

    wrapped = torch.nn.parallel.DistributedDataParallel(
        model,
        # Only meaningful for CUDA.  Passing a CPU device here is an error in
        # Torch rather than a no-op, so it is set conditionally.
        device_ids=[device.index if device.index is not None else local_rank()]
        if device.type == "cuda"
        else None,
    )
    _LOGGER.info("wrapped model for distributed training across %d process(es)", world_size())
    return wrapped, True


def undistribute(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying model from inside any wrapper.

    Unwraps a distributed wrapper and a compiled one, and both together.
    Used to populate :attr:`~rade_qnet.engines.base.ModelHandle.unwrapped`, so
    that a checkpoint carries plain parameter names and loads into a
    single-process model -- see :mod:`.checkpoint`.

    Parameters
    ----------
    model
        A model, possibly wrapped.

    Returns
    -------
    torch.nn.Module
        The innermost model.
    """
    inner = model
    # Looped because the wrappers nest: `torch.compile` applied to a
    # distributed model leaves both layers in place.
    while True:
        if isinstance(inner, torch.nn.parallel.DistributedDataParallel):
            inner = inner.module
            continue
        original = getattr(inner, "_orig_mod", None)
        if isinstance(original, torch.nn.Module):
            inner = original
            continue
        return inner


def shutdown_distributed() -> None:
    """
    Tear the process group down, if one was started.

    Called at the end of a fit. Without it a process that finishes early
    leaves its peers blocked on a collective operation until they time out,
    which presents as a job set that hangs rather than as an error.
    """
    if distributed_backend.is_available() and distributed_backend.is_initialized():
        _LOGGER.info("destroying the distributed process group")
        distributed_backend.destroy_process_group()
```

---

## 5. `src/rade_qnet/engines/torch/engine.py`

19124 bytes · SHA-256 `2d7faa241a18fb2b`

```python
"""
``TorchEngine`` -- the ``Engine`` implementation for PyTorch.

Thin by design. Every decision of substance lives in a focused module --
:mod:`.hardware`, :mod:`.materialise`, :mod:`.loaders`, :mod:`.loops`,
:mod:`.callbacks`, :mod:`.checkpoint`, :mod:`.distributed` -- and this class
sequences them. If a method here grows a second decision, that decision
belongs in one of those modules.

The order in :meth:`TorchEngine.prepare` is the contract
--------------------------------------------------------
``materialise`` is a *pipeline* stage and runs before ``prepare`` is called at
all. Inside ``prepare`` the remaining order is::

    device placement -> compile -> distribute -> optimiser

and every step depends on the one before it. Compiling before placement traces
a graph for the wrong device. Distributing before compiling means the compiler
sees the wrapper rather than the model. Building the optimiser before
distributing hands it parameters the wrapper is about to replace.

The ordering is enforced by it being written once, here, rather than by a
comment asking callers to be careful.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.runtime.components import engine as register_engine
from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from ...core.spec.training import CheckpointSpec, EarlyStoppingSpec, TorchTrainingSpec
from ..base import EngineCapabilities, ModelHandle
from .callbacks import BestCheckpoint, EarlyStopping, GradientNorms, LearningRateSchedule
from .checkpoint import load_weights, save_weights
from .distributed import distribute, shutdown_distributed, undistribute
from .hardware import (
    ResolvedHardware,
    available_accelerators,
    compile_if_requested,
    resolve_hardware,
)
from .learners.supervised import SupervisedLearner
from .loops import fit_epochs
from .losses import build_loss
from .materialise import count_parameters
from .materialise import materialise as materialise_model
from .predictor import predict_batches

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

    from ...core.contract.data import TensorLike
    from ...core.contract.result import FitOutcome
    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource
    from ...core.spec.hardware import HardwareSpec
    from .callbacks import Callback

__all__ = ["TorchEngine"]

_LOGGER = get_logger(__name__)

#: Registered name, used by the engine registry and by ``TorchTrainingSpec``'s
#: ``engine`` discriminator.
ENGINE_NAME = "torch"

#: Optimiser constructors by spec name. A table rather than a chain of
#: conditionals, so adding one is a line here and nothing else.
_OPTIMISERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}

#: Keys the handle's ``extras`` carries. Named so that :meth:`TorchEngine.fit`
#: and :meth:`TorchEngine.prepare` cannot disagree on a string literal.
_OPTIMISER_KEY = "optimiser"
_HARDWARE_KEY = "hardware"
_SIGNATURE_KEY = "signature"

#: Monitor used in place of a validation metric when a run has no validation
#: split. Always available, since every epoch has a training loss.
_FALLBACK_MONITOR = "train_loss"


def _without_validation_monitor[SpecT: (EarlyStoppingSpec, CheckpointSpec)](
    spec: SpecT,
) -> SpecT:
    """
    Return a spec whose monitor does not require a validation split.

    Parameters
    ----------
    spec
        An early-stopping or checkpoint specification.

    Returns
    -------
    EarlyStoppingSpec or CheckpointSpec
        The spec unchanged when it is disabled or already monitors a training
        metric, and otherwise a copy monitoring :data:`_FALLBACK_MONITOR`.

        A copy rather than a mutation, because the spec belongs to the run
        specification and is shared -- rewriting it in place would change what
        a later report says the run was configured to do.
    """
    if not spec.enabled or not spec.monitor.startswith("val"):
        return spec
    return spec.model_copy(update={"monitor": _FALLBACK_MONITOR})


@register_engine(ENGINE_NAME)
class TorchEngine:
    """
    Trains ``torch.nn.Module`` models.

    Satisfies :class:`~rade_qnet.engines.base.Engine` structurally, without
    inheriting from it, so the conformance suite checks it against the same
    definition a user's engine is held to.

    Parameters
    ----------
    checkpoint_directory
        Where :class:`~.callbacks.BestCheckpoint` writes per-epoch files.
        ``None`` keeps the best weights in memory only, which is the right
        default: a tuning sweep of several hundred trials would otherwise
        leave several hundred checkpoint files behind, and only the final
        bundle's weights are of any interest.
    """

    def __init__(self, *, checkpoint_directory: Path | None = None) -> None:
        self.checkpoint_directory = checkpoint_directory

    def capabilities(self) -> EngineCapabilities:
        """
        Declare what this engine supports.

        Returns
        -------
        EngineCapabilities
            The engine's declared abilities.
        """
        return EngineCapabilities(
            name=ENGINE_NAME,
            supports_epochs=True,
            supports_validation_during_fit=True,
            supports_checkpointing=True,
            supports_distributed=True,
            supports_lazy_materialisation=True,
            accelerators=available_accelerators(),
        )

    def materialise(self, model: object, signature: InputSignature) -> object:
        """
        Give a lazily shaped model its parameters.

        Parameters
        ----------
        model
            The untrained model.
        signature
            The declared interface.

        Returns
        -------
        object
            The model, with parameters.

        Raises
        ------
        EngineError
            If the model is not a Torch module, or the dummy pass fails.
        """
        return materialise_model(self._module(model), signature)

    def prepare(
        self,
        model: object,
        *,
        hardware: HardwareSpec,
        training: object,
        static: Mapping[str, TensorLike] | None = None,
    ) -> ModelHandle:
        """
        Place the model on its device and build its training apparatus.

        Parameters
        ----------
        model
            The materialised model.
        hardware
            Device, precision, compilation and distribution settings.
        training
            A :class:`~rade_qnet.core.spec.training.TorchTrainingSpec`.
        static
            Static inputs from the data build, uploaded once.

        Returns
        -------
        ModelHandle
            The prepared model and its apparatus.

        Raises
        ------
        EngineError
            If ``training`` is not a Torch training spec, or the model is not
            a Torch module.
        """
        module = self._module(model)
        spec = self._training_spec(training)
        resolved = resolve_hardware(hardware)

        # The order below is the contract; see the module docstring.
        module = module.to(device=resolved.device)
        prepared = compile_if_requested(module, spec=hardware)
        prepared, is_distributed = distribute(prepared, spec=hardware, device=resolved.device)

        optimiser = self._build_optimiser(prepared, spec)
        n_parameters = count_parameters(undistribute(prepared), trainable_only=True)
        if n_parameters == 0:
            raise EngineError(
                "the prepared model has no trainable parameters, so the optimiser "
                "would have nothing to update and training would run without "
                "changing anything. Either every parameter has requires_grad=False, "
                "or the model was not materialised before prepare()"
            )

        handle = ModelHandle(
            model=prepared,
            unwrapped=undistribute(prepared),
            device=str(resolved.device),
            precision=resolved.precision,
            is_distributed=is_distributed,
            static=dict(static or {}),
            extras={
                _OPTIMISER_KEY: optimiser,
                _HARDWARE_KEY: resolved,
            },
        )
        _LOGGER.info("prepared %s with %d trainable parameter(s)", handle.describe(), n_parameters)
        return handle

    def fit(
        self,
        handle: ModelHandle,
        sources: Mapping[str, BatchSource],
        training: object,
        *,
        on_epoch_end: object = None,
    ) -> FitOutcome:
        """
        Train the model and report what happened.

        Parameters
        ----------
        handle
            The prepared model.
        sources
            One source per split. ``train`` is required.
        training
            A :class:`~rade_qnet.core.spec.training.TorchTrainingSpec`.
        on_epoch_end
            Currently accepted and not yet forwarded; see the note below.

        Returns
        -------
        FitOutcome
            History, best epoch, and whether the best weights were restored.

        Raises
        ------
        EngineError
            If a required source is missing, or the spec is not a Torch one.

        Notes
        -----
        ``on_epoch_end`` is accepted so the signature matches the protocol,
        and is not yet wired through. The seam exists for the pipeline to
        forward live epoch metrics to hooks and trackers, and those consumers
        arrive with ``orchestration``'s tracker integration -- so wiring it
        now would mean a callback with no reader. It is recorded as an open
        item rather than quietly dropped.
        """
        spec = self._training_spec(training)
        if on_epoch_end is not None:
            _LOGGER.debug("on_epoch_end was supplied but is not yet forwarded")

        module = self._module(handle.model)
        unwrapped = self._module(handle.unwrapped)
        resolved = self._resolved_hardware(handle)
        optimiser = self._optimiser(handle)

        has_validation = "validation" in sources
        early_stopping_spec = spec.early_stopping
        checkpoint_spec = spec.checkpoint
        if not has_validation:
            # Both of these monitor 'val_loss' by default, so a run with no
            # validation split would be refused by the callbacks on the
            # default configuration.  Substituting the training loss here
            # rather than raising is what makes a train-only fit possible at
            # all -- and it is the honest reading of the situation: the run
            # still has a best epoch, it is just chosen on less evidence.
            early_stopping_spec = _without_validation_monitor(early_stopping_spec)
            checkpoint_spec = _without_validation_monitor(checkpoint_spec)
            _LOGGER.warning(
                "no validation source: early stopping and best-epoch selection "
                "fall back to monitoring %r. A best epoch chosen on the training "
                "loss is the epoch that fitted the training data most closely, "
                "which is not the same thing as the one that generalises best",
                _FALLBACK_MONITOR,
            )

        gradient_norms = GradientNorms(clip_norm=spec.gradient_clip_norm)
        checkpoint = BestCheckpoint(
            checkpoint_spec,
            model=unwrapped,
            has_validation=has_validation,
            directory=self.checkpoint_directory,
        )
        callbacks: list[Callback] = [
            EarlyStopping(early_stopping_spec, has_validation=has_validation),
            LearningRateSchedule(
                spec.scheduler,
                optimiser=optimiser,
                epochs=spec.epochs,
                has_validation=has_validation,
            ),
        ]

        learner = SupervisedLearner(
            loss=build_loss(spec.loss),
            optimiser=optimiser,
            hardware=resolved,
            gradient_clip_norm=spec.gradient_clip_norm,
            gradient_norms=gradient_norms,
        )

        try:
            return fit_epochs(
                module,
                sources,
                learner=learner,
                signature=sources["train"].signature,
                device=resolved.device,
                epochs=spec.epochs,
                callbacks=callbacks,
                gradient_norms=gradient_norms,
                checkpoint=checkpoint,
                learning_rate=spec.learning_rate,
            )
        finally:
            # Torn down even when the fit raised.  A process that exits
            # holding a process group leaves its peers blocked on a collective
            # until they time out, which presents as a job set that hangs
            # rather than as the error that actually occurred.
            if handle.is_distributed:
                shutdown_distributed()

    def predict(self, handle: ModelHandle, source: BatchSource) -> NDArray[np.floating]:
        """
        Run a forward pass over a source and return the raw output.

        Returns the model's own output space: inverting the target transform
        is the pipeline's job, using the data build's fitted state. An engine
        that inverted it here would invert it twice.

        Parameters
        ----------
        handle
            The prepared model.
        source
            Batches to predict over. Must be bounded.

        Returns
        -------
        numpy.ndarray
            Predictions in source order, one row per sample.

        Raises
        ------
        EngineError
            If the source is unbounded.
        """
        return predict_batches(
            self._module(handle.model),
            source,
            resolved=self._resolved_hardware(handle),
        )

    def save_weights(self, handle: ModelHandle, path: Path) -> None:
        """
        Write the model's parameters to a path.

        Parameters
        ----------
        handle
            The trained model.
        path
            Destination file.
        """
        save_weights(handle, path)

    def load_weights(self, model: object, path: Path) -> object:
        """
        Load parameters into a freshly built model.

        Parameters
        ----------
        model
            A model of the same architecture.
        path
            File written by :meth:`save_weights`.

        Returns
        -------
        object
            The model with parameters loaded.
        """
        return load_weights(model, path)

    # -- Narrowing helpers -------------------------------------------------

    @staticmethod
    def _module(model: object) -> torch.nn.Module:
        """
        Narrow an engine-opaque model to a Torch module.

        Parameters
        ----------
        model
            The model object.

        Returns
        -------
        torch.nn.Module
            The same object, narrowed.

        Raises
        ------
        EngineError
            If it is not a Torch module. The protocol types model objects as
            ``object`` so one signature serves every engine; this is where
            the Torch engine makes its requirement explicit, and the message
            names the likely cause.
        """
        if not isinstance(model, torch.nn.Module):
            raise EngineError(
                f"the Torch engine requires a torch.nn.Module, received "
                f"{type(model).__name__}. Check that the run spec's training "
                f"engine matches what the model definition builds"
            )
        return model

    @staticmethod
    def _training_spec(training: object) -> TorchTrainingSpec:
        """
        Narrow an engine-opaque training spec to a Torch one.

        Parameters
        ----------
        training
            The training spec.

        Returns
        -------
        TorchTrainingSpec
            The same spec, narrowed.

        Raises
        ------
        EngineError
            If it is another engine's spec.
        """
        if not isinstance(training, TorchTrainingSpec):
            raise EngineError(
                f"the Torch engine requires a TorchTrainingSpec, received "
                f"{type(training).__name__}. The run spec's training.engine "
                f"field selects both, so they cannot normally disagree -- this "
                f"usually means an engine was constructed directly"
            )
        return training

    @staticmethod
    def _optimiser(handle: ModelHandle) -> torch.optim.Optimizer:
        """
        Return the optimiser carried on a handle.

        Parameters
        ----------
        handle
            The prepared model.

        Returns
        -------
        torch.optim.Optimizer
            The optimiser.

        Raises
        ------
        EngineError
            If the handle carries none, which means it was built by a
            different engine.
        """
        optimiser = handle.extras.get(_OPTIMISER_KEY)
        if not isinstance(optimiser, torch.optim.Optimizer):
            raise EngineError(
                "this handle carries no optimiser; it was not prepared by the Torch engine"
            )
        return optimiser

    @staticmethod
    def _resolved_hardware(handle: ModelHandle) -> ResolvedHardware:
        """
        Return the resolved hardware carried on a handle.

        Parameters
        ----------
        handle
            The prepared model.

        Returns
        -------
        ResolvedHardware
            Device, precision and gradient scaler.

        Raises
        ------
        EngineError
            If the handle carries none.
        """
        resolved = handle.extras.get(_HARDWARE_KEY)
        if not isinstance(resolved, ResolvedHardware):
            raise EngineError(
                "this handle carries no resolved hardware; it was not prepared by the Torch engine"
            )
        return resolved

    def _build_optimiser(
        self, model: torch.nn.Module, spec: TorchTrainingSpec
    ) -> torch.optim.Optimizer:
        """
        Construct the optimiser a spec names.

        Parameters
        ----------
        model
            The prepared model, whose parameters the optimiser will track.
        spec
            The training spec.

        Returns
        -------
        torch.optim.Optimizer
            The optimiser.

        Raises
        ------
        EngineError
            If the optimiser name is unrecognised.
        """
        if spec.optimiser not in _OPTIMISERS:
            raise EngineError(
                f"unknown optimiser {spec.optimiser!r}; supported optimisers are "
                f"{sorted(_OPTIMISERS)}"
            )
        return _OPTIMISERS[spec.optimiser](
            model.parameters(),
            lr=spec.learning_rate,
            weight_decay=spec.weight_decay,
        )
```

---

## 6. `src/rade_qnet/engines/torch/hardware.py`

12848 bytes · SHA-256 `1e6d10f5dc810e55`

```python
"""
Resolving a ``HardwareSpec`` into the Torch objects a fit actually uses.

One module turns a declarative request -- "cuda, bf16, compiled" -- into a
device, an autocast context and a gradient scaler. Everything that inspects
``torch.cuda`` or ``torch.backends`` lives here, so the rest of the engine
never asks what machine it is on.

A named device is a request, not a guarantee
--------------------------------------------
:func:`resolve_device` warns and degrades when the requested accelerator is
absent, rather than raising. That is a deliberate asymmetry with the rest of
the framework, which refuses almost everything it cannot honour exactly.

The reasoning is that an absent accelerator does not make a run *wrong*, only
slower -- whereas the configurations this framework does refuse (a leaky
split, an unsupported precision) make the result incorrect or meaningless. A
job set submitted against a GPU host and rerun on a laptop should produce the
same numbers more slowly, not fail. The warning is what keeps it from being
silent, and the chosen device is recorded in the model handle so a report can
show that a run intended for CUDA ran on the CPU.

``fp16`` is the one exception, and it is handled by the spec rather than here:
``HardwareSpec`` already refuses ``fp16`` on a CPU at load time, because that
combination is not slow-but-correct -- most CPU kernels have no ``fp16`` path.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from ...core.spec.hardware import HardwareSpec

__all__ = [
    "ResolvedHardware",
    "apply_thread_budget",
    "autocast_for",
    "available_accelerators",
    "compile_if_requested",
    "resolve_device",
    "resolve_hardware",
]

_LOGGER = get_logger(__name__)

#: Spec precision names mapped to Torch dtypes. ``fp32`` is absent on purpose:
#: full precision means "do not autocast at all", which is a different
#: decision from "autocast to float32" and is handled by returning a null
#: context rather than by looking up a dtype.
_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

#: Device types that support autocast. Autocast on an MPS device is not
#: implemented in Torch, so requesting reduced precision there degrades to
#: full precision with a warning rather than erroring.
_AUTOCAST_DEVICES = frozenset({"cpu", "cuda"})


class ResolvedHardware:
    """
    The concrete Torch objects one fit runs against.

    Parameters
    ----------
    device
        Where tensors and parameters live.
    precision
        The precision actually in effect, which may differ from the one
        requested if the chosen device could not honour it.
    requested_device
        What the specification asked for, kept so a report can show that a
        run intended for an accelerator ran on the CPU.
    scaler
        Gradient scaler, present only for ``fp16``. ``float16`` has roughly
        five exponent bits fewer than ``float32``, so small gradients
        underflow to zero and the affected parameters simply stop learning --
        quietly, and only for the layers with the smallest gradients.
        ``bfloat16`` keeps ``float32``'s exponent range and needs no scaler.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        precision: str,
        requested_device: str,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        self.device = device
        self.precision = precision
        self.requested_device = requested_device
        self.scaler = scaler

    @property
    def degraded(self) -> bool:
        """Whether the resolved device differs from the one requested."""
        return self.requested_device not in ("auto", self.device.type)

    def autocast(self) -> AbstractContextManager[None]:
        """
        Return the context a forward pass runs inside.

        Returns
        -------
        contextlib.AbstractContextManager
            An autocast context, or a null context under full precision.
        """
        return autocast_for(self.device, self.precision)

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "device": str(self.device),
            "precision": self.precision,
            "requested_device": self.requested_device,
            "degraded": self.degraded,
            "gradient_scaler": self.scaler is not None,
        }


def available_accelerators() -> tuple[str, ...]:
    """
    Return the accelerator types this machine can actually use.

    Returns
    -------
    tuple of str
        Device types, best first, always ending with ``cpu``.
    """
    accelerators: list[str] = []
    if torch.cuda.is_available():
        accelerators.append("cuda")
    # `mps.is_available()` is false on non-Apple builds, and the attribute
    # itself is absent on sufficiently old Torch, so both are checked.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accelerators.append("mps")
    accelerators.append("cpu")
    return tuple(accelerators)


def resolve_device(spec: HardwareSpec) -> torch.device:
    """
    Return the device a fit will run on.

    Parameters
    ----------
    spec
        The hardware request.

    Returns
    -------
    torch.device
        The chosen device, with an index when one was requested.
    """
    accelerators = available_accelerators()

    if spec.device == "auto":
        chosen = accelerators[0]
        _LOGGER.info("device='auto' resolved to %s (available: %s)", chosen, accelerators)
        return torch.device(chosen)

    if spec.device not in accelerators:
        # Warn and degrade rather than raise: see the module docstring.
        _LOGGER.warning(
            "device=%r was requested but is not available on this machine "
            "(available: %s); running on cpu instead. Results will match, "
            "throughput will not",
            spec.device,
            accelerators,
        )
        return torch.device("cpu")

    if spec.device_index is None:
        return torch.device(spec.device)
    return torch.device(f"{spec.device}:{spec.device_index}")


def autocast_for(device: torch.device, precision: str) -> AbstractContextManager[None]:
    """
    Return the autocast context for a device and precision.

    Parameters
    ----------
    device
        The resolved device.
    precision
        ``fp32``, ``fp16`` or ``bf16``.

    Returns
    -------
    contextlib.AbstractContextManager
        An autocast context, or a null context when no casting applies.

    Raises
    ------
    EngineError
        If the precision name is not recognised. Unlike an absent
        accelerator, an unrecognised precision is a programming error rather
        than a property of the machine, so it raises.
    """
    if precision == "fp32":
        return nullcontext()
    if precision not in _AUTOCAST_DTYPES:
        raise EngineError(
            f"unknown precision {precision!r}; expected one of "
            f"{['fp32', *sorted(_AUTOCAST_DTYPES)]}"
        )
    if device.type not in _AUTOCAST_DEVICES:
        _LOGGER.warning(
            "precision=%r is not supported on a %s device; running in fp32",
            precision,
            device.type,
        )
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPES[precision])


def apply_thread_budget(spec: HardwareSpec) -> int:
    """
    Apply ``threads_per_worker`` to this process's intra-op thread pool.

    **Why this is not left to the environment.** A worker process has its
    thread budget set by environment variable before it imports anything,
    which works because the worker is a fresh interpreter. The process that
    launched it has already imported Torch, so the same variable does
    nothing there. The budget would then apply to pooled jobs and not to
    sequential ones -- and because the number of threads decides the order
    in which a reduction accumulates, the same job would score differently
    depending on where it ran.

    That difference is small and entirely real: measured on the flagship
    model it moves the test :math:`R^2` in the seventh significant figure,
    reproducibly, with each value stable for its own thread count. Setting
    the budget from the specification in both paths is what makes placement
    an operational choice rather than a modelling one.

    Parameters
    ----------
    spec
        The hardware request. ``threads_per_worker=None`` means "whatever
        this machine chose", which is left untouched.

    Returns
    -------
    int
        The thread count in effect afterwards, whether or not it changed.
    """
    if spec.threads_per_worker is None:
        return torch.get_num_threads()

    # Idempotent, and cheap enough to call per job. Torch tolerates being
    # set repeatedly in a process; it is resizing a pool, not creating one.
    torch.set_num_threads(spec.threads_per_worker)
    _LOGGER.debug("set torch intra-op threads to %d", spec.threads_per_worker)
    return torch.get_num_threads()


def resolve_hardware(spec: HardwareSpec) -> ResolvedHardware:
    """
    Resolve a hardware request into the objects a fit uses.

    Parameters
    ----------
    spec
        The hardware request.

    Returns
    -------
    ResolvedHardware
        Device, effective precision and gradient scaler.
    """
    apply_thread_budget(spec)
    device = resolve_device(spec)

    precision = spec.precision
    if precision != "fp32" and device.type not in _AUTOCAST_DEVICES:
        _LOGGER.warning(
            "precision=%r cannot be honoured on a %s device; using fp32",
            precision,
            device.type,
        )
        precision = "fp32"

    # `HardwareSpec` refuses fp16-on-CPU at load time, but degrading an absent
    # accelerator can recreate the combination it refused: `device='cuda',
    # precision='fp16'` on a machine with no CUDA lands on the CPU still
    # asking for fp16.  The same rule has to hold however the pairing arose,
    # or the spec's guarantee would depend on which machine ran the job.
    if precision == "fp16" and device.type == "cpu":
        _LOGGER.warning(
            "precision='fp16' is not usable on a cpu device, which this run "
            "degraded to; using fp32. Specify precision='bf16' for reduced "
            "precision that works on both"
        )
        precision = "fp32"

    # A scaler is only meaningful for fp16, and only on a device where
    # autocast runs.  See `ResolvedHardware.scaler` for why bf16 needs none.
    scaler = (
        torch.amp.GradScaler(device=device.type)
        if precision == "fp16" and device.type in _AUTOCAST_DEVICES
        else None
    )

    resolved = ResolvedHardware(
        device=device,
        precision=precision,
        requested_device=spec.device,
        scaler=scaler,
    )
    _LOGGER.info("resolved hardware: %s", resolved.describe())
    return resolved


def compile_if_requested(model: torch.nn.Module, *, spec: HardwareSpec) -> torch.nn.Module:
    """
    Apply Torch's graph compiler when the specification asks for it.

    Must be called *after* lazy parameters are materialised. Compiling an
    unmaterialised module traces a graph whose shapes are not yet known, which
    is the same ordering mistake as wrapping for distributed training too
    early -- defect 6.

    Parameters
    ----------
    model
        The materialised model.
    spec
        The hardware request.

    Returns
    -------
    torch.nn.Module
        The compiled model, or the original if compilation was not requested
        or is unavailable. Compilation is a throughput optimisation and never
        changes what the model computes, so an unavailable compiler warns and
        returns the original rather than failing the run.
    """
    if not spec.compile_model:
        return model

    compiler = getattr(torch, "compile", None)
    if compiler is None:
        _LOGGER.warning(
            "compile_model=True but this Torch build has no torch.compile; running uncompiled"
        )
        return model

    _LOGGER.info("compiling model with torch.compile")
    # The return type is a callable wrapper rather than a Module as far as the
    # type checker is concerned, but it behaves as one -- including
    # `state_dict` -- so it is narrowed here once instead of at each use.
    compiled: torch.nn.Module = compiler(model)
    return compiled
```

---

## 7. `src/rade_qnet/engines/torch/loaders.py`

10182 bytes · SHA-256 `856ddcfb1c2ca4e9`

```python
"""
Turning a ``BatchSource``'s NumPy batches into device-resident tensors.

The one place in the framework where a NumPy array becomes a
``torch.Tensor``. Everything upstream -- splitting, windowing, batch ordering
-- is engine-neutral, so every engine inherits one implementation of the parts
that are easy to get subtly wrong.

Static inputs leave per-sample collation
----------------------------------------
**Defect 4.** The implementation this framework replaces merged every static
tensor into every sample, and its collation function then ran ``torch.equal``
across the batch for each static key and returned ``values[0]``.

So for every batch of every epoch, a graph adjacency matrix was compared
against itself ``batch_size`` times to confirm something true by
construction. On a batch size of 64 and a 400-node graph that is 64
comparisons of a 160,000-element tensor, per batch, per epoch, to learn
nothing.

:class:`StaticInputs` replaces all of it. The static tensors are uploaded once
at the start of a fit and passed to every forward call by reference. This is
not a behavioural change and is worth being precise about why: the old
collation *already* returned exactly one copy, and the network *already*
received exactly one tensor. The same object reaches the same place; only the
comparison is gone.

Workers, and why the default is none
------------------------------------
``LoaderSpec.num_workers`` defaults to zero, meaning load in the main process.
That is the right default here and not merely a conservative one: the
framework fans a job set out across a process pool, and a worker process that
spawns its own loader workers oversubscribes the machine -- N jobs each
claiming W workers on a C-core box is NW processes competing for C cores, and
throughput collapses rather than degrading.

Because a ``BatchSource`` already yields whole batches, this module does not
construct a ``torch.utils.data.DataLoader`` at all. Prefetching across
processes would mean pickling the source, and the win for an in-memory array
is not worth the failure modes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.contract.data import TARGET_KEY
from ...core.runtime.errors import ContractError
from ...core.runtime.logging import get_logger
from .materialise import torch_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource

__all__ = ["TARGET_KEY", "StaticInputs", "to_device_batches", "to_tensor"]

_LOGGER = get_logger(__name__)


def to_tensor(
    value: object,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Convert one batch entry to a tensor on a device.

    Accepts a tensor as well as an array, and returns an existing tensor
    unchanged when it already has the right dtype and device. That matters for
    a static input: re-uploading a 400-node adjacency matrix on every forward
    call would reintroduce defect 4's cost by a different route.

    Parameters
    ----------
    value
        A NumPy array, a Torch tensor, or anything ``torch.as_tensor``
        accepts.
    dtype
        Target dtype. ``None`` keeps whatever the input has.
    device
        Target device. ``None`` keeps the input where it is.

    Returns
    -------
    torch.Tensor
        The converted tensor.
    """
    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(np.asarray(value))
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if device is not None and tensor.device != device:
        # `non_blocking` is deliberately not set.  It is only an advantage
        # from pinned host memory, and setting it without pinning silently
        # does nothing on some backends and returns an unready tensor on
        # others.
        tensor = tensor.to(device=device)
    return tensor


class StaticInputs:
    """
    The inputs that are the same for every batch, uploaded once.

    Parameters
    ----------
    tensors
        Static inputs, already on the target device.

    Notes
    -----
    Held in a small class rather than a bare dict so that the single upload is
    a visible event with a place to log it, and so the set can report itself in
    a run summary. An absent static set is an empty instance rather than
    ``None``, which keeps the forward call free of a branch.
    """

    def __init__(self, tensors: Mapping[str, torch.Tensor] | None = None) -> None:
        self.tensors: dict[str, torch.Tensor] = dict(tensors or {})

    @classmethod
    def from_source(
        cls,
        source: BatchSource,
        *,
        signature: InputSignature,
        device: torch.device,
    ) -> StaticInputs:
        """
        Upload a source's static inputs to a device, once.

        Parameters
        ----------
        source
            The batch source.
        signature
            The declared interface, used for the dtype of each static input.
        device
            Where to upload.

        Returns
        -------
        StaticInputs
            The uploaded set, empty when the source has none.

        Raises
        ------
        ContractError
            If the source supplies a static input the signature does not
            declare. A tensor nothing declared will be passed to ``forward``
            as a keyword argument and rejected there, with a message naming
            neither side.
        """
        supplied = dict(source.static)
        if not supplied:
            return cls()

        undeclared = sorted(set(supplied) - set(signature.static))
        if undeclared:
            raise ContractError(
                f"the source supplies static input(s) {undeclared} that the "
                f"signature does not declare; it declares "
                f"{sorted(signature.static)}"
            )

        tensors = {
            name: to_tensor(value, dtype=torch_dtype(signature.static[name].dtype), device=device)
            for name, value in supplied.items()
        }
        _LOGGER.info(
            "uploaded %d static input(s) to %s once: %s",
            len(tensors),
            device,
            {name: tuple(tensor.shape) for name, tensor in sorted(tensors.items())},
        )
        return cls(tensors)

    def __bool__(self) -> bool:
        """Return whether any static input is present."""
        return bool(self.tensors)

    def __len__(self) -> int:
        """Return how many static inputs are present."""
        return len(self.tensors)

    def merge_into(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Add the static inputs to a batch, by reference.

        Parameters
        ----------
        batch
            A batch of dynamic inputs. Mutated and returned, because this runs
            once per batch per epoch and allocating a fresh dict each time is
            measurable at small batch sizes.

        Returns
        -------
        dict
            The same dict, with the static inputs added.
        """
        batch.update(self.tensors)
        return batch

    def describe(self) -> dict[str, list[int]]:
        """
        Return the shape of each static input, for reports and logs.

        Returns
        -------
        dict
            Input name to shape.
        """
        return {name: list(tensor.shape) for name, tensor in sorted(self.tensors.items())}


def to_device_batches(
    source: BatchSource,
    *,
    signature: InputSignature,
    device: torch.device,
    static: StaticInputs | None = None,
    validate_first: bool = True,
) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
    """
    Convert one pass of a source into device-resident tensor batches.

    Yields the inputs and the target separately rather than as one mapping,
    because the learner needs them apart -- the inputs go to ``forward`` as
    keyword arguments and the target goes to the loss -- and splitting them
    here means the learner never has to know which key the target uses.

    Parameters
    ----------
    source
        The batch source.
    signature
        The declared interface, supplying each input's dtype.
    device
        Where to place the tensors.
    static
        Already-uploaded static inputs. ``None`` means none; pass the result
        of :meth:`StaticInputs.from_source` to avoid re-uploading per epoch.
    validate_first
        Whether to key-check the first batch against the signature. Only the
        first: the check is cheap but not free, and a source that produced a
        correctly keyed first batch and a differently keyed tenth one is a
        failure mode worth catching in the conformance suite rather than on
        every batch of every epoch.

    Yields
    ------
    tuple
        The inputs as a mapping, and the target.

    Raises
    ------
    ContractError
        If a batch carries no target, or -- on the first batch -- disagrees
        with the signature's declared keys.
    """
    statics = static if static is not None else StaticInputs()
    dtypes = {name: torch_dtype(spec.dtype) for name, spec in signature.dynamic.items()}
    target_dtype = torch_dtype(signature.target.dtype)

    for index, batch in enumerate(source.batches()):
        if index == 0 and validate_first:
            signature.validate_batch_keys(batch, where="to_device_batches", target_key=TARGET_KEY)
        if TARGET_KEY not in batch:
            raise ContractError(
                f"batch {index} carries no {TARGET_KEY!r}; a training batch must "
                f"carry the target. An inference stream without one belongs in "
                f"predict(), not here"
            )

        inputs = {
            name: to_tensor(value, dtype=dtypes.get(name), device=device)
            for name, value in batch.items()
            if name != TARGET_KEY
        }
        target = to_tensor(batch[TARGET_KEY], dtype=target_dtype, device=device)
        yield statics.merge_into(inputs), target
```

---

## 8. `src/rade_qnet/engines/torch/loops.py`

14963 bytes · SHA-256 `f43d783b05f0da87`

```python
"""
``fit_epochs`` -- passes over a finite source, driving a learner.

The library split described in ``ARCHITECTURE.md`` §6: the loop decides *when*
to step, validate, checkpoint and stop; the learner decides *what* one update
means. A supervised regression, a deep Q-network and a pathwise hedging
objective are then three learners sharing this one function, rather than three
training scripts.

Why the ``Learner`` protocol is declared here
---------------------------------------------
The loop is the consumer, so the loop declares the interface it needs. The
same dependency inversion puts ``Catalog`` and ``Tracker`` in
``core.runtime.context`` rather than in ``storage``: the party that depends on
a capability owns its definition, and the party that provides it satisfies a
contract it need not import.

Two drivers, one of them deferred
---------------------------------
:attr:`~rade_qnet.core.contract.source.BatchSource.steps_per_epoch` returning
``None`` means the source is unbounded, and that single value selects the
driver. A fixed dataset has a meaningful notion of a pass, so it is driven
here by epochs. An environment does not, so Phase 7 adds ``fit_steps``.

:func:`fit_epochs` refuses an unbounded source rather than inventing a pass
length for it, because the invented number would silently become the
denominator of every reported metric and the period of every schedule.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Protocol

import torch

from ...core.contract.result import EpochRecord, FitOutcome
from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from .callbacks import EpochContext
from .loaders import StaticInputs, to_device_batches

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ...core.contract.signature import InputSignature
    from ...core.contract.source import BatchSource
    from .callbacks import BestCheckpoint, Callback, GradientNorms

__all__ = ["Learner", "fit_epochs"]

_LOGGER = get_logger(__name__)

#: Key every learner reports its scalar objective under.
LOSS_KEY = "loss"

#: Key a learner uses to flag a step it skipped for a non-finite loss.
_NON_FINITE_KEY = "non_finite_steps"


class Learner(Protocol):
    """
    What one optimisation step means.

    Implemented by :class:`~.learners.supervised.SupervisedLearner` and, from
    Phase 7, by the reinforcement-learning update rules.
    """

    def train_step(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Perform one update and report its scalars.

        Parameters
        ----------
        model
            The prepared model.
        inputs
            Batch inputs.
        target
            Observed values.

        Returns
        -------
        dict
            Must carry ``loss``; any other key is averaged into the epoch's
            metrics.
        """
        ...

    def eval_step(
        self,
        model: torch.nn.Module,
        inputs: Mapping[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, float]:
        """
        Compute scalars for one batch without updating anything.

        Parameters
        ----------
        model
            The prepared model.
        inputs
            Batch inputs.
        target
            Observed values.

        Returns
        -------
        dict
            Must carry ``loss``.
        """
        ...


def _mean_of(totals: Mapping[str, float], n_batches: int) -> dict[str, float]:
    """
    Average a pass's accumulated scalars.

    Parameters
    ----------
    totals
        Summed scalars.
    n_batches
        Number of batches summed.

    Returns
    -------
    dict
        Means, or an empty mapping when no batch ran.
    """
    if n_batches == 0:
        return {}
    # The non-finite counter is a count and not an average: "0.03 non-finite
    # steps" is meaningless, "3 non-finite steps" is actionable.
    return {
        name: total if name == _NON_FINITE_KEY else total / n_batches
        for name, total in totals.items()
    }


def _run_pass(
    source: BatchSource,
    *,
    model: torch.nn.Module,
    learner: Learner,
    signature: InputSignature,
    device: torch.device,
    static: StaticInputs,
    training: bool,
) -> dict[str, float]:
    """
    Run one pass over a source and return its mean scalars.

    Parameters
    ----------
    source
        The batch source.
    model
        The prepared model.
    learner
        The update rule.
    signature
        The declared interface.
    device
        Where tensors are placed.
    static
        Already-uploaded static inputs.
    training
        Whether to update parameters. Also sets the model's mode, which is
        what makes dropout and batch normalisation behave correctly -- an
        evaluation pass left in training mode reports a loss computed with
        dropout active and batch statistics from the held-out batch.

    Returns
    -------
    dict
        Mean scalars for the pass.
    """
    model.train(training)
    step = learner.train_step if training else learner.eval_step

    totals: dict[str, float] = {}
    n_batches = 0
    for inputs, target in to_device_batches(
        source, signature=signature, device=device, static=static
    ):
        for name, value in step(model, inputs, target).items():
            totals[name] = totals.get(name, 0.0) + value
        n_batches += 1

    return _mean_of(totals, n_batches)


def fit_epochs(
    model: torch.nn.Module,
    sources: Mapping[str, BatchSource],
    *,
    learner: Learner,
    signature: InputSignature,
    device: torch.device,
    epochs: int,
    callbacks: Sequence[Callback] = (),
    gradient_norms: GradientNorms | None = None,
    checkpoint: BestCheckpoint | None = None,
    learning_rate: float | None = None,
) -> FitOutcome:
    """
    Train for a budget of passes over the training source.

    Parameters
    ----------
    model
        The prepared model. Must already be materialised: an optimiser built
        over unmaterialised parameters tracks nothing, which is defect 6.
    sources
        Batch sources by split name. ``train`` is required.
    learner
        The update rule.
    signature
        The declared interface.
    device
        Where tensors are placed.
    epochs
        Maximum passes over the training source.
    callbacks
        Consulted at the end of each epoch, in order. Any one of them may end
        the run.
    gradient_norms
        Tracker whose per-epoch summary is folded into the epoch's metrics.
    checkpoint
        Best-weight tracker, asked to restore at the end. Passed separately
        from ``callbacks`` because the loop needs its answer -- which epoch
        was best, and whether its weights were restored -- for the outcome,
        and a bare ``Callback`` cannot be asked. It also supplies the name
        recorded as the monitored metric, which is taken from it rather than
        passed separately so the reported monitor cannot disagree with the one
        that actually chose the best epoch.
    learning_rate
        Rate to record against the first epoch when no schedule is in use.

    Returns
    -------
    FitOutcome
        The epoch history, the best epoch, and whether the run stopped early
        and restored the best weights.

    Raises
    ------
    EngineError
        If there is no training source, if it is unbounded, or if a learner
        reports no ``loss``.
    """
    if "train" not in sources:
        raise EngineError(f"fit_epochs needs a 'train' source; received {sorted(sources)}")

    train_source = sources["train"]
    if train_source.steps_per_epoch is None:
        raise EngineError(
            "fit_epochs cannot drive an unbounded source, which reports "
            "steps_per_epoch=None. An unbounded source has no notion of a pass, "
            "so an epoch count would be arbitrary and would silently become the "
            "denominator of every reported metric. Use fit_steps instead"
        )

    validation_source = sources.get("validation")
    static = StaticInputs.from_source(train_source, signature=signature, device=device)

    history: list[EpochRecord] = []
    stopped_early = False
    started = time.perf_counter()

    for epoch in range(epochs):
        epoch_started = time.perf_counter()

        train_scalars = _run_pass(
            train_source,
            model=model,
            learner=learner,
            signature=signature,
            device=device,
            static=static,
            training=True,
        )
        if LOSS_KEY not in train_scalars:
            raise EngineError(
                f"the learner reported {sorted(train_scalars)} for epoch {epoch} "
                f"but not {LOSS_KEY!r}; every learner must report its objective "
                f"under that key, because the loop and the callbacks are written "
                f"against it"
            )

        # A non-finite training loss means the parameters are already NaN, and
        # nothing in a training loop can recover from that: every subsequent
        # epoch computes NaN gradients over NaN weights.  Stopping here rather
        # than finishing the budget matters because the failure otherwise
        # surfaces in the evaluate stage, as a message about non-finite
        # predictions -- which points at the metrics rather than at the epoch
        # the run actually diverged in.
        if not math.isfinite(train_scalars[LOSS_KEY]):
            raise EngineError(
                f"the training loss became {train_scalars[LOSS_KEY]} at epoch "
                f"{epoch + 1}, so the parameters are no longer finite and no "
                f"later epoch can recover. The usual causes are a non-finite "
                f"input reaching the model, a learning rate too high for the "
                f"objective, or a loss that divides by a quantity that reached "
                f"zero. Check the data build's quality metrics first, then "
                f"lower the learning rate or enable gradient clipping"
            )

        val_scalars = (
            _run_pass(
                validation_source,
                model=model,
                learner=learner,
                signature=signature,
                device=device,
                static=static,
                training=False,
            )
            if validation_source is not None
            else {}
        )

        # Validation scalars are prefixed, so `loss` from the two passes
        # cannot collide in one metrics mapping.
        metrics = {name: value for name, value in train_scalars.items() if name != LOSS_KEY}
        metrics.update(
            {f"val_{name}": value for name, value in val_scalars.items() if name != LOSS_KEY}
        )
        if gradient_norms is not None:
            metrics.update(gradient_norms.summarise())

        record = EpochRecord(
            epoch=epoch,
            train_loss=train_scalars[LOSS_KEY],
            val_loss=val_scalars.get(LOSS_KEY),
            metrics=metrics,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - epoch_started,
        )
        history.append(record)
        _log_epoch(record, epochs)

        context = _context_from(record)
        # Checkpointing runs before the other callbacks so that the snapshot
        # is taken from the epoch's own weights, whatever a schedule does next.
        if checkpoint is not None:
            checkpoint.on_epoch_end(context)
        for callback in callbacks:
            callback.on_epoch_end(context)

        if any(callback.should_stop for callback in callbacks):
            stopped_early = True
            break

        # Read after the callbacks, so a schedule's change is recorded against
        # the epoch it takes effect in rather than the one that caused it.
        learning_rate = _current_rate(callbacks, default=learning_rate)

    restored = checkpoint.restore_best() if checkpoint is not None else False
    best_epoch = checkpoint.monitor.best_epoch if checkpoint is not None else None
    best_value = checkpoint.monitor.best_value if checkpoint is not None else None
    monitor = checkpoint.monitor.monitor if checkpoint is not None else "val_loss"

    outcome = FitOutcome(
        history=tuple(history),
        monitor=monitor,
        best_epoch=best_epoch,
        best_monitor_value=best_value,
        stopped_early=stopped_early,
        restored_best=restored,
        total_seconds=time.perf_counter() - started,
    )
    # `EpochRecord.epoch` is zero-based, but every per-epoch line above was
    # logged one-based in the `n of m` form.  Reporting the raw index here
    # would put "epoch 1/3" and "best ... at epoch 0" in the same log, which
    # reads as an off-by-one in the training rather than in the numbering.
    _LOGGER.info(
        "fit finished: %d epoch(s) in %.1fs, best %s=%s at epoch %s%s",
        outcome.n_epochs,
        outcome.total_seconds,
        monitor,
        None if best_value is None else f"{best_value:.6g}",
        "n/a" if best_epoch is None else best_epoch + 1,
        " (weights restored)" if restored else "",
    )
    return outcome


def _context_from(record: EpochRecord) -> EpochContext:
    """
    Build the callback context for a finished epoch.

    Parameters
    ----------
    record
        The epoch's record.

    Returns
    -------
    EpochContext
        The context callbacks are given.
    """
    return EpochContext(
        epoch=record.epoch,
        train_loss=record.train_loss,
        val_loss=record.val_loss,
        metrics=record.metrics,
        learning_rate=record.learning_rate,
    )


def _current_rate(callbacks: Sequence[Callback], *, default: float | None) -> float | None:
    """
    Return the learning rate a schedule callback reports, if there is one.

    Parameters
    ----------
    callbacks
        The callbacks in use.
    default
        Rate to return when no callback reports one.

    Returns
    -------
    float or None
        The current rate.
    """
    for callback in callbacks:
        rate = getattr(callback, "current_rate", None)
        if rate is not None:
            return float(rate)
    return default


def _log_epoch(record: EpochRecord, epochs: int) -> None:
    """
    Log one epoch's headline numbers.

    Parameters
    ----------
    record
        The epoch's record.
    epochs
        The epoch budget, for the ``n of m`` form.
    """
    validation = "" if record.val_loss is None else f" val_loss={record.val_loss:.6g}"
    _LOGGER.info(
        "epoch %d/%d train_loss=%.6g%s (%.2fs)",
        record.epoch + 1,
        epochs,
        record.train_loss,
        validation,
        record.seconds,
    )
```

---

## 9. `src/rade_qnet/engines/torch/losses.py`

9939 bytes · SHA-256 `43b81664fc961647`

```python
"""
The loss registry.

A loss is named in the specification and looked up here, rather than chosen by
an ``if`` chain inside the learner. That buys two things worth the indirection:
a loss can be registered by a model package without touching the engine, and
an unrecognised name fails when the spec is read -- listing what is available
-- instead of partway into the first epoch.

The asymmetric losses, and why they are in the framework
--------------------------------------------------------
``asymmetric`` and ``quantile`` look domain-specific, but neither is: each is
a general statement about how to weigh errors -- over-prediction against
under-prediction, or one quantile of the error distribution -- and both are
used well outside P&L work.

They live here because a loss is consumed by the *learner*, and the learner
is engine code. A loss that only one model needs can still be registered from
that model's package without touching this module; the registry is what makes
that possible.

What is problem-specific is the *choice* of loss and the value of its
asymmetry, and those live in a configuration file where they belong.

Both reduce to a mean over the batch rather than a sum, so a loss value is
comparable between runs with different batch sizes. A sum is not: halving the
batch size halves the reported loss while changing nothing about the model.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import torch

from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "LossFunction",
    "asymmetric_loss",
    "build_loss",
    "quantile_loss",
    "register_loss",
    "registered_losses",
]

_LOGGER = get_logger(__name__)

#: Default penalty multiplier applied to under-prediction by ``asymmetric``.
#: Chosen as a neutral-but-visible default: 1.0 would make the loss identical
#: to ``mae`` and hide the fact that an asymmetric loss was configured at all.
_DEFAULT_UNDER_PENALTY = 2.0

#: Default quantile for ``quantile``. The median, which makes the loss equal
#: to half the mean absolute error and so gives a recognisable baseline.
_DEFAULT_QUANTILE = 0.5


class LossFunction(Protocol):
    """
    A loss: predictions and a target in, one scalar out.

    A protocol rather than a type alias so that a registered loss carrying
    configuration -- a quantile level, an asymmetry factor -- satisfies the
    same contract as a plain function.
    """

    def __call__(self, predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Return the loss for one batch.

        Parameters
        ----------
        predictions
            Model output.
        target
            Observed values, broadcastable to ``predictions``.

        Returns
        -------
        torch.Tensor
            A zero-dimensional tensor, so that ``backward`` can be called on
            it without a further reduction.
        """
        ...


def asymmetric_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    under_penalty: float = _DEFAULT_UNDER_PENALTY,
) -> torch.Tensor:
    """
    Absolute error with under-prediction penalised more heavily.

    The objective for a quantity where the two directions of error cost
    different amounts. Under-predicting a risk figure leaves a position
    unhedged; over-predicting it costs some funding. A symmetric loss asserts
    those are equally bad, which for a risk number is simply false.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.
    under_penalty
        Multiplier applied where the prediction falls below the target. One
        reduces this to the mean absolute error.

    Returns
    -------
    torch.Tensor
        Scalar mean loss.
    """
    error = target - predictions
    # `error > 0` is exactly the under-prediction case: the target exceeded
    # what was predicted.
    weight = torch.where(error > 0, torch.full_like(error, under_penalty), torch.ones_like(error))
    return (weight * error.abs()).mean()


def quantile_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    *,
    quantile: float = _DEFAULT_QUANTILE,
) -> torch.Tensor:
    """
    Pinball loss, which fits a conditional quantile rather than a mean.

    Used when the useful output is an interval rather than a point: training
    at 0.05 and 0.95 gives a band, and a band is what a risk limit is set
    against.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.
    quantile
        The quantile to fit, strictly between zero and one.

    Returns
    -------
    torch.Tensor
        Scalar mean loss.

    Raises
    ------
    EngineError
        If the quantile is not in ``(0, 1)``. At zero or one the loss is
        one-sided and the minimiser is unbounded, so it would train to an
        extreme value rather than failing visibly.
    """
    if not 0.0 < quantile < 1.0:
        raise EngineError(
            f"quantile must be strictly between 0 and 1, received {quantile}; "
            f"at the boundary the loss is one-sided and has no finite minimiser"
        )
    error = target - predictions
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def _mse(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return the mean squared error.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.nn.functional.mse_loss(predictions, target)


def _mae(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return the mean absolute error.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.nn.functional.l1_loss(predictions, target)


def _huber(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Return the Huber loss with the Torch default transition point.

    Squared near zero and absolute in the tails, which is the usual choice
    for a P&L series: it keeps the smooth gradient of a squared error without
    letting one outlying day dominate the epoch.

    Parameters
    ----------
    predictions
        Model output.
    target
        Observed values.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    return torch.nn.functional.huber_loss(predictions, target)


#: The registry. Mutable by `register_loss` only, so every entry arrives
#: through the duplicate check there.
_REGISTRY: dict[str, LossFunction] = {
    "mse": _mse,
    "mae": _mae,
    "huber": _huber,
    "asymmetric": asymmetric_loss,
    "quantile": quantile_loss,
}


def register_loss(name: str, loss: LossFunction, *, replace: bool = False) -> None:
    """
    Add a loss to the registry.

    Parameters
    ----------
    name
        Name the specification will use.
    loss
        The loss function.
    replace
        Whether to overwrite an existing entry.

    Raises
    ------
    EngineError
        If the name is taken and ``replace`` is false. Silently overwriting
        would mean two packages registering the same name produce different
        objectives depending on import order, which is a difference nothing
        would report.
    """
    if name in _REGISTRY and not replace:
        raise EngineError(
            f"a loss named {name!r} is already registered; pass replace=True to "
            f"override it deliberately"
        )
    _REGISTRY[name] = loss
    _LOGGER.debug("registered loss %r", name)


def registered_losses() -> tuple[str, ...]:
    """
    Return the registered loss names, sorted.

    Returns
    -------
    tuple of str
        Available names.
    """
    return tuple(sorted(_REGISTRY))


def build_loss(name: str, *, options: Mapping[str, object] | None = None) -> LossFunction:
    """
    Return a loss by name, with its options bound.

    Parameters
    ----------
    name
        A registered loss name.
    options
        Keyword arguments to bind -- ``quantile`` for the pinball loss,
        ``under_penalty`` for the asymmetric one. Bound here rather than
        passed per call, so the learner calls every loss identically.

    Returns
    -------
    LossFunction
        The loss, ready to call with predictions and a target.

    Raises
    ------
    EngineError
        If the name is unknown, listing what is available, or if an option is
        not accepted by the chosen loss -- which is usually a quantile set on
        a loss that has no quantile, and would otherwise be ignored in
        silence.
    """
    if name not in _REGISTRY:
        raise EngineError(
            f"unknown loss {name!r}; registered losses are {list(registered_losses())}"
        )

    loss = _REGISTRY[name]
    if not options:
        return loss

    import inspect  # noqa: PLC0415  Needed only on the configured-options path.

    accepted = set(inspect.signature(loss).parameters)
    rejected = sorted(set(options) - accepted)
    if rejected:
        raise EngineError(
            f"loss {name!r} does not accept option(s) {rejected}; it accepts "
            f"{sorted(accepted - {'predictions', 'target'})}"
        )

    def configured(predictions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Return the configured loss for one batch.

        Parameters
        ----------
        predictions
            Model output.
        target
            Observed values.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        return loss(predictions, target, **options)

    return configured
```

---

## 10. `src/rade_qnet/engines/torch/materialise.py`

11265 bytes · SHA-256 `b19a9c1b5031cb03`

```python
"""
Giving a lazily shaped model its parameters, before anything else touches it.

**Defect 6.** A module built from ``LazyLinear`` and friends has no parameters
until it has seen one input: the shapes are inferred from the first forward
pass. The implementation this framework replaces wrapped such a model for
distributed training before that point, which hands the wrapper an empty
parameter group to synchronise. The outcomes are a crash deep inside the
distributed library, or -- considerably worse -- a wrapper that synchronises
nothing, so every rank trains its own private copy and the run reports a
plausible loss curve for a model that was never actually distributed.

The same ordering trap catches three other things, which is why this runs
first and not just before the distributed wrapper:

* an **optimiser** constructed over an empty parameter list tracks nothing,
  and ``step()`` is then a no-op that raises no error;
* a **checkpoint** of an unmaterialised module has no tensors in it, and
  loading it back succeeds while restoring nothing;
* ``torch.compile`` traces a graph whose shapes are not yet known.

So the order is fixed: materialise, then hardware, then distribute, then
optimiser. ``ARCHITECTURE.md`` §5 makes ``materialise`` its own pipeline stage
rather than a detail inside the engine, precisely so that this ordering is
visible where runs are defined and cannot be rearranged by accident.

Why a signature is enough
-------------------------
The dummy forward needs exact shapes and dtypes with no data present, and
that is the whole reason :class:`~rade_qnet.core.contract.signature.InputSignature`
exists. Synthesising the batch from the signature rather than borrowing a real
one keeps materialisation independent of the data build, which is what makes a
six-month-old bundle reconstructible: saved signature plus saved weights, no
dataset required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature, TensorSpec

__all__ = [
    "count_parameters",
    "dummy_batch",
    "has_lazy_parameters",
    "materialise",
    "torch_dtype",
]

_LOGGER = get_logger(__name__)

#: Batch size for the dummy forward. Two rather than one, so that a module
#: which collapses a singleton batch dimension -- or a batch-norm layer, which
#: refuses a batch of one in training mode -- is exercised honestly.
_DUMMY_BATCH_SIZE = 2

#: Spec dtype names mapped to Torch dtypes. A fixed table rather than
#: ``getattr(torch, name)``, so an unrecognised name produces a message naming
#: the supported set instead of an ``AttributeError``.
_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

#: Dtypes for which a plain ``ones`` fill is the right dummy value. Integer
#: inputs are usually indices, and zero is the one index guaranteed to be
#: inside any embedding table -- ones would be out of range for a table of
#: size one, which is exactly what a minimal test fixture has.
_INTEGER_DTYPES = frozenset({torch.int8, torch.int16, torch.int32, torch.int64, torch.bool})


def torch_dtype(name: str) -> torch.dtype:
    """
    Return the Torch dtype for a signature's dtype name.

    Parameters
    ----------
    name
        A library-agnostic dtype name, as carried by
        :class:`~rade_qnet.core.contract.signature.TensorSpec`.

    Returns
    -------
    torch.dtype
        The corresponding Torch dtype.

    Raises
    ------
    EngineError
        If the name is not recognised, listing the supported names.
    """
    try:
        return _DTYPES[name]
    except KeyError:
        raise EngineError(
            f"dtype {name!r} is not supported by the Torch engine; supported "
            f"dtypes are {sorted(_DTYPES)}"
        ) from None


def _dummy_tensor(spec: TensorSpec, *, batch_size: int, device: torch.device) -> torch.Tensor:
    """
    Build one synthetic tensor from a spec.

    Parameters
    ----------
    spec
        The declared shape and dtype.
    batch_size
        Size to substitute for each wildcard dimension.
    device
        Where to allocate.

    Returns
    -------
    torch.Tensor
        A tensor matching the spec.
    """
    shape = spec.concrete_shape(batch_size)
    dtype = torch_dtype(spec.dtype)
    # Zeros for integers and booleans, which are indices or masks: see the
    # note on `_INTEGER_DTYPES`.  Ones for floats, because a zero-filled float
    # input makes a multiplicative layer's output independent of its weights,
    # so a shape error in the weight initialisation would not show up.
    if dtype in _INTEGER_DTYPES:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.ones(shape, dtype=dtype, device=device)


def dummy_batch(
    signature: InputSignature,
    *,
    batch_size: int = _DUMMY_BATCH_SIZE,
    device: torch.device | None = None,
    include_static: bool = True,
) -> dict[str, torch.Tensor]:
    """
    Synthesise a batch from a signature, with no data present.

    Parameters
    ----------
    signature
        The declared interface.
    batch_size
        Size for each wildcard dimension.
    device
        Where to allocate. ``None`` means the CPU.
    include_static
        Whether to include the static inputs. False when the caller already
        holds the real static tensors and wants only the dynamic part
        synthesised -- which is the better path when they are available,
        because a graph adjacency matrix of ones is a complete graph and some
        models will not run on one.

    Returns
    -------
    dict
        Input name to synthetic tensor.
    """
    target_device = device or torch.device("cpu")
    batch = {
        name: _dummy_tensor(spec, batch_size=batch_size, device=target_device)
        for name, spec in signature.dynamic.items()
    }
    if include_static:
        batch.update(
            {
                name: _dummy_tensor(spec, batch_size=batch_size, device=target_device)
                for name, spec in signature.static.items()
            }
        )
    return batch


def has_lazy_parameters(model: torch.nn.Module) -> bool:
    """
    Return whether any parameter is still waiting for its shape.

    Parameters
    ----------
    model
        The model to inspect.

    Returns
    -------
    bool
        True if the model contains an uninitialised parameter or buffer.
    """
    uninitialised = (
        torch.nn.parameter.UninitializedParameter,
        torch.nn.parameter.UninitializedBuffer,
    )
    return any(
        isinstance(tensor, uninitialised)
        for tensor in (*model.parameters(recurse=True), *model.buffers(recurse=True))
    )


def count_parameters(model: torch.nn.Module, *, trainable_only: bool = False) -> int:
    """
    Return the number of parameter elements.

    Zero for an unmaterialised model, which is the signal that makes defect 6
    detectable: an optimiser built over zero parameters is a silent no-op, so
    the count is asserted after materialisation rather than trusted.

    Parameters
    ----------
    model
        The model to inspect.
    trainable_only
        Count only parameters that require a gradient.

    Returns
    -------
    int
        Number of elements.
    """
    if has_lazy_parameters(model):
        return 0
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def materialise(
    model: torch.nn.Module,
    signature: InputSignature,
    *,
    static: Mapping[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """
    Run one dummy forward pass so lazy parameters acquire their shapes.

    Safe to call on an already-materialised model, in which case it returns
    immediately. That makes it callable unconditionally from the pipeline,
    which is what keeps the ordering guarantee from depending on a model
    author remembering to declare that their model is lazy.

    Parameters
    ----------
    model
        The constructed model, possibly unmaterialised.
    signature
        The declared interface, from which the dummy batch is built.
    static
        Real static tensors, when the caller has them. Preferred over
        synthetic ones: a synthetic adjacency matrix of ones is a complete
        graph, and a model that normalises by node degree may not accept one.
    device
        Where to run the pass.

    Returns
    -------
    torch.nn.Module
        The same model, materialised.

    Raises
    ------
    EngineError
        If the dummy pass fails, or if it completes without materialising the
        parameters. The second case is the quiet one: it means the model's
        forward does not route through its lazy submodules, so the run would
        proceed with an optimiser tracking nothing.
    """
    if not has_lazy_parameters(model):
        _LOGGER.debug("model has no lazy parameters; nothing to materialise")
        return model

    batch = dummy_batch(signature, device=device, include_static=not static)
    if static:
        batch.update(static)

    _LOGGER.info(
        "materialising lazy parameters with a dummy batch: %s",
        {name: tuple(tensor.shape) for name, tensor in sorted(batch.items())},
    )

    was_training = model.training
    model.eval()
    try:
        # `no_grad` because this pass exists only to fix shapes.  Without it
        # the dummy batch would build a graph and the first real backward pass
        # could be computed against synthetic activations.
        with torch.no_grad():
            model(**batch)
    except Exception as error:
        raise EngineError(
            f"the dummy forward pass used to materialise lazy parameters failed: "
            f"{type(error).__name__}: {error}. The batch was built from the input "
            f"signature:\n{signature.describe()}\nEither the signature disagrees "
            f"with what forward() accepts, or forward() needs real values rather "
            f"than synthetic ones -- in which case pass the real static inputs"
        ) from error
    finally:
        # Restored rather than left in eval mode: a model that arrived in
        # training mode must leave in training mode, or dropout and batch-norm
        # would behave differently for reasons invisible at the call site.
        model.train(was_training)

    if has_lazy_parameters(model):
        raise EngineError(
            "the dummy forward pass completed but the model still has "
            "uninitialised parameters; its forward() does not route through "
            "every lazy submodule. An optimiser built now would track nothing "
            "and training would appear to run while changing no weights"
        )

    _LOGGER.info("materialised %d parameter element(s)", count_parameters(model))
    return model
```

---

## 11. `src/rade_qnet/engines/torch/predictor.py`

6452 bytes · SHA-256 `87c7e1bff8d8aba1`

```python
"""
Batched inference, including the precompute path.

Training and inference ask a model for the same thing -- a forward pass --
under opposite constraints. Training updates parameters, so nothing computed
from them survives a step. Inference does not, so anything that does not vary
across batches can be computed once.

For the flagship that distinction is most of the runtime. Its graph encoder
turns a static adjacency into node embeddings, and during training that work
is unavoidable because the encoder's weights are moving. During a prediction
pass the weights are frozen and the graph does not change, so recomputing the
embeddings for every batch repeats identical arithmetic -- which on a long
evaluation pass can dominate everything else.

:class:`~rade_qnet.core.capability.protocols.Precomputable` is how a model says
so. This module is where the framework acts on the declaration, and the
critical property is that it must not change any number. A performance path
that quietly perturbs results is worse than no performance path at all,
because the discrepancy shows up as a model that scores differently in
evaluation than it did in training, with nothing pointing at the cause.

So the two routes are kept as close as possible: the same device placement,
the same static inputs, the same batch order, the same dtype on the way out.
The only difference is which method the module is called through, and
``test_precompute_matches_plain_path`` asserts the outputs are identical
rather than merely close.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from ...core.capability.protocols import Precomputable
from ...core.runtime.errors import EngineError
from ...core.runtime.logging import get_logger
from .loaders import StaticInputs, to_device_batches

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from ...core.contract.source import BatchSource
    from .hardware import ResolvedHardware

__all__ = ["predict_batches"]

_LOGGER = get_logger(__name__)


def predict_batches(
    module: torch.nn.Module,
    source: BatchSource,
    *,
    resolved: ResolvedHardware,
    allow_precompute: bool = True,
) -> NDArray[np.floating]:
    """
    Run a forward pass over a bounded source and return the raw output.

    Returns the model's own output space. Inverting the target transform is
    the pipeline's job, using the fitted state -- an engine that inverted it
    here would invert it twice, once for the pipeline and once for itself.

    Parameters
    ----------
    module
        The model, already on the right device. Put into evaluation mode
        here rather than by the caller, because a forward pass left in
        training mode would apply dropout and update batch-norm statistics,
        and the resulting predictions would be both wrong and irreproducible.
    source
        Batches to predict over. Must be bounded: an unbounded source has no
        last batch, so there is nothing to return.
    resolved
        The hardware placement, supplying the device and the autocast
        context.
    allow_precompute
        Whether to use the precompute path when the model declares it.
        Defaults to true. The escape hatch exists so a test can run both
        routes over one model and compare them, which is the only way to
        know the optimisation is safe.

    Returns
    -------
    numpy.ndarray
        Predictions in source order, one row per sample.

    Raises
    ------
    EngineError
        If the source is unbounded, or yields no batches at all.
    """
    if source.steps_per_epoch is None:
        raise EngineError(
            "predict needs a bounded source; this one reports "
            "steps_per_epoch=None, so it has no last batch and there is "
            "nothing to return"
        )

    signature = source.signature
    static = StaticInputs.from_source(source, signature=signature, device=resolved.device)

    module.eval()
    blocks: list[NDArray[np.floating]] = []

    with torch.no_grad(), resolved.autocast():
        precomputed = _precompute(module, static, enabled=allow_precompute)

        for inputs, _ in to_device_batches(
            source, signature=signature, device=resolved.device, static=static
        ):
            if precomputed is None:
                output = module(**inputs)
            else:
                output = module.forward_with_precomputed(inputs, precomputed)
            # Cast to float32 before leaving Torch: NumPy has no bfloat16, so
            # a reduced-precision tensor would fail to convert, and a float16
            # one would silently lose decimal digits of a P&L figure.
            blocks.append(output.detach().to(dtype=torch.float32).cpu().numpy())

    if not blocks:
        raise EngineError(
            f"the source yielded no batches, so there is nothing to predict "
            f"(it reports steps_per_epoch={source.steps_per_epoch})"
        )
    return np.concatenate(blocks, axis=0)


def _precompute(
    module: torch.nn.Module,
    static: StaticInputs,
    *,
    enabled: bool,
) -> Mapping[str, torch.Tensor] | None:
    """
    Compute the reusable static encoding, when the model offers one.

    Parameters
    ----------
    module
        The model, already in evaluation mode and inside ``no_grad``. Both
        matter: the encoding is computed once and reused, so it must be
        produced under exactly the conditions the batches will be.
    static
        The static inputs, already on the device.
    enabled
        Whether to try at all.

    Returns
    -------
    Mapping or None
        The precomputed tensors, or ``None`` to take the ordinary forward
        path -- which is also what a model with no static inputs gets, since
        there is nothing for it to encode.
    """
    if not enabled or not isinstance(module, Precomputable):
        return None

    tensors = static.tensors
    if not tensors:
        _LOGGER.debug(
            "%s declares precompute() but the source has no static inputs; "
            "using the ordinary forward path",
            type(module).__name__,
        )
        return None

    precomputed = module.precompute(tensors)
    _LOGGER.debug(
        "precomputed %d static tensor(s) for %s, reused across the pass",
        len(precomputed),
        type(module).__name__,
    )
    return precomputed
```

---

## 12. `src/rade_qnet/engines/torch/seeding.py`

5679 bytes · SHA-256 `34f9f4cfe1207350`

```python
"""
Seeds Torch's random number generators, and makes ``strict`` mean something.

``core.runtime.seeding`` seeds Python and NumPy and then calls out to whatever
seeders have registered. It cannot seed Torch itself: ``core`` has an empty
dependency set, so it cannot import a training library. The registration
therefore belongs here, in the package that owns the library -- which is the
same dependency inversion the engine registry uses, applied to a different
concern.

Without this module, ``seed_everything`` leaves Torch unseeded. Every weight
initialisation, every dropout mask and every shuffled batch order would differ
between two runs of the same configuration, and the framework would be
reporting a seed that it had not actually applied to the one library that
matters most. Nothing in the output would say so: both runs complete, both
report the seed they were given, and the scores differ.

**What ``strict`` does and does not buy.** Seeding makes the *sampling*
reproducible. It does not make the *arithmetic* reproducible: a GPU reduction
sums in an order that depends on how the work was scheduled, and cuDNN selects
among algorithms by benchmarking them. ``strict`` therefore also asks Torch to
use deterministic algorithm implementations and turns the benchmark selector
off. That costs throughput, which is exactly why it is not the default.

Some operations have no deterministic implementation at all. Under ``strict``
Torch raises when one is reached, and that error propagates rather than being
caught: a user who asked for reproducibility must be told it is unavailable
for their model rather than receiving results that quietly are not
reproducible. Under ``warn`` the same situation logs and continues, which is
the level to use when reproducibility is desirable but not worth refusing a
run over.
"""

from __future__ import annotations

import os

import torch

from ...core.runtime.logging import get_logger
from ...core.runtime.seeding import Determinism, register_seeder

__all__ = ["SEEDER_NAME", "seed_torch"]

#: Name this seeder registers under. Exported so a test can unregister it and
#: so a user replacing it does not have to guess the string.
SEEDER_NAME = "torch"

#: Environment variable cuBLAS reads to select a deterministic reduction
#: workspace. It is read by the library at its first allocation, so setting it
#: later in a process has no effect -- which is why it is set here, at seeding
#: time, rather than inside the fit.
_CUBLAS_WORKSPACE_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"

#: The configuration cuBLAS requires for deterministic reductions. ``:4096:8``
#: is the larger of the two documented settings; it costs a little more memory
#: than ``:16:8`` and does not restrict the stream count.
_CUBLAS_WORKSPACE_VALUE = ":4096:8"

_LOGGER = get_logger(__name__)


def seed_torch(seed: int, determinism: Determinism = "off") -> None:
    """
    Seed every Torch generator and apply the requested determinism level.

    Parameters
    ----------
    seed
        A non-negative integer below ``2 ** 32``.
    determinism
        ``off`` seeds and nothing more. ``warn`` additionally asks for
        deterministic algorithms but tolerates an operation that has none.
        ``strict`` refuses such an operation, so that a run claiming
        reproducibility has it.

    Raises
    ------
    RuntimeError
        Propagated from Torch under ``strict`` when an operation reached
        during the run has no deterministic implementation. Deliberately not
        caught: ``seed_everything`` turns it into a specification error naming
        this seeder, and the alternative is a run that reports a seed it could
        not honour.
    """
    # `manual_seed` covers the CPU generator and every visible CUDA device, so
    # a per-device loop is unnecessary and would miss a device that appears
    # later.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Seeds all devices, including ones not yet initialised.
        torch.cuda.manual_seed_all(seed)

    if determinism == "off":
        return

    # cuDNN benchmarks its convolution algorithms on the first call and caches
    # the winner. The benchmark is timing-dependent, so the chosen algorithm --
    # and therefore the arithmetic -- can differ between two runs on the same
    # machine. Turning it off is a throughput cost and a prerequisite for
    # reproducibility.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if determinism == "strict":
        # Must be set before cuBLAS allocates its workspace, which happens on
        # the first matrix multiply. Setting it here means seeding is the last
        # moment it can still take effect.
        os.environ.setdefault(_CUBLAS_WORKSPACE_VARIABLE, _CUBLAS_WORKSPACE_VALUE)

    # `warn_only=True` logs and proceeds where no deterministic implementation
    # exists; under `strict` the same situation raises. That is the whole
    # difference between the two levels, and it is a difference in what the
    # user is promised rather than in what the framework attempts.
    torch.use_deterministic_algorithms(True, warn_only=determinism == "warn")
    _LOGGER.debug("seeded torch with %d at determinism=%r", seed, determinism)


# Registered at import, so importing the engine is enough to make
# `seed_everything` cover Torch. The train pipeline imports the engine before
# it seeds, which is the ordering this relies on -- and the ordering is tested,
# because a seeder registered after seeding is a seeder that did nothing.
register_seeder(SEEDER_NAME, seed_torch, replace=True)
```

