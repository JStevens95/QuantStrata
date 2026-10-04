# `src/rade_qnet/engines/torch/training`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 39 | 1832 | `1513f7834fd4c319` |
| 2 | `callbacks.py` | 689 | 21962 | `4f364430b3ee327c` |
| 3 | `checkpoint.py` | 300 | 10642 | `df134fb5e194563f` |
| 4 | `loops.py` | 908 | 31738 | `4cb0b2dc45846113` |
| 5 | `losses.py` | 348 | 9946 | `e2df8fc6f1313596` |

---

## 1. `src/rade_qnet/engines/torch/training/__init__.py`

1832 bytes · SHA-256 `1513f7834fd4c319`

```python
"""
How a fit is executed: the drivers, and everything that watches them.

The split from ``learners``
---------------------------
A learner decides *what one update means*.  This package decides *when* an
update happens, when to validate, when to write a checkpoint and when to
stop.  That is the whole of the Phase 2 loop/learner split, and the reason it
is worth a package boundary is that the two change for entirely different
reasons: a new algorithm adds a learner and touches nothing here, while a new
stopping rule changes ``callbacks.py`` and touches no algorithm.

Why this is not a framework-level package
------------------------------------------
Every module here is irreducibly PyTorch.  The loop calls ``optimiser.step``,
the callbacks save ``state_dict`` objects and drive
``torch.optim.lr_scheduler``, the losses are ``torch.nn`` modules.  The part
that genuinely *is* library-agnostic was hoisted long ago and lives in
:mod:`rade_qnet.core.contract.result` as ``FitOutcome`` and ``EpochRecord`` --
which is why the xgboost engine, whose fit is a single call with no loop at
all, still produces the same history type as a hundred-epoch gradient run.

Modules
-------
``loops.py``
    The two drivers.  ``fit_epochs`` makes passes over a finite dataset;
    ``fit_steps`` spends a budget of interaction against an unbounded one.
    Also the two learner protocols, ``Learner`` and ``PolicyLearner``.
``callbacks.py``
    Observation and intervention between epochs: ``EarlyStopping``,
    ``BestCheckpoint``, ``LearningRateSchedule``, ``GradientNorms``, and the
    ``EpochContext`` they all read.
``losses.py``
    Objective functions, resolved by name from a training specification.
``checkpoint.py``
    Reading and writing weights, so a run can be resumed or a bundle loaded.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/engines/torch/training/callbacks.py`

21962 bytes · SHA-256 `4f364430b3ee327c`

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

from ....core.lifecycle.errors import EngineError
from ....core.provenance.logging import get_logger
from .checkpoint import restore_state_dict, snapshot_state_dict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ....core.spec.training import CheckpointSpec, EarlyStoppingSpec, SchedulerSpec

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

## 3. `src/rade_qnet/engines/torch/training/checkpoint.py`

10642 bytes · SHA-256 `df134fb5e194563f`

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

from ....core.lifecycle.errors import BundleError, EngineError
from ....core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from ...base import ModelHandle

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

## 4. `src/rade_qnet/engines/torch/training/loops.py`

31738 bytes · SHA-256 `4cb0b2dc45846113`

```python
"""
The two loop drivers: ``fit_epochs`` over a dataset, ``fit_steps`` over an environment.

The library split described in ``ARCHITECTURE.md`` §6: the loop decides *when*
to step, validate, checkpoint and stop; the learner decides *what* one update
means. A supervised regression, a deep Q-network and a pathwise hedging
objective are then three learners sharing a driver, rather than three
training scripts.

Why the learner protocols are declared here
-------------------------------------------
The loop is the consumer, so the loop declares the interface it needs. The
same dependency inversion puts ``Catalog`` and ``Tracker`` in
``core.lifecycle.context`` rather than in ``storage``: the party that depends on
a capability owns its definition, and the party that provides it satisfies a
contract it need not import.

Two drivers, selected by the source
-----------------------------------
:attr:`~rade_qnet.core.contract.source.BatchSource.steps_per_epoch` returning
``None`` means the source is unbounded, and that single value selects the
driver. A fixed dataset has a meaningful notion of a pass, so it is driven by
epochs. An environment does not, so it is driven by a step budget. The source
declares which it is; the engine does not guess.

Each driver refuses the other's source rather than coping, because the
invented number -- a pass length for an environment, a step budget for a
dataset -- would silently become the denominator of every reported metric and
the period of every schedule.

Record: why there are two learner protocols
-------------------------------------------
``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.1 claimed Phase 7 would add its
learners "against an unchanged loop", and said that if a second loop were
needed, *that* was the finding rather than a licence to fork. So, recorded as
the finding:

The second **driver** was never the surprise -- ``core.contract.source`` has
specified since Phase 1 that an unbounded source "is driven by ``fit_steps``",
and :func:`fit_epochs` has always refused one by name. What did not survive
contact is the **learner protocol**. :class:`Learner` takes
``(model, inputs, target)``, because :func:`fit_epochs` splits each batch into
inputs and a target using the signature's declared target.

A policy has no target. ``PolicySignature`` has an observation space and an
action space and nothing resembling one, and what an interactive update
consumes is a whole transition -- observation, action, reward, next
observation, and the two episode-end flags -- not a pair. Passing a reward as
``target`` would type-check and would be a lie: a reward is not a label, and
a learner reading it as one is the exact confusion the paradigm split exists
to prevent.

Hence :class:`PolicyLearner`, with its own two methods. The alternative --
widening :class:`Learner` to make ``target`` optional and adding the five
other fields -- would push the branch *into every supervised learner*, which
is the coupling the split was for. The loop/learner split itself held: both
drivers below are the same shape, and the pipeline is a sibling of the
supervised one rather than a forked lifecycle.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import torch

from ....core.contract.result import EpochRecord, FitOutcome
from ....core.lifecycle.errors import EngineError
from ....core.provenance.logging import get_logger
from ..loaders import TARGET_KEY, StaticInputs, to_device_batches, to_tensor
from .callbacks import EpochContext

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from ....core.contract.signature import InputSignature
    from ....core.contract.source import BatchSource
    from .callbacks import BestCheckpoint, Callback, GradientNorms

__all__ = ["ExperienceSummary", "Learner", "PolicyLearner", "fit_epochs", "fit_steps"]

_LOGGER = get_logger(__name__)

#: Key every learner reports its scalar objective under.
LOSS_KEY = "loss"

#: Key a learner uses to flag a step it skipped for a non-finite loss.
_NON_FINITE_KEY = "non_finite_steps"


class Learner(Protocol):
    """
    What one optimisation step means, for supervised learning.

    Implemented by :class:`~.learners.supervised.SupervisedLearner`. An
    interactive update rule implements :class:`PolicyLearner` instead -- see
    the record in the module docstring for why the two are separate.
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


class PolicyLearner(Protocol):
    """
    What one interactive update means, and how the policy chooses an action.

    Two methods rather than :class:`Learner`'s two, because interaction needs
    a different pair.

    :meth:`act` is here, and not on the model, because exploration is a
    property of the *algorithm*: the same network is acted on greedily by one
    learner, epsilon-greedily by another, and sampled from by a third. Putting
    it on the policy would force every network to carry every exploration
    scheme, and would make a saved policy's behaviour depend on which
    algorithm had trained it.

    There is no ``eval_step``. A dataset has a held-out split; an environment
    does not, and the only honest way to evaluate a policy is to run more
    episodes with exploration turned off. That is a different *amount of
    interaction*, not a different step, so it belongs to the pipeline which
    can collect it -- not here.
    """

    def act(self, policy: torch.nn.Module, observation: object) -> object:
        """
        Choose one action for one observation.

        Called by the collector, not by the driver, and therefore outside any
        gradient graph: an implementation must detach or use ``no_grad``, or
        the graph of every action taken is retained until the update, which
        for a replay buffer means retaining it for the whole run.

        Parameters
        ----------
        policy
            The prepared policy.
        observation
            One observation, already placed on the policy's device by the
            engine and not batched. Device placement happens there rather
            than here for the same reason it does for a batch: a learner
            that had to do its own would get it wrong once per learner, and
            the symptom -- a forward pass failing inside a linear layer -- is
            a long way from the cause.

        Returns
        -------
        object
            One action the environment will accept. Not necessarily a
            tensor: a discrete environment wants an index.
        """
        ...

    def update(
        self,
        policy: torch.nn.Module,
        experience: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Perform one update from a batch of experience and report its scalars.

        Parameters
        ----------
        policy
            The prepared policy.
        experience
            One batch of transitions, on the device, keyed as the source's
            signature declares: observations, actions, next observations, the
            two episode-end flags, and the reward under
            :data:`~rade_qnet.core.contract.data.TARGET_KEY`.

            Passed whole rather than pre-split, because what an algorithm
            reads varies -- a policy-gradient method ignores the next
            observation entirely, a value-based one depends on it -- and a
            driver that split it would have to know which algorithm it was
            driving.

        Returns
        -------
        dict
            Must carry ``loss``; any other key is averaged into the reported
            block's metrics. A learner that performs no update still reports
            one, so that "nothing happened" is visible rather than absent.
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


@runtime_checkable
class ExperienceSummary(Protocol):
    """
    An opt-in capability: a source that can summarise the episodes it has seen.

    Checked with ``isinstance`` and skipped when absent, which is the
    framework's standard capability pattern -- see
    ``core.authoring.capabilities``. Optional rather than part of
    :class:`~rade_qnet.core.contract.source.BatchSource` because a replay
    source serving stored transitions has no episodes of its own to report,
    and widening the source protocol would force it to invent an answer.
    """

    def episode_summary(self) -> Mapping[str, float]:
        """
        Summarise the finished episodes.

        Returns
        -------
        Mapping
            Named statistics, or empty when no episode has finished.
        """
        ...


def fit_steps(
    policy: torch.nn.Module,
    source: BatchSource,
    *,
    learner: PolicyLearner,
    device: torch.device,
    total_steps: int,
    report_every_steps: int,
    callbacks: Sequence[Callback] = (),
    gradient_norms: GradientNorms | None = None,
    checkpoint: BestCheckpoint | None = None,
    learning_rate: float | None = None,
) -> FitOutcome:
    """
    Train for a budget of environment steps against an unbounded source.

    The interactive counterpart of :func:`fit_epochs`, and deliberately the
    same shape: collect, update, record, consult the callbacks, maybe stop.
    What differs is only the unit of progress -- steps taken rather than
    passes completed -- and that the experience arrives whole rather than
    split into inputs and a target.

    How the budget is counted
    ------------------------
    One update per batch, and the steps in that batch are counted from the
    batch itself rather than from a configured batch size. The source decides
    how much experience one update gets; the driver counts what it was
    handed. A second number here could disagree with the source's, and the
    disagreement would be invisible -- the run would simply stop at the wrong
    time.

    Parameters
    ----------
    policy
        The prepared policy. Must already be materialised, for the reason in
        :func:`fit_epochs`.
    source
        An unbounded source of experience, normally a
        :class:`~rade_qnet.sources.batching.rollout.RolloutSource`.
    learner
        The update rule, which also chooses the actions.
    device
        Where tensors are placed.
    total_steps
        Environment steps to run before stopping.
    report_every_steps
        Steps per reported block. One :class:`EpochRecord` is emitted per
        block, and the callbacks are consulted once per block.

        Blocks rather than updates because an interactive run does thousands
        of small updates, and a history with one entry each would be useless
        to read and would make early stopping fire on the noise of a single
        batch.
    callbacks
        Consulted at the end of each block, in order. Any one of them may end
        the run.
    gradient_norms
        Tracker whose summary is folded into each block's metrics.
    checkpoint
        Best-weight tracker. For an interactive run the metric worth
        monitoring is the episode return, not a loss, because a policy-
        gradient loss falling says almost nothing about whether the agent
        improved.
    learning_rate
        Rate to record against the first block when no schedule is in use.

    Returns
    -------
    FitOutcome
        The block history, the best block, and whether the run stopped early
        and restored the best weights.

    Raises
    ------
    EngineError
        If the source is bounded, if the budget or block size is not
        positive, if a learner reports no ``loss``, or if the loss becomes
        non-finite.
    """
    if source.steps_per_epoch is not None:
        raise EngineError(
            f"fit_steps drives an unbounded source, but this one reports "
            f"steps_per_epoch={source.steps_per_epoch}. A bounded source has a "
            f"meaningful pass, so a step budget would cut it off mid-dataset or "
            f"silently repeat it. Use fit_epochs instead"
        )
    if total_steps <= 0 or report_every_steps <= 0:
        raise EngineError(
            f"fit_steps needs positive total_steps and report_every_steps; "
            f"received {total_steps} and {report_every_steps}"
        )

    history: list[EpochRecord] = []
    stopped_early = False
    started = time.perf_counter()

    steps_taken = 0
    block = 0
    block_started = time.perf_counter()
    totals: dict[str, float] = {}
    updates_in_block = 0
    next_report = min(report_every_steps, total_steps)

    policy.train(True)
    for batch in source.batches():
        experience = {name: to_tensor(value, device=device) for name, value in batch.items()}
        scalars = learner.update(policy, experience)
        if LOSS_KEY not in scalars:
            raise EngineError(
                f"the learner reported {sorted(scalars)} for the update at step "
                f"{steps_taken} but not {LOSS_KEY!r}; every learner must report "
                f"its objective under that key, because the loop and the "
                f"callbacks are written against it"
            )

        for name, value in scalars.items():
            totals[name] = totals.get(name, 0.0) + value
        updates_in_block += 1
        steps_taken += _steps_in(experience)

        if steps_taken < next_report:
            continue

        record = _block_record(
            block,
            totals=totals,
            updates=updates_in_block,
            source=source,
            gradient_norms=gradient_norms,
            learning_rate=learning_rate,
            seconds=time.perf_counter() - block_started,
        )
        # Non-finite for the same reason as in `fit_epochs`: the parameters
        # are already NaN and no later update recovers.  Raised here rather
        # than after the callbacks so a checkpoint is never taken from
        # weights that are known to be broken.
        if not math.isfinite(record.train_loss):
            raise EngineError(
                f"the training loss became {record.train_loss} after "
                f"{steps_taken} step(s), so the parameters are no longer finite "
                f"and no later update can recover. In an interactive run the "
                f"usual causes are an unbounded reward, a value target that "
                f"bootstrapped past a terminal state, or a learning rate too "
                f"high for the objective. Check the episode return first: if it "
                f"diverged before the loss did, the reward is the problem"
            )

        history.append(record)
        _log_block(record, steps_taken=steps_taken, total_steps=total_steps)

        context = _context_from(record)
        if checkpoint is not None:
            checkpoint.on_epoch_end(context)
        for callback in callbacks:
            callback.on_epoch_end(context)

        if any(callback.should_stop for callback in callbacks):
            stopped_early = True
            break

        learning_rate = _current_rate(callbacks, default=learning_rate)

        if steps_taken >= total_steps:
            break

        block += 1
        block_started = time.perf_counter()
        totals = {}
        updates_in_block = 0
        next_report = min(steps_taken + report_every_steps, total_steps)

    restored = checkpoint.restore_best() if checkpoint is not None else False
    outcome = FitOutcome(
        history=tuple(history),
        monitor=checkpoint.monitor.monitor if checkpoint is not None else LOSS_KEY,
        best_epoch=checkpoint.monitor.best_epoch if checkpoint is not None else None,
        best_monitor_value=checkpoint.monitor.best_value if checkpoint is not None else None,
        stopped_early=stopped_early,
        restored_best=restored,
        total_seconds=time.perf_counter() - started,
    )
    _LOGGER.info(
        "fit finished: %d step(s) over %d block(s) in %.1fs%s",
        steps_taken,
        outcome.n_epochs,
        outcome.total_seconds,
        " (weights restored)" if restored else "",
    )
    return outcome


def _steps_in(experience: Mapping[str, torch.Tensor]) -> int:
    """
    Count the transitions in one batch of experience.

    Read from the target -- the reward -- because it is the one entry
    guaranteed to be exactly one value per transition. An observation may
    carry extra dimensions, and a source may add keys this driver has never
    heard of.

    Parameters
    ----------
    experience
        One batch, on the device.

    Returns
    -------
    int
        The number of transitions.

    Raises
    ------
    EngineError
        If the batch carries no reward, which means it is not experience.
    """
    reward = experience.get(TARGET_KEY)
    if reward is None:
        raise EngineError(
            f"the experience batch has {sorted(experience)} but no {TARGET_KEY!r}; "
            f"fit_steps counts its budget in transitions, and a batch with no "
            f"reward is not experience"
        )
    return int(reward.shape[0]) if reward.ndim else 1


def _block_record(
    block: int,
    *,
    totals: Mapping[str, float],
    updates: int,
    source: BatchSource,
    gradient_norms: GradientNorms | None,
    learning_rate: float | None,
    seconds: float,
) -> EpochRecord:
    """
    Summarise one block of updates as a history record.

    Parameters
    ----------
    block
        Zero-based block index.
    totals
        Scalars summed over the block's updates.
    updates
        How many updates the block contained.
    source
        Asked for its episode summary, if it can give one.
    gradient_norms
        Tracker whose summary is folded in, if present.
    learning_rate
        Rate in effect for the block.
    seconds
        Wall-clock duration of the block.

    Returns
    -------
    EpochRecord
        The block's record, with ``val_loss`` left ``None`` -- an environment
        has no held-out split, and reporting the training loss there would
        make a monitor watching it think it was watching validation.
    """
    means = _mean_of(totals, updates)
    metrics = {name: value for name, value in means.items() if name != LOSS_KEY}
    if isinstance(source, ExperienceSummary):
        metrics.update(source.episode_summary())
    if gradient_norms is not None:
        metrics.update(gradient_norms.summarise())

    return EpochRecord(
        epoch=block,
        train_loss=means[LOSS_KEY],
        val_loss=None,
        metrics=metrics,
        learning_rate=learning_rate,
        seconds=seconds,
    )


def _log_block(record: EpochRecord, *, steps_taken: int, total_steps: int) -> None:
    """
    Log one block's headline numbers.

    Parameters
    ----------
    record
        The block's record.
    steps_taken
        Steps completed so far, logged instead of a block index because the
        budget is in steps and that is what a reader is tracking.
    total_steps
        The step budget.
    """
    episode_return = record.metrics.get("episode_return")
    reported = "" if episode_return is None else f" episode_return={episode_return:.6g}"
    _LOGGER.info(
        "step %d/%d loss=%.6g%s (%.2fs)",
        steps_taken,
        total_steps,
        record.train_loss,
        reported,
        record.seconds,
    )
```

---

## 5. `src/rade_qnet/engines/torch/training/losses.py`

9946 bytes · SHA-256 `e2df8fc6f1313596`

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

from ....core.lifecycle.errors import EngineError
from ....core.provenance.logging import get_logger

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

