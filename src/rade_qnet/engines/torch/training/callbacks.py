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
