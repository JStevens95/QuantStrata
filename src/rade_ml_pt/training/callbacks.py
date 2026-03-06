"""
Hook-based training callbacks for the PyTorch training loop.

Ports Keras-style callbacks (``tf.keras.callbacks``) to lightweight classes
whose hook methods (``on_train_begin``, ``on_epoch_end``, …) are invoked by
:class:`~src.rade_ml_pt.training.trainer.Trainer`.

Provided callbacks:
    * :class:`EarlyStopping` – stop when a metric plateaus.
    * :class:`ModelCheckpoint` – persist best / periodic ``state_dict`` snapshots.
    * :class:`ReduceLROnPlateau` – thin wrapper around the PyTorch scheduler.
    * :class:`TensorBoardLogger` – scalar logging via ``SummaryWriter``.
    * :class:`MetricsLogger` – epoch-level JSON log with timing and summary.
    * :func:`get_standard_callbacks` – factory driven by ``TrainingConfig``.
"""
from __future__ import annotations

import copy
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from src.rade_ml_pt.core.config import TrainingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base callback
# ---------------------------------------------------------------------------

class Callback:
    """Base callback with no-op hooks for training loop integration."""

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        pass


# ---------------------------------------------------------------------------
# EarlyStopping
# ---------------------------------------------------------------------------

class EarlyStopping(Callback):
    """Stop training when *monitor* has not improved for *patience* epochs.

    When ``restore_best_weights=True`` the model's ``state_dict`` is rolled
    back to the best-seen snapshot upon stopping.

    :param patience: epochs to wait before stopping.
    :param min_delta: minimum absolute change that qualifies as an improvement.
    :param monitor: metric key to watch in ``logs``.
    :param mode: ``"min"`` (lower is better) or ``"max"``.
    :param restore_best_weights: reload best weights when stopping.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        monitor: str = "val_loss",
        mode: str = "min",
        restore_best_weights: bool = True,
    ) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_value: float = float("inf") if mode == "min" else float("-inf")
        self.best_weights: Optional[Dict[str, Any]] = None
        self.wait: int = 0
        self.stopped_epoch: int = 0
        self.stop_training: bool = False

    # Injected by the Trainer before the loop starts
    _model: Optional[torch.nn.Module] = None

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        is_improvement = (
            (current < self.best_value - self.min_delta)
            if self.mode == "min"
            else (current > self.best_value + self.min_delta)
        )

        if is_improvement:
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights and self._model is not None:
                self.best_weights = copy.deepcopy(self._model.state_dict())
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                if (
                    self.restore_best_weights
                    and self.best_weights is not None
                    and self._model is not None
                ):
                    self._model.load_state_dict(self.best_weights)
                    logger.info(
                        "EarlyStopping: restoring best weights from epoch %d.",
                        epoch - self.wait,
                    )


# ---------------------------------------------------------------------------
# ModelCheckpoint
# ---------------------------------------------------------------------------

class ModelCheckpoint(Callback):
    """Save ``state_dict`` snapshots to *checkpoint_dir* during training.

    :param checkpoint_dir: directory for ``.pt`` files.
    :param save_best_only: only save when *monitor* improves.
    :param monitor: metric key to watch.
    :param mode: ``"min"`` or ``"max"``.
    :param save_weights_only: reserved for future use (always saves state_dict).
    :param save_freq: reserved for future use (always per-epoch).
    """

    _model: Optional[torch.nn.Module] = None

    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        monitor: str = "val_loss",
        mode: str = "min",
        save_weights_only: bool = False,
        save_freq: str = "epoch",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.save_freq = save_freq
        self.best_value: float = float("inf") if mode == "min" else float("-inf")

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (
            (current < self.best_value) if self.mode == "min" else (current > self.best_value)
        )

        if self.save_best_only and not improved:
            return
        if improved:
            self.best_value = current

        path = self.checkpoint_dir / f"model_{epoch + 1:03d}.pt"
        if self._model is not None:
            torch.save(self._model.state_dict(), path)
            logger.info("Checkpoint saved: %s", path)


# ---------------------------------------------------------------------------
# ReduceLROnPlateau
# ---------------------------------------------------------------------------

class ReduceLROnPlateau(Callback):
    """Thin wrapper around ``torch.optim.lr_scheduler.ReduceLROnPlateau``.

    Call :meth:`set_optimizer` before training to bind the internal scheduler.

    :param monitor: metric key to watch.
    :param factor: factor by which LR is reduced.
    :param patience: epochs with no improvement before reducing.
    :param mode: ``"min"`` or ``"max"``.
    :param min_lr: lower bound on the learning rate.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        factor: float = 0.1,
        patience: int = 5,
        mode: str = "min",
        min_lr: float = 1e-6,
    ) -> None:
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.mode = mode
        self.min_lr = min_lr
        self._scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None

    def set_optimizer(self, optimizer: torch.optim.Optimizer) -> None:
        """Create the internal ``ReduceLROnPlateau`` scheduler."""
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            min_lrs=[self.min_lr],
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}
        value = logs.get(self.monitor)
        if value is not None and self._scheduler is not None:
            self._scheduler.step(value)


# ---------------------------------------------------------------------------
# TensorBoardLogger
# ---------------------------------------------------------------------------

class TensorBoardLogger(Callback):
    """Log epoch-level scalars to TensorBoard via ``SummaryWriter``.

    Falls back silently if ``tensorboard`` is not installed.

    :param log_dir: directory for TensorBoard event files.
    """

    def __init__(self, log_dir: str = "./logs/tensorboard") -> None:
        self.log_dir = log_dir
        self._writer: Optional[Any] = None

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._writer = SummaryWriter(log_dir=self.log_dir)
        except ImportError:
            logger.warning("tensorboard not installed; skipping TensorBoard logging.")

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is None:
            return
        logs = logs or {}
        for key, value in logs.items():
            self._writer.add_scalar(f"training/{key}", value, epoch)

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        if self._writer is not None:
            self._writer.close()


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------

class MetricsLogger(Callback):
    """Persist per-epoch metrics and timing information to a JSON file.

    At ``on_train_end`` the full history, per-epoch wall-clock times and a
    compact summary block are written to ``<log_dir>/<log_file>``.

    :param log_dir: directory for the JSON log.
    :param log_file: filename inside *log_dir*.
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        log_file: str = "training_log.json",
    ) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.log_path = self.log_dir / self.log_file

        self.history: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.start_time: Optional[float] = None
        self._epoch_start: Optional[float] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

    def on_train_begin(self, logs: Optional[Dict[str, Any]] = None) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.history = {}
        self.epoch_times = []
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        logs = logs or {}

        # Record wall-clock time for this epoch
        epoch_duration = time.time() - self._epoch_start if self._epoch_start else 0.0
        self.epoch_times.append(epoch_duration)

        # Accumulate per-metric history
        for key, value in logs.items():
            self.history.setdefault(key, []).append(value)

        # Track best validation loss
        val_loss = logs.get("val_loss")
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1

        # Log a concise one-liner per epoch
        parts = [f"{k}={v:.6f}" for k, v in logs.items()]
        logger.info(
            "Epoch %d/%s  %s  [%.1fs]",
            epoch + 1,
            "?",
            "  ".join(parts),
            epoch_duration,
        )

    def on_train_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        total_time = time.time() - self.start_time if self.start_time else 0.0

        payload: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "total_training_time_seconds": round(total_time, 2),
            "total_epochs": len(self.epoch_times),
            "epoch_times_seconds": [round(t, 4) for t in self.epoch_times],
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss if self.best_val_loss < float("inf") else None,
            "history": self.history,
            "summary": self._build_summary(total_time),
        }

        try:
            with open(self.log_path, "w") as fh:
                json.dump(payload, fh, indent=2)
            logger.info("Training log saved to %s", self.log_path)
        except OSError as exc:
            logger.warning("Failed to write training log: %s", exc)

    def _build_summary(self, total_time: float) -> Dict[str, Any]:
        """Compile a concise summary block for the JSON payload."""
        summary: Dict[str, Any] = {
            "total_time_seconds": round(total_time, 2),
            "epochs_completed": len(self.epoch_times),
            "best_epoch": self.best_epoch,
        }

        if self.best_val_loss < float("inf"):
            summary["best_val_loss"] = round(self.best_val_loss, 6)

        # Final-epoch metrics
        for key, values in self.history.items():
            if values:
                summary[f"final_{key}"] = round(values[-1], 6)

        # Average epoch duration
        if self.epoch_times:
            summary["avg_epoch_time_seconds"] = round(
                sum(self.epoch_times) / len(self.epoch_times), 4
            )

        return summary


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_standard_callbacks(config: "TrainingConfig") -> List[Callback]:
    """Build the standard callback list from a :class:`TrainingConfig`.

    :param config: training configuration dataclass.
    :returns: list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if config.early_stopping is not None:
        callbacks.append(
            EarlyStopping(
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                monitor=config.early_stopping.monitor,
                mode=config.early_stopping.mode,
                restore_best_weights=config.early_stopping.restore_best_weights,
            )
        )

    if config.checkpoint is not None:
        callbacks.append(
            ModelCheckpoint(
                checkpoint_dir=config.checkpoint.checkpoint_dir,
                save_best_only=config.checkpoint.save_best_only,
                monitor=config.checkpoint.monitor,
                mode=config.checkpoint.mode,
            )
        )

    if config.lr_reduction is not None:
        callbacks.append(
            ReduceLROnPlateau(
                monitor=config.lr_reduction.monitor,
                factor=config.lr_reduction.factor,
                patience=config.lr_reduction.patience,
                mode=config.lr_reduction.mode,
                min_lr=config.lr_reduction.min_lr,
            )
        )

    if config.log_dir is not None:
        tb_dir = str(Path(config.log_dir) / "tensorboard")
        callbacks.append(TensorBoardLogger(log_dir=tb_dir))
        callbacks.append(MetricsLogger(log_dir=config.log_dir))

    return callbacks
