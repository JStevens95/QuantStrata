"""
Generic training loop for QuantStrata ML models.

Provides run_training() and TrainingLoop for any model conforming to the
Trainable protocol.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, Union

import numpy as np

from src.machine_learning.core.config import TrainingConfig
from src.machine_learning.core.protocols import Trainable
from src.machine_learning.core.types import (
    TrainingResult,
    CheckpointInfo,
)

logger = logging.getLogger(__name__)


def _batch_iterator(
    features: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield batches of (features, targets)."""
    n = len(features)
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_idx = indices[start:end]
        yield features[batch_idx], targets[batch_idx]


def _split_data(
    features: np.ndarray,
    targets: np.ndarray,
    validation_split: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Split data into train and validation sets."""
    if validation_split <= 0.0 or validation_split >= 1.0:
        return features, targets, None, None
    n = len(features)
    split_idx = int(n * (1 - validation_split))
    indices = np.arange(n)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]
    return features[train_idx], targets[train_idx], features[val_idx], targets[val_idx]


class TrainingLoop:
    """
    Generic training loop for Trainable models.

    This loop works with any model conforming to the ``Trainable`` protocol
    and uses the rich ``TrainingConfig`` from ``core.config``.  It reads
    nested sub-configs (``early_stopping``, ``checkpoint``, etc.) to drive
    early-stopping, checkpointing, and logging.

    Supports
    --------
    - Epoch-based training with configurable batch size
    - Validation split or separate validation data
    - Checkpointing (save best, periodic)
    - Early stopping (via ``config.early_stopping``)
    - Per-epoch logging

    Example
    -------
    >>> loop = TrainingLoop(model, config)
    >>> result = loop.run(train_features, train_targets)
    """

    def __init__(
        self,
        model: Trainable,
        config: TrainingConfig,
        train_step_fn: Optional[Callable[[Any, Any], float]] = None,
        loss_fn: Optional[Callable[[Any, Any], float]] = None,
        log_every: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        model : Trainable
            Model conforming to the Trainable protocol.
        config : TrainingConfig
            Training configuration (from ``core.config``).
        train_step_fn : callable, optional
            Custom training step ``(inputs, targets) -> loss``.
            If *None*, falls back to ``model.train_step`` or
            ``model.forward`` + ``model.compute_loss``.
        loss_fn : callable, optional
            Standalone loss function ``(y_true, y_pred) -> float``.
            Used when neither *train_step_fn* nor ``model.train_step``
            are available.
        log_every : int
            Log metrics every N epochs (default 1).
        """
        self.model = model
        self.config = config
        self._train_step_fn = train_step_fn
        self._loss_fn = loss_fn
        self._log_every = log_every

        self._history: dict = {"loss": [], "val_loss": []}
        self._checkpoints: List[CheckpointInfo] = []
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._epochs_without_improvement = 0

    # -- helpers that read nested config safely -----------------------------

    @property
    def _early_stopping_patience(self) -> int:
        """Return patience from nested early-stopping config, or 0."""
        es = self.config.early_stopping
        return es.patience if es is not None else 0

    @property
    def _checkpoint_dir(self) -> Optional[str]:
        """Return checkpoint directory, or *None* if checkpointing is off."""
        ck = self.config.checkpoint
        return ck.checkpoint_dir if ck is not None else None

    @property
    def _save_best_only(self) -> bool:
        ck = self.config.checkpoint
        return ck.save_best_only if ck is not None else True

    # -- training / validation steps ----------------------------------------

    def _train_step(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Run a single training step and return loss."""
        if self._train_step_fn is not None:
            return self._train_step_fn(features, targets)
        if hasattr(self.model, "train_step"):
            return self.model.train_step(features, targets)
        y_pred = self.model.forward(features)
        if self._loss_fn is not None:
            return float(self._loss_fn(targets, y_pred))
        return self.model.compute_loss(targets, y_pred)

    def _validate(
        self,
        val_features: np.ndarray,
        val_targets: np.ndarray,
    ) -> float:
        """Compute validation loss."""
        y_pred = self.model.forward(val_features)
        if self._loss_fn is not None:
            return float(self._loss_fn(val_targets, y_pred))
        return self.model.compute_loss(val_targets, y_pred)

    # -- checkpointing ------------------------------------------------------

    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        is_best: bool,
    ) -> Optional[CheckpointInfo]:
        """Save checkpoint if configured."""
        ckpt_dir_str = self._checkpoint_dir
        if ckpt_dir_str is None:
            return None
        ckpt_dir = Path(ckpt_dir_str)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        suffix = "best" if is_best else f"epoch_{epoch}"
        ckpt_path = ckpt_dir / f"checkpoint_{suffix}.json"

        params = self.model.get_parameters()
        serialisable_params = {}
        for k, v in params.items():
            if isinstance(v, np.ndarray):
                serialisable_params[k] = v.tolist()
            elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
                serialisable_params[k] = [arr.tolist() for arr in v]
            else:
                serialisable_params[k] = v
        with open(ckpt_path, "w") as f:
            json.dump(serialisable_params, f)

        info = CheckpointInfo(
            path=str(ckpt_path),
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            is_best=is_best,
        )
        self._checkpoints.append(info)
        if self.config.verbose >= 1:
            logger.info(
                "Checkpoint saved: %s (epoch %d, is_best=%s)",
                ckpt_path, epoch, is_best,
            )
        return info

    # -- main loop ----------------------------------------------------------

    def run(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        val_features: Optional[np.ndarray] = None,
        val_targets: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """
        Run the training loop.

        Parameters
        ----------
        features : ndarray
            Training features.
        targets : ndarray
            Training targets.
        val_features : ndarray, optional
            Validation features (overrides ``config.validation_split``).
        val_targets : ndarray, optional
            Validation targets.

        Returns
        -------
        TrainingResult
            Training history, checkpoints, and metadata.
        """
        start_time = time.time()

        # Split data if no explicit validation set provided
        if val_features is None and self.config.validation_split > 0:
            features, targets, val_features, val_targets = _split_data(
                features, targets, self.config.validation_split
            )
        has_validation = val_features is not None and val_targets is not None
        early_stopped = False

        for epoch in range(1, self.config.epochs + 1):
            # -- training epoch --
            epoch_losses = []
            for batch_x, batch_y in _batch_iterator(
                features, targets, self.config.batch_size
            ):
                loss = self._train_step(batch_x, batch_y)
                epoch_losses.append(loss)
            train_loss = float(np.mean(epoch_losses))
            self._history["loss"].append(train_loss)

            # -- validation --
            val_loss = None
            if has_validation:
                val_loss = self._validate(val_features, val_targets)
                self._history["val_loss"].append(val_loss)

            # -- logging --
            if self.config.verbose >= 1 and epoch % self._log_every == 0:
                msg = f"Epoch {epoch}/{self.config.epochs} — loss: {train_loss:.5f}"
                if val_loss is not None:
                    msg += f", val_loss: {val_loss:.5f}"
                logger.info(msg)

            # -- best-model tracking --
            compare_loss = val_loss if has_validation else train_loss
            is_best = compare_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = compare_loss
                self._best_epoch = epoch
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

            # -- checkpointing --
            should_save = False
            if self._checkpoint_dir is not None:
                if self._save_best_only and is_best:
                    should_save = True
            if should_save:
                self._save_checkpoint(epoch, train_loss, val_loss, is_best)

            # -- early stopping --
            patience = self._early_stopping_patience
            if patience > 0 and self._epochs_without_improvement >= patience:
                if self.config.verbose >= 1:
                    logger.info(
                        "Early stopping at epoch %d (no improvement for %d epochs)",
                        epoch, patience,
                    )
                early_stopped = True
                break

        training_time = time.time() - start_time
        return TrainingResult(
            history=self._history,
            final_epoch=epoch,
            best_epoch=self._best_epoch,
            best_train_loss=min(self._history["loss"]),
            best_val_loss=(
                min(self._history["val_loss"])
                if self._history["val_loss"]
                else None
            ),
            checkpoints=self._checkpoints,
            config=self.config.to_dict(),
            training_time_seconds=training_time,
            stopped_early=early_stopped,
        )


def run_training(
    model: Trainable,
    features: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    val_features: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None,
    train_step_fn: Optional[Callable[[Any, Any], float]] = None,
    loss_fn: Optional[Callable[[Any, Any], float]] = None,
) -> TrainingResult:
    """
    Run training for a Trainable model.

    This is the main entry point for the generic (framework-agnostic)
    training pipeline.

    Parameters
    ----------
    model : Trainable
        Model conforming to the Trainable protocol.
    features : ndarray
        Training features.
    targets : ndarray
        Training targets.
    config : TrainingConfig
        Training configuration (from ``core.config``).
    val_features : ndarray, optional
        Validation features.
    val_targets : ndarray, optional
        Validation targets.
    train_step_fn : callable, optional
        Custom training step function.
    loss_fn : callable, optional
        Loss function ``(y_true, y_pred) -> float``.

    Returns
    -------
    TrainingResult
        Training history, checkpoints, and metadata.

    Example
    -------
    >>> from src.machine_learning.core.config import TrainingConfig
    >>> config = TrainingConfig(epochs=50)
    >>> result = run_training(model, X_train, y_train, config)
    >>> print(result.best_train_loss)
    """
    loop = TrainingLoop(
        model, config,
        train_step_fn=train_step_fn,
        loss_fn=loss_fn,
    )
    return loop.run(features, targets, val_features, val_targets)


__all__ = ["run_training", "TrainingLoop"]
