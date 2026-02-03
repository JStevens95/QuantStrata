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

from src.machine_learning.core.protocols import Trainable
from src.machine_learning.core.types import (
    TrainingConfig,
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

    Supports:
    - Epoch-based training with configurable batch size
    - Validation split or separate validation data
    - Checkpointing (save best, periodic)
    - Early stopping
    - Logging

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
    ) -> None:
        """
        Parameters
        ----------
        model : Trainable
            Model conforming to the Trainable protocol.
        config : TrainingConfig
            Training configuration.
        train_step_fn : callable, optional
            Custom training step function (inputs, targets) -> loss.
            If None, uses model.compute_loss after model.forward.
        """
        self.model = model
        self.config = config
        self._train_step_fn = train_step_fn
        self._history: dict = {"loss": [], "val_loss": []}
        self._checkpoints: List[CheckpointInfo] = []
        self._best_val_loss = float("inf")
        self._best_epoch = 0
        self._epochs_without_improvement = 0

    def _train_step(self, features: np.ndarray, targets: np.ndarray) -> float:
        """Run a single training step and return loss."""
        if self._train_step_fn is not None:
            return self._train_step_fn(features, targets)
        # Default: forward + compute_loss (no gradient update here; use adapter)
        if hasattr(self.model, "train_step"):
            return self.model.train_step(features, targets)
        y_pred = self.model.forward(features)
        if self.config.loss_fn is not None:
            return float(self.config.loss_fn(targets, y_pred))
        return self.model.compute_loss(targets, y_pred)

    def _validate(
        self,
        val_features: np.ndarray,
        val_targets: np.ndarray,
    ) -> float:
        """Compute validation loss."""
        y_pred = self.model.forward(val_features)
        if self.config.loss_fn is not None:
            return float(self.config.loss_fn(val_targets, y_pred))
        return self.model.compute_loss(val_targets, y_pred)

    def _save_checkpoint(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        is_best: bool,
    ) -> Optional[CheckpointInfo]:
        """Save checkpoint if configured."""
        if self.config.checkpoint_dir is None:
            return None
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        suffix = "best" if is_best else f"epoch_{epoch}"
        ckpt_path = ckpt_dir / f"checkpoint_{suffix}.json"
        params = self.model.get_parameters()
        # Serialise numpy arrays
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
            logger.info(f"Checkpoint saved: {ckpt_path} (epoch {epoch}, is_best={is_best})")
        return info

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
            Validation features (overrides validation_split).
        val_targets : ndarray, optional
            Validation targets.

        Returns
        -------
        TrainingResult
            Training history, checkpoints, and metadata.
        """
        start_time = time.time()
        # Split data if no validation provided and split > 0
        if val_features is None and self.config.validation_split > 0:
            features, targets, val_features, val_targets = _split_data(
                features, targets, self.config.validation_split
            )
        has_validation = val_features is not None and val_targets is not None

        for epoch in range(1, self.config.epochs + 1):
            # Training epoch
            epoch_losses = []
            for batch_x, batch_y in _batch_iterator(
                features, targets, self.config.batch_size
            ):
                loss = self._train_step(batch_x, batch_y)
                epoch_losses.append(loss)
            train_loss = float(np.mean(epoch_losses))
            self._history["loss"].append(train_loss)

            # Validation
            val_loss = None
            if has_validation:
                val_loss = self._validate(val_features, val_targets)
                self._history["val_loss"].append(val_loss)

            # Logging
            if self.config.verbose >= 1 and epoch % self.config.log_every == 0:
                msg = f"Epoch {epoch}/{self.config.epochs} — loss: {train_loss:.5f}"
                if val_loss is not None:
                    msg += f", val_loss: {val_loss:.5f}"
                logger.info(msg)

            # Checkpointing
            compare_loss = val_loss if has_validation else train_loss
            is_best = compare_loss < self._best_val_loss
            if is_best:
                self._best_val_loss = compare_loss
                self._best_epoch = epoch
                self._epochs_without_improvement = 0
            else:
                self._epochs_without_improvement += 1

            should_save = False
            if self.config.checkpoint_dir:
                if self.config.save_best_only and is_best:
                    should_save = True
                elif self.config.checkpoint_frequency > 0 and epoch % self.config.checkpoint_frequency == 0:
                    should_save = True
            if should_save:
                self._save_checkpoint(epoch, train_loss, val_loss, is_best)

            # Early stopping
            if (
                self.config.early_stopping_patience > 0
                and self._epochs_without_improvement >= self.config.early_stopping_patience
            ):
                if self.config.verbose >= 1:
                    logger.info(f"Early stopping at epoch {epoch} (no improvement for {self.config.early_stopping_patience} epochs)")
                break

        training_time = time.time() - start_time
        result = TrainingResult(
            history=self._history,
            final_epoch=epoch,
            best_epoch=self._best_epoch,
            best_train_loss=min(self._history["loss"]),
            best_val_loss=min(self._history["val_loss"]) if self._history["val_loss"] else None,
            checkpoints=self._checkpoints,
            config=self.config,
            training_time_seconds=training_time,
        )
        return result


def run_training(
    model: Trainable,
    features: np.ndarray,
    targets: np.ndarray,
    config: TrainingConfig,
    val_features: Optional[np.ndarray] = None,
    val_targets: Optional[np.ndarray] = None,
    train_step_fn: Optional[Callable[[Any, Any], float]] = None,
) -> TrainingResult:
    """
    Run training for a Trainable model.

    This is the main entry point for the generic training pipeline.

    Parameters
    ----------
    model : Trainable
        Model conforming to the Trainable protocol.
    features : ndarray
        Training features.
    targets : ndarray
        Training targets.
    config : TrainingConfig
        Training configuration.
    val_features : ndarray, optional
        Validation features.
    val_targets : ndarray, optional
        Validation targets.
    train_step_fn : callable, optional
        Custom training step function.

    Returns
    -------
    TrainingResult
        Training history, checkpoints, and metadata.

    Example
    -------
    >>> from src.machine_learning.core import TrainingConfig
    >>> from src.machine_learning.pipeline import run_training
    >>> config = TrainingConfig(epochs=50, learning_rate=0.001)
    >>> result = run_training(model, X_train, y_train, config)
    >>> print(result.best_train_loss)
    """
    loop = TrainingLoop(model, config, train_step_fn=train_step_fn)
    return loop.run(features, targets, val_features, val_targets)


__all__ = ["run_training", "TrainingLoop"]
