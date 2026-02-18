"""
Custom Keras callbacks for ML training.

This module provides domain-specific callbacks that extend Keras built-ins:
    - ``MetricsLogger``:  JSON training log (no Keras equivalent).
    - ``PricingErrorCallback``:  Pricing-specific error tracking in financial terms.
    - ``get_standard_callbacks``:  Factory that reads ``TrainingConfig`` and returns
      a list of standard Keras callbacks (EarlyStopping, ModelCheckpoint, TensorBoard)
      plus the custom ones above.

Keras built-in callbacks that are used directly (NOT reimplemented here):
    - ``tf.keras.callbacks.EarlyStopping``
    - ``tf.keras.callbacks.ModelCheckpoint``
    - ``tf.keras.callbacks.TensorBoard``  (histogram_freq=1 for gradient/weight monitoring)
    - ``verbose=1`` in ``model.fit()`` for progress bars / ETA

Usage:
    callbacks = get_standard_callbacks(config, val_data)
    model.fit(train_ds, callbacks=callbacks)
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf


class MetricsLogger(tf.keras.callbacks.Callback):
    """
    Callback to log training metrics to a JSON file.

    Creates a detailed training log with:
        - Per-epoch metrics (loss, validation loss, custom metrics)
        - Training timestamps
        - Best epoch tracking
        - Total training time

    Attributes
    ----------
    log_dir : Path
        Directory to save logs.
    log_file : str
        Name of the log file.
    """

    def __init__(self, log_dir: str = "./logs", log_file: str = "training_log.json"):
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.log_path = self.log_dir / log_file

        self.history: Dict[str, List[float]] = {}
        self.epoch_times: List[float] = []
        self.start_time: Optional[float] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0

    def on_train_begin(self, logs=None):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time = time.time()
        self.history = {}
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        epoch_time = time.time() - self._epoch_start
        self.epoch_times.append(epoch_time)

        # Update history
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))

        # Track best epoch
        val_loss = logs.get("val_loss", logs.get("loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time if self.start_time else 0

        log_data = {
            "training_completed": datetime.utcnow().isoformat(),
            "total_epochs": len(self.epoch_times),
            "total_time_seconds": total_time,
            "best_epoch": self.best_epoch,
            "best_val_loss": self.best_val_loss,
            "avg_epoch_time": (
                sum(self.epoch_times) / len(self.epoch_times) if self.epoch_times else 0
            ),
            "history": self.history,
            "epoch_times": self.epoch_times,
        }

        with open(self.log_path, "w") as f:
            json.dump(log_data, f, indent=2)


class PricingErrorCallback(tf.keras.callbacks.Callback):
    """
    Callback to track pricing-specific error metrics.

    Computes and logs (every ``log_every`` epochs):
        - Mean absolute pricing error (in original scale)
        - Mean percentage pricing error
        - Max pricing error
        - 95th-percentile error

    Useful for monitoring model performance in financial terms.

    Attributes
    ----------
    val_features : np.ndarray
        Validation features (original scale).
    val_prices : np.ndarray
        Validation target prices (original scale).
    price_scaler : sklearn scaler, optional
        Scaler to denormalise predictions back to original price scale.
        Must implement ``inverse_transform(array)``.
    log_every : int
        Log every N epochs.
    """

    def __init__(
        self,
        val_features: np.ndarray,
        val_prices: np.ndarray,
        price_scaler: Optional[Any] = None,
        log_every: int = 5,
    ):
        super().__init__()
        self.val_features = val_features
        self.val_prices = val_prices
        self.price_scaler = price_scaler
        self.log_every = log_every

        self.pricing_errors: Dict[str, List[float]] = {
            "mae": [],
            "mape": [],
            "max_error": [],
            "p95_error": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.log_every != 0:
            return

        # Get predictions
        preds = self.model.predict(self.val_features, verbose=0).flatten()

        # Denormalise using sklearn scaler if provided
        if self.price_scaler is not None:
            if hasattr(self.price_scaler, "inverse_transform"):
                preds = self.price_scaler.inverse_transform(
                    preds.reshape(-1, 1)
                ).flatten()

        # Compute errors
        errors = np.abs(self.val_prices - preds)
        mae = float(np.mean(errors))
        mape = float(np.mean(errors / (np.abs(self.val_prices) + 1e-8)) * 100)
        max_error = float(np.max(errors))
        p95_error = float(np.percentile(errors, 95))

        self.pricing_errors["mae"].append(mae)
        self.pricing_errors["mape"].append(mape)
        self.pricing_errors["max_error"].append(max_error)
        self.pricing_errors["p95_error"].append(p95_error)

        print(
            f"  Pricing: MAE=${mae:.2f}, MAPE={mape:.2f}%, "
            f"Max=${max_error:.2f}, P95=${p95_error:.2f}"
        )


def get_standard_callbacks(
    config: "TrainingConfig",
    val_data: Optional[Any] = None,
) -> List[tf.keras.callbacks.Callback]:
    """
    Build a standard callback list from ``TrainingConfig``.

    Returns standard Keras callbacks (EarlyStopping, ModelCheckpoint,
    TensorBoard) plus the custom ``MetricsLogger``.

    Parameters
    ----------
    config : TrainingConfig
        Training configuration.
    val_data : optional
        Validation data (unused, kept for API compatibility).

    Returns
    -------
    list of tf.keras.callbacks.Callback
    """
    from src.machine_learning.core.config import TrainingConfig

    callbacks: List[tf.keras.callbacks.Callback] = []

    # Early stopping (Keras built-in)
    if config.early_stopping is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            patience=config.early_stopping.patience,
            min_delta=config.early_stopping.min_delta,
            monitor=config.early_stopping.monitor,
            mode=config.early_stopping.mode,
            restore_best_weights=config.early_stopping.restore_best_weights,
            verbose=1,
        ))

    # Model checkpointing (Keras built-in)
    if config.checkpoint is not None:
        checkpoint_path = Path(config.checkpoint.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        callbacks.append(tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path / "model_{epoch:03d}.keras"),
            save_freq=config.checkpoint.save_freq,
            save_best_only=config.checkpoint.save_best_only,
            monitor=config.checkpoint.monitor,
            mode=config.checkpoint.mode,
            save_weights_only=config.checkpoint.save_weights_only,
            verbose=1,
        ))

    # TensorBoard (Keras built-in — histogram_freq=1 replaces GradientMonitorCallback)
    if config.log_dir is not None:
        callbacks.append(tf.keras.callbacks.TensorBoard(
            log_dir=config.log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq="epoch",
        ))

    # MetricsLogger (custom — JSON training log)
    if config.log_dir is not None:
        callbacks.append(MetricsLogger(log_dir=config.log_dir))

    return callbacks
