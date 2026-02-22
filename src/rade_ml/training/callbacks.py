"""
Custom Keras callbacks for ML training.

This module provides specialised callbacks that extend Keras built-ins:
    - MetricsLogger: JSON training log.
    - get_standard_callbacks: Factory that reads TrainingConfig and returns
      a list of standard Keras callbacks (EarlyStopping, ModelCheckpoint,
      ReduceLROnPlateau, TensorBoard) plus MetricsLogger.

Keras built-in callbacks that are used directly (NOT reimplemented here):
    - tf.keras.callbacks.EarlyStopping
    - tf.keras.callbacks.ModelCheckpoint
    - tf.keras.callbacks.ReduceLROnPlateau
    - tf.keras.callbacks.TensorBoard (histogram_freq=1 for gradient/weight monitoring)
"""
from __future__ import annotations

import json
import time
import logging

import tensorflow as tf

from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.rade_ml.core.config import TrainingConfig

# define module level logging.
logger = logging.getLogger(__name__)


class MetricsLogger(tf.keras.callbacks.Callback):
    """
    Callback to log training metrics to JSON file.

    Create a detailed training log with:
        - per-epoch metrics (loss, validation loss, custom metrics)
        - best epoch tracking
        - total training time.
    """

    def __init__(self, log_dir: str = "./logs", log_file: str = "training_log.json") -> None:
        """
        Initiate MetricsLogger.

        :param log_dir:
        :param log_file:
        """
        super().__init__()
        self.log_dir = Path(log_dir)
        self.log_file = log_file
        self.log_path = self.log_dir / self.log_file

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

        # update history
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(float(value))

        # track best epoch
        val_loss = logs.get("val_loss", logs.get("loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1

    def on_train_end(self, logs=None):
        total_time = time.time() - self.start_time if self.start_time else 0

        # log data
        log_data = {
            "training_completed": datetime.now().isoformat(),
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
        logger.info(f"Training log saved to {self.log_path}")


def get_standard_callbacks(config: "TrainingConfig") -> List[tf.keras.callbacks.Callback]:
    """
    Build a standard callbacks list from "TrainingConfig" object.

    :param config: training config
    :return:
    """
    # define callbacks output.
    callbacks: List[tf.keras.callbacks.Callback] = []

    # early stopping.
    if config.early_stopping is not None:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                patience=config.early_stopping.patience,
                min_delta=config.early_stopping.min_delta,
                monitor=config.early_stopping.monitor,
                mode=config.early_stopping.mode,
                restore_best_weights=config.early_stopping.restore_best_weights,
                verbose=1,
            )
        )

    # model checkpointing.
    if config.checkpoint is not None:
        checkpoint_path = Path(config.checkpoint.checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(checkpoint_path / "model_{epoch:03d}.keras"),
                save_freq=config.checkpoint.save_freq,
                save_best_only=config.checkpoint.save_best_only,
                monitor=config.checkpoint.monitor,
                mode=config.checkpoint.mode,
                save_weights_only=config.checkpoint.save_weights_only,
                verbose=1,
            )
        )

    # reduce learning rate on plateau.
    if config.lr_reduction is not None:
        callbacks.append(
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=config.lr_reduction.monitor,
                factor=config.lr_reduction.factor,
                patience=config.lr_reduction.patience,
                mode=config.lr_reduction.mode,
                min_lr=config.lr_reduction.min_lr,
            )
        )

    # tensorboard (histogram_freq=1 replaces manual gradient monitoring).
    if config.log_dir is not None:
        tb_dir = str(Path(config.log_dir) / "tensorboard")
        callbacks.append(
            tf.keras.callbacks.TensorBoard(
                log_dir=tb_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq="epoch",
            )
        )

    # MetricsLogger (JSON training log).
    if config.log_dir is not None:
        callbacks.append(MetricsLogger(log_dir=config.log_dir))

    return callbacks
