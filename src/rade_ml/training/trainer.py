"""
High-level Trainer class for TensorFlow model training.

This module wraps Keras model.fit() with:
    - Serialisable configuration management (TrainingConfig)
    - Callback orchestration via get_standard_callbacks
    - Mixed precision and XLA compilation support
    - Structured TrainingResult output

The Trainer accepts **only** tf.data.Dataset inputs -- callers are responsible for building their datasets
(use build_tf_dataset).

Usage:
    model = MyModel()
    config = TrainingConfig(epochs=100)

    trainer = Trainer(model, config)
    result = trainer.fit(train_ds, val_ds)

    print(result.history)
    print(f"Best epoch: {result.best_epoch}")
"""
from __future__ import annotations

import time
import logging

import numpy as np
import tensorflow as tf

from typing import Any, Dict, List, Optional

from src.rade_ml.core.config import TrainingConfig
from src.rade_ml.core.types import TrainingResult
from src.rade_ml.training.callbacks import get_standard_callbacks

# define module level logging.
logger = logging.getLogger(__name__)


class Trainer:
    """
    High-level trainer for TensorFlow models.

    Provides a clean interface for training with best practices:
        - Automatic callback configuration from TrainingConfig
        - Mixed precision support
        - XLA compilation
        - Structured TrainingResult output

    Example:
        config = TrainingConfig(
            epochs=100,
            optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
            early_stopping=EarlyStoppingConfig(patience=10),
        )
        trainer = Trainer(model, config)
        result = trainer.fit(train_ds, val_ds)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: TrainingConfig,
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ):
        """
        Initiate Trainer instance.

        :param model: TensorFlow/Keras model to train.
        :param config: training configuration.
        :param custom_callbacks: additional user-supplied callbacks.
        """
        self.model = model
        self.config = config
        self.custom_callbacks = custom_callbacks or []
        self._is_compiled = False

        # set up training environment.
        self._setup_environment()

    # ------------------------------------------------------------------
    # Environment setup
    # ------------------------------------------------------------------

    def _setup_environment(self) -> None:
        """Set up training environment (seeds, precision policy)."""
        if self.config.seed is not None:
            tf.random.set_seed(self.config.seed)
            np.random.seed(self.config.seed)

        if self.config.mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------

    def compile(
        self,
        loss: Optional[str] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        metrics: Optional[List[str]] = None,
    ) -> "Trainer":
        """
        Compile the model using Keras built-in loss / metric resolution.

        :param loss: loss function name (overrides config).
        :param optimizer: optimizer instance (overrides config).
        :param metrics: metric names (overrides config).
        :return: self for chaining.
        """
        # build optimizer from config if not provided.
        if optimizer is None:
            optimizer = self.config.optimizer.build()

        # resolve loss -- Keras handles string lookup natively.
        loss_fn = tf.keras.losses.get(loss or self.config.loss)

        # resolve metrics -- Keras handles string lookup natively.
        metric_names = metrics or self.config.metrics
        metric_fns = [tf.keras.metrics.get(m) for m in metric_names]

        # compile model.
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metric_fns,
            jit_compile=self.config.xla_compile,
        )

        self._is_compiled = True
        return self

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        train_data: tf.data.Dataset,
        val_data: Optional[tf.data.Dataset] = None,
        class_weight: Optional[Dict[int, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """
        Train the model.

        :param train_data: training data (already batched/shuffled via build_tf_dataset).
        :param val_data: validation data.
        :param class_weight: class weights for imbalanced data.
        :param sample_weight: per-sample weights.
        :return: TrainingResult with history, best epoch, timing and model summary.
        """
        # auto-compile if not already done.
        if not self._is_compiled:
            self.compile()

        # build callbacks from config + user-supplied.
        callbacks = get_standard_callbacks(self.config)
        callbacks.extend(self.custom_callbacks)

        # train.
        start_time = time.time()

        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=self.config.verbose,
            class_weight=class_weight,
            sample_weight=sample_weight,
        )

        total_time = time.time() - start_time

        # extract results from keras history.
        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        # determine best epoch from val_loss (or loss if no validation).
        val_losses = hist_dict.get("val_loss", hist_dict.get("loss", []))
        best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else 1
        best_val_loss = float(min(val_losses)) if val_losses else 0.0
        best_train_loss = float(hist_dict["loss"][best_epoch - 1]) if "loss" in hist_dict else 0.0

        # detect early stopping.
        final_epoch = len(hist_dict.get("loss", []))
        stopped_early = final_epoch < self.config.epochs

        return TrainingResult(
            history=hist_dict,
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            best_train_loss=best_train_loss,
            final_epoch=final_epoch,
            training_time_seconds=total_time,
            stopped_early=stopped_early,
            config=self.config.to_dict(),
            model_summary=self._get_model_summary(),
        )

    # ------------------------------------------------------------------
    # Evaluate / predict convenience
    # ------------------------------------------------------------------

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate model on test data.

        :param test_data: test dataset.
        :return: metric name -> value.
        """
        results = self.model.evaluate(test_data, verbose=0, return_dict=True)
        return {k: float(v) for k, v in results.items()}

    def predict(
        self,
        data: tf.data.Dataset,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate predictions.

        :param data: input data.
        :param batch_size: batch size for prediction.
        :return: predictions array.
        """
        return self.model.predict(data, verbose=0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_model_summary(self) -> Dict[str, Any]:
        """Get model summary as a serialisable dictionary."""
        try:
            return {
                "name": self.model.name,
                "trainable_params": int(sum(
                    tf.reduce_prod(w.shape) for w in self.model.trainable_weights
                )),
                "non_trainable_params": int(sum(
                    tf.reduce_prod(w.shape) for w in self.model.non_trainable_weights
                )),
                "layers": len(self.model.layers),
            }
        except Exception:
            return {}
