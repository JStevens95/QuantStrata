"""
High-level Trainer class for TensorFlow model training.

This module wraps Keras model.fit() with:
    - Serialisable configuration management (TrainingConfig)
    - Callback orchestration via get_standard_callbacks
    - Mixed precision and XLA compilation support
    - Reproducibility (seed setting)
    - Structured TrainingResult output

The Trainer accepts **only** tf.data.Dataset inputs -- callers are
responsible for building their datasets (use build_tf_dataset).

Usage:
    model = MyModel()
    config = TrainingConfig(epochs=100, optimizer=OptimizerConfig(name="adam", learning_rate=1e-3))

    trainer = Trainer(model, config)
    result = trainer.fit(train_ds, val_ds)

    print(result.history)
    print(f"Best epoch: {result.best_epoch}")
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np
import tensorflow as tf

from src.rade_ml.core.config import TrainingConfig
from src.rade_ml.training.callbacks import get_standard_callbacks
from src.rade_ml.core.types import TrainingResult


class Trainer:
    """
    High-level trainer for TensorFlow models.

    Provides a clean interface for training with best practices:
        - Automatic callback configuration from TrainingConfig
        - Mixed precision support
        - XLA compilation
        - Reproducibility (seed setting)

    Parameters
    ----------
    model : tf.keras.Model
        TensorFlow/Keras model to train.
    config : TrainingConfig
        Training configuration.
    custom_callbacks : list of tf.keras.callbacks.Callback, optional
        Additional user-supplied callbacks appended after standard ones.

    Example
    -------
    >>> config = TrainingConfig(
    ...     epochs=100,
    ...     optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
    ...     early_stopping=EarlyStoppingConfig(patience=10),
    ... )
    >>> trainer = Trainer(model, config)
    >>> result = trainer.fit(train_ds, val_ds)
    """

    def __init__(
        self,
        model: tf.keras.Model,
        config: TrainingConfig,
        custom_callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    ):
        self.model = model
        self.config = config
        self.custom_callbacks = custom_callbacks or []
        self._is_compiled = False
        self._setup_environment()

    def _setup_environment(self) -> None:
        """Set up training environment (seeds, precision)."""
        if self.config.seed is not None:
            tf.random.set_seed(self.config.seed)
            np.random.seed(self.config.seed)

        if self.config.mixed_precision:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)

    def compile(
        self,
        loss: Optional[str] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        metrics: Optional[List[str]] = None,
    ) -> "Trainer":
        """
        Compile the model using Keras built-in loss / metric resolution.

        Parameters
        ----------
        loss : str, optional
            Loss function name (overrides config).
        optimizer : tf.keras.optimizers.Optimizer, optional
            Optimizer instance (overrides config).
        metrics : list of str, optional
            Metric names (overrides config).

        Returns
        -------
        Trainer
            Self for chaining.
        """
        if optimizer is None:
            optimizer = self.config.optimizer.build()

        loss_fn = tf.keras.losses.get(loss or self.config.loss)

        metric_names = metrics or self.config.metrics
        metric_fns = [tf.keras.metrics.get(m) for m in metric_names]

        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metric_fns,
            jit_compile=self.config.xla_compile,
        )

        self._is_compiled = True
        return self

    def fit(
        self,
        train_data: tf.data.Dataset,
        val_data: Optional[tf.data.Dataset] = None,
        class_weight: Optional[Dict[int, float]] = None,
        sample_weight: Optional[np.ndarray] = None,
    ) -> TrainingResult:
        """
        Train the model.

        Parameters
        ----------
        train_data : tf.data.Dataset
            Training data (already batched/shuffled via build_tf_dataset).
        val_data : tf.data.Dataset, optional
            Validation data.
        class_weight : dict, optional
            Class weights for imbalanced data.
        sample_weight : np.ndarray, optional
            Per-sample weights.

        Returns
        -------
        TrainingResult
            Training history, best epoch, timing, and model summary.
        """
        if not self._is_compiled:
            self.compile()

        callbacks = get_standard_callbacks(self.config)
        callbacks.extend(self.custom_callbacks)

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

        hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}

        val_losses = hist_dict.get("val_loss", hist_dict.get("loss", []))
        best_epoch = int(np.argmin(val_losses)) + 1 if val_losses else 1
        best_val_loss = float(min(val_losses)) if val_losses else 0.0
        best_train_loss = float(hist_dict["loss"][best_epoch - 1]) if "loss" in hist_dict else 0.0

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

    def evaluate(self, test_data: tf.data.Dataset) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Parameters
        ----------
        test_data : tf.data.Dataset
            Test dataset.

        Returns
        -------
        dict
            Metric name -> value.
        """
        results = self.model.evaluate(test_data, verbose=0, return_dict=True)
        return {k: float(v) for k, v in results.items()}

    def predict(self, data: tf.data.Dataset) -> np.ndarray:
        """
        Generate predictions.

        Parameters
        ----------
        data : tf.data.Dataset
            Input data.

        Returns
        -------
        np.ndarray
            Predictions array.
        """
        return self.model.predict(data, verbose=0)

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
