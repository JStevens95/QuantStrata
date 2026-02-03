"""
Trainable protocol for QuantStrata ML models.

Any model that conforms to this protocol can be trained via the generic
training pipeline (src.machine_learning.pipelines.training).

The protocol is framework-agnostic in definition; implementations may wrap
Keras, PyTorch, or pure NumPy/JAX models.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Trainable(Protocol):
    """
    Minimal interface for a trainable ML model.

    Required methods:
        forward(inputs) -> outputs
        compute_loss(y_true, y_pred) -> loss (scalar)
        get_parameters() -> parameter dict (for checkpointing)
        set_parameters(params) -> None (for loading)

    Optional:
        train_step(batch) -> loss (if model handles its own gradient update)
        validate_step(batch) -> metrics dict
    """

    def forward(self, inputs: Any) -> Any:
        """
        Forward pass: inputs -> predictions.

        Parameters
        ----------
        inputs : Any
            Model inputs (e.g. ndarray, dict of tensors).

        Returns
        -------
        Any
            Model outputs (e.g. predictions).
        """
        ...

    def compute_loss(self, y_true: Any, y_pred: Any) -> float:
        """
        Compute scalar loss given true and predicted values.

        Parameters
        ----------
        y_true : Any
            Ground truth targets.
        y_pred : Any
            Model predictions.

        Returns
        -------
        float
            Scalar loss value.
        """
        ...

    def get_parameters(self) -> Dict[str, Any]:
        """
        Return model parameters for checkpointing.

        Returns
        -------
        dict
            Parameter dict (e.g. weights, biases).
        """
        ...

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """
        Load model parameters from checkpoint.

        Parameters
        ----------
        params : dict
            Parameter dict as returned by get_parameters().
        """
        ...


class KerasTrainableAdapter:
    """
    Adapter to wrap a Keras model as a Trainable.

    This allows Keras models (e.g. HybridGnnRnn) to be used with the generic
    training pipeline. For full Keras training with callbacks, use TrainingManager.
    """

    def __init__(self, keras_model: Any, loss_fn: Any = None) -> None:
        """
        Parameters
        ----------
        keras_model : tf.keras.Model
            A compiled (or uncompiled) Keras model.
        loss_fn : callable, optional
            Loss function (y_true, y_pred) -> scalar. If None, uses model.loss.
        """
        self.model = keras_model
        self._loss_fn = loss_fn

    def forward(self, inputs: Any) -> Any:
        """Forward pass via model.predict or __call__."""
        return self.model(inputs, training=False)

    def compute_loss(self, y_true: Any, y_pred: Any) -> float:
        """Compute loss using the provided or compiled loss function."""
        import tensorflow as tf

        if self._loss_fn is not None:
            loss = self._loss_fn(y_true, y_pred)
        elif self.model.loss is not None:
            # In Keras 3.0, model.loss is a string; convert to callable
            if isinstance(self.model.loss, str):
                loss_fn = tf.keras.losses.get(self.model.loss)
                loss = loss_fn(y_true, y_pred)
            else:
                loss = self.model.loss(y_true, y_pred)
        else:
            loss = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
        return float(tf.reduce_mean(loss).numpy())

    def get_parameters(self) -> Dict[str, Any]:
        """Return model weights as a dict."""
        return {
            "weights": [w.numpy() for w in self.model.weights],
            "weight_names": [w.name for w in self.model.weights],
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Load model weights from a dict."""
        weights = params.get("weights", [])
        for w, val in zip(self.model.weights, weights):
            w.assign(val)

    def train_step(self, inputs: Any, targets: Any) -> float:
        """
        Run a single training step (forward + backward + optimizer step).

        Requires the model to be compiled with an optimizer.
        """
        import tensorflow as tf

        with tf.GradientTape() as tape:
            y_pred = self.model(inputs, training=True)
            # Compute loss as tensor for gradient tracking
            loss_tensor = self._compute_loss_tensor(targets, y_pred)
        grads = tape.gradient(loss_tensor, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return float(tf.reduce_mean(loss_tensor).numpy())

    def _compute_loss_tensor(self, y_true: Any, y_pred: Any) -> "tf.Tensor":
        """Compute loss as a tensor (for gradient tracking)."""
        import tensorflow as tf

        if self._loss_fn is not None:
            return self._loss_fn(y_true, y_pred)
        elif self.model.loss is not None:
            if isinstance(self.model.loss, str):
                loss_fn = tf.keras.losses.get(self.model.loss)
                return loss_fn(y_true, y_pred)
            else:
                return self.model.loss(y_true, y_pred)
        else:
            return tf.keras.losses.MeanSquaredError()(y_true, y_pred)


__all__ = ["Trainable", "KerasTrainableAdapter"]
