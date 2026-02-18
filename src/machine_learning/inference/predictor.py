"""
Efficient model inference utilities.

This module provides:
    - ``Predictor``: Single-model inference (ndarray and dict features),
      MC Dropout uncertainty estimation, DataFrame prediction.
    - ``BatchPredictor``: Multi-model ensemble prediction and comparison.
    - ``create_serving_function``: TF Serving deployment helper.

Scaler convention:
    All scalers are sklearn objects (``StandardScaler``, ``MinMaxScaler``, etc.)
    that implement ``.transform()`` and ``.inverse_transform()``.

Usage:
    predictor = Predictor(model, feature_scaler=scaler_X, target_scaler=scaler_y)

    # Standard ndarray features
    prices = predictor.predict(features)

    # Dict features (GNN model)
    pnl = predictor.predict({"trade_features": tf, "adjacency_matrix": adj})

    # Pre-batched tf.data.Dataset
    pnl = predictor.predict_dataset(test_ds)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

# Type alias — sklearn scaler or any object with transform / inverse_transform
Scaler = Any

# Features can be arrays or dicts of arrays/tensors
Features = Union[np.ndarray, Dict[str, Any]]


class Predictor:
    """
    High-level predictor for TensorFlow models.

    Supports ndarray features (MLP models) and dict features (GNN / graph
    models), with optional normalisation via sklearn scalers.

    Attributes
    ----------
    model : tf.keras.Model
        TensorFlow model for inference.
    feature_scaler : sklearn scaler, optional
        Applied to ndarray features before prediction.
    target_scaler : sklearn scaler, optional
        Applied to predictions for denormalisation.
    """

    def __init__(
        self,
        model: tf.keras.Model,
        feature_scaler: Optional[Scaler] = None,
        target_scaler: Optional[Scaler] = None,
    ):
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

    def set_scalers(
        self,
        feature_scaler: Optional[Scaler] = None,
        target_scaler: Optional[Scaler] = None,
    ) -> "Predictor":
        """
        Set normalisation scalers.

        Returns
        -------
        Predictor
            Self for method chaining.
        """
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        if target_scaler is not None:
            self.target_scaler = target_scaler
        return self

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        features: Features,
        normalize: bool = True,
        denormalize: bool = True,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Generate predictions from ndarray or dict features.

        Dict features (GNN / graph models) bypass normalisation — those models
        manage their own feature preprocessing.

        Parameters
        ----------
        features : np.ndarray or dict
            Input features.
        normalize : bool
            Normalise ndarray features before prediction (requires scaler).
        denormalize : bool
            Denormalise predictions (requires target scaler).
        batch_size : int
            Batch size for ndarray prediction.

        Returns
        -------
        np.ndarray
            Predictions (flattened to 1-D).
        """
        if isinstance(features, dict):
            predictions = self._predict_dict(features)
        else:
            features = np.asarray(features, dtype=np.float32)

            if normalize and self.feature_scaler is not None:
                features = self.feature_scaler.transform(features)

            predictions = self.model.predict(
                features, batch_size=batch_size, verbose=0,
            ).flatten()

        if denormalize and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions

    def predict_dataset(
        self,
        dataset: tf.data.Dataset,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Generate predictions from a pre-batched ``tf.data.Dataset``.

        Parameters
        ----------
        dataset : tf.data.Dataset
            May yield ``(features, targets)`` tuples or feature-only batches.
        denormalize : bool
            Denormalise predictions using the target scaler.

        Returns
        -------
        np.ndarray
            Concatenated predictions (flattened).
        """
        all_preds: List[np.ndarray] = []

        for batch in dataset:
            if isinstance(batch, (tuple, list)):
                batch_features = batch[0]
            else:
                batch_features = batch

            batch_preds = self.model(batch_features, training=False)
            all_preds.append(tf.reshape(batch_preds, [-1]).numpy())

        predictions = np.concatenate(all_preds)

        if denormalize and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions

    def predict_with_uncertainty(
        self,
        features: Features,
        n_samples: int = 100,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using MC Dropout.

        Requires the model to have dropout layers.

        Parameters
        ----------
        features : np.ndarray or dict
            Input features.
        n_samples : int
            Number of stochastic forward passes.
        normalize : bool
            Normalise ndarray features.
        denormalize : bool
            Denormalise predictions.

        Returns
        -------
        mean_pred : np.ndarray
            Mean predictions.
        std_pred : np.ndarray
            Standard deviation of predictions (uncertainty).
        """
        if isinstance(features, dict):
            prepared = {
                k: tf.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v
                for k, v in features.items()
            }
        else:
            prepared = np.asarray(features, dtype=np.float32)
            if normalize and self.feature_scaler is not None:
                prepared = self.feature_scaler.transform(prepared)

        # Multiple forward passes with dropout enabled (training=True)
        predictions_list = []
        for _ in range(n_samples):
            preds = self.model(prepared, training=True)
            predictions_list.append(tf.reshape(preds, [-1]).numpy())

        predictions = np.array(predictions_list)
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)

        if denormalize and self.target_scaler is not None:
            mean_pred = self.target_scaler.inverse_transform(
                mean_pred.reshape(-1, 1)
            ).flatten()
            # Scale std by the scaler's standard deviation
            if hasattr(self.target_scaler, "scale_"):
                std_pred = std_pred * self.target_scaler.scale_.flatten()[0]

        return mean_pred, std_pred

    def predict_dataframe(
        self,
        df: "pd.DataFrame",
        feature_columns: List[str],
        output_column: str = "prediction",
        normalize: bool = True,
        denormalize: bool = True,
    ) -> "pd.DataFrame":
        """
        Predict and add results to a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        feature_columns : list of str
            Column names to use as features.
        output_column : str
            Name for the prediction column.
        normalize : bool
            Normalise features.
        denormalize : bool
            Denormalise predictions.

        Returns
        -------
        pd.DataFrame
            DataFrame with predictions added.
        """
        import pandas as pd

        features = df[feature_columns].values
        predictions = self.predict(features, normalize=normalize, denormalize=denormalize)

        result = df.copy()
        result[output_column] = predictions
        return result

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _predict_dict(self, features: Dict[str, Any]) -> np.ndarray:
        """Run inference with dict features (GNN / graph models)."""
        tensor_features = {
            k: tf.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v
            for k, v in features.items()
        }
        output = self.model(tensor_features, training=False)
        return tf.reshape(output, [-1]).numpy()


class BatchPredictor:
    """
    Batch predictor for multiple models — ensemble predictions and comparison.

    Example
    -------
    >>> bp = BatchPredictor()
    >>> bp.add_model("v1", model1, target_scaler=scaler1)
    >>> bp.add_model("v2", model2, target_scaler=scaler2)
    >>> results = bp.predict_all(features)
    >>> ensemble = bp.predict_ensemble(features, weights={"v1": 0.6, "v2": 0.4})
    """

    def __init__(self):
        self.models: Dict[str, Predictor] = {}

    def add_model(
        self,
        name: str,
        model: Union[tf.keras.Model, Predictor],
        feature_scaler: Optional[Scaler] = None,
        target_scaler: Optional[Scaler] = None,
    ) -> "BatchPredictor":
        """
        Register a model.

        Parameters
        ----------
        name : str
            Model identifier.
        model : tf.keras.Model or Predictor
            Keras model or wrapped Predictor.
        feature_scaler, target_scaler : sklearn scaler, optional

        Returns
        -------
        BatchPredictor
            Self for chaining.
        """
        if isinstance(model, Predictor):
            self.models[name] = model
        else:
            self.models[name] = Predictor(model, feature_scaler, target_scaler)
        return self

    def predict_all(
        self,
        features: Features,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all registered models.

        Returns
        -------
        dict
            Model name -> predictions array.
        """
        return {
            name: predictor.predict(features, normalize=normalize, denormalize=denormalize)
            for name, predictor in self.models.items()
        }

    def predict_ensemble(
        self,
        features: Features,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Generate weighted ensemble predictions.

        Parameters
        ----------
        features : Features
        weights : dict, optional
            Model weights (default: equal).

        Returns
        -------
        np.ndarray
            Weighted average predictions.
        """
        predictions = self.predict_all(features, normalize=normalize, denormalize=denormalize)

        if weights is None:
            weights = {name: 1.0 / len(self.models) for name in self.models}

        ensemble = np.zeros_like(list(predictions.values())[0])
        for name, preds in predictions.items():
            ensemble += weights.get(name, 0) * preds

        return ensemble

    def compare(
        self,
        features: Features,
        targets: np.ndarray,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all models on the same data.

        Returns
        -------
        dict
            Model name -> metrics dict.
        """
        from src.machine_learning.evaluation.evaluator import compute_metrics

        predictions = self.predict_all(features, normalize=normalize, denormalize=denormalize)
        targets = np.asarray(targets).flatten()

        metric_names = ["mse", "mae", "rmse", "r2"]
        return {
            name: compute_metrics(targets, preds, metric_names)
            for name, preds in predictions.items()
        }


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------

def create_serving_function(
    model: tf.keras.Model,
    feature_scaler: Optional[Scaler] = None,
    target_scaler: Optional[Scaler] = None,
) -> tf.types.experimental.ConcreteFunction:
    """
    Create a TensorFlow serving function with pre/post-processing baked in.

    Useful for TensorFlow Serving deployment.

    Parameters
    ----------
    model : tf.keras.Model
    feature_scaler : sklearn scaler, optional
        Feature normalisation stats (mean/scale baked as TF constants).
    target_scaler : sklearn scaler, optional
        Target denormalisation stats (mean/scale baked as TF constants).

    Returns
    -------
    tf.function
    """
    # Bake scaler stats as TF constants for graph execution
    if feature_scaler is not None and hasattr(feature_scaler, "mean_"):
        feat_mean = tf.constant(feature_scaler.mean_, dtype=tf.float32)
        feat_scale = tf.constant(feature_scaler.scale_, dtype=tf.float32)
    else:
        feat_mean = None
        feat_scale = None

    if target_scaler is not None and hasattr(target_scaler, "mean_"):
        target_mean = tf.constant(target_scaler.mean_, dtype=tf.float32)
        target_scale = tf.constant(target_scaler.scale_, dtype=tf.float32)
    else:
        target_mean = None
        target_scale = None

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="features")
    ])
    def serve(features):
        # Normalise features
        if feat_mean is not None:
            features = (features - feat_mean) / (feat_scale + 1e-8)

        predictions = model(features, training=False)

        # Denormalise predictions
        if target_mean is not None:
            predictions = predictions * target_scale + target_mean

        return {"predictions": predictions}

    return serve
