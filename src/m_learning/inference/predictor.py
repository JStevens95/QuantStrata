"""
Efficient model inference utilities.

This module provides optimized prediction classes for:
    - Single model inference
    - Batch processing
    - Streaming predictions
    - GPU memory management

Usage:
    predictor = Predictor(model)
    
    # Single batch prediction
    prices = predictor.predict(features)
    
    # Large dataset prediction with automatic batching
    prices = predictor.predict_large(large_features, batch_size=1024)
    
    # Prediction with automatic denormalization
    predictor.set_scalers(feature_scaler, target_scaler)
    prices = predictor.predict(raw_features, denormalize=True)
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.m_learning.data.dataset import NormalizationStats


class Predictor:
    """
    High-level predictor for TensorFlow models.
    
    Provides:
        - Efficient batch prediction
        - Automatic feature normalization
        - Automatic prediction denormalization
        - Memory-efficient large dataset processing
    
    Attributes:
        model: TensorFlow model
        feature_scaler: Optional feature normalization stats
        target_scaler: Optional target normalization stats
    
    Example:
        predictor = Predictor(model)
        predictor.set_scalers(train_dataset.feature_stats, train_dataset.target_stats)
        
        # Predict with automatic normalization/denormalization
        prices = predictor.predict(raw_features, normalize=True, denormalize=True)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        feature_scaler: Optional[NormalizationStats] = None,
        target_scaler: Optional[NormalizationStats] = None,
    ):
        """
        Initialize predictor.
        
        Args:
            model: TensorFlow/Keras model
            feature_scaler: Feature normalization statistics
            target_scaler: Target normalization statistics
        """
        self.model = model
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
    
    def set_scalers(
        self,
        feature_scaler: Optional[NormalizationStats] = None,
        target_scaler: Optional[NormalizationStats] = None,
    ) -> "Predictor":
        """
        Set normalization scalers.
        
        Args:
            feature_scaler: Feature normalization stats
            target_scaler: Target normalization stats
        
        Returns:
            Self for chaining
        """
        if feature_scaler is not None:
            self.feature_scaler = feature_scaler
        if target_scaler is not None:
            self.target_scaler = target_scaler
        return self
    
    def predict(
        self,
        features: np.ndarray,
        normalize: bool = True,
        denormalize: bool = True,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Generate predictions.
        
        Args:
            features: Input features
            normalize: Whether to normalize features (if scaler available)
            denormalize: Whether to denormalize predictions (if scaler available)
            batch_size: Batch size for prediction
        
        Returns:
            Predictions array
        """
        features = np.asarray(features, dtype=np.float32)
        
        # Normalize features
        if normalize and self.feature_scaler is not None:
            features = self.feature_scaler.normalize(features)
        
        # Predict
        predictions = self.model.predict(features, batch_size=batch_size, verbose=0)
        predictions = predictions.flatten()
        
        # Denormalize predictions
        if denormalize and self.target_scaler is not None:
            predictions = self.target_scaler.denormalize(predictions)
        
        return predictions
    
    def predict_large(
        self,
        features: np.ndarray,
        batch_size: int = 1024,
        normalize: bool = True,
        denormalize: bool = True,
        progress: bool = True,
    ) -> np.ndarray:
        """
        Predict on large dataset with memory-efficient batching.
        
        Args:
            features: Large feature array
            batch_size: Batch size for prediction
            normalize: Whether to normalize features
            denormalize: Whether to denormalize predictions
            progress: Whether to show progress
        
        Returns:
            Predictions array
        """
        features = np.asarray(features, dtype=np.float32)
        n_samples = len(features)
        
        # Normalize all features at once (more efficient)
        if normalize and self.feature_scaler is not None:
            features = self.feature_scaler.normalize(features)
        
        # Predict in batches
        predictions = []
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start = i * batch_size
            end = min(start + batch_size, n_samples)
            
            batch_preds = self.model.predict(features[start:end], verbose=0)
            predictions.append(batch_preds.flatten())
            
            if progress and (i + 1) % 10 == 0:
                print(f"  Batch {i+1}/{n_batches} ({end}/{n_samples} samples)")
        
        predictions = np.concatenate(predictions)
        
        # Denormalize
        if denormalize and self.target_scaler is not None:
            predictions = self.target_scaler.denormalize(predictions)
        
        return predictions
    
    def predict_with_uncertainty(
        self,
        features: np.ndarray,
        n_samples: int = 100,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimation using MC Dropout.
        
        Requires model to have dropout layers.
        
        Args:
            features: Input features
            n_samples: Number of forward passes for uncertainty estimation
            normalize: Whether to normalize features
            denormalize: Whether to denormalize predictions
        
        Returns:
            Tuple of (mean predictions, std predictions)
        """
        features = np.asarray(features, dtype=np.float32)
        
        if normalize and self.feature_scaler is not None:
            features = self.feature_scaler.normalize(features)
        
        # Multiple forward passes with dropout enabled
        predictions_list = []
        for _ in range(n_samples):
            preds = self.model(features, training=True)  # training=True enables dropout
            predictions_list.append(preds.numpy().flatten())
        
        predictions = np.array(predictions_list)
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # Denormalize
        if denormalize and self.target_scaler is not None:
            mean_pred = self.target_scaler.denormalize(mean_pred)
            std_pred = std_pred * self.target_scaler.std  # Scale std appropriately
        
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
        Predict and add results to DataFrame.
        
        Args:
            df: Input DataFrame
            feature_columns: Column names to use as features
            output_column: Name for prediction column
            normalize: Whether to normalize features
            denormalize: Whether to denormalize predictions
        
        Returns:
            DataFrame with predictions added
        """
        import pandas as pd
        
        features = df[feature_columns].values
        predictions = self.predict(features, normalize=normalize, denormalize=denormalize)
        
        result = df.copy()
        result[output_column] = predictions
        return result


class BatchPredictor:
    """
    Batch predictor for processing multiple models or configurations.
    
    Useful for:
        - Ensemble predictions
        - Model comparison
        - A/B testing
    
    Example:
        predictor = BatchPredictor()
        predictor.add_model("model_v1", model1)
        predictor.add_model("model_v2", model2)
        
        results = predictor.predict_all(features)
        # Returns: {"model_v1": predictions1, "model_v2": predictions2}
    """
    
    def __init__(self):
        """Initialize batch predictor."""
        self.models: Dict[str, Predictor] = {}
    
    def add_model(
        self,
        name: str,
        model: Union[tf.keras.Model, Predictor],
        feature_scaler: Optional[NormalizationStats] = None,
        target_scaler: Optional[NormalizationStats] = None,
    ) -> "BatchPredictor":
        """
        Add a model to the batch.
        
        Args:
            name: Model name/identifier
            model: Keras model or Predictor
            feature_scaler: Feature normalization stats
            target_scaler: Target normalization stats
        
        Returns:
            Self for chaining
        """
        if isinstance(model, Predictor):
            self.models[name] = model
        else:
            self.models[name] = Predictor(model, feature_scaler, target_scaler)
        return self
    
    def predict_all(
        self,
        features: np.ndarray,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate predictions from all models.
        
        Args:
            features: Input features
            normalize: Whether to normalize
            denormalize: Whether to denormalize
        
        Returns:
            Dict mapping model names to predictions
        """
        return {
            name: predictor.predict(features, normalize=normalize, denormalize=denormalize)
            for name, predictor in self.models.items()
        }
    
    def predict_ensemble(
        self,
        features: np.ndarray,
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> np.ndarray:
        """
        Generate weighted ensemble predictions.
        
        Args:
            features: Input features
            weights: Optional model weights (default: equal weights)
            normalize: Whether to normalize
            denormalize: Whether to denormalize
        
        Returns:
            Weighted average predictions
        """
        predictions = self.predict_all(features, normalize=normalize, denormalize=denormalize)
        
        if weights is None:
            weights = {name: 1.0 / len(self.models) for name in self.models}
        
        # Weighted average
        ensemble = np.zeros_like(list(predictions.values())[0])
        for name, preds in predictions.items():
            ensemble += weights.get(name, 0) * preds
        
        return ensemble
    
    def compare(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        normalize: bool = True,
        denormalize: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare all models on the same data.
        
        Args:
            features: Input features
            targets: Ground truth targets
            normalize: Whether to normalize
            denormalize: Whether to denormalize
        
        Returns:
            Dict mapping model names to metric dicts
        """
        from src.m_learning.evaluation.evaluator import compute_metrics
        
        predictions = self.predict_all(features, normalize=normalize, denormalize=denormalize)
        targets = np.asarray(targets).flatten()
        
        metrics = ["mse", "mae", "rmse", "r2"]
        
        return {
            name: compute_metrics(targets, preds, metrics)
            for name, preds in predictions.items()
        }


def create_serving_function(
    model: tf.keras.Model,
    feature_scaler: Optional[NormalizationStats] = None,
    target_scaler: Optional[NormalizationStats] = None,
) -> tf.function:
    """
    Create a TensorFlow serving function with preprocessing.
    
    Useful for TensorFlow Serving deployment.
    
    Args:
        model: Keras model
        feature_scaler: Feature normalization stats
        target_scaler: Target normalization stats
    
    Returns:
        tf.function for serving
    """
    # Convert scalers to TF constants
    if feature_scaler is not None:
        feat_mean = tf.constant(feature_scaler.mean, dtype=tf.float32)
        feat_std = tf.constant(feature_scaler.std, dtype=tf.float32)
    
    if target_scaler is not None:
        target_mean = tf.constant(target_scaler.mean, dtype=tf.float32)
        target_std = tf.constant(target_scaler.std, dtype=tf.float32)
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="features")
    ])
    def serve(features):
        # Normalize features
        if feature_scaler is not None:
            features = (features - feat_mean) / (feat_std + 1e-8)
        
        # Predict
        predictions = model(features, training=False)
        
        # Denormalize predictions
        if target_scaler is not None:
            predictions = predictions * target_std + target_mean
        
        return {"predictions": predictions}
    
    return serve
