"""
Model I/O utilities for TensorFlow models.

This module provides comprehensive model serialization using:
    - TensorFlow SavedModel format (primary)
    - Keras native format (.keras)
    - Weights-only format (for architecture reuse)

Artifact Structure:
    model_dir/
    ├── saved_model/           # TensorFlow SavedModel
    │   ├── saved_model.pb
    │   ├── variables/
    │   └── assets/
    ├── model.keras            # Keras native format (optional)
    ├── config.json            # Training configuration
    ├── metadata.json          # Model metadata
    ├── normalization.json     # Feature/target normalization stats
    └── training_history.json  # Training history (optional)

Usage:
    # Save model with all artifacts
    save_model(
        model=trained_model,
        path="models/my_pricer",
        config=training_config,
        metadata={"version": "1.0", "description": "FX vanilla pricer"},
        normalization_stats=dataset.feature_stats,
    )
    
    # Load model
    loaded = load_model("models/my_pricer")
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow as tf

from src.m_learning.data.dataset import NormalizationStats


@dataclass
class ModelArtifact:
    """
    Container for loaded model artifacts.
    
    Attributes:
        model: Loaded TensorFlow model
        config: Training configuration (if saved)
        metadata: Model metadata
        feature_stats: Feature normalization statistics
        target_stats: Target normalization statistics
        training_history: Training history (if saved)
    """
    model: tf.keras.Model
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    feature_stats: Optional[NormalizationStats] = None
    target_stats: Optional[NormalizationStats] = None
    training_history: Optional[Dict[str, Any]] = None
    
    def predict(self, features: np.ndarray, denormalize: bool = True) -> np.ndarray:
        """
        Generate predictions with optional denormalization.
        
        Args:
            features: Input features (raw or normalized depending on feature_stats)
            denormalize: Whether to denormalize predictions
        
        Returns:
            Predictions (denormalized if requested and stats available)
        """
        # Normalize features if stats available
        if self.feature_stats is not None:
            features = self.feature_stats.normalize(features)
        
        # Predict
        predictions = self.model.predict(features, verbose=0).flatten()
        
        # Denormalize predictions if requested
        if denormalize and self.target_stats is not None:
            predictions = self.target_stats.denormalize(predictions)
        
        return predictions


def save_model(
    model: tf.keras.Model,
    path: Union[str, Path],
    config: Optional["TrainingConfig"] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feature_stats: Optional[NormalizationStats] = None,
    target_stats: Optional[NormalizationStats] = None,
    training_history: Optional[Dict[str, Any]] = None,
    save_format: str = "tf",
    include_optimizer: bool = False,
) -> Path:
    """
    Save a trained model with all artifacts.
    
    Creates a complete model artifact including:
        - Model weights and architecture
        - Training configuration
        - Normalization statistics
        - Metadata
    
    Args:
        model: Trained Keras model
        path: Directory to save artifacts
        config: Training configuration
        metadata: Additional metadata
        feature_stats: Feature normalization statistics
        target_stats: Target normalization statistics
        training_history: Training history dict
        save_format: 'tf' (SavedModel) or 'keras' (native Keras)
        include_optimizer: Whether to save optimizer state
    
    Returns:
        Path to saved model directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model
    if save_format == "tf":
        # TensorFlow SavedModel format
        model_path = path / "saved_model"
        model.save(str(model_path), save_format="tf", include_optimizer=include_optimizer)
    else:
        # Keras native format
        model_path = path / "model.keras"
        model.save(str(model_path), save_format="keras", include_optimizer=include_optimizer)
    
    # 2. Save configuration
    if config is not None:
        config_path = path / "config.json"
        if hasattr(config, "to_dict"):
            config_data = config.to_dict()
        else:
            config_data = config
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)
    
    # 3. Save metadata
    metadata = metadata or {}
    metadata.update({
        "saved_at": datetime.utcnow().isoformat(),
        "tensorflow_version": tf.__version__,
        "model_name": model.name,
        "save_format": save_format,
    })
    
    # Add model summary if available
    try:
        metadata["trainable_params"] = int(sum(
            tf.reduce_prod(w.shape) for w in model.trainable_weights
        ))
        metadata["total_params"] = int(sum(
            tf.reduce_prod(w.shape) for w in model.weights
        ))
    except Exception:
        pass
    
    with open(path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # 4. Save normalization statistics
    norm_data = {}
    if feature_stats is not None:
        norm_data["feature_stats"] = feature_stats.to_dict()
    if target_stats is not None:
        norm_data["target_stats"] = target_stats.to_dict()
    
    if norm_data:
        with open(path / "normalization.json", "w") as f:
            json.dump(norm_data, f, indent=2)
    
    # 5. Save training history
    if training_history is not None:
        with open(path / "training_history.json", "w") as f:
            json.dump(training_history, f, indent=2)
    
    print(f"Model saved to: {path}")
    return path


def load_model(
    path: Union[str, Path],
    custom_objects: Optional[Dict[str, Any]] = None,
    compile_model: bool = False,
) -> ModelArtifact:
    """
    Load a saved model with all artifacts.
    
    Args:
        path: Path to model directory
        custom_objects: Custom Keras objects (layers, losses, etc.)
        compile_model: Whether to compile the loaded model
    
    Returns:
        ModelArtifact with model and associated data
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")
    
    # 1. Load model
    saved_model_path = path / "saved_model"
    keras_path = path / "model.keras"
    
    if saved_model_path.exists():
        model = tf.keras.models.load_model(
            str(saved_model_path),
            custom_objects=custom_objects,
            compile=compile_model,
        )
    elif keras_path.exists():
        model = tf.keras.models.load_model(
            str(keras_path),
            custom_objects=custom_objects,
            compile=compile_model,
        )
    else:
        raise FileNotFoundError(f"No model found in {path}")
    
    # 2. Load configuration
    config = None
    config_path = path / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)
    
    # 3. Load metadata
    metadata = None
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    
    # 4. Load normalization statistics
    feature_stats = None
    target_stats = None
    norm_path = path / "normalization.json"
    if norm_path.exists():
        with open(norm_path, "r") as f:
            norm_data = json.load(f)
        if "feature_stats" in norm_data:
            feature_stats = NormalizationStats.from_dict(norm_data["feature_stats"])
        if "target_stats" in norm_data:
            target_stats = NormalizationStats.from_dict(norm_data["target_stats"])
    
    # 5. Load training history
    training_history = None
    history_path = path / "training_history.json"
    if history_path.exists():
        with open(history_path, "r") as f:
            training_history = json.load(f)
    
    print(f"Model loaded from: {path}")
    
    return ModelArtifact(
        model=model,
        config=config,
        metadata=metadata,
        feature_stats=feature_stats,
        target_stats=target_stats,
        training_history=training_history,
    )


def export_saved_model(
    model: tf.keras.Model,
    path: Union[str, Path],
    signatures: Optional[Dict[str, tf.function]] = None,
) -> Path:
    """
    Export model as TensorFlow SavedModel for serving.
    
    This creates a production-ready SavedModel with optional
    custom signatures for TensorFlow Serving.
    
    Args:
        model: Keras model to export
        path: Export path
        signatures: Optional custom serving signatures
    
    Returns:
        Path to exported model
    """
    path = Path(path)
    
    if signatures is None:
        # Create default serving signature
        @tf.function(input_signature=[
            tf.TensorSpec(shape=[None, None], dtype=tf.float32, name="features")
        ])
        def serve(features):
            return {"predictions": model(features, training=False)}
        
        signatures = {"serving_default": serve}
    
    tf.saved_model.save(model, str(path), signatures=signatures)
    print(f"SavedModel exported to: {path}")
    
    return path


def list_model_artifacts(path: Union[str, Path]) -> Dict[str, bool]:
    """
    List available artifacts in a model directory.
    
    Args:
        path: Model directory path
    
    Returns:
        Dict mapping artifact names to existence status
    """
    path = Path(path)
    
    return {
        "saved_model": (path / "saved_model").exists(),
        "model.keras": (path / "model.keras").exists(),
        "config.json": (path / "config.json").exists(),
        "metadata.json": (path / "metadata.json").exists(),
        "normalization.json": (path / "normalization.json").exists(),
        "training_history.json": (path / "training_history.json").exists(),
    }


def copy_model(
    source: Union[str, Path],
    destination: Union[str, Path],
) -> Path:
    """
    Copy a model artifact directory.
    
    Args:
        source: Source model path
        destination: Destination path
    
    Returns:
        Destination path
    """
    source = Path(source)
    destination = Path(destination)
    
    if destination.exists():
        shutil.rmtree(destination)
    
    shutil.copytree(source, destination)
    print(f"Model copied to: {destination}")
    
    return destination
