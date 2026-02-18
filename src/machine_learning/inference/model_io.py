"""
Model I/O utilities for TensorFlow models.

This module provides comprehensive model serialization using:
    - TensorFlow SavedModel format (primary)
    - Keras native format (.keras)
    - Weights-only format (for architecture reuse)

Scaler convention:
    Normalization is handled by sklearn scalers (``StandardScaler``,
    ``MinMaxScaler``).  Scalers are persisted via ``joblib.dump`` /
    ``joblib.load`` alongside the model artifacts.

Artifact structure:
    model_dir/
    ├── saved_model/           # TensorFlow SavedModel
    │   ├── saved_model.pb
    │   ├── variables/
    │   └── assets/
    ├── model.keras            # Keras native format (optional)
    ├── config.json            # Training configuration
    ├── metadata.json          # Model metadata
    ├── feature_scaler.joblib  # sklearn feature scaler
    ├── target_scaler.joblib   # sklearn target scaler
    └── training_history.json  # Training history (optional)

Usage:
    save_model(
        model=trained_model,
        path="models/my_pricer",
        config=training_config,
        feature_scaler=scaler_X,
        target_scaler=scaler_y,
    )

    artifact = load_model("models/my_pricer")
    predictions = artifact.predict(features)
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

if TYPE_CHECKING:
    from src.machine_learning.core.config import TrainingConfig

import numpy as np
import tensorflow as tf

# Scaler type — any sklearn scaler with transform / inverse_transform
Scaler = Any


def _ensure_custom_models_registered() -> None:
    """Import modules that define custom Keras models so @register_keras_serializable runs."""
    try:
        import src.machine_learning.core.base  # noqa: F401
        import src.machine_learning.models.pricing.model  # noqa: F401
    except ImportError:
        pass


def _save_keras_model(model: tf.keras.Model, path: Path, include_optimizer: bool) -> None:
    """Save Keras model to .keras path (Keras 3 compatible: no deprecated save_format)."""
    path = Path(path)
    kwargs = {}
    if include_optimizer:
        kwargs["include_optimizer"] = True
    model.save(str(path), **kwargs)


def _load_keras_model(path: Path, custom_objects: Optional[Dict], compile_model: bool) -> tf.keras.Model:
    """Load Keras model from .keras file or SavedModel directory."""
    _ensure_custom_models_registered()
    return tf.keras.models.load_model(
        str(path),
        custom_objects=custom_objects or {},
        compile=compile_model,
    )


def _save_scaler(scaler: Any, path: Path) -> None:
    """Save an sklearn scaler via joblib."""
    import joblib
    joblib.dump(scaler, str(path))


def _load_scaler(path: Path) -> Any:
    """Load an sklearn scaler via joblib."""
    import joblib
    return joblib.load(str(path))


@dataclass
class ModelArtifact:
    """
    Container for loaded model artifacts.

    Attributes
    ----------
    model : tf.keras.Model
        Loaded TensorFlow model.
    config : dict, optional
        Training configuration.
    metadata : dict, optional
        Model metadata.
    feature_scaler : sklearn scaler, optional
        Fitted feature scaler (transform / inverse_transform).
    target_scaler : sklearn scaler, optional
        Fitted target scaler (for denormalising predictions).
    training_history : dict, optional
        Training history.
    """

    model: tf.keras.Model
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    feature_scaler: Optional[Scaler] = None
    target_scaler: Optional[Scaler] = None
    training_history: Optional[Dict[str, Any]] = None

    def predict(
        self,
        features: np.ndarray,
        normalize: bool = True,
        denormalize: bool = True,
        batch_size: int = 256,
    ) -> np.ndarray:
        """
        Generate predictions with optional normalisation / denormalisation.

        Parameters
        ----------
        features : np.ndarray
            Raw input features.
        normalize : bool
            Whether to normalise features using the saved scaler.
        denormalize : bool
            Whether to denormalise predictions using the saved scaler.
        batch_size : int
            Batch size for prediction.

        Returns
        -------
        np.ndarray
            Predictions (denormalised if requested and scaler available).
        """
        if normalize and self.feature_scaler is not None:
            features = self.feature_scaler.transform(features)

        predictions = self.model.predict(features, batch_size=batch_size, verbose=0).flatten()

        if denormalize and self.target_scaler is not None:
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()

        return predictions


def save_model(
    model: tf.keras.Model,
    path: Union[str, Path],
    config: Optional["TrainingConfig"] = None,
    metadata: Optional[Dict[str, Any]] = None,
    feature_scaler: Optional[Scaler] = None,
    target_scaler: Optional[Scaler] = None,
    training_history: Optional[Dict[str, Any]] = None,
    save_format: str = "keras",
    include_optimizer: bool = False,
) -> Path:
    """
    Save a trained model with all artifacts.

    Creates a complete model artifact including:
        - Model weights and architecture
        - Training configuration
        - sklearn scalers (via joblib)
        - Metadata

    Parameters
    ----------
    model : tf.keras.Model
        Trained Keras model.
    path : str or Path
        Directory to save artifacts.
    config : TrainingConfig, optional
        Training configuration.
    metadata : dict, optional
        Additional metadata.
    feature_scaler : sklearn scaler, optional
        Fitted feature scaler.
    target_scaler : sklearn scaler, optional
        Fitted target scaler.
    training_history : dict, optional
        Training history dict.
    save_format : str
        ``'keras'`` (default, .keras file) or ``'tf'`` (SavedModel).
    include_optimizer : bool
        Whether to save optimizer state.

    Returns
    -------
    Path
        Path to saved model directory.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # 1. Save model
    if save_format == "tf":
        model_dir = path / "saved_model"
        tf.saved_model.save(model, str(model_dir))
    else:
        model_path = path / "model.keras"
        _save_keras_model(model, model_path, include_optimizer)

    # 2. Save configuration
    if config is not None:
        config_path = path / "config.json"
        config_data = config.to_dict() if hasattr(config, "to_dict") else config
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

    # 4. Save sklearn scalers via joblib
    if feature_scaler is not None:
        _save_scaler(feature_scaler, path / "feature_scaler.joblib")
    if target_scaler is not None:
        _save_scaler(target_scaler, path / "target_scaler.joblib")

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

    Parameters
    ----------
    path : str or Path
        Path to model directory.
    custom_objects : dict, optional
        Custom Keras objects (layers, losses, etc.).
    compile_model : bool
        Whether to compile the loaded model.

    Returns
    -------
    ModelArtifact
        Model and associated data.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    # 1. Load model (.keras preferred for Keras 3 compatibility)
    keras_path = path / "model.keras"
    saved_model_path = path / "saved_model"

    if keras_path.exists():
        model = _load_keras_model(keras_path, custom_objects, compile_model)
    elif saved_model_path.exists():
        model = _load_keras_model(saved_model_path, custom_objects, compile_model)
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

    # 4. Load sklearn scalers
    feature_scaler = None
    target_scaler = None

    feat_scaler_path = path / "feature_scaler.joblib"
    tgt_scaler_path = path / "target_scaler.joblib"

    if feat_scaler_path.exists():
        feature_scaler = _load_scaler(feat_scaler_path)
    if tgt_scaler_path.exists():
        target_scaler = _load_scaler(tgt_scaler_path)

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
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        training_history=training_history,
    )


def export_saved_model(
    model: tf.keras.Model,
    path: Union[str, Path],
    signatures: Optional[Dict[str, tf.function]] = None,
) -> Path:
    """
    Export model as TensorFlow SavedModel for serving.

    Parameters
    ----------
    model : tf.keras.Model
        Keras model to export.
    path : str or Path
        Export path.
    signatures : dict, optional
        Custom serving signatures.

    Returns
    -------
    Path
        Path to exported model.
    """
    path = Path(path)

    if signatures is None:
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

    Parameters
    ----------
    path : str or Path
        Model directory path.

    Returns
    -------
    dict
        Artifact names -> existence status.
    """
    path = Path(path)

    return {
        "saved_model": (path / "saved_model").exists(),
        "model.keras": (path / "model.keras").exists(),
        "config.json": (path / "config.json").exists(),
        "metadata.json": (path / "metadata.json").exists(),
        "feature_scaler.joblib": (path / "feature_scaler.joblib").exists(),
        "target_scaler.joblib": (path / "target_scaler.joblib").exists(),
        "training_history.json": (path / "training_history.json").exists(),
    }


def copy_model(
    source: Union[str, Path],
    destination: Union[str, Path],
) -> Path:
    """
    Copy a model artifact directory.

    Parameters
    ----------
    source : str or Path
        Source model path.
    destination : str or Path
        Destination path.

    Returns
    -------
    Path
        Destination path.
    """
    source = Path(source)
    destination = Path(destination)

    if destination.exists():
        shutil.rmtree(destination)

    shutil.copytree(source, destination)
    print(f"Model copied to: {destination}")

    return destination
