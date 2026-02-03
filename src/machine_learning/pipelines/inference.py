"""
Generalised inference pipeline for QuantStrata ML models.

Provides:
- save_model(): Save a trained model's parameters and config.
- load_model(): Load a model from an artifact directory.
- predict(): Run inference on inputs.

Artifact layout convention:
    artifact_dir/
    ├── config.json       # TrainingConfig or model config
    ├── parameters.json   # Model parameters (weights)
    └── metadata.json     # Optional metadata (model class, version, etc.)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type, Union

import numpy as np

from src.machine_learning.core.protocols import Trainable
from src.machine_learning.core.types import TrainingConfig

logger = logging.getLogger(__name__)


def _serialise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert numpy arrays to lists for JSON serialisation."""
    serialisable = {}
    for k, v in params.items():
        if isinstance(v, np.ndarray):
            serialisable[k] = v.tolist()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            serialisable[k] = [arr.tolist() for arr in v]
        else:
            serialisable[k] = v
    return serialisable


def _deserialise_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert lists back to numpy arrays where appropriate."""
    deserialised = {}
    for k, v in params.items():
        if isinstance(v, list):
            # Check if it's a list of lists (array) vs list of arrays
            if len(v) > 0 and isinstance(v[0], list):
                # Could be list of arrays or 2D array; try to infer
                if len(v) > 0 and isinstance(v[0][0], (int, float)):
                    # Looks like a 2D array
                    deserialised[k] = np.array(v)
                else:
                    # List of arrays
                    deserialised[k] = [np.array(arr) for arr in v]
            elif len(v) > 0 and isinstance(v[0], (int, float)):
                deserialised[k] = np.array(v)
            else:
                deserialised[k] = v
        else:
            deserialised[k] = v
    return deserialised


def save_model(
    model: Trainable,
    artifact_dir: str,
    config: Optional[Union[TrainingConfig, Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Save a trained model to an artifact directory.

    Parameters
    ----------
    model : Trainable
        Trained model.
    artifact_dir : str
        Directory to save artifacts.
    config : TrainingConfig or dict, optional
        Training config or model config to save.
    metadata : dict, optional
        Additional metadata (e.g. model_class, version).

    Returns
    -------
    str
        Path to the artifact directory.

    Example
    -------
    >>> save_model(model, "artifacts/my_model", config=training_config)
    'artifacts/my_model'
    """
    path = Path(artifact_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save parameters
    params = model.get_parameters()
    serialisable_params = _serialise_params(params)
    with open(path / "parameters.json", "w") as f:
        json.dump(serialisable_params, f, indent=2)

    # Save config
    if config is not None:
        config_dict = config.to_dict() if hasattr(config, "to_dict") else config
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    # Save metadata
    meta = metadata or {}
    meta["model_class"] = type(model).__name__
    with open(path / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved to {artifact_dir}")
    return str(path)


def load_model(
    artifact_dir: str,
    model_factory: Callable[..., Trainable],
    factory_kwargs: Optional[Dict[str, Any]] = None,
) -> Trainable:
    """
    Load a model from an artifact directory.

    Parameters
    ----------
    artifact_dir : str
        Directory containing saved artifacts.
    model_factory : callable
        Factory function or class to instantiate the model.
        Called as model_factory(**factory_kwargs) if factory_kwargs provided,
        else model_factory().
    factory_kwargs : dict, optional
        Arguments to pass to the model factory.

    Returns
    -------
    Trainable
        Loaded model with parameters set.

    Example
    -------
    >>> model = load_model("artifacts/my_model", MyModel)
    >>> predictions = predict(model, X_test)
    """
    path = Path(artifact_dir)
    if not path.exists():
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    # Load parameters
    params_path = path / "parameters.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Parameters file not found: {params_path}")
    with open(params_path) as f:
        params = json.load(f)
    params = _deserialise_params(params)

    # Load config if needed by factory
    config_path = path / "config.json"
    config = None
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Instantiate model
    kwargs = factory_kwargs or {}
    if config is not None and "config" not in kwargs:
        kwargs["config"] = config
    model = model_factory(**kwargs) if kwargs else model_factory()

    # Load parameters
    model.set_parameters(params)
    logger.info(f"Model loaded from {artifact_dir}")
    return model


def predict(
    model: Trainable,
    inputs: Any,
    batch_size: Optional[int] = None,
) -> np.ndarray:
    """
    Run inference on inputs.

    Parameters
    ----------
    model : Trainable
        Trained model.
    inputs : array-like
        Inputs to the model.
    batch_size : int, optional
        If provided, run inference in batches.

    Returns
    -------
    ndarray
        Model predictions.

    Example
    -------
    >>> predictions = predict(model, X_test)
    >>> print(predictions.shape)
    (100, 1)
    """
    inputs = np.asarray(inputs)
    if batch_size is None or len(inputs) <= batch_size:
        outputs = model.forward(inputs)
        return np.asarray(outputs)

    # Batch inference
    results = []
    for start in range(0, len(inputs), batch_size):
        end = min(start + batch_size, len(inputs))
        batch_out = model.forward(inputs[start:end])
        results.append(np.asarray(batch_out))
    return np.concatenate(results, axis=0)


__all__ = ["save_model", "load_model", "predict"]
