"""
Model inference and deployment utilities.

This module provides:
    - ModelIO: Save/load models with sklearn scalers via joblib
    - Predictor: Efficient inference (ndarray, dict, tf.data.Dataset)
    - BatchPredictor: Multi-model ensemble inference and comparison
    - Serving: create_serving_function for TF Serving deployment

Usage:
    from src.machine_learning.inference import save_model, load_model, Predictor

    save_model(model, "models/my_pricer", feature_scaler=scaler_X, target_scaler=scaler_y)

    artifact = load_model("models/my_pricer")
    predictor = Predictor(artifact.model, target_scaler=artifact.target_scaler)
    prices = predictor.predict(features)
"""
from src.machine_learning.inference.model_io import (
    save_model,
    load_model,
    export_saved_model,
    ModelArtifact,
)
from src.machine_learning.inference.predictor import (
    Predictor,
    BatchPredictor,
    Features,
    create_serving_function,
)

__all__ = [
    "save_model",
    "load_model",
    "export_saved_model",
    "ModelArtifact",
    "Predictor",
    "BatchPredictor",
    "Features",
    "create_serving_function",
]
