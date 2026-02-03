"""
Model inference and deployment utilities.

This module provides:
    - ModelIO: Save/load models using TensorFlow SavedModel format
    - Predictor: Efficient batch inference
    - Model serving utilities

Usage:
    from src.machine_learning.inference import save_model, load_model, Predictor
    
    # Save trained model
    save_model(model, "models/my_pricer", metadata={"version": "1.0"})
    
    # Load for inference
    loaded_model = load_model("models/my_pricer")
    
    # Efficient batch prediction
    predictor = Predictor(loaded_model)
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
)

__all__ = [
    "save_model",
    "load_model",
    "export_saved_model",
    "ModelArtifact",
    "Predictor",
    "BatchPredictor",
]
