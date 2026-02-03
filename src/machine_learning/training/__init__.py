"""
TensorFlow-native training infrastructure.

This module provides:
    - Trainer: High-level training interface
    - Custom training loops for advanced use cases
    - Training utilities and helpers

Usage:
    from src.machine_learning.training import Trainer
    from src.machine_learning.core import TrainingConfig
    
    trainer = Trainer(model, config)
    history = trainer.fit(train_dataset, val_dataset)
"""
from src.machine_learning.training.trainer import (
    Trainer,
    TrainingResult,
    compile_model,
    fit_model,
)

__all__ = [
    "Trainer",
    "TrainingResult",
    "compile_model",
    "fit_model",
]
