"""
TensorFlow-native training infrastructure.

This module provides:
    - Trainer: High-level Keras training interface (accepts tf.data.Dataset only)
    - Custom callbacks: MetricsLogger, PricingErrorCallback
    - Learning rate schedules: WarmupCosineSchedule
    - Training utilities and helpers

Usage:
    from src.machine_learning.training import Trainer
    from src.machine_learning.core import TrainingConfig, TrainingResult

    trainer = Trainer(model, config)
    result = trainer.fit(train_ds, val_ds)
"""
# Canonical result type from core (single source of truth)
from src.machine_learning.core.types import TrainingResult

# Trainer class and convenience functions
from src.machine_learning.training.trainer import (
    Trainer,
    compile_model,
    fit_model,
)

# Callbacks
from src.machine_learning.training.callbacks import (
    MetricsLogger,
    PricingErrorCallback,
    get_standard_callbacks,
)

# Learning rate schedules
from src.machine_learning.training.schedules import WarmupCosineSchedule

__all__ = [
    # Trainer
    "Trainer",
    "TrainingResult",
    "compile_model",
    "fit_model",
    # Callbacks
    "MetricsLogger",
    "PricingErrorCallback",
    "get_standard_callbacks",
    # Schedules
    "WarmupCosineSchedule",
]
