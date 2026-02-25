"""
Training loop, callbacks, learning-rate schedules, and distribution strategy.
"""
from src.rade_ml.training.schedules import WarmupCosineSchedule
from src.rade_ml.training.callbacks import get_standard_callbacks
from src.rade_ml.training.trainer import Trainer, setup_training_environment
from src.rade_ml.training.strategy import get_training_strategy

__all__ = [
    "WarmupCosineSchedule",
    "get_standard_callbacks",
    "get_training_strategy",
    "setup_training_environment",
    "Trainer",
]
