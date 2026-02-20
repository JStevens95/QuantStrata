"""
rade ML framework -- machine learning library for quantitative finance.

Top-level re-exports for convenience:
    from rade_ml import BaseModel, TrainingConfig, Trainer
"""
from src.rade_ml.core.base import BaseModel
from src.rade_ml.core.config import TrainingConfig, DataPipelineConfig

__all__ = ["BaseModel", "TrainingConfig", "DataPipelineConfig"]
