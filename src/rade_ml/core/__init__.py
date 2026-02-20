"""
Core abstractions: base model, configuration dataclasses, result types.
"""
from src.rade_ml.core.base import BaseModel
from src.rade_ml.core.config import (
    DataPipelineConfig,
    TrainingConfig,
    OptimizerConfig,
    LrScheduleConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    ReduceLrConfig,
)
from src.rade_ml.core.types import (
    TrainingResult,
    EvaluationResult,
    InferenceResult,
    TuningResult,
    CheckpointInfo,
)

__all__ = [
    "BaseModel",
    "DataPipelineConfig",
    "TrainingConfig",
    "OptimizerConfig",
    "LrScheduleConfig",
    "EarlyStoppingConfig",
    "CheckpointConfig",
    "ReduceLrConfig",
    "TrainingResult",
    "EvaluationResult",
    "InferenceResult",
    "TuningResult",
    "CheckpointInfo",
]
