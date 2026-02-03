"""
Core ML components: base classes, configuration, and callbacks.

This module provides the foundational components for the ML framework:
    - Base model classes (BaseModel, PricingModel, CalibrationModel, PortfolioModel)
    - Configuration dataclasses (TrainingConfig, OptimizerConfig, etc.)
    - Custom Keras callbacks

Usage:
    from src.m_learning.core import (
        BaseModel,
        PricingModel,
        TrainingConfig,
        OptimizerConfig,
        EarlyStoppingConfig,
    )
"""
from src.m_learning.core.base import (
    BaseModel,
    PricingModel,
    CalibrationModel,
    PortfolioModel,
)
from src.m_learning.core.config import (
    TrainingConfig,
    OptimizerConfig,
    LRScheduleConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    DataConfig,
    ModelConfig,
)
from src.m_learning.core.callbacks import (
    MetricsLogger,
    PricingErrorCallback,
    TrainingProgressCallback,
    GradientMonitorCallback,
    get_standard_callbacks,
)

# Result types (used by pipelines)
from src.m_learning.core.types import (
    TrainingConfig as TypesTrainingConfig,
    TrainingResult,
    EvaluationResult,
    CheckpointInfo,
    TuningResult,
)
# Legacy aliases
from src.m_learning.core.types import (
    TrainingResult as LegacyTrainingResult,
    EvaluationResult as LegacyEvaluationResult,
)

__all__ = [
    # Base models
    "BaseModel",
    "PricingModel",
    "CalibrationModel",
    "PortfolioModel",
    # Configuration
    "TrainingConfig",
    "OptimizerConfig",
    "LRScheduleConfig",
    "EarlyStoppingConfig",
    "CheckpointConfig",
    "DataConfig",
    "ModelConfig",
    # Callbacks
    "MetricsLogger",
    "PricingErrorCallback",
    "TrainingProgressCallback",
    "GradientMonitorCallback",
    "get_standard_callbacks",
    # Result types
    "TypesTrainingConfig",
    "TrainingResult",
    "EvaluationResult",
    "CheckpointInfo",
    "TuningResult",
    # Legacy
    "LegacyTrainingResult",
    "LegacyEvaluationResult",
]
