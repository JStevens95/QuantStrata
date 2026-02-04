"""
Core ML components: base classes, configuration, and callbacks.

This module provides the foundational components for the ML framework:
    - Base model classes (BaseModel, PricingModel, CalibrationModel, PortfolioModel)
    - Configuration dataclasses (TrainingConfig, OptimizerConfig, etc.)
    - Custom Keras callbacks

Usage:
    from src.machine_learning.core import (
        BaseModel,
        PricingModel,
        TrainingConfig,
        OptimizerConfig,
        EarlyStoppingConfig,
    )
"""
from src.machine_learning.core.base import (
    BaseModel,
    PricingModel,
    CalibrationModel,
    PortfolioModel,
)
from src.machine_learning.core.config import (
    TrainingConfig,
    OptimizerConfig,
    LRScheduleConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    DataConfig,
    ModelConfig,
)
from src.machine_learning.core.callbacks import (
    MetricsLogger,
    PricingErrorCallback,
    TrainingProgressCallback,
    GradientMonitorCallback,
    get_standard_callbacks,
)

# Result types (used by pipelines)
from src.machine_learning.core.types import (
    TrainingConfig as TypesTrainingConfig,
    TrainingResult,
    EvaluationResult,
    CheckpointInfo,
    TuningResult,
)
# Legacy aliases
from src.machine_learning.core.types import (
    TrainingResult as LegacyTrainingResult,
    EvaluationResult as LegacyEvaluationResult,
)

# Experiment tracking (no tensorflow dependency)
from src.machine_learning.core.tracking import (
    ExperimentTracker,
    RunInfo,
    InMemoryTracker,
    MLflowTracker,
    WandBTracker,
    create_tracker,
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
    # Experiment tracking
    "ExperimentTracker",
    "RunInfo",
    "InMemoryTracker",
    "MLflowTracker",
    "WandBTracker",
    "create_tracker",
]
