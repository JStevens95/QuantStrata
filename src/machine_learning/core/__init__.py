"""
Core ML components: base classes, configuration, result types, and tracking.

This module provides the foundational components for the ML framework:
    - Base model classes (BaseModel, PricingModel, CalibrationModel, PortfolioModel)
    - Configuration dataclasses (TrainingConfig, OptimizerConfig, etc.)
    - Result types (TrainingResult, EvaluationResult, CheckpointInfo, TuningResult)
    - Experiment tracking (ExperimentTracker, MLflowTracker, etc.)
    - Trainable protocol and KerasTrainableAdapter

Usage:
    from src.machine_learning.core import (
        BaseModel,
        PricingModel,
        TrainingConfig,
        OptimizerConfig,
        EarlyStoppingConfig,
        TrainingResult,
        EvaluationResult,
    )
"""
# Base model hierarchy
from src.machine_learning.core.base import (
    BaseModel,
    PricingModel,
    CalibrationModel,
    PortfolioModel,
)

# Configuration (canonical TrainingConfig lives here)
from src.machine_learning.core.config import (
    TrainingConfig,
    OptimizerConfig,
    LRScheduleConfig,
    EarlyStoppingConfig,
    CheckpointConfig,
    ModelConfig,
    DataPipelineConfig,
)

# Result types (canonical definitions — single source of truth)
from src.machine_learning.core.types import (
    TrainingResult,
    EvaluationResult,
    CheckpointInfo,
    TuningResult,
)

# Trainable protocol
from src.machine_learning.core.protocols import (
    Trainable,
    KerasTrainableAdapter,
)

# Experiment tracking (no TensorFlow dependency at import time)
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
    "ModelConfig",
    "DataPipelineConfig",
    # Result types
    "TrainingResult",
    "EvaluationResult",
    "CheckpointInfo",
    "TuningResult",
    # Protocols
    "Trainable",
    "KerasTrainableAdapter",
    # Experiment tracking
    "ExperimentTracker",
    "RunInfo",
    "InMemoryTracker",
    "MLflowTracker",
    "WandBTracker",
    "create_tracker",
]
