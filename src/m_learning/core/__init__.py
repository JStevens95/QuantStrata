"""
Core protocols and types for the QuantStrata ML framework.

- protocols: Trainable interface for models
- types: TrainingConfig, TrainingResult, EvaluationResult, etc.
"""

from src.m_learning.core.protocols import Trainable
from src.m_learning.core.types import (
    TrainingConfig,
    TrainingResult,
    EvaluationResult,
    CheckpointInfo,
)

__all__ = [
    "Trainable",
    "TrainingConfig",
    "TrainingResult",
    "EvaluationResult",
    "CheckpointInfo",
]
