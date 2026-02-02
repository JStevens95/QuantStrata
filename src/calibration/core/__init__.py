"""
Calibration Core Framework.

This module provides a unified interface for model calibration:
- CalibrationEngine: Generic optimizer orchestration
- ObjectiveFunction: Pluggable objective functions
- OptimizerConfig: Configurable optimization backends
"""

from src.calibration.core.engine import (
    CalibrationEngine,
    CalibrationResult,
    CalibrationConfig,
)
from src.calibration.core.objectives import (
    ObjectiveFunction,
    WeightedLeastSquares,
    PenalizedObjective,
    MaxLikelihood,
)
from src.calibration.core.optimizers import (
    OptimizerConfig,
    LBFGSBConfig,
    LevenbergMarquardtConfig,
    DifferentialEvolutionConfig,
)

__all__ = [
    # Engine
    "CalibrationEngine",
    "CalibrationResult",
    "CalibrationConfig",
    # Objectives
    "ObjectiveFunction",
    "WeightedLeastSquares",
    "PenalizedObjective",
    "MaxLikelihood",
    # Optimizers
    "OptimizerConfig",
    "LBFGSBConfig",
    "LevenbergMarquardtConfig",
    "DifferentialEvolutionConfig",
]
