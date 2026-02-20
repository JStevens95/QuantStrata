"""
Evaluation module: model evaluation, metrics, and diagnostic plots.
"""
from src.rade_ml.evaluation.evaluator import Evaluator
from src.rade_ml.evaluation.metrics import (
    rmse,
    mape,
    mae,
    mse,
    max_absolute_error,
    percentile_absolute_error,
    r_squared,
)

__all__ = [
    "Evaluator",
    "rmse",
    "mape",
    "mae",
    "mse",
    "max_absolute_error",
    "percentile_absolute_error",
    "r_squared",
]
