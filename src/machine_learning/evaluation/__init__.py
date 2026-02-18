"""
Model evaluation utilities.

This module provides:
    - Evaluator: Comprehensive model evaluation (sklearn.metrics + domain metrics)
    - compute_metrics: Standalone metric computation
    - Visualization: Prediction plots, residual analysis, Q-Q plots

Usage:
    from src.machine_learning.evaluation import Evaluator, evaluate_model

    evaluator = Evaluator(model, target_scaler=scaler)
    result = evaluator.evaluate(test_ds)
    evaluator.plot_predictions(test_ds)
"""
from src.machine_learning.core.types import EvaluationResult
from src.machine_learning.evaluation.evaluator import (
    Evaluator,
    evaluate_model,
    compute_metrics,
    EvalData,
)
from src.machine_learning.evaluation.metrics import (
    PricingMetrics,
    CalibrationMetrics,
)

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "evaluate_model",
    "compute_metrics",
    "EvalData",
    "PricingMetrics",
    "CalibrationMetrics",
]
