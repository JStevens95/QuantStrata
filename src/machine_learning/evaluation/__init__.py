"""
Model evaluation utilities.

This module provides:
    - Evaluator: Comprehensive model evaluation
    - Metrics: Custom TensorFlow metrics
    - Visualization: Training and evaluation plots

Usage:
    from src.machine_learning.evaluation import Evaluator, evaluate_model
    
    evaluator = Evaluator(model)
    results = evaluator.evaluate(test_dataset)
    evaluator.plot_predictions(test_dataset)
"""
from src.machine_learning.evaluation.evaluator import (
    Evaluator,
    EvaluationResult,
    evaluate_model,
    compute_metrics,
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
    "PricingMetrics",
    "CalibrationMetrics"
]
