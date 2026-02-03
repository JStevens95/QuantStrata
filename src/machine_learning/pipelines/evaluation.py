"""
Standardised model evaluation for QuantStrata ML models.

Provides evaluate_model() that computes common metrics and returns an
EvaluationResult with a consistent schema for comparison and serialisation.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from src.machine_learning.core.protocols import Trainable
from src.machine_learning.core.types import EvaluationResult, TrainingResult

logger = logging.getLogger(__name__)


def _compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((y_true - y_pred) ** 2))


def _compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def _compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _compute_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean absolute percentage error."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def _compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


METRIC_FUNCTIONS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "mse": _compute_mse,
    "mae": _compute_mae,
    "rmse": _compute_rmse,
    "mape": _compute_mape,
    "r2": _compute_r2,
}


def evaluate_model(
    model: Trainable,
    features: np.ndarray,
    targets: np.ndarray,
    metrics: Optional[List[str]] = None,
    loss_fn: Optional[Callable[[Any, Any], float]] = None,
    training_result: Optional[TrainingResult] = None,
    benchmark_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> EvaluationResult:
    """
    Evaluate a trained model on a dataset.

    Parameters
    ----------
    model : Trainable
        Trained model conforming to the Trainable protocol.
    features : ndarray
        Evaluation features.
    targets : ndarray
        Ground truth targets.
    metrics : list of str, optional
        Metrics to compute (e.g. ["mse", "mae", "r2"]).
        Defaults to ["mse", "mae"].
    loss_fn : callable, optional
        Custom loss function (y_true, y_pred) -> scalar.
        If None, uses model.compute_loss.
    training_result : TrainingResult, optional
        Training result to include loss curves in output.
    benchmark_fn : callable, optional
        Function to compute benchmark predictions for pricing error.
        If provided, pricing_error = MAE(model_pred, benchmark_pred).
    metadata : dict, optional
        Additional metadata to include in result.

    Returns
    -------
    EvaluationResult
        Evaluation loss, metrics, and optional loss curves / pricing error.

    Example
    -------
    >>> from src.machine_learning.pipeline import evaluate_model
    >>> result = evaluate_model(model, X_test, y_test, metrics=["mse", "mae", "r2"])
    >>> print(result.metrics)
    {'mse': 0.001, 'mae': 0.02, 'r2': 0.95}
    """
    if metrics is None:
        metrics = ["mse", "mae"]

    # Forward pass
    y_pred = model.forward(features)
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(targets)

    # Compute loss
    if loss_fn is not None:
        loss = float(loss_fn(y_true, y_pred))
    else:
        loss = model.compute_loss(y_true, y_pred)

    # Compute metrics
    computed_metrics: Dict[str, float] = {}
    for metric in metrics:
        if metric in METRIC_FUNCTIONS:
            computed_metrics[metric] = METRIC_FUNCTIONS[metric](y_true, y_pred)
        else:
            logger.warning(f"Unknown metric '{metric}', skipping.")

    # Loss curves from training
    loss_curves = None
    if training_result is not None and training_result.history:
        loss_curves = training_result.history

    # Pricing error vs benchmark
    pricing_error = None
    if benchmark_fn is not None:
        benchmark_pred = benchmark_fn(features)
        benchmark_pred = np.asarray(benchmark_pred)
        pricing_error = _compute_mae(y_pred, benchmark_pred)

    result = EvaluationResult(
        loss=loss,
        metrics=computed_metrics,
        loss_curves=loss_curves,
        pricing_error=pricing_error,
        metadata=metadata or {},
    )
    return result


__all__ = ["evaluate_model", "METRIC_FUNCTIONS"]
