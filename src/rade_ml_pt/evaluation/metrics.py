"""
Standard evaluation metric functions for the rade ML framework.

All functions follow the signature ``fn(y_true, y_pred) -> float`` so they
can be passed directly to ``Evaluator.run(additional_metrics={...})``.

These operate on numpy arrays and are framework-agnostic.

Usage:
    from rade_ml.evaluation.metrics import rmse, mape

    evaluator = Evaluator(model)
    result = evaluator.run(test_ds, additional_metrics={"rmse": rmse, "mape": mape})
"""
from __future__ import annotations

import numpy as np


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :return: RMSE scalar.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error.

    Uses an epsilon floor on the denominator to avoid division by zero
    for near-zero targets (common in PnL data).

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :param eps: small constant to prevent division by zero.
    :return: MAPE as a percentage (0-100+ scale).
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


def max_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Maximum Absolute Error across all samples.

    Useful as a worst-case risk metric for pricing / revaluation models.

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :return: max absolute error scalar.
    """
    return float(np.max(np.abs(y_true - y_pred)))


def percentile_absolute_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """
    Percentile of the absolute error distribution.

    Common choices are P95 and P99 for tail-risk monitoring.

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :param percentile: percentile to compute (0-100).
    :return: the requested percentile of absolute errors.
    """
    return float(np.percentile(np.abs(y_true - y_pred), percentile))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :return: MAE scalar.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :return: MSE scalar.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Coefficient of determination (R-squared).

    Returns 1.0 for a perfect fit, 0.0 when the model predicts the mean,
    and negative values when the model is worse than predicting the mean.

    :param y_true: ground truth values.
    :param y_pred: predicted values.
    :return: R-squared scalar.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return float(1.0 - ss_res / ss_tot)
