"""
Comprehensive model evaluation utilities.

This module provides:
    - ``compute_metrics``:  Delegates to ``sklearn.metrics`` for standard regression
      metrics and adds domain-specific pricing metrics (MAPE, percentile errors).
    - ``Evaluator``:  Full evaluation pipeline with visualisations.
    - ``evaluate_model``:  One-liner convenience function.

Supported input formats:
    - ``tf.data.Dataset`` (including dict-feature batches for GNN / graph models)
    - ``(features, targets)`` tuples (ndarray or Dict[str, ndarray])

Usage:
    evaluator = Evaluator(model)
    result = evaluator.evaluate(test_ds)
    print(result.summary())

    evaluator.plot_predictions(test_ds)
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    max_error as sklearn_max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from src.machine_learning.core.types import EvaluationResult

logger = logging.getLogger(__name__)

# Type alias — sklearn scaler (StandardScaler / MinMaxScaler) or any object
# with an ``inverse_transform`` method.
Scaler = Any

# Accepted evaluation data formats
EvalData = Union[
    tf.data.Dataset,
    Tuple[np.ndarray, np.ndarray],
    Tuple[Dict[str, np.ndarray], np.ndarray],
]


# ---------------------------------------------------------------------------
# Metrics (delegates to sklearn)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Compute evaluation metrics.

    Standard metrics delegate to ``sklearn.metrics``.  Domain-specific metrics
    (percentile errors) are computed directly.

    Available metrics:
        mse, mae, rmse, mape, r2, max_error, p95_error, p99_error, median_error

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values.
    y_pred : np.ndarray
        Predicted values.
    metrics : list of str
        Metric names to compute.

    Returns
    -------
    dict
        Metric name -> float value.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    abs_errors = np.abs(y_true - y_pred)

    result: Dict[str, float] = {}

    for metric in metrics:
        m = metric.lower()

        if m == "mse":
            result["mse"] = float(mean_squared_error(y_true, y_pred))
        elif m == "mae":
            result["mae"] = float(mean_absolute_error(y_true, y_pred))
        elif m == "rmse":
            result["rmse"] = float(mean_squared_error(y_true, y_pred, squared=False))
        elif m == "mape":
            # sklearn MAPE returns fraction (0-1+), we report as percentage
            mask = np.abs(y_true) > 1e-8
            if mask.any():
                result["mape"] = float(
                    mean_absolute_percentage_error(y_true[mask], y_pred[mask]) * 100
                )
            else:
                result["mape"] = 0.0
        elif m == "r2":
            result["r2"] = float(r2_score(y_true, y_pred))
        elif m == "max_error":
            result["max_error"] = float(np.max(abs_errors))
        elif m == "p95_error":
            result["p95_error"] = float(np.percentile(abs_errors, 95))
        elif m == "p99_error":
            result["p99_error"] = float(np.percentile(abs_errors, 99))
        elif m == "median_error":
            result["median_error"] = float(np.median(abs_errors))
        else:
            raise ValueError(f"Unknown metric: {metric}")

    return result


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class Evaluator:
    """
    Comprehensive model evaluator.

    Handles ndarray features (MLP pricers), dict features (GNN / graph models),
    and pre-batched ``tf.data.Dataset`` pipelines.

    Attributes
    ----------
    model : tf.keras.Model
        Model to evaluate.
    target_scaler : sklearn scaler, optional
        Scaler for denormalising predictions (must have ``inverse_transform``).

    Example
    -------
    >>> evaluator = Evaluator(model, target_scaler=scaler)
    >>> result = evaluator.evaluate(test_ds)
    >>> print(result.summary())
    """

    def __init__(
        self,
        model: tf.keras.Model,
        target_scaler: Optional[Scaler] = None,
    ):
        self.model = model
        self.target_scaler = target_scaler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        data: EvalData,
        metrics: Optional[List[str]] = None,
        include_predictions: bool = True,
        batch_size: int = 256,
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.

        Parameters
        ----------
        data : EvalData
            Evaluation data in any supported format.
        metrics : list of str, optional
            Metric names (default: standard regression suite).
        include_predictions : bool
            Whether to include raw predictions/targets in the result.
        batch_size : int
            Batch size when ``data`` is an array tuple.

        Returns
        -------
        EvaluationResult
        """
        predictions, targets, dataset_info = self._extract_predictions_and_targets(
            data, batch_size,
        )

        # Denormalise if scaler is available
        if self.target_scaler is not None and hasattr(self.target_scaler, "inverse_transform"):
            predictions = self.target_scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            targets = self.target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).flatten()

        if metrics is None:
            metrics = ["mse", "mae", "rmse", "mape", "r2", "max_error", "p95_error"]

        computed_metrics = compute_metrics(y_true=targets, y_pred=predictions, metrics=metrics)

        residuals = targets - predictions if include_predictions else None

        return EvaluationResult(
            metrics=computed_metrics,
            predictions=predictions if include_predictions else None,
            targets=targets if include_predictions else None,
            residuals=residuals,
            dataset_info=dataset_info,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_predictions_and_targets(
        self,
        data: EvalData,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Run inference and collect (predictions, targets, info) from any format."""

        # ----- tf.data.Dataset ------------------------------------------------
        if isinstance(data, tf.data.Dataset):
            predictions, targets = self._predict_from_tf_dataset(data)
            return predictions, targets, {"n_samples": len(predictions)}

        # ----- (features, targets) tuple --------------------------------------
        if isinstance(data, tuple) and len(data) == 2:
            features, targets_raw = data
            targets = np.asarray(targets_raw).flatten()

            if isinstance(features, dict):
                predictions = self._predict_dict_features(features)
            else:
                predictions = self.model.predict(
                    features, batch_size=batch_size, verbose=0,
                ).flatten()

            return predictions, targets, {"n_samples": len(targets)}

        raise ValueError(
            f"Unsupported data type: {type(data)}. "
            "Expected tf.data.Dataset or (features, targets) tuple."
        )

    def _predict_from_tf_dataset(
        self,
        dataset: tf.data.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Iterate over batched tf.data.Dataset, run inference, collect results."""
        all_preds: List[np.ndarray] = []
        all_targets: List[np.ndarray] = []

        for batch in dataset:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                batch_features, batch_targets = batch
            else:
                raise ValueError(
                    "tf.data.Dataset must yield (features, targets) tuples."
                )

            batch_preds = self.model(batch_features, training=False)
            all_preds.append(tf.reshape(batch_preds, [-1]).numpy())
            all_targets.append(tf.reshape(batch_targets, [-1]).numpy())

        return np.concatenate(all_preds), np.concatenate(all_targets)

    def _predict_dict_features(self, features: Dict[str, Any]) -> np.ndarray:
        """Predict from dict features (GNN / graph models)."""
        tensor_features = {
            k: tf.convert_to_tensor(v) if not isinstance(v, tf.Tensor) else v
            for k, v in features.items()
        }
        output = self.model(tensor_features, training=False)
        return tf.reshape(output, [-1]).numpy()

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------

    def plot_predictions(
        self,
        data: EvalData,
        title: str = "Predicted vs Actual",
        figsize: Tuple[int, int] = (10, 8),
        sample_size: Optional[int] = None,
    ) -> None:
        """
        Plot predictions vs actual values (scatter, residual distribution, Q-Q).

        Parameters
        ----------
        data : EvalData
            Evaluation data.
        title : str
            Plot title.
        figsize : tuple
            Figure size.
        sample_size : int, optional
            Sub-sample for large datasets.
        """
        import matplotlib.pyplot as plt

        result = self.evaluate(data, include_predictions=True)
        y_true = result.targets
        y_pred = result.predictions

        if sample_size and len(y_true) > sample_size:
            idx = np.random.choice(len(y_true), sample_size, replace=False)
            y_true = y_true[idx]
            y_pred = y_pred[idx]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # Scatter: predicted vs actual
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", linewidth=2, label="Perfect")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f'{title}\nR² = {result.metrics.get("r2", 0):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Residual distribution
        ax = axes[0, 1]
        residuals = y_true - y_pred
        ax.hist(residuals, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Residual (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title(f'Residual Distribution\nMAE = {result.metrics.get("mae", 0):.4f}')
        ax.grid(True, alpha=0.3)

        # Residual vs predicted
        ax = axes[1, 0]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(0, color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual")
        ax.set_title("Residuals vs Predicted")
        ax.grid(True, alpha=0.3)

        # Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("Q-Q Plot (Normality Check)")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_error_analysis(
        self,
        data: EvalData,
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> None:
        """
        Plot error analysis by feature.

        Parameters
        ----------
        data : EvalData
            Evaluation data.
        feature_names : list of str, optional
            Feature column names.
        figsize : tuple
            Figure size.
        """
        import matplotlib.pyplot as plt

        if isinstance(data, tuple) and len(data) == 2:
            features, _ = data
        else:
            raise ValueError("plot_error_analysis requires (features, targets) tuple input.")

        if isinstance(features, dict):
            raise ValueError("plot_error_analysis does not support dict features.")

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]

        result = self.evaluate(data, include_predictions=True)
        abs_errors = np.abs(result.residuals)

        n_features = min(len(feature_names), 6)
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()

        for i, (ax, name) in enumerate(zip(axes, feature_names[:n_features])):
            ax.scatter(features[:, i], abs_errors, alpha=0.5, s=10)
            ax.set_xlabel(name)
            ax.set_ylabel("Absolute Error")
            ax.set_title(f"Error vs {name}")
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.suptitle("Error Analysis by Feature", y=1.02)
        plt.show()

    def compare_with_benchmark(
        self,
        data: Tuple[np.ndarray, np.ndarray],
        benchmark_fn: Callable[[np.ndarray], np.ndarray],
        benchmark_name: str = "Benchmark",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model with a benchmark (e.g. Black-Scholes).

        Parameters
        ----------
        data : tuple of (features, targets)
            Evaluation data.
        benchmark_fn : callable
            ``features -> predictions``.
        benchmark_name : str
            Label for the benchmark.

        Returns
        -------
        dict
            ``{"model": {metrics}, benchmark_name: {metrics}}``.
        """
        features, targets = data
        targets = np.asarray(targets).flatten()

        model_preds = self.model.predict(features, verbose=0).flatten()
        benchmark_preds = benchmark_fn(features)

        if self.target_scaler is not None and hasattr(self.target_scaler, "inverse_transform"):
            model_preds = self.target_scaler.inverse_transform(
                model_preds.reshape(-1, 1)
            ).flatten()
            targets = self.target_scaler.inverse_transform(
                targets.reshape(-1, 1)
            ).flatten()

        metric_names = ["mse", "mae", "rmse", "mape", "r2"]
        return {
            "model": compute_metrics(targets, model_preds, metric_names),
            benchmark_name: compute_metrics(targets, benchmark_preds, metric_names),
        }


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: tf.keras.Model,
    data: EvalData,
    metrics: Optional[List[str]] = None,
    target_scaler: Optional[Scaler] = None,
) -> EvaluationResult:
    """
    Evaluate a model (one-liner convenience).

    Parameters
    ----------
    model : tf.keras.Model
    data : EvalData
    metrics : list of str, optional
    target_scaler : sklearn scaler, optional

    Returns
    -------
    EvaluationResult
    """
    evaluator = Evaluator(model, target_scaler=target_scaler)
    return evaluator.evaluate(data, metrics=metrics)
