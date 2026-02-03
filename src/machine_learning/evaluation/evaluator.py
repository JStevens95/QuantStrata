"""
Comprehensive model evaluation utilities.

This module provides professional-grade model evaluation including:
    - Standard regression metrics (MSE, MAE, R², etc.)
    - Domain-specific pricing metrics (MAPE, max error, percentiles)
    - Visualization tools (prediction plots, residual analysis)
    - Comparison against benchmarks

Usage:
    evaluator = Evaluator(model)
    
    # Full evaluation
    result = evaluator.evaluate(test_dataset)
    print(result.summary())
    
    # Visualizations
    evaluator.plot_predictions(test_dataset)
    evaluator.plot_residuals(test_dataset)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from src.machine_learning.data.dataset import TFDataset, NormalizationStats


@dataclass
class EvaluationResult:
    """
    Container for evaluation results.
    
    Attributes:
        metrics: Dictionary of computed metrics
        predictions: Model predictions array
        targets: Ground truth targets
        residuals: Prediction errors (targets - predictions)
        dataset_info: Information about the evaluated dataset
        timestamp: Evaluation timestamp
    """
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"EvaluationResult({metrics_str})"
    
    def summary(self) -> str:
        """Return formatted summary string."""
        lines = [
            "=" * 50,
            "EVALUATION RESULTS",
            "=" * 50,
            f"Timestamp: {self.timestamp}",
            f"Samples: {self.dataset_info.get('n_samples', 'N/A')}",
            "",
            "Metrics:",
            "-" * 30,
        ]
        
        for name, value in sorted(self.metrics.items()):
            if "error" in name.lower() or "loss" in name.lower():
                lines.append(f"  {name:20s}: {value:12.6f}")
            elif "r2" in name.lower() or "score" in name.lower():
                lines.append(f"  {name:20s}: {value:12.4f}")
            else:
                lines.append(f"  {name:20s}: {value:12.6f}")
        
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "metrics": self.metrics,
            "dataset_info": self.dataset_info,
            "timestamp": self.timestamp,
        }
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EvaluationResult":
        """Load from JSON file."""
        with open(path, "r") as f:
            d = json.load(f)
        return cls(
            metrics=d["metrics"],
            dataset_info=d.get("dataset_info", {}),
            timestamp=d.get("timestamp", ""),
        )


class Evaluator:
    """
    Comprehensive model evaluator.
    
    Provides:
        - Standard regression metrics
        - Domain-specific metrics for pricing/calibration
        - Visualization tools
        - Benchmark comparison
    
    Attributes:
        model: TensorFlow model to evaluate
        target_scaler: Optional scaler for denormalizing predictions
    
    Example:
        evaluator = Evaluator(model, target_scaler=dataset.target_stats)
        result = evaluator.evaluate(test_dataset)
        
        print(result.summary())
        evaluator.plot_predictions(test_dataset)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        target_scaler: Optional[NormalizationStats] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Keras model to evaluate
            target_scaler: Normalization stats for denormalizing predictions
        """
        self.model = model
        self.target_scaler = target_scaler
    
    def evaluate(
        self,
        data: Union[TFDataset, tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
        metrics: Optional[List[str]] = None,
        include_predictions: bool = True,
        batch_size: int = 256,
    ) -> EvaluationResult:
        """
        Evaluate model on dataset.
        
        Args:
            data: Evaluation data
            metrics: List of metrics to compute (default: standard regression metrics)
            include_predictions: Whether to include predictions in result
            batch_size: Batch size for prediction
        
        Returns:
            EvaluationResult with computed metrics
        """
        # Extract features and targets
        if isinstance(data, TFDataset):
            features = data.features
            targets = data.targets.flatten()
            dataset_info = {"n_samples": len(data), **data.metadata}
            target_scaler = self.target_scaler or data.target_stats
        elif isinstance(data, tuple):
            features, targets = data
            targets = np.asarray(targets).flatten()
            dataset_info = {"n_samples": len(features)}
            target_scaler = self.target_scaler
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Generate predictions
        predictions = self.model.predict(features, batch_size=batch_size, verbose=0)
        predictions = predictions.flatten()
        
        # Denormalize if scaler available
        if target_scaler is not None:
            predictions_orig = target_scaler.denormalize(predictions)
            targets_orig = target_scaler.denormalize(targets)
        else:
            predictions_orig = predictions
            targets_orig = targets
        
        # Compute metrics
        if metrics is None:
            metrics = ["mse", "mae", "rmse", "mape", "r2", "max_error", "p95_error"]
        
        computed_metrics = compute_metrics(
            y_true=targets_orig,
            y_pred=predictions_orig,
            metrics=metrics,
        )
        
        # Build result
        residuals = targets_orig - predictions_orig if include_predictions else None
        
        return EvaluationResult(
            metrics=computed_metrics,
            predictions=predictions_orig if include_predictions else None,
            targets=targets_orig if include_predictions else None,
            residuals=residuals,
            dataset_info=dataset_info,
        )
    
    def plot_predictions(
        self,
        data: Union[TFDataset, Tuple[np.ndarray, np.ndarray]],
        title: str = "Predicted vs Actual",
        figsize: Tuple[int, int] = (10, 8),
        sample_size: Optional[int] = None,
    ) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            data: Evaluation data
            title: Plot title
            figsize: Figure size
            sample_size: Optional sample size for large datasets
        """
        import matplotlib.pyplot as plt
        
        result = self.evaluate(data, include_predictions=True)
        
        y_true = result.targets
        y_pred = result.predictions
        
        # Subsample if needed
        if sample_size and len(y_true) > sample_size:
            idx = np.random.choice(len(y_true), sample_size, replace=False)
            y_true = y_true[idx]
            y_pred = y_pred[idx]
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Scatter plot: predicted vs actual
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, alpha=0.5, s=10)
        
        # Perfect prediction line
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect')
        
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{title}\nR² = {result.metrics.get("r2", 0):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Residual distribution
        ax = axes[0, 1]
        residuals = y_true - y_pred
        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Residual (Actual - Predicted)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Residual Distribution\nMAE = {result.metrics.get("mae", 0):.4f}')
        ax.grid(True, alpha=0.3)
        
        # 3. Residual vs predicted
        ax = axes[1, 0]
        ax.scatter(y_pred, residuals, alpha=0.5, s=10)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Residual')
        ax.set_title('Residuals vs Predicted')
        ax.grid(True, alpha=0.3)
        
        # 4. Q-Q plot
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot (Normality Check)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_error_analysis(
        self,
        data: Union[TFDataset, Tuple[np.ndarray, np.ndarray]],
        feature_names: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
    ) -> None:
        """
        Plot error analysis by feature.
        
        Shows how prediction error varies with each input feature.
        
        Args:
            data: Evaluation data
            feature_names: Optional feature names
            figsize: Figure size
        """
        import matplotlib.pyplot as plt
        
        if isinstance(data, TFDataset):
            features = data.features
            feature_names = feature_names or data.feature_names
        else:
            features, _ = data
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(features.shape[1])]
        
        result = self.evaluate(data, include_predictions=True)
        abs_errors = np.abs(result.residuals)
        
        n_features = min(len(feature_names), 6)  # Limit to 6 features
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        for i, (ax, name) in enumerate(zip(axes, feature_names[:n_features])):
            ax.scatter(features[:, i], abs_errors, alpha=0.5, s=10)
            ax.set_xlabel(name)
            ax.set_ylabel('Absolute Error')
            ax.set_title(f'Error vs {name}')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('Error Analysis by Feature', y=1.02)
        plt.show()
    
    def compare_with_benchmark(
        self,
        data: Union[TFDataset, Tuple[np.ndarray, np.ndarray]],
        benchmark_fn: Callable[[np.ndarray], np.ndarray],
        benchmark_name: str = "Benchmark",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare model with a benchmark (e.g., Black-Scholes).
        
        Args:
            data: Evaluation data
            benchmark_fn: Function that takes features and returns predictions
            benchmark_name: Name for the benchmark
        
        Returns:
            Dictionary with metrics for both model and benchmark
        """
        if isinstance(data, TFDataset):
            features = data.features
            targets = data.targets.flatten()
        else:
            features, targets = data
            targets = np.asarray(targets).flatten()
        
        # Model predictions
        model_preds = self.model.predict(features, verbose=0).flatten()
        
        # Benchmark predictions
        benchmark_preds = benchmark_fn(features)
        
        # Denormalize if needed
        if self.target_scaler:
            model_preds = self.target_scaler.denormalize(model_preds)
            targets = self.target_scaler.denormalize(targets)
        
        # Compute metrics for both
        metrics = ["mse", "mae", "rmse", "mape", "r2"]
        
        return {
            "model": compute_metrics(targets, model_preds, metrics),
            benchmark_name: compute_metrics(targets, benchmark_preds, metrics),
        }


def evaluate_model(
    model: tf.keras.Model,
    data: Union[TFDataset, tf.data.Dataset, Tuple[np.ndarray, np.ndarray]],
    metrics: Optional[List[str]] = None,
    target_scaler: Optional[NormalizationStats] = None,
) -> EvaluationResult:
    """
    Evaluate a model (convenience function).
    
    Args:
        model: Keras model
        data: Evaluation data
        metrics: List of metrics to compute
        target_scaler: Optional normalization stats for denormalizing
    
    Returns:
        EvaluationResult
    """
    evaluator = Evaluator(model, target_scaler=target_scaler)
    return evaluator.evaluate(data, metrics=metrics)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str],
) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Available metrics:
        - mse: Mean Squared Error
        - mae: Mean Absolute Error
        - rmse: Root Mean Squared Error
        - mape: Mean Absolute Percentage Error
        - r2: R² (coefficient of determination)
        - max_error: Maximum absolute error
        - p95_error: 95th percentile error
        - p99_error: 99th percentile error
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        metrics: List of metric names
    
    Returns:
        Dictionary of metric values
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    errors = y_true - y_pred
    abs_errors = np.abs(errors)
    
    result = {}
    
    for metric in metrics:
        metric = metric.lower()
        
        if metric == "mse":
            result["mse"] = float(np.mean(errors ** 2))
        elif metric == "mae":
            result["mae"] = float(np.mean(abs_errors))
        elif metric == "rmse":
            result["rmse"] = float(np.sqrt(np.mean(errors ** 2)))
        elif metric == "mape":
            # Avoid division by zero
            mask = np.abs(y_true) > 1e-8
            if mask.any():
                result["mape"] = float(np.mean(abs_errors[mask] / np.abs(y_true[mask])) * 100)
            else:
                result["mape"] = 0.0
        elif metric == "r2":
            ss_res = np.sum(errors ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            result["r2"] = float(1 - ss_res / (ss_tot + 1e-8))
        elif metric == "max_error":
            result["max_error"] = float(np.max(abs_errors))
        elif metric == "p95_error":
            result["p95_error"] = float(np.percentile(abs_errors, 95))
        elif metric == "p99_error":
            result["p99_error"] = float(np.percentile(abs_errors, 99))
        elif metric == "median_error":
            result["median_error"] = float(np.median(abs_errors))
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    return result
