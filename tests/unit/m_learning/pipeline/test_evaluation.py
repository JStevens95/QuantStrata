"""Tests for m_learning.pipeline.evaluation."""

import numpy as np
import pytest

from src.m_learning.core.types import TrainingResult
from src.m_learning.pipeline.evaluation import evaluate_model, METRIC_FUNCTIONS


class DummyModel:
    """Simple model for testing evaluation."""

    def __init__(self, weights: np.ndarray):
        self.weights = weights

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return inputs @ self.weights

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def get_parameters(self):
        return {"weights": self.weights}

    def set_parameters(self, params):
        self.weights = params["weights"]


class TestMetricFunctions:
    """Tests for individual metric functions."""

    def test_mse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        mse = METRIC_FUNCTIONS["mse"](y_true, y_pred)
        expected = np.mean((y_true - y_pred) ** 2)
        assert abs(mse - expected) < 1e-8

    def test_mae(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        mae = METRIC_FUNCTIONS["mae"](y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        assert abs(mae - expected) < 1e-8

    def test_rmse(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.1, 2.1, 2.9])
        rmse = METRIC_FUNCTIONS["rmse"](y_true, y_pred)
        expected = np.sqrt(np.mean((y_true - y_pred) ** 2))
        assert abs(rmse - expected) < 1e-8

    def test_r2_perfect(self):
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        r2 = METRIC_FUNCTIONS["r2"](y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-8

    def test_r2_imperfect(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.8, 4.1, 4.9])
        r2 = METRIC_FUNCTIONS["r2"](y_true, y_pred)
        assert 0 < r2 < 1


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_basic_evaluation(self):
        """Test basic evaluation."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0])

        result = evaluate_model(model, X, y)

        assert result.loss < 1e-8  # Perfect prediction
        assert "mse" in result.metrics
        assert "mae" in result.metrics

    def test_evaluation_with_metrics(self):
        """Test evaluation with custom metrics."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0]) + 0.1 * np.random.randn(50)

        result = evaluate_model(model, X, y, metrics=["mse", "mae", "r2"])

        assert "mse" in result.metrics
        assert "mae" in result.metrics
        assert "r2" in result.metrics
        assert result.metrics["r2"] > 0.9  # Good fit

    def test_evaluation_with_loss_curves(self):
        """Test evaluation with training history."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0])

        training_result = TrainingResult(
            history={"loss": [0.5, 0.3, 0.1], "val_loss": [0.6, 0.4, 0.2]},
            final_epoch=3,
            best_epoch=3,
            best_train_loss=0.1,
        )

        result = evaluate_model(model, X, y, training_result=training_result)

        assert result.loss_curves is not None
        assert "loss" in result.loss_curves

    def test_evaluation_with_benchmark(self):
        """Test evaluation with benchmark function."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0])

        def benchmark_fn(inputs):
            # Slightly different predictions
            return inputs @ np.array([1.01, 1.99])

        result = evaluate_model(model, X, y, benchmark_fn=benchmark_fn)

        assert result.pricing_error is not None
        assert result.pricing_error < 1.0  # Small difference

    def test_evaluation_with_metadata(self):
        """Test evaluation with metadata."""
        model = DummyModel(np.array([1.0, 2.0]))
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, 2.0])

        result = evaluate_model(
            model, X, y,
            metadata={"model_name": "test_model", "version": "1.0"}
        )

        assert result.metadata["model_name"] == "test_model"
        assert result.metadata["version"] == "1.0"
