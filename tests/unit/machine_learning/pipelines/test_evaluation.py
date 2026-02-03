"""
Unit tests for src.machine_learning.pipelines.evaluation module.

Tests evaluate_model() and metric functions.
"""

import pytest
import numpy as np

# Skip entire module if TensorFlow is not available
tf = pytest.importorskip("tensorflow")

from src.machine_learning.pipelines.evaluation import (
    evaluate_model,
    METRIC_FUNCTIONS,
    _compute_mse,
    _compute_mae,
    _compute_rmse,
    _compute_mape,
    _compute_r2,
)
from src.machine_learning.core.types import EvaluationResult, TrainingResult
from src.machine_learning.core.protocols import KerasTrainableAdapter


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_model():
    """Create a simple trained model."""
    np.random.seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Quick training
    X = np.random.randn(100, 4).astype(np.float32)
    y = X.sum(axis=1, keepdims=True).astype(np.float32)
    model.fit(X, y, epochs=10, verbose=0)
    
    return KerasTrainableAdapter(model)


@pytest.fixture
def sample_data():
    """Sample evaluation data."""
    np.random.seed(42)
    X = np.random.randn(50, 4).astype(np.float32)
    y = X.sum(axis=1, keepdims=True).astype(np.float32)
    return X, y


# =============================================================================
# Metric Function Tests
# =============================================================================


class TestMetricFunctions:
    """Tests for individual metric functions."""

    def test_compute_mse(self):
        """MSE computation is correct."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        mse = _compute_mse(y_true, y_pred)
        expected = 0.25  # (0.5^2 + 0.5^2 + 0.5^2) / 3 = 0.25
        assert abs(mse - expected) < 1e-6

    def test_compute_mae(self):
        """MAE computation is correct."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 3.5])
        
        mae = _compute_mae(y_true, y_pred)
        expected = (0.5 + 0.0 + 0.5) / 3  # 0.333...
        assert abs(mae - expected) < 1e-6

    def test_compute_rmse(self):
        """RMSE computation is correct."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 4.0])
        
        rmse = _compute_rmse(y_true, y_pred)
        expected = np.sqrt((0 + 0 + 1) / 3)  # sqrt(1/3)
        assert abs(rmse - expected) < 1e-6

    def test_compute_mape(self):
        """MAPE computation is correct."""
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 220.0, 330.0])
        
        mape = _compute_mape(y_true, y_pred)
        # 10% error for each
        expected = 10.0
        assert abs(mape - expected) < 1e-4

    def test_compute_r2_perfect(self):
        """R² is 1.0 for perfect predictions."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        r2 = _compute_r2(y_true, y_pred)
        assert abs(r2 - 1.0) < 1e-6

    def test_compute_r2_baseline(self):
        """R² is 0 for mean prediction."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        
        r2 = _compute_r2(y_true, y_pred)
        assert abs(r2) < 1e-6

    def test_compute_r2_worse_than_baseline(self):
        """R² is negative for very bad predictions."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])  # Way off
        
        r2 = _compute_r2(y_true, y_pred)
        assert r2 < 0

    def test_metric_functions_dict(self):
        """METRIC_FUNCTIONS contains expected metrics."""
        assert "mse" in METRIC_FUNCTIONS
        assert "mae" in METRIC_FUNCTIONS
        assert "rmse" in METRIC_FUNCTIONS
        assert "mape" in METRIC_FUNCTIONS
        assert "r2" in METRIC_FUNCTIONS


# =============================================================================
# evaluate_model Tests
# =============================================================================


class TestEvaluateModel:
    """Tests for evaluate_model function."""

    def test_returns_evaluation_result(self, simple_model, sample_data):
        """evaluate_model returns EvaluationResult."""
        features, targets = sample_data
        
        result = evaluate_model(simple_model, features, targets)
        
        assert isinstance(result, EvaluationResult)

    def test_computes_loss(self, simple_model, sample_data):
        """Loss is computed."""
        features, targets = sample_data
        
        result = evaluate_model(simple_model, features, targets)
        
        assert result.loss >= 0

    def test_default_metrics(self, simple_model, sample_data):
        """Default metrics are mse and mae."""
        features, targets = sample_data
        
        result = evaluate_model(simple_model, features, targets)
        
        assert "mse" in result.metrics
        assert "mae" in result.metrics

    def test_custom_metrics(self, simple_model, sample_data):
        """Custom metrics can be specified."""
        features, targets = sample_data
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            metrics=["mse", "mae", "rmse", "r2"],
        )
        
        assert "mse" in result.metrics
        assert "mae" in result.metrics
        assert "rmse" in result.metrics
        assert "r2" in result.metrics

    def test_custom_loss_fn(self, simple_model, sample_data):
        """Custom loss function is used."""
        features, targets = sample_data
        
        def custom_loss(y_true, y_pred):
            return 42.0  # Fixed value for testing
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            loss_fn=custom_loss,
        )
        
        assert result.loss == 42.0

    def test_with_training_result(self, simple_model, sample_data):
        """Loss curves from training_result are included."""
        features, targets = sample_data
        
        training_result = TrainingResult(
            history={"loss": [1.0, 0.5, 0.2], "val_loss": [1.1, 0.6, 0.3]},
            final_epoch=3,
            best_epoch=3,
            best_train_loss=0.2,
        )
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            training_result=training_result,
        )
        
        assert result.loss_curves is not None
        assert "loss" in result.loss_curves
        assert "val_loss" in result.loss_curves

    def test_with_benchmark_fn(self, simple_model, sample_data):
        """Benchmark function computes pricing_error."""
        features, targets = sample_data
        
        def benchmark_fn(x):
            return np.zeros((len(x), 1))  # Always predict 0
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            benchmark_fn=benchmark_fn,
        )
        
        assert result.pricing_error is not None
        assert result.pricing_error >= 0

    def test_with_metadata(self, simple_model, sample_data):
        """Custom metadata is included."""
        features, targets = sample_data
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            metadata={"dataset": "test", "version": "1.0"},
        )
        
        assert result.metadata["dataset"] == "test"
        assert result.metadata["version"] == "1.0"

    def test_unknown_metric_warning(self, simple_model, sample_data, caplog):
        """Unknown metric logs warning."""
        features, targets = sample_data
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            metrics=["mse", "unknown_metric"],
        )
        
        # Unknown metric should not be in results
        assert "unknown_metric" not in result.metrics
        # Warning should be logged
        assert "Unknown metric" in caplog.text or "mse" in result.metrics

    def test_metrics_are_reasonable(self, simple_model, sample_data):
        """Computed metrics are reasonable values."""
        features, targets = sample_data
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            metrics=["mse", "mae", "r2"],
        )
        
        # MSE and MAE should be non-negative
        assert result.metrics["mse"] >= 0
        assert result.metrics["mae"] >= 0
        # R² should be between -inf and 1
        assert result.metrics["r2"] <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestEvaluationIntegration:
    """Integration tests for evaluation pipeline."""

    def test_full_evaluation_pipeline(self):
        """Full evaluation pipeline with trained model."""
        from src.machine_learning.models.pricing.model import create_mlp_pricer
        from src.machine_learning.data.pricing.build import build_pricing_data
        
        # Build and train model
        model = create_mlp_pricer(n_features=6, hidden_units=[32, 16])
        model.compile(optimizer='adam', loss='mse')
        
        data = build_pricing_data(n_samples=200, batch_size=32, seed=42)
        model.fit(data.train_ds, epochs=5, verbose=0)
        
        # Extract test data
        test_features = []
        test_targets = []
        for x, y in data.test_ds.unbatch():
            test_features.append(x.numpy())
            test_targets.append(y.numpy())
        test_features = np.stack(test_features)
        test_targets = np.stack(test_targets)
        
        # Evaluate
        adapter = KerasTrainableAdapter(model)
        result = evaluate_model(
            adapter,
            test_features,
            test_targets,
            metrics=["mse", "mae", "rmse", "r2"],
        )
        
        # Result should be valid
        assert result.loss >= 0
        assert all(m in result.metrics for m in ["mse", "mae", "rmse", "r2"])

    def test_evaluation_result_serializable(self, simple_model, sample_data):
        """Evaluation result can be serialized to JSON."""
        features, targets = sample_data
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            metrics=["mse", "mae", "r2"],
            metadata={"model": "test"},
        )
        
        # Should convert to dict without errors
        result_dict = result.to_dict()
        
        import json
        json_str = json.dumps(result_dict)
        restored = json.loads(json_str)
        
        assert restored["loss"] == result.loss
        assert restored["metrics"]["mse"] == result.metrics["mse"]

    def test_evaluation_summary(self, simple_model, sample_data):
        """Evaluation summary is formatted correctly."""
        features, targets = sample_data
        
        result = evaluate_model(
            simple_model,
            features,
            targets,
            metrics=["mse", "mae"],
        )
        
        summary = result.summary()
        
        assert "EVALUATION RESULTS" in summary
        assert "Loss:" in summary
        assert "mse" in summary
        assert "mae" in summary
