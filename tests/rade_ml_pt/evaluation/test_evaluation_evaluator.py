"""Unit tests for rade_ml_pt.evaluation.evaluator -- Evaluator."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.rade_ml_pt.core.types import EvaluationResult
from src.rade_ml_pt.evaluation.evaluator import Evaluator


def _make_model():
    """Create a simple linear model for testing."""
    model = nn.Linear(3, 1)
    return model


def _make_test_loader(n=40, batch_size=10):
    """Create a DataLoader yielding (X, y) batches."""
    np.random.seed(42)
    X = torch.from_numpy(np.random.randn(n, 3).astype(np.float32))
    y = torch.from_numpy(np.random.randn(n, 1).astype(np.float32))
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


class TestEvaluatorRun:
    def test_returns_evaluation_result(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())
        result = evaluator.run(_make_test_loader())
        assert isinstance(result, EvaluationResult)

    def test_contains_loss_and_metrics(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())
        result = evaluator.run(_make_test_loader())
        assert "loss" in result.metrics
        assert result.loss is not None

    def test_predictions_and_targets_present(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())
        result = evaluator.run(_make_test_loader(n=20))
        assert result.predictions is not None
        assert result.targets is not None
        assert result.residuals is not None
        assert len(result.predictions) == 20
        assert len(result.targets) == 20

    def test_return_predictions_false(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())
        result = evaluator.run(_make_test_loader(), return_predictions=False)
        assert result.predictions is None
        assert result.targets is None

    def test_additional_metrics(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())

        def custom_metric(y_true, y_pred):
            return float(np.mean(np.abs(y_true - y_pred)))

        result = evaluator.run(
            _make_test_loader(),
            additional_metrics={"custom_mae": custom_metric},
        )
        assert "custom_mae" in result.metrics

    def test_residual_stats_present(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())
        result = evaluator.run(_make_test_loader())
        assert "residual_mean" in result.metrics
        assert "residual_std" in result.metrics
        assert "residual_mae" in result.metrics
        assert "residual_max" in result.metrics
        assert "residual_p95" in result.metrics
        assert "residual_p99" in result.metrics

    def test_dataset_info(self):
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=nn.MSELoss())
        result = evaluator.run(_make_test_loader(n=30))
        assert result.dataset_info["samples"] == 30
        assert result.dataset_info["eval_time_seconds"] > 0

    def test_no_loss_fn(self):
        """When no loss_fn is provided, loss should be None."""
        model = _make_model()
        evaluator = Evaluator(model, loss_fn=None)
        result = evaluator.run(_make_test_loader())
        assert result.loss is None
        assert "loss" not in result.metrics
