"""Unit tests for rade_ml.evaluation.evaluator -- Evaluator."""
import numpy as np
import pytest
import tensorflow as tf

from src.rade_ml.core.types import EvaluationResult
from src.rade_ml.evaluation.evaluator import Evaluator


def _make_compiled_model():
    inp = tf.keras.Input(shape=(3,))
    out = tf.keras.layers.Dense(1)(inp)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def _make_test_ds(n=40, batch_size=10):
    np.random.seed(42)
    X = np.random.randn(n, 3).astype(np.float32)
    y = np.random.randn(n, 1).astype(np.float32)
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size)


class TestEvaluatorRun:
    def test_returns_evaluation_result(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)
        result = evaluator.run(_make_test_ds())
        assert isinstance(result, EvaluationResult)

    def test_contains_loss_and_metrics(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)
        result = evaluator.run(_make_test_ds())
        assert "loss" in result.metrics
        assert "mae" in result.metrics
        assert result.loss is not None

    def test_predictions_and_targets_present(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)
        result = evaluator.run(_make_test_ds(n=20))
        assert result.predictions is not None
        assert result.targets is not None
        assert result.residuals is not None
        assert len(result.predictions) == 20
        assert len(result.targets) == 20

    def test_return_predictions_false(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)
        result = evaluator.run(_make_test_ds(), return_predictions=False)
        assert result.predictions is None
        assert result.targets is None

    def test_additional_metrics(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)

        def custom_metric(y_true, y_pred):
            return float(np.mean(np.abs(y_true - y_pred)))

        result = evaluator.run(
            _make_test_ds(),
            additional_metrics={"custom_mae": custom_metric}
        )
        assert "custom_mae" in result.metrics

    def test_residual_stats_present(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)
        result = evaluator.run(_make_test_ds())
        assert "residual_mean" in result.metrics
        assert "residual_std" in result.metrics
        assert "residual_mae" in result.metrics
        assert "residual_max" in result.metrics
        assert "residual_p95" in result.metrics
        assert "residual_p99" in result.metrics

    def test_dataset_info(self):
        model = _make_compiled_model()
        evaluator = Evaluator(model)
        result = evaluator.run(_make_test_ds(n=30))
        assert result.dataset_info["samples"] == 30
        assert result.dataset_info["eval_time_seconds"] > 0
