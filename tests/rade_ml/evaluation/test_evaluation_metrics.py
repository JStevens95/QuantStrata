"""Unit tests for rade_ml.evaluation.metrics."""
import numpy as np
import pytest

from src.rade_ml.evaluation.metrics import (
    rmse, mape, mae, mse, max_absolute_error,
    percentile_absolute_error, r_squared,
)


@pytest.fixture
def perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return y, y.copy()


@pytest.fixture
def offset_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = y_true + 1.0
    return y_true, y_pred


class TestRmse:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert rmse(y_true, y_pred) == pytest.approx(0.0)

    def test_known_offset(self, offset_predictions):
        y_true, y_pred = offset_predictions
        assert rmse(y_true, y_pred) == pytest.approx(1.0)


class TestMae:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert mae(y_true, y_pred) == pytest.approx(0.0)

    def test_known_offset(self, offset_predictions):
        y_true, y_pred = offset_predictions
        assert mae(y_true, y_pred) == pytest.approx(1.0)


class TestMse:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert mse(y_true, y_pred) == pytest.approx(0.0)

    def test_known_offset(self, offset_predictions):
        y_true, y_pred = offset_predictions
        assert mse(y_true, y_pred) == pytest.approx(1.0)


class TestMape:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert mape(y_true, y_pred) == pytest.approx(0.0, abs=1e-5)

    def test_known_offset(self):
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 22.0])
        result = mape(y_true, y_pred)
        assert result > 0.0
        assert result < 100.0


class TestMaxAbsoluteError:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert max_absolute_error(y_true, y_pred) == pytest.approx(0.0)

    def test_known_max(self):
        y_true = np.array([0.0, 0.0, 0.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert max_absolute_error(y_true, y_pred) == pytest.approx(3.0)


class TestPercentileAbsoluteError:
    def test_95th(self):
        np.random.seed(0)
        y_true = np.zeros(1000)
        y_pred = np.random.randn(1000)
        p95 = percentile_absolute_error(y_true, y_pred, percentile=95.0)
        p50 = percentile_absolute_error(y_true, y_pred, percentile=50.0)
        assert p95 > p50


class TestRSquared:
    def test_perfect(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        assert r_squared(y_true, y_pred) == pytest.approx(1.0)

    def test_mean_predictor(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.full_like(y_true, y_true.mean())
        assert r_squared(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)

    def test_constant_target(self):
        y_true = np.array([5.0, 5.0, 5.0])
        y_pred = np.array([5.0, 5.0, 5.0])
        assert r_squared(y_true, y_pred) == 1.0

    def test_negative_r2(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([10.0, 20.0, 30.0])
        assert r_squared(y_true, y_pred) < 0
