"""
Unit tests for src.q_learning.evaluation.metrics.

Tests sharpe_ratio, max_drawdown, win_rate.
"""

import pytest

from src.q_learning.evaluation.metrics import sharpe_ratio, max_drawdown, win_rate


class TestSharpeRatio:
    def test_empty_returns_zero(self):
        assert sharpe_ratio([]) == 0.0

    def test_constant_returns_zero_std(self):
        assert sharpe_ratio([1.0, 1.0, 1.0]) == 0.0

    def test_positive_mean_positive_sharpe(self):
        r = sharpe_ratio([1.0, 2.0, 3.0])
        assert r > 0

    def test_risk_free_used(self):
        r = sharpe_ratio([1.0, 2.0, 3.0], risk_free=1.5)
        assert isinstance(r, float)


class TestMaxDrawdown:
    def test_empty_zero(self):
        assert max_drawdown([]) == 0.0

    def test_monotonic_zero_drawdown(self):
        assert max_drawdown([1.0, 2.0, 3.0]) == 0.0

    def test_drawdown_captured(self):
        cum = [10.0, 12.0, 8.0, 9.0, 7.0]  # peak 12, trough 7 -> dd 5
        assert max_drawdown(cum) == 5.0


class TestWinRate:
    def test_empty_zero(self):
        assert win_rate([]) == 0.0

    def test_all_positive_one(self):
        assert win_rate([0.1, 0.2, 0.3]) == 1.0

    def test_mixed(self):
        assert win_rate([1.0, -1.0, 1.0]) == pytest.approx(2 / 3)
