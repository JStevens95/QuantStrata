"""Tests for m_learning.evaluation.delta_hedging_backtest."""

import numpy as np
import pytest

from src.m_learning.data.delta_hedging import simulate_hedging_path
from src.m_learning.evaluation.delta_hedging_backtest import (
    BacktestResult,
    backtest_summary_stats,
    run_delta_hedging_backtest,
    run_single_path_backtest,
)


class TestRunSinglePathBacktest:
    def test_no_cost_bsm_hedge_small_variance(self):
        path = simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 50, seed=42)
        pnl, cost = run_single_path_backtest(
            path, K=100, option_type=1, cost_rate=0.0, delta_sequence=path.delta
        )
        assert cost == 0.0
        assert np.isfinite(pnl)
        # With BSM delta and no cost, PnL should be small (discretisation error)
        assert abs(pnl) < 2.0

    def test_with_cost_increases_total_cost(self):
        path = simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 20, seed=42)
        _, cost_zero = run_single_path_backtest(
            path, K=100, option_type=1, cost_rate=0.0, delta_sequence=path.delta
        )
        _, cost_pos = run_single_path_backtest(
            path, K=100, option_type=1, cost_rate=0.01, delta_sequence=path.delta
        )
        assert cost_pos >= cost_zero
        assert cost_pos > 0


class TestRunDeltaHedgingBacktest:
    def test_bsm_only(self):
        paths = [
            simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 10, seed=i)
            for i in range(5)
        ]
        result = run_delta_hedging_backtest(
            paths, K=100, T=1, r=0.05, sigma=0.2, option_type=1, cost_rate=0
        )
        assert result.terminal_pnl_bsm.shape == (5,)
        assert result.cost_bsm.shape == (5,)
        assert result.terminal_pnl_ml is None
        assert result.cost_ml is None

    def test_bsm_and_ml(self):
        paths = [
            simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 8, seed=i)
            for i in range(3)
        ]

        def ml_delta_fn(features):
            return np.full(features.shape[0], 0.5)  # constant delta

        result = run_delta_hedging_backtest(
            paths, K=100, T=1, r=0.05, sigma=0.2, option_type=1,
            cost_rate=0.001, ml_delta_fn=ml_delta_fn
        )
        assert result.terminal_pnl_ml is not None
        assert result.terminal_pnl_ml.shape == (3,)
        assert result.cost_ml is not None


class TestBacktestSummaryStats:
    def test_bsm_only_keys(self):
        paths = [
            simulate_hedging_path(100, 100, 1.0, 0.05, 0.2, 1, 10, seed=i)
            for i in range(4)
        ]
        result = run_delta_hedging_backtest(
            paths, K=100, T=1, r=0.05, sigma=0.2, option_type=1, cost_rate=0
        )
        stats = backtest_summary_stats(result, var_percentile=5.0)
        assert "mean_pnl_bsm" in stats
        assert "std_pnl_bsm" in stats
        assert "var_bsm" in stats
        assert "cvar_bsm" in stats
        assert "mean_cost_bsm" in stats
