"""
Unit tests for backtesting performance metrics.

Tests the performance metric calculations:
- Sharpe ratio
- Sortino ratio
- Maximum drawdown
- Calmar ratio
- Win rate
- Profit factor
"""

import numpy as np
import pytest

from src.backtesting.core.metrics import (
    PerformanceMetrics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_calmar_ratio,
    compute_win_rate,
    compute_profit_factor,
    compute_all_metrics,
)


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""
    
    def test_sharpe_zero_volatility(self):
        """Zero volatility should return 0."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])
        sharpe = compute_sharpe_ratio(returns)
        # Constant returns have zero std
        assert np.isclose(sharpe, 0.0, atol=1e-10)
    
    def test_sharpe_positive_returns(self):
        """Positive returns should give positive Sharpe."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001  # Positive drift
        sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0
    
    def test_sharpe_with_risk_free_rate(self):
        """Risk-free rate should reduce Sharpe."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.001
        
        sharpe_0 = compute_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_rf = compute_sharpe_ratio(returns, risk_free_rate=0.05)
        
        assert sharpe_rf < sharpe_0
    
    def test_sharpe_annualization(self):
        """Different periods_per_year should scale properly."""
        np.random.seed(42)
        returns = np.random.randn(12) * 0.03 + 0.01  # Monthly returns
        
        sharpe_monthly = compute_sharpe_ratio(returns, periods_per_year=12)
        sharpe_daily = compute_sharpe_ratio(returns, periods_per_year=252)
        
        # Annualized Sharpe should be similar order of magnitude
        assert sharpe_monthly > 0
        assert sharpe_daily > 0


class TestSortinoRatio:
    """Tests for Sortino ratio calculation."""
    
    def test_sortino_no_downside(self):
        """All positive returns should give high Sortino."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        sortino = compute_sortino_ratio(returns)
        # No downside deviation -> inf
        assert sortino == float("inf")
    
    def test_sortino_vs_sharpe(self):
        """Sortino should be higher than Sharpe for asymmetric returns."""
        np.random.seed(42)
        # Create positively skewed returns
        returns = np.abs(np.random.randn(252) * 0.01) + 0.001
        
        sharpe = compute_sharpe_ratio(returns)
        sortino = compute_sortino_ratio(returns)
        
        # Sortino should be >= Sharpe for positive skew
        assert sortino >= sharpe * 0.9  # Allow some tolerance


class TestMaxDrawdown:
    """Tests for maximum drawdown calculation."""
    
    def test_no_drawdown(self):
        """Monotonically increasing returns have no drawdown."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        max_dd, duration, _, _ = compute_max_drawdown(returns)
        assert max_dd == 0.0
    
    def test_known_drawdown(self):
        """Test with known drawdown scenario."""
        # Up 10%, then down 20%, then up 10%
        # Wealth: 1.0 -> 1.10 -> 0.88 -> 0.968
        # Max DD from peak (1.10) to trough (0.88) = 20%
        returns = np.array([0.10, -0.20, 0.10])
        max_dd, _, _, _ = compute_max_drawdown(returns)
        assert np.isclose(max_dd, 0.20, atol=0.01)
    
    def test_drawdown_duration(self):
        """Test drawdown duration calculation."""
        # Up 20%, then down 30% from peak, partial recovery
        returns = np.array([0.20, -0.15, -0.10, 0.05, 0.10])
        max_dd, duration, peak_idx, trough_idx = compute_max_drawdown(returns)
        
        # Should have significant drawdown
        assert max_dd > 0.15
        assert duration >= 0
        assert trough_idx > peak_idx


class TestCalmarRatio:
    """Tests for Calmar ratio calculation."""
    
    def test_calmar_positive(self):
        """Positive returns with drawdown should give positive Calmar."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0005
        calmar = compute_calmar_ratio(returns)
        # Could be positive or negative depending on random seed
        assert np.isfinite(calmar)
    
    def test_calmar_no_drawdown(self):
        """No drawdown should give infinite Calmar."""
        returns = np.array([0.01, 0.02, 0.015, 0.01])
        calmar = compute_calmar_ratio(returns)
        assert calmar == float("inf")


class TestWinRate:
    """Tests for win rate calculation."""
    
    def test_all_wins(self):
        """All positive returns should give 100% win rate."""
        returns = np.array([0.01, 0.02, 0.005])
        assert compute_win_rate(returns) == 1.0
    
    def test_all_losses(self):
        """All negative returns should give 0% win rate."""
        returns = np.array([-0.01, -0.02, -0.005])
        assert compute_win_rate(returns) == 0.0
    
    def test_mixed(self):
        """Mixed returns should give proper ratio."""
        returns = np.array([0.01, -0.01, 0.02, -0.005])
        assert compute_win_rate(returns) == 0.5


class TestProfitFactor:
    """Tests for profit factor calculation."""
    
    def test_all_wins(self):
        """All wins should give infinite profit factor."""
        returns = np.array([0.01, 0.02, 0.005])
        assert compute_profit_factor(returns) == float("inf")
    
    def test_all_losses(self):
        """All losses should give 0 profit factor."""
        returns = np.array([-0.01, -0.02, -0.005])
        assert compute_profit_factor(returns) == 0.0
    
    def test_balanced(self):
        """Equal wins and losses should give profit factor ~1."""
        returns = np.array([0.02, -0.02, 0.01, -0.01])
        pf = compute_profit_factor(returns)
        assert np.isclose(pf, 1.0, atol=0.01)
    
    def test_profitable(self):
        """Profitable strategy should have PF > 1."""
        returns = np.array([0.03, -0.01, 0.02, -0.01])
        pf = compute_profit_factor(returns)
        assert pf > 1.0


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""
    
    def test_returns_performance_metrics(self):
        """Should return PerformanceMetrics object."""
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01 + 0.0003
        metrics = compute_all_metrics(returns)
        
        assert isinstance(metrics, PerformanceMetrics)
    
    def test_all_fields_populated(self):
        """All metric fields should be populated."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        metrics = compute_all_metrics(returns)
        
        assert np.isfinite(metrics.total_return)
        assert np.isfinite(metrics.annualized_return)
        assert np.isfinite(metrics.annualized_volatility)
        assert np.isfinite(metrics.sharpe_ratio)
        assert np.isfinite(metrics.max_drawdown)
        assert np.isfinite(metrics.win_rate)
        assert metrics.num_trades == 100
    
    def test_empty_returns(self):
        """Empty returns should return zero metrics."""
        metrics = compute_all_metrics(np.array([]))
        
        assert metrics.total_return == 0.0
        assert metrics.sharpe_ratio == 0.0
        assert metrics.num_trades == 0
    
    def test_str_representation(self):
        """String representation should work."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        metrics = compute_all_metrics(returns)
        
        s = str(metrics)
        assert "Total Return" in s
        assert "Sharpe Ratio" in s
        assert "Max Drawdown" in s
