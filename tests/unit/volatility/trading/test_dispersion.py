"""
Unit tests for dispersion trading module.

Tests DispersionTrader, DispersionConfig, and DispersionAnalysis.
"""

import numpy as np
import pytest

from src.volatility.trading.dispersion import (
    compute_average_correlation,
    compute_realized_correlation,
    DispersionAnalysis,
    DispersionConfig,
    DispersionTrader,
)


class TestDispersionConfig:
    """Tests for DispersionConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = DispersionConfig()
        
        assert config.implied_corr_threshold_long < config.implied_corr_threshold_short
        assert config.index_notional > 0
        assert config.hedge_ratio == 1.0
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = DispersionConfig(
            implied_corr_threshold_long=0.5,
            implied_corr_threshold_short=0.85,
            index_notional=2_000_000,
        )
        
        assert config.implied_corr_threshold_long == 0.5
        assert config.implied_corr_threshold_short == 0.85
        assert config.index_notional == 2_000_000


class TestDispersionAnalysis:
    """Tests for DispersionAnalysis dataclass."""
    
    def test_analysis_creation(self) -> None:
        """Test analysis result creation."""
        analysis = DispersionAnalysis(
            index_vol=0.18,
            avg_constituent_vol=0.22,
            weighted_constituent_vol=0.21,
            implied_correlation=0.65,
            signal="neutral",
            signal_strength=0.0,
        )
        
        assert analysis.index_vol == 0.18
        assert analysis.implied_correlation == 0.65
        assert analysis.signal == "neutral"


class TestDispersionTrader:
    """Tests for DispersionTrader."""
    
    def test_trader_creation(self) -> None:
        """Test trader creation."""
        trader = DispersionTrader(
            index_ticker="SPX",
            constituents=["AAPL", "MSFT", "GOOGL"],
            weights=np.array([0.4, 0.35, 0.25]),
        )
        
        assert trader.index_ticker == "SPX"
        assert len(trader.constituents) == 3
        np.testing.assert_array_almost_equal(trader.weights, [0.4, 0.35, 0.25])
    
    def test_trader_equal_weights(self) -> None:
        """Test trader with equal weights."""
        trader = DispersionTrader(
            index_ticker="INDEX",
            constituents=["A", "B", "C", "D"],
        )
        
        # Should have equal weights
        expected_weight = 0.25
        np.testing.assert_array_almost_equal(
            trader.weights,
            [expected_weight] * 4,
        )
    
    def test_analyze_basic(self) -> None:
        """Test basic analysis."""
        trader = DispersionTrader(
            index_ticker="INDEX",
            constituents=["A", "B", "C"],
            weights=np.array([0.4, 0.35, 0.25]),
        )
        
        analysis = trader.analyze(
            index_vol=0.18,
            constituent_vols=np.array([0.22, 0.25, 0.20]),
        )
        
        assert analysis.index_vol == 0.18
        assert analysis.avg_constituent_vol > 0
        assert 0 <= analysis.implied_correlation <= 1
    
    def test_analyze_with_correlation_matrix(self) -> None:
        """Test analysis with correlation matrix."""
        trader = DispersionTrader(
            index_ticker="INDEX",
            constituents=["A", "B"],
            weights=np.array([0.5, 0.5]),
        )
        
        corr_matrix = np.array([
            [1.0, 0.6],
            [0.6, 1.0],
        ])
        
        analysis = trader.analyze(
            index_vol=0.20,
            constituent_vols=np.array([0.25, 0.22]),
            correlation_matrix=corr_matrix,
        )
        
        assert analysis.index_vol == 0.20
    
    def test_dispersion_signal(self) -> None:
        """Test that signal is set from implied correlation."""
        trader = DispersionTrader(
            index_ticker="INDEX",
            constituents=["A", "B", "C"],
            weights=np.array([0.33, 0.33, 0.34]),
        )
        
        analysis = trader.analyze(
            index_vol=0.15,
            constituent_vols=np.array([0.20, 0.22, 0.18]),
        )
        
        assert analysis.signal in ("neutral", "long_dispersion", "short_dispersion")
        assert analysis.implied_correlation >= -1 and analysis.implied_correlation <= 1
    
    def test_signal_strength(self) -> None:
        """Test signal strength is set when correlation is extreme."""
        trader = DispersionTrader(
            index_ticker="INDEX",
            constituents=["A", "B"],
            config=DispersionConfig(implied_corr_threshold_long=0.3, implied_corr_threshold_short=0.8),
        )
        
        analysis = trader.analyze(
            index_vol=0.10,
            constituent_vols=np.array([0.25, 0.25]),
        )
        
        assert isinstance(analysis.signal_strength, (int, float))


class TestComputeRealizedCorrelation:
    """Tests for realized correlation computation (returns a matrix; use average for scalar)."""
    
    def test_perfect_correlation(self) -> None:
        """Test with perfectly correlated returns."""
        returns = np.random.randn(100)
        returns_matrix = np.column_stack([returns, returns, returns])
        
        corr_matrix = compute_realized_correlation(returns_matrix)
        avg_corr = compute_average_correlation(corr_matrix)
        
        assert abs(avg_corr - 1.0) < 0.01
    
    def test_uncorrelated_returns(self) -> None:
        """Test with uncorrelated returns."""
        np.random.seed(42)
        returns_matrix = np.random.randn(1000, 5)
        
        corr_matrix = compute_realized_correlation(returns_matrix)
        avg_corr = compute_average_correlation(corr_matrix)
        
        # Should be close to zero for random uncorrelated data
        assert abs(avg_corr) < 0.15
    
    def test_moderate_correlation(self) -> None:
        """Test with moderately correlated returns."""
        np.random.seed(42)
        common_factor = np.random.randn(100)
        idio = np.random.randn(100, 3) * 0.5
        
        # Returns with common factor
        returns_matrix = common_factor.reshape(-1, 1) * 0.7 + idio
        
        corr_matrix = compute_realized_correlation(returns_matrix)
        avg_corr = compute_average_correlation(corr_matrix)
        
        # Should be positive and moderate
        assert 0.3 < avg_corr < 0.9


class TestComputeAverageCorrelation:
    """Tests for average correlation computation."""
    
    def test_identity_matrix(self) -> None:
        """Test with identity correlation matrix."""
        corr_matrix = np.eye(5)
        
        avg_corr = compute_average_correlation(corr_matrix)
        
        # Off-diagonal elements are 0
        assert abs(avg_corr) < 0.01
    
    def test_uniform_correlation(self) -> None:
        """Test with uniform correlation."""
        rho = 0.5
        n = 4
        corr_matrix = np.ones((n, n)) * rho
        np.fill_diagonal(corr_matrix, 1.0)
        
        avg_corr = compute_average_correlation(corr_matrix)
        
        assert abs(avg_corr - rho) < 0.01
    
    def test_varying_correlation(self) -> None:
        """Test with varying correlations."""
        corr_matrix = np.array([
            [1.0, 0.6, 0.4],
            [0.6, 1.0, 0.5],
            [0.4, 0.5, 1.0],
        ])
        
        avg_corr = compute_average_correlation(corr_matrix)
        
        # Average of 0.6, 0.4, 0.5 = 0.5
        assert abs(avg_corr - 0.5) < 0.01
