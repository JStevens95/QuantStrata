"""
Unit tests for volatility-of-volatility analytics module.

Tests VolOfVolAnalyzer and VolOfVolMetrics.
"""

import numpy as np
import pytest

from src.volatility.analytics.vol_of_vol import (
    VolOfVolAnalyzer,
    VolOfVolMetrics,
)


class TestVolOfVolMetrics:
    """Tests for VolOfVolMetrics dataclass."""
    
    def test_metrics_creation(self) -> None:
        """Test metrics creation."""
        metrics = VolOfVolMetrics(
            vol_of_iv=0.05,
            vol_of_rv=0.04,
            mean_iv=0.20,
            mean_rv=0.18,
            iv_rv_spread=0.02,
            vol_persistence=0.7,
            vol_mean_reversion=0.1,
            regime="normal",
            regime_score=0.5,
        )
        
        assert metrics.vol_of_iv == 0.05
        assert metrics.vol_of_rv == 0.04
        assert metrics.regime == "normal"
    
    def test_metrics_summary(self) -> None:
        """Test summary method."""
        metrics = VolOfVolMetrics(
            vol_of_iv=0.05,
            vol_of_rv=0.04,
            mean_iv=0.20,
            mean_rv=0.18,
            iv_rv_spread=0.02,
            vol_persistence=0.7,
            vol_mean_reversion=0.1,
            regime="normal",
            regime_score=0.5,
        )
        
        summary = metrics.summary()
        assert "vol_of_iv" in summary
        assert "regime" in summary


class TestVolOfVolAnalyzer:
    """Tests for VolOfVolAnalyzer."""
    
    def test_analyzer_creation(self) -> None:
        """Test analyzer creation."""
        analyzer = VolOfVolAnalyzer(window=20)
        
        assert analyzer.window == 20
        assert analyzer.annualization == 252
    
    def test_analyzer_custom_thresholds(self) -> None:
        """Test analyzer with custom regime thresholds."""
        thresholds = {"low": 0.10, "high": 0.30, "crisis": 0.50}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        assert analyzer.regime_thresholds["low"] == 0.10
    
    def test_analyze_implied_vol_only(self) -> None:
        """Test analysis with implied vol only."""
        analyzer = VolOfVolAnalyzer(window=10)
        
        # Generate synthetic implied vol series
        np.random.seed(42)
        implied_vols = 0.20 + 0.02 * np.random.randn(100)
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.vol_of_iv > 0
        assert metrics.mean_iv > 0
    
    def test_analyze_with_realized_vol(self) -> None:
        """Test analysis with both implied and realized vol."""
        analyzer = VolOfVolAnalyzer(window=10)
        
        np.random.seed(42)
        implied_vols = 0.22 + 0.03 * np.random.randn(100)
        realized_vols = 0.18 + 0.02 * np.random.randn(100)
        
        metrics = analyzer.analyze(
            implied_vols=implied_vols,
            realized_vols=realized_vols,
        )
        
        assert metrics.vol_of_iv > 0
        assert metrics.vol_of_rv is not None
        assert metrics.vol_of_rv >= 0
    
    def test_analyze_with_prices(self) -> None:
        """Test analysis with price data."""
        analyzer = VolOfVolAnalyzer(window=20)
        
        np.random.seed(42)
        returns = np.random.randn(200) * 0.01
        prices = 100 * np.cumprod(1 + returns)
        
        implied_vols = 0.20 + 0.02 * np.random.randn(200)
        
        metrics = analyzer.analyze(
            implied_vols=implied_vols,
            prices=prices,
        )
        
        assert metrics.vol_of_iv > 0
    
    def test_vol_premium_calculation(self) -> None:
        """Test vol premium (IV - RV) calculation."""
        analyzer = VolOfVolAnalyzer(window=10)
        
        # IV consistently higher than RV
        implied_vols = np.ones(100) * 0.22
        realized_vols = np.ones(100) * 0.18
        
        metrics = analyzer.analyze(
            implied_vols=implied_vols,
            realized_vols=realized_vols,
        )
        
        # IV-RV spread should be positive when IV > RV
        assert metrics.iv_rv_spread is not None
        assert metrics.iv_rv_spread > 0
    
    def test_regime_detection_low(self) -> None:
        """Test regime detection for low volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # Low volatility
        implied_vols = np.ones(50) * 0.12
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.regime == "low"
    
    def test_regime_detection_normal(self) -> None:
        """Test regime detection for normal volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # Normal volatility
        implied_vols = np.ones(50) * 0.20
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.regime == "normal"
    
    def test_regime_detection_high(self) -> None:
        """Test regime detection for high volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # High volatility
        implied_vols = np.ones(50) * 0.30
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.regime == "high"
    
    def test_regime_detection_crisis(self) -> None:
        """Test regime detection for crisis volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # Crisis volatility
        implied_vols = np.ones(50) * 0.50
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.regime == "crisis"
    
    def test_vol_of_vol_increases_in_crisis(self) -> None:
        """Test that vol-of-vol is higher during volatile periods."""
        analyzer = VolOfVolAnalyzer(window=10)
        
        np.random.seed(42)
        
        # Stable period
        stable_vols = 0.15 + 0.01 * np.random.randn(100)
        stable_metrics = analyzer.analyze(implied_vols=stable_vols)
        
        # Volatile period
        volatile_vols = 0.30 + 0.05 * np.random.randn(100)
        volatile_metrics = analyzer.analyze(implied_vols=volatile_vols)
        
        # Vol-of-vol should be higher in volatile period
        assert volatile_metrics.vol_of_iv > stable_metrics.vol_of_iv
    
    def test_short_data_handling(self) -> None:
        """Test handling of data shorter than window."""
        analyzer = VolOfVolAnalyzer(window=50)
        
        # Only 20 data points (less than window)
        implied_vols = 0.20 + 0.02 * np.random.randn(20)
        
        # Should still work (using available data)
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.vol_of_iv >= 0
