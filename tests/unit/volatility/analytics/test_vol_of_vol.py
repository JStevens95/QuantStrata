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
            vol_of_implied_vol=0.05,
            vol_of_realized_vol=0.04,
            mean_implied_vol=0.20,
            mean_realized_vol=0.18,
            vol_premium=0.02,
            current_regime="normal",
        )
        
        assert metrics.vol_of_implied_vol == 0.05
        assert metrics.vol_of_realized_vol == 0.04
        assert metrics.current_regime == "normal"
    
    def test_default_values(self) -> None:
        """Test default values."""
        metrics = VolOfVolMetrics(
            vol_of_implied_vol=0.05,
        )
        
        assert metrics.vol_of_realized_vol is None
        assert metrics.current_regime is None


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
        
        assert metrics.vol_of_implied_vol > 0
        assert metrics.mean_implied_vol > 0
    
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
        
        assert metrics.vol_of_implied_vol > 0
        assert metrics.vol_of_realized_vol is not None
        assert metrics.vol_of_realized_vol > 0
    
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
        
        assert metrics.vol_of_implied_vol > 0
    
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
        
        # Premium should be positive
        assert metrics.vol_premium is not None
        assert metrics.vol_premium > 0
    
    def test_regime_detection_low(self) -> None:
        """Test regime detection for low volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # Low volatility
        implied_vols = np.ones(50) * 0.12
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.current_regime == "low"
    
    def test_regime_detection_normal(self) -> None:
        """Test regime detection for normal volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # Normal volatility
        implied_vols = np.ones(50) * 0.20
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.current_regime == "normal"
    
    def test_regime_detection_high(self) -> None:
        """Test regime detection for high volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # High volatility
        implied_vols = np.ones(50) * 0.30
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.current_regime == "high"
    
    def test_regime_detection_crisis(self) -> None:
        """Test regime detection for crisis volatility."""
        thresholds = {"low": 0.15, "high": 0.25, "crisis": 0.40}
        analyzer = VolOfVolAnalyzer(regime_thresholds=thresholds)
        
        # Crisis volatility
        implied_vols = np.ones(50) * 0.50
        
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.current_regime == "crisis"
    
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
        assert volatile_metrics.vol_of_implied_vol > stable_metrics.vol_of_implied_vol
    
    def test_short_data_handling(self) -> None:
        """Test handling of data shorter than window."""
        analyzer = VolOfVolAnalyzer(window=50)
        
        # Only 20 data points (less than window)
        implied_vols = 0.20 + 0.02 * np.random.randn(20)
        
        # Should still work (using available data)
        metrics = analyzer.analyze(implied_vols=implied_vols)
        
        assert metrics.vol_of_implied_vol >= 0
