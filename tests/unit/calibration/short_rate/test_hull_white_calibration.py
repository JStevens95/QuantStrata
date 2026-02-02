"""
Unit tests for Hull-White model calibration.

Tests the calibration of Hull-White short rate model parameters
to swaption and cap volatilities.
"""

import numpy as np
import pytest

from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    hw_zc_bond_price,
)
from src.calibration.short_rate.hull_white import (
    HullWhiteCalibrationConfig,
    HullWhiteCalibrationResult,
    calibrate_hull_white_to_swaptions,
    calibrate_hull_white_to_caps,
)


class TestHullWhiteCalibrationConfig:
    """Tests for HullWhiteCalibrationConfig."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = HullWhiteCalibrationConfig()
        
        assert config.vol_type == "normal"
        assert config.use_atm_only is True
        assert config.max_iter == 500
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HullWhiteCalibrationConfig(
            vol_type="lognormal",
            weight_by_vega=True,
            max_iter=100,
        )
        
        assert config.vol_type == "lognormal"
        assert config.weight_by_vega is True


class TestHullWhiteSwaptionCalibration:
    """Tests for Hull-White calibration to swaptions."""
    
    @pytest.fixture
    def simple_yield_curve(self):
        """Create a simple flat yield curve."""
        r = 0.03  # 3% flat rate
        
        def df(t):
            return np.exp(-r * t)
        
        return df, r
    
    @pytest.fixture
    def synthetic_swaption_vols(self, simple_yield_curve):
        """Generate synthetic swaption vols."""
        df, r0 = simple_yield_curve
        
        # Simple grid
        expiries = np.array([1.0, 2.0])
        tenors = np.array([5.0, 10.0])
        
        # Use flat vol as synthetic market
        # Typical normal vol: ~50bp
        vols = np.full((len(expiries), len(tenors)), 0.005)
        
        return vols, expiries, tenors, df, r0
    
    def test_calibration_runs(self, synthetic_swaption_vols):
        """Test that calibration runs without error."""
        vols, expiries, tenors, df, r0 = synthetic_swaption_vols
        
        result = calibrate_hull_white_to_swaptions(
            swaption_vols=vols,
            expiries=expiries,
            tenors=tenors,
            yield_curve_df=df,
            r0=r0,
            config=HullWhiteCalibrationConfig(max_iter=50),
        )
        
        assert isinstance(result, HullWhiteCalibrationResult)
        assert result.params is not None
    
    def test_calibration_result_properties(self, synthetic_swaption_vols):
        """Test calibration result properties."""
        vols, expiries, tenors, df, r0 = synthetic_swaption_vols
        
        result = calibrate_hull_white_to_swaptions(
            swaption_vols=vols,
            expiries=expiries,
            tenors=tenors,
            yield_curve_df=df,
            r0=r0,
            config=HullWhiteCalibrationConfig(max_iter=50),
        )
        
        # Check parameter bounds
        assert result.params.a > 0
        assert result.params.sigma > 0
        
        # Check error metrics
        assert result.rmse >= 0
        assert result.max_error >= 0
        
        # Check arrays
        assert result.market_vols.shape == vols.shape
        assert result.model_vols.shape == vols.shape
    
    def test_calibrated_params_reasonable(self, synthetic_swaption_vols):
        """Test that calibrated parameters are in reasonable range."""
        vols, expiries, tenors, df, r0 = synthetic_swaption_vols
        
        result = calibrate_hull_white_to_swaptions(
            swaption_vols=vols,
            expiries=expiries,
            tenors=tenors,
            yield_curve_df=df,
            r0=r0,
            config=HullWhiteCalibrationConfig(max_iter=100),
        )
        
        # Mean reversion should be small but positive
        assert 0.001 < result.params.a < 2.0
        
        # Volatility should be small (rates vol is in bp)
        assert 0.0001 < result.params.sigma < 0.1


class TestHullWhiteCapCalibration:
    """Tests for Hull-White calibration to caps."""
    
    @pytest.fixture
    def simple_yield_curve(self):
        """Create a simple flat yield curve."""
        r = 0.03
        
        def df(t):
            return np.exp(-r * t)
        
        return df, r
    
    @pytest.fixture
    def synthetic_cap_vols(self, simple_yield_curve):
        """Generate synthetic cap vols."""
        df, r0 = simple_yield_curve
        
        expiries = np.array([1.0, 2.0, 3.0])
        vols = np.array([0.004, 0.005, 0.006])  # Increasing term structure
        
        return vols, expiries, df, r0
    
    def test_cap_calibration_runs(self, synthetic_cap_vols):
        """Test that cap calibration runs without error."""
        vols, expiries, df, r0 = synthetic_cap_vols
        
        result = calibrate_hull_white_to_caps(
            cap_vols=vols,
            expiries=expiries,
            yield_curve_df=df,
            r0=r0,
            config=HullWhiteCalibrationConfig(max_iter=50),
        )
        
        assert isinstance(result, HullWhiteCalibrationResult)
        assert result.params is not None
    
    def test_cap_calibration_result_str(self, synthetic_cap_vols):
        """Test string representation of result."""
        vols, expiries, df, r0 = synthetic_cap_vols
        
        result = calibrate_hull_white_to_caps(
            cap_vols=vols,
            expiries=expiries,
            yield_curve_df=df,
            r0=r0,
            config=HullWhiteCalibrationConfig(max_iter=50),
        )
        
        s = str(result)
        assert "a (mean reversion)" in s
        assert "σ (volatility)" in s


class TestHullWhiteCalibrationWithInitialGuess:
    """Tests for calibration with initial guesses."""
    
    def test_initial_guess_used(self):
        """Test that initial guess is used."""
        r0 = 0.03
        df = lambda t: np.exp(-r0 * t)
        
        expiries = np.array([1.0, 2.0])
        tenors = np.array([5.0])
        vols = np.full((2, 1), 0.005)
        
        # Run with specific initial guess
        result = calibrate_hull_white_to_swaptions(
            swaption_vols=vols,
            expiries=expiries,
            tenors=tenors,
            yield_curve_df=df,
            r0=r0,
            initial_guess=(0.05, 0.008),
            config=HullWhiteCalibrationConfig(max_iter=10),
        )
        
        # Should run without error
        assert result is not None
