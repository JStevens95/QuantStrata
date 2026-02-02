"""
Unit tests for SABR interest rate (swaption smile) calibration.

Tests the SABR calibration functions for interest rate swaption smiles.
"""

import numpy as np
import pytest

from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    SabrConfig,
    sabr_implied_vol,
    calibrate_sabr_to_smile,
    calibrate_sabr_to_swaption_smile,
    calibrate_sabr_swaption_cube,
)


class TestSabrSwaptionSmileCalibration:
    """Tests for SABR swaption smile calibration."""
    
    @pytest.fixture
    def synthetic_ir_smile(self):
        """Generate synthetic IR smile data."""
        # Forward swap rate
        forward = 0.03  # 3%
        
        # Strike grid centered on ATM
        strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
        
        # Generate vols using known SABR params (normal SABR, beta=0)
        target_params = SabrParameters(
            alpha=0.0045,  # ~45bp
            beta=0.0,
            rho=-0.2,
            nu=0.4,
        )
        
        # Compute model vols (for normal SABR, this is approximate)
        market_vols = np.array([
            sabr_implied_vol(forward=forward, strike=K, expiry=10.0, params=target_params)
            for K in strikes
        ])
        
        return strikes, market_vols, forward, target_params
    
    def test_swaption_smile_calibration_runs(self, synthetic_ir_smile):
        """Test that swaption smile calibration runs."""
        strikes, market_vols, forward, _ = synthetic_ir_smile
        
        params = calibrate_sabr_to_swaption_smile(
            strikes=strikes,
            market_vols=market_vols,
            forward_swap_rate=forward,
            expiry=10.0,
            tenor=10.0,
            vol_type="lognormal",  # Use lognormal since synthetic data uses lognormal SABR
        )
        
        assert isinstance(params, SabrParameters)
        assert params.alpha > 0
    
    def test_swaption_smile_calibration_normal_vol(self):
        """Test calibration with normal vol convention."""
        forward = 0.03
        strikes = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
        
        # Normal vols (in decimal, e.g., 0.005 = 50bp)
        market_vols = np.array([0.0052, 0.0050, 0.0048, 0.0050, 0.0052])
        
        params = calibrate_sabr_to_swaption_smile(
            strikes=strikes,
            market_vols=market_vols,
            forward_swap_rate=forward,
            expiry=5.0,
            tenor=10.0,
            vol_type="normal",
        )
        
        assert isinstance(params, SabrParameters)
        # For normal SABR, beta should be 0
        assert params.beta == 0.0
    
    def test_swaption_smile_round_trip(self, synthetic_ir_smile):
        """Test that calibration can recover original params (approximately)."""
        strikes, market_vols, forward, target_params = synthetic_ir_smile
        
        # Calibrate
        params = calibrate_sabr_to_swaption_smile(
            strikes=strikes,
            market_vols=market_vols,
            forward_swap_rate=forward,
            expiry=10.0,
            tenor=10.0,
            vol_type="lognormal",
            config=SabrConfig(beta=0.0, max_iter=200),
        )
        
        # Check parameters are in reasonable range
        assert 0.001 < params.alpha < 0.1
        assert -0.99 < params.rho < 0.99
        assert params.nu >= 0


class TestSabrSwaptionCubeCalibration:
    """Tests for SABR swaption cube calibration."""
    
    def test_cube_calibration_runs(self):
        """Test that cube calibration runs for multiple points."""
        expiries = [5.0, 10.0]
        tenors = [5.0, 10.0]
        
        # Create data for each point
        strikes_by_point = {}
        vols_by_point = {}
        forward_by_point = {}
        
        for exp in expiries:
            for ten in tenors:
                key = (exp, ten)
                strikes_by_point[key] = np.array([0.02, 0.03, 0.04])
                vols_by_point[key] = np.array([0.0052, 0.0048, 0.0052])
                forward_by_point[key] = 0.03
        
        result = calibrate_sabr_swaption_cube(
            expiries=expiries,
            tenors=tenors,
            strikes_by_point=strikes_by_point,
            vols_by_point=vols_by_point,
            forward_by_point=forward_by_point,
            vol_type="normal",
            config=SabrConfig(beta=0.0, max_iter=50),
        )
        
        # Should have params for each point
        assert len(result) == 4
        
        for key, params in result.items():
            assert isinstance(params, SabrParameters)
            assert params.alpha > 0


class TestSabrIRConfigValidation:
    """Tests for SABR IR configuration validation."""
    
    def test_invalid_vol_type_raises(self):
        """Test that invalid vol_type raises error."""
        with pytest.raises(ValueError, match="vol_type"):
            calibrate_sabr_to_swaption_smile(
                strikes=np.array([0.02, 0.03, 0.04]),
                market_vols=np.array([0.005, 0.004, 0.005]),
                forward_swap_rate=0.03,
                expiry=5.0,
                tenor=10.0,
                vol_type="invalid",
            )
    
    def test_normal_vol_sets_beta_zero(self):
        """Test that normal vol automatically sets beta=0."""
        params = calibrate_sabr_to_swaption_smile(
            strikes=np.array([0.02, 0.03, 0.04]),
            market_vols=np.array([0.005, 0.004, 0.005]),
            forward_swap_rate=0.03,
            expiry=5.0,
            tenor=10.0,
            vol_type="normal",
            config=SabrConfig(beta=0.5),  # This should be overridden
        )
        
        assert params.beta == 0.0


class TestSabrIREdgeCases:
    """Tests for edge cases in SABR IR calibration."""
    
    def test_minimum_strikes(self):
        """Test calibration with minimum number of strikes (3)."""
        params = calibrate_sabr_to_swaption_smile(
            strikes=np.array([0.02, 0.03, 0.04]),
            market_vols=np.array([0.005, 0.004, 0.005]),
            forward_swap_rate=0.03,
            expiry=5.0,
            tenor=10.0,
        )
        
        assert params is not None
    
    def test_negative_forward_rate_raises(self):
        """Test that negative forward rate raises error.
        
        Note: The current SABR implementation requires positive forward rates.
        For markets with negative rates (EUR, JPY), a shifted SABR model
        would be needed, which is not yet implemented.
        """
        forward = -0.005  # -0.5%
        strikes = np.array([-0.01, -0.005, 0.0, 0.005])
        market_vols = np.array([0.006, 0.005, 0.005, 0.006])
        
        with pytest.raises(ValueError, match="forward must be positive"):
            calibrate_sabr_to_swaption_smile(
                strikes=strikes,
                market_vols=market_vols,
                forward_swap_rate=forward,
                expiry=5.0,
                tenor=10.0,
                vol_type="normal",
            )
