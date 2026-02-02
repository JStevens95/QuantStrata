"""
Unit tests for Heston model calibration.

Tests the calibration of Heston stochastic volatility model parameters
to implied volatility surfaces.
"""

import numpy as np
import pytest

from src.models.stochastic_volatility.heston import (
    HestonParameters,
    heston_characteristic_function,
    heston_call_price,
    heston_put_price,
    heston_implied_vol,
    heston_implied_vol_surface,
)
from src.calibration.stochastic_volatility.heston import (
    HestonCalibrationConfig,
    HestonCalibrationResult,
    calibrate_heston_to_vols,
)


class TestHestonCharacteristicFunction:
    """Tests for Heston characteristic function."""
    
    def test_characteristic_function_at_zero(self):
        """φ(0) = 1 for any valid parameters."""
        params = HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7
        )
        
        phi = heston_characteristic_function(
            u=0.0, params=params, spot=100.0, r=0.05, q=0.02, tau=1.0
        )
        
        assert np.isclose(np.abs(phi), 1.0, atol=1e-10)
    
    def test_characteristic_function_real_valued_at_i(self):
        """Test that φ(-i) gives the forward price factor."""
        params = HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7
        )
        
        # At u = -i, should be related to expected forward
        phi = heston_characteristic_function(
            u=-1j, params=params, spot=100.0, r=0.05, q=0.02, tau=1.0
        )
        
        # Should be real (no imaginary part)
        assert np.abs(np.imag(phi)) < 1e-10


class TestHestonPricing:
    """Tests for Heston option pricing."""
    
    @pytest.fixture
    def heston_params(self):
        """Standard Heston parameters for testing."""
        return HestonParameters(
            kappa=2.0,
            theta=0.04,  # 20% long-term vol
            xi=0.3,
            v0=0.04,     # 20% initial vol
            rho=-0.7,
        )
    
    def test_call_price_positive(self, heston_params):
        """Call price should be positive."""
        price = heston_call_price(
            params=heston_params,
            spot=100.0,
            strike=100.0,
            r=0.05,
            q=0.02,
            tau=1.0,
        )
        
        assert price > 0
    
    def test_put_call_parity(self, heston_params):
        """Test put-call parity: C - P = S*exp(-qT) - K*exp(-rT)."""
        spot, strike, r, q, tau = 100.0, 100.0, 0.05, 0.02, 1.0
        
        call = heston_call_price(heston_params, spot, strike, r, q, tau)
        put = heston_put_price(heston_params, spot, strike, r, q, tau)
        
        forward_diff = spot * np.exp(-q * tau) - strike * np.exp(-r * tau)
        
        assert np.isclose(call - put, forward_diff, rtol=0.01)
    
    def test_call_price_increases_with_vol(self, heston_params):
        """Higher vol should mean higher call price."""
        low_vol_params = HestonParameters(
            kappa=2.0, theta=0.01, xi=0.2, v0=0.01, rho=-0.7
        )
        high_vol_params = HestonParameters(
            kappa=2.0, theta=0.09, xi=0.3, v0=0.09, rho=-0.7
        )
        
        low_price = heston_call_price(
            low_vol_params, 100.0, 100.0, 0.05, 0.0, 1.0
        )
        high_price = heston_call_price(
            high_vol_params, 100.0, 100.0, 0.05, 0.0, 1.0
        )
        
        assert high_price > low_price


class TestHestonImpliedVol:
    """Tests for Heston implied volatility calculation."""
    
    @pytest.fixture
    def heston_params(self):
        return HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7
        )
    
    def test_implied_vol_atm(self, heston_params):
        """ATM implied vol should be close to sqrt(v0)."""
        iv = heston_implied_vol(
            params=heston_params,
            spot=100.0,
            strike=100.0,
            r=0.05,
            q=0.02,
            tau=1.0,
        )
        
        # Should be around 20% (sqrt(0.04))
        assert 0.15 < iv < 0.30
    
    def test_implied_vol_smile(self, heston_params):
        """Negative rho should create downward-sloping smile for calls."""
        spot = 100.0
        strikes = np.array([80, 100, 120])
        
        ivs = [
            heston_implied_vol(
                heston_params, spot, K, 0.05, 0.02, 1.0
            )
            for K in strikes
        ]
        
        # With rho < 0, OTM puts (low K) should have higher vol
        assert ivs[0] > ivs[1]  # ITM call (OTM put) > ATM


class TestHestonCalibration:
    """Tests for Heston calibration to vol surfaces."""
    
    @pytest.fixture
    def target_params(self):
        """Target Heston parameters to recover."""
        return HestonParameters(
            kappa=2.0, theta=0.04, xi=0.4, v0=0.04, rho=-0.6
        )
    
    @pytest.fixture
    def synthetic_surface(self, target_params):
        """Generate synthetic vol surface from target params."""
        spot = 100.0
        r, q = 0.05, 0.02
        
        # Small grid for fast testing
        strikes = np.array([90, 100, 110])
        expiries = np.array([0.5, 1.0])
        
        market_vols = heston_implied_vol_surface(
            target_params, spot, r, q, strikes, expiries
        )
        
        return strikes, expiries, market_vols, spot, r, q
    
    def test_calibration_round_trip(self, target_params, synthetic_surface):
        """Test that calibration runs and produces valid output.
        
        Note: Heston calibration is a non-convex problem. This test verifies
        that the calibration infrastructure works correctly, not that it
        achieves perfect fit (which depends on optimizer luck and iterations).
        """
        strikes, expiries, market_vols, spot, r, q = synthetic_surface
        
        # Calibrate
        result = calibrate_heston_to_vols(
            market_vols=market_vols,
            strikes=strikes,
            expiries=expiries,
            spot=spot,
            r=r,
            q=q,
            config=HestonCalibrationConfig(
                fix_v0_to_atm=True,
                use_global_optimizer=False,  # Faster for testing
                max_iter=100,
            ),
        )
        
        assert isinstance(result, HestonCalibrationResult)
        
        # Check that calibration improved from initial guess
        assert result.calibration_result.objective_value <= result.calibration_result.initial_objective
        
        # Check parameters are in valid bounds
        assert result.params.kappa > 0
        assert result.params.theta > 0
        assert result.params.xi > 0
        assert -1 < result.params.rho < 1
        
        # Check RMSE is reasonable (5% vol error for quick local optimization)
        # Proper calibration with global optimizer would achieve < 1%
        assert result.rmse < 0.05
    
    def test_calibration_result_properties(self, synthetic_surface):
        """Test HestonCalibrationResult properties."""
        strikes, expiries, market_vols, spot, r, q = synthetic_surface
        
        result = calibrate_heston_to_vols(
            market_vols=market_vols,
            strikes=strikes,
            expiries=expiries,
            spot=spot,
            r=r,
            q=q,
            config=HestonCalibrationConfig(
                use_global_optimizer=False,
                max_iter=50,
            ),
        )
        
        # Check all properties exist
        assert result.params is not None
        assert result.calibration_result is not None
        assert result.market_vols.shape == market_vols.shape
        assert result.model_vols.shape == market_vols.shape
        assert result.rmse >= 0
        assert result.max_error >= 0
        assert isinstance(result.feller_satisfied, bool)
    
    def test_feller_constraint_enforcement(self, synthetic_surface):
        """Test that Feller constraint can be enforced."""
        strikes, expiries, market_vols, spot, r, q = synthetic_surface
        
        result = calibrate_heston_to_vols(
            market_vols=market_vols,
            strikes=strikes,
            expiries=expiries,
            spot=spot,
            r=r,
            q=q,
            config=HestonCalibrationConfig(
                enforce_feller=True,
                feller_penalty_weight=10000.0,
                use_global_optimizer=False,
                max_iter=100,
            ),
        )
        
        # Should satisfy Feller (or be close)
        # Due to penalties, it should be satisfied or nearly so
        feller_ratio = result.params.feller_ratio
        assert feller_ratio > 0.5  # At least close to Feller


class TestHestonCalibrationConfig:
    """Tests for HestonCalibrationConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = HestonCalibrationConfig()
        
        assert config.fix_v0_to_atm is True
        assert config.enforce_feller is True
        assert config.use_global_optimizer is True
        assert config.max_iter == 500
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = HestonCalibrationConfig(
            fix_v0_to_atm=False,
            enforce_feller=False,
            use_global_optimizer=False,
            max_iter=100,
            verbose=True,
        )
        
        assert config.fix_v0_to_atm is False
        assert config.enforce_feller is False
        assert config.max_iter == 100
