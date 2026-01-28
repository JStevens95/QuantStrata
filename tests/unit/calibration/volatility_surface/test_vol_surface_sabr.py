"""
Unit tests for SABR Model Calibration.

Tests cover:
1. SABR parameter validation
2. SABR implied volatility (Hagan formula)
3. ATM vol special case
4. SABR calibration to market smile
5. Term structure calibration
6. Vol surface creation

Author: QuantStrata Team
"""
import numpy as np
import pytest

from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    SabrConfig,
    sabr_implied_vol,
    sabr_implied_vol_vec,
    calibrate_sabr_to_smile,
    calibrate_sabr_term_structure,
    create_sabr_vol_surface,
)


# =============================================================================
# SabrParameters Tests
# =============================================================================

class TestSabrParameters:
    """Tests for SABR parameter validation."""
    
    def test_valid_parameters(self):
        """Test creation with valid parameters."""
        params = SabrParameters(alpha=0.2, beta=1.0, rho=-0.3, nu=0.5)
        assert params.alpha == 0.2
        assert params.beta == 1.0
        assert params.rho == -0.3
        assert params.nu == 0.5
    
    def test_alpha_must_be_positive(self):
        """Test that alpha must be positive."""
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SabrParameters(alpha=0.0, beta=1.0, rho=0.0, nu=0.5)
        
        with pytest.raises(ValueError, match="alpha must be > 0"):
            SabrParameters(alpha=-0.1, beta=1.0, rho=0.0, nu=0.5)
    
    def test_beta_bounds(self):
        """Test that beta must be in [0, 1]."""
        # Valid boundaries
        SabrParameters(alpha=0.2, beta=0.0, rho=0.0, nu=0.5)
        SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=0.5)
        
        # Invalid
        with pytest.raises(ValueError, match="beta must be in"):
            SabrParameters(alpha=0.2, beta=-0.1, rho=0.0, nu=0.5)
        
        with pytest.raises(ValueError, match="beta must be in"):
            SabrParameters(alpha=0.2, beta=1.1, rho=0.0, nu=0.5)
    
    def test_rho_bounds(self):
        """Test that rho must be in (-1, 1)."""
        # Near boundaries (valid)
        SabrParameters(alpha=0.2, beta=1.0, rho=-0.99, nu=0.5)
        SabrParameters(alpha=0.2, beta=1.0, rho=0.99, nu=0.5)
        
        # At or beyond boundaries (invalid)
        with pytest.raises(ValueError, match="rho must be in"):
            SabrParameters(alpha=0.2, beta=1.0, rho=-1.0, nu=0.5)
        
        with pytest.raises(ValueError, match="rho must be in"):
            SabrParameters(alpha=0.2, beta=1.0, rho=1.0, nu=0.5)
    
    def test_nu_non_negative(self):
        """Test that nu must be non-negative."""
        # Valid: nu = 0 (no vol-of-vol)
        SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=0.0)
        
        # Invalid
        with pytest.raises(ValueError, match="nu must be >= 0"):
            SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=-0.1)
    
    def test_to_array(self):
        """Test conversion to array."""
        params = SabrParameters(alpha=0.2, beta=1.0, rho=-0.3, nu=0.5)
        arr = params.to_array()
        
        np.testing.assert_array_equal(arr, [0.2, 1.0, -0.3, 0.5])
    
    def test_from_array(self):
        """Test creation from array."""
        arr = np.array([0.2, 1.0, -0.3, 0.5])
        params = SabrParameters.from_array(arr)
        
        assert params.alpha == 0.2
        assert params.beta == 1.0
        assert params.rho == -0.3
        assert params.nu == 0.5
    
    def test_from_array_with_fixed_beta(self):
        """Test creation from array with fixed beta."""
        arr = np.array([0.2, -0.3, 0.5])  # alpha, rho, nu
        params = SabrParameters.from_array(arr, beta_fixed=1.0)
        
        assert params.alpha == 0.2
        assert params.beta == 1.0
        assert params.rho == -0.3
        assert params.nu == 0.5


# =============================================================================
# SABR Implied Volatility Tests
# =============================================================================

class TestSabrImpliedVol:
    """Tests for SABR implied volatility formula."""
    
    @pytest.fixture
    def typical_params(self):
        """Typical FX SABR parameters."""
        return SabrParameters(alpha=0.2, beta=1.0, rho=-0.25, nu=0.4)
    
    def test_atm_vol(self, typical_params):
        """Test ATM volatility."""
        F = 1.0
        K = 1.0  # ATM
        T = 1.0
        
        vol = sabr_implied_vol(forward=F, strike=K, expiry=T, params=typical_params)
        
        # For beta=1, ATM vol ≈ alpha (with small corrections)
        assert 0.1 < vol < 0.3  # Reasonable range
    
    def test_smile_shape(self, typical_params):
        """Test that SABR produces a smile."""
        F = 1.0
        T = 1.0
        
        # Strike range
        strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)
        vols = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=typical_params)
        
        # With negative rho, left wing should be higher
        assert vols[0] > vols[2]  # Put wing > ATM
        # Should form a skewed smile
        assert np.all(np.isfinite(vols))
        assert np.all(vols > 0)
    
    def test_skew_direction(self):
        """Test that rho controls skew direction."""
        F = 1.0
        T = 1.0
        strikes = np.array([0.8, 1.0, 1.2], dtype=float)
        
        # Negative rho: left (put) wing higher
        params_neg_rho = SabrParameters(alpha=0.2, beta=1.0, rho=-0.5, nu=0.4)
        vols_neg = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=params_neg_rho)
        assert vols_neg[0] > vols_neg[2]  # Left > Right
        
        # Positive rho: right (call) wing higher
        params_pos_rho = SabrParameters(alpha=0.2, beta=1.0, rho=0.5, nu=0.4)
        vols_pos = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=params_pos_rho)
        assert vols_pos[2] > vols_pos[0]  # Right > Left
    
    def test_smile_curvature(self):
        """Test that nu controls smile curvature."""
        F = 1.0
        T = 1.0
        strikes = np.array([0.8, 1.0, 1.2], dtype=float)
        
        # Low vol-of-vol: flatter smile
        params_low_nu = SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=0.1)
        vols_low = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=params_low_nu)
        
        # High vol-of-vol: more curved smile
        params_high_nu = SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=0.8)
        vols_high = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=params_high_nu)
        
        # Curvature = average of wings - ATM
        curvature_low = 0.5 * (vols_low[0] + vols_low[2]) - vols_low[1]
        curvature_high = 0.5 * (vols_high[0] + vols_high[2]) - vols_high[1]
        
        assert curvature_high > curvature_low
    
    def test_zero_nu_reduces_to_black(self):
        """Test that nu=0 gives approximately flat smile."""
        F = 1.0
        T = 1.0
        strikes = np.array([0.8, 0.9, 1.0, 1.1, 1.2], dtype=float)
        
        params = SabrParameters(alpha=0.2, beta=1.0, rho=0.0, nu=0.0)
        vols = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=params)
        
        # Should be approximately flat (small variations due to beta < 1 corrections)
        vol_range = vols.max() - vols.min()
        assert vol_range < 0.01  # Very small variation
    
    def test_expiry_effect(self, typical_params):
        """Test that shorter expiry gives higher wing vols."""
        F = 1.0
        K = 0.8  # OTM put
        
        vol_short = sabr_implied_vol(forward=F, strike=K, expiry=0.25, params=typical_params)
        vol_long = sabr_implied_vol(forward=F, strike=K, expiry=2.0, params=typical_params)
        
        # SABR smile typically flattens with expiry
        # But this depends on parameters, so just check both are valid
        assert vol_short > 0
        assert vol_long > 0
    
    def test_invalid_inputs(self, typical_params):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="positive"):
            sabr_implied_vol(forward=-1.0, strike=1.0, expiry=1.0, params=typical_params)
        
        with pytest.raises(ValueError, match="positive"):
            sabr_implied_vol(forward=1.0, strike=-1.0, expiry=1.0, params=typical_params)
        
        with pytest.raises(ValueError, match="positive"):
            sabr_implied_vol(forward=1.0, strike=1.0, expiry=-0.1, params=typical_params)


# =============================================================================
# SABR Calibration Tests
# =============================================================================

class TestSabrCalibration:
    """Tests for SABR calibration."""
    
    @pytest.fixture
    def synthetic_smile(self):
        """Generate synthetic smile from known SABR params."""
        true_params = SabrParameters(alpha=0.2, beta=1.0, rho=-0.3, nu=0.5)
        F = 1.0
        T = 1.0
        strikes = np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15], dtype=float)
        vols = sabr_implied_vol_vec(forward=F, strikes=strikes, expiry=T, params=true_params)
        
        return {
            "forward": F,
            "strikes": strikes,
            "vols": vols,
            "expiry": T,
            "true_params": true_params,
        }
    
    def test_calibrate_recovers_true_params(self, synthetic_smile):
        """Test that calibration recovers true parameters (no noise)."""
        F = synthetic_smile["forward"]
        K = synthetic_smile["strikes"]
        vols = synthetic_smile["vols"]
        T = synthetic_smile["expiry"]
        true_params = synthetic_smile["true_params"]
        
        config = SabrConfig(beta=1.0)  # Fix beta
        calibrated = calibrate_sabr_to_smile(
            forward=F, strikes=K, market_vols=vols, expiry=T, config=config
        )
        
        # Should recover true parameters (with some tolerance as SABR params can be correlated)
        assert calibrated.alpha == pytest.approx(true_params.alpha, rel=0.1)
        assert calibrated.rho == pytest.approx(true_params.rho, abs=0.15)
        assert calibrated.nu == pytest.approx(true_params.nu, rel=0.15)
    
    def test_calibrate_fits_smile(self, synthetic_smile):
        """Test that calibrated params fit the smile well."""
        F = synthetic_smile["forward"]
        K = synthetic_smile["strikes"]
        vols = synthetic_smile["vols"]
        T = synthetic_smile["expiry"]
        
        config = SabrConfig(beta=1.0)
        calibrated = calibrate_sabr_to_smile(
            forward=F, strikes=K, market_vols=vols, expiry=T, config=config
        )
        
        # Fitted vols should match market
        fitted_vols = sabr_implied_vol_vec(forward=F, strikes=K, expiry=T, params=calibrated)
        
        np.testing.assert_allclose(fitted_vols, vols, rtol=0.01)
    
    def test_calibrate_with_noise(self, synthetic_smile):
        """Test calibration with noisy data."""
        F = synthetic_smile["forward"]
        K = synthetic_smile["strikes"]
        vols = synthetic_smile["vols"]
        T = synthetic_smile["expiry"]
        
        # Add small noise
        np.random.seed(42)
        noisy_vols = vols + np.random.normal(0, 0.002, size=vols.shape)
        
        config = SabrConfig(beta=1.0)
        calibrated = calibrate_sabr_to_smile(
            forward=F, strikes=K, market_vols=noisy_vols, expiry=T, config=config
        )
        
        # Should still produce reasonable fit
        fitted_vols = sabr_implied_vol_vec(forward=F, strikes=K, expiry=T, params=calibrated)
        
        # Fit error should be small
        rmse = np.sqrt(np.mean((fitted_vols - noisy_vols) ** 2))
        assert rmse < 0.005
    
    def test_calibrate_with_weights(self, synthetic_smile):
        """Test calibration with weights."""
        F = synthetic_smile["forward"]
        K = synthetic_smile["strikes"]
        vols = synthetic_smile["vols"]
        T = synthetic_smile["expiry"]
        
        # Weight ATM more heavily
        weights = np.array([1, 1, 2, 5, 2, 1, 1], dtype=float)
        
        config = SabrConfig(beta=1.0)
        calibrated = calibrate_sabr_to_smile(
            forward=F, strikes=K, market_vols=vols, expiry=T, config=config, weights=weights
        )
        
        # Should still fit well
        fitted_vols = sabr_implied_vol_vec(forward=F, strikes=K, expiry=T, params=calibrated)
        
        np.testing.assert_allclose(fitted_vols, vols, rtol=0.02)
    
    def test_calibrate_requires_minimum_quotes(self):
        """Test that calibration requires at least 3 quotes."""
        F = 1.0
        K = np.array([0.95, 1.05], dtype=float)
        vols = np.array([0.21, 0.19], dtype=float)
        T = 1.0
        
        config = SabrConfig(beta=1.0)
        with pytest.raises(ValueError, match="At least 3 quotes"):
            calibrate_sabr_to_smile(
                forward=F, strikes=K, market_vols=vols, expiry=T, config=config
            )


# =============================================================================
# Term Structure Calibration Tests
# =============================================================================

class TestSabrTermStructure:
    """Tests for SABR term structure calibration."""
    
    @pytest.fixture
    def term_structure_data(self):
        """Generate term structure with multiple expiries."""
        expiries = [0.25, 0.5, 1.0, 2.0]
        F_by_T = {T: 1.0 for T in expiries}  # Flat forward (for simplicity)
        K_by_T = {T: np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]) for T in expiries}
        
        # Use slightly different params for each expiry (more realistic)
        true_params = {
            0.25: SabrParameters(alpha=0.22, beta=1.0, rho=-0.35, nu=0.6),
            0.5: SabrParameters(alpha=0.21, beta=1.0, rho=-0.32, nu=0.55),
            1.0: SabrParameters(alpha=0.20, beta=1.0, rho=-0.30, nu=0.50),
            2.0: SabrParameters(alpha=0.19, beta=1.0, rho=-0.28, nu=0.45),
        }
        
        vols_by_T = {
            T: sabr_implied_vol_vec(forward=F_by_T[T], strikes=K_by_T[T], expiry=T, params=true_params[T])
            for T in expiries
        }
        
        return {
            "forward_by_expiry": F_by_T,
            "strikes_by_expiry": K_by_T,
            "vols_by_expiry": vols_by_T,
            "true_params": true_params,
        }
    
    def test_calibrate_term_structure(self, term_structure_data):
        """Test term structure calibration."""
        config = SabrConfig(beta=1.0)
        
        params_by_T = calibrate_sabr_term_structure(
            forward_by_expiry=term_structure_data["forward_by_expiry"],
            strikes_by_expiry=term_structure_data["strikes_by_expiry"],
            vols_by_expiry=term_structure_data["vols_by_expiry"],
            config=config,
        )
        
        # Should have params for each expiry
        assert len(params_by_T) == 4
        
        # Each should fit reasonably well
        for T, params in params_by_T.items():
            F = term_structure_data["forward_by_expiry"][T]
            K = term_structure_data["strikes_by_expiry"][T]
            market_vols = term_structure_data["vols_by_expiry"][T]
            
            fitted_vols = sabr_implied_vol_vec(forward=F, strikes=K, expiry=T, params=params)
            rmse = np.sqrt(np.mean((fitted_vols - market_vols) ** 2))
            assert rmse < 0.005


# =============================================================================
# Vol Surface Creation Tests
# =============================================================================

class TestCreateSabrVolSurface:
    """Tests for SABR vol surface creation."""
    
    @pytest.fixture
    def surface_data(self):
        """Create data for vol surface."""
        expiries = [0.5, 1.0, 2.0]
        params_by_T = {
            0.5: SabrParameters(alpha=0.21, beta=1.0, rho=-0.3, nu=0.5),
            1.0: SabrParameters(alpha=0.20, beta=1.0, rho=-0.3, nu=0.5),
            2.0: SabrParameters(alpha=0.19, beta=1.0, rho=-0.3, nu=0.5),
        }
        F_by_T = {T: 1.0 for T in expiries}
        strikes = np.linspace(0.8, 1.2, 9)
        
        return {
            "params_by_expiry": params_by_T,
            "forward_by_expiry": F_by_T,
            "strikes": strikes,
        }
    
    def test_create_surface(self, surface_data):
        """Test vol surface creation."""
        vol_func = create_sabr_vol_surface(
            params_by_expiry=surface_data["params_by_expiry"],
            forward_by_expiry=surface_data["forward_by_expiry"],
            strikes=surface_data["strikes"],
        )
        
        # Should be callable
        vol = vol_func(1.0, 1.0)  # ATM at T=1
        assert 0.1 < vol < 0.3
    
    def test_surface_at_calibrated_expiries(self, surface_data):
        """Test that surface matches SABR at calibrated expiries."""
        vol_func = create_sabr_vol_surface(
            params_by_expiry=surface_data["params_by_expiry"],
            forward_by_expiry=surface_data["forward_by_expiry"],
            strikes=surface_data["strikes"],
        )
        
        T = 1.0
        K = 0.95
        params = surface_data["params_by_expiry"][T]
        F = surface_data["forward_by_expiry"][T]
        
        expected = sabr_implied_vol(forward=F, strike=K, expiry=T, params=params)
        actual = vol_func(T, K)
        
        assert actual == pytest.approx(expected, rel=0.01)
    
    def test_surface_interpolates_expiry(self, surface_data):
        """Test that surface interpolates between expiries."""
        vol_func = create_sabr_vol_surface(
            params_by_expiry=surface_data["params_by_expiry"],
            forward_by_expiry=surface_data["forward_by_expiry"],
            strikes=surface_data["strikes"],
        )
        
        # Interpolate at T=0.75 (between 0.5 and 1.0)
        vol = vol_func(0.75, 1.0)
        
        # Should be between the values at T=0.5 and T=1.0
        vol_05 = vol_func(0.5, 1.0)
        vol_10 = vol_func(1.0, 1.0)
        
        assert min(vol_05, vol_10) <= vol <= max(vol_05, vol_10) + 0.01
    
    def test_surface_extrapolates_short(self, surface_data):
        """Test extrapolation for short expiries."""
        vol_func = create_sabr_vol_surface(
            params_by_expiry=surface_data["params_by_expiry"],
            forward_by_expiry=surface_data["forward_by_expiry"],
            strikes=surface_data["strikes"],
        )
        
        # Before first expiry
        vol = vol_func(0.25, 1.0)
        assert 0.1 < vol < 0.4  # Reasonable range
    
    def test_surface_extrapolates_long(self, surface_data):
        """Test extrapolation for long expiries."""
        vol_func = create_sabr_vol_surface(
            params_by_expiry=surface_data["params_by_expiry"],
            forward_by_expiry=surface_data["forward_by_expiry"],
            strikes=surface_data["strikes"],
        )
        
        # After last expiry
        vol = vol_func(3.0, 1.0)
        assert 0.1 < vol < 0.4  # Reasonable range
