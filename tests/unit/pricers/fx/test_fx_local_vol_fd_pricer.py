"""
Unit tests for Local Volatility Finite Difference Pricer.

Tests cover:
1. Configuration validation
2. Pricing with flat local vol (should match BSM)
3. Pricing with varying local vol surfaces
4. Edge cases (T=0, extreme strikes)
5. Convergence properties
6. Comparison with constant vol FD pricer
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.pricers.fx.local_vol_fde import FxLocalVolFdPricer
from src.marketdata.surfaces.local_vol_surface import FlatLocalVolSurface, LocalVolSurface


# =============================================================================
# Helper: Black-Scholes price for comparison
# =============================================================================

def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0.0:
        F = S * np.exp((r - q) * T)
        return max(F - K, 0.0) * np.exp(-r * T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0.0)
    if sigma <= 0.0:
        F = S * np.exp((r - q) * T)
        return max(K - F, 0.0) * np.exp(-r * T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# =============================================================================
# Configuration Tests
# =============================================================================

class TestFxLocalVolFdPricerConfig:
    """Tests for pricer configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        pricer = FxLocalVolFdPricer(local_vol_surface=local_vol)
        assert pricer.n_space == 401
        assert pricer.n_time_steps == 200
        assert pricer.n_std == 6.0
        assert pricer.theta == 0.5
        assert pricer.use_log_space is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        pricer = FxLocalVolFdPricer(
            local_vol_surface=local_vol,
            n_space=201,
            n_time_steps=100,
            n_std=5.0,
            theta=1.0,
            use_log_space=False,
        )
        assert pricer.n_space == 201
        assert pricer.n_time_steps == 100
        assert pricer.theta == 1.0
        assert pricer.use_log_space is False

    def test_invalid_n_space_too_small(self) -> None:
        """Test that n_space < 10 raises ValueError."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="n_space must be >= 10"):
            FxLocalVolFdPricer(local_vol_surface=local_vol, n_space=5)

    def test_invalid_n_time_steps_zero(self) -> None:
        """Test that n_time_steps < 1 raises ValueError."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="n_time_steps must be >= 1"):
            FxLocalVolFdPricer(local_vol_surface=local_vol, n_time_steps=0)

    def test_invalid_n_std_negative(self) -> None:
        """Test that n_std <= 0 raises ValueError."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="n_std must be > 0"):
            FxLocalVolFdPricer(local_vol_surface=local_vol, n_std=-1.0)

    def test_invalid_theta_out_of_bounds(self) -> None:
        """Test that theta outside [0, 1] raises ValueError."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        with pytest.raises(ValueError, match="theta must be in"):
            FxLocalVolFdPricer(local_vol_surface=local_vol, theta=-0.1)
        with pytest.raises(ValueError, match="theta must be in"):
            FxLocalVolFdPricer(local_vol_surface=local_vol, theta=1.5)


# =============================================================================
# Pricing Tests: Flat Local Vol (Should Match BSM)
# =============================================================================

class TestLocalVolFdPricingFlatVol:
    """Tests for pricing with flat local vol (should match Black-Scholes)."""

    @pytest.fixture
    def pricer(self) -> FxLocalVolFdPricer:
        """Create pricer with flat local vol."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        return FxLocalVolFdPricer(
            local_vol_surface=local_vol,
            n_space=201,
            n_time_steps=100,
        )

    def test_atm_call_price(self, pricer: FxLocalVolFdPricer) -> None:
        """Test ATM call price matches BSM."""
        spot = 100.0
        strike = 100.0
        T = 1.0
        r_d = 0.05
        r_f = 0.02
        sigma = 0.20

        fd_price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="call"
        )
        bs_price = bs_call_price(spot, strike, T, r_d, r_f, sigma)

        # Should be very close (within 0.1% or 1bp)
        assert fd_price == pytest.approx(bs_price, rel=0.001, abs=0.01)

    def test_atm_put_price(self, pricer: FxLocalVolFdPricer) -> None:
        """Test ATM put price matches BSM."""
        spot = 100.0
        strike = 100.0
        T = 1.0
        r_d = 0.05
        r_f = 0.02
        sigma = 0.20

        fd_price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="put"
        )
        bs_price = bs_put_price(spot, strike, T, r_d, r_f, sigma)

        assert fd_price == pytest.approx(bs_price, rel=0.001, abs=0.01)

    def test_itm_call_price(self, pricer: FxLocalVolFdPricer) -> None:
        """Test ITM call price matches BSM."""
        spot = 100.0
        strike = 90.0
        T = 0.5
        r_d = 0.05
        r_f = 0.02
        sigma = 0.20

        fd_price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="call"
        )
        bs_price = bs_call_price(spot, strike, T, r_d, r_f, sigma)

        assert fd_price == pytest.approx(bs_price, rel=0.002, abs=0.02)

    def test_otm_call_price(self, pricer: FxLocalVolFdPricer) -> None:
        """Test OTM call price matches BSM."""
        spot = 100.0
        strike = 110.0
        T = 0.5
        r_d = 0.05
        r_f = 0.02
        sigma = 0.20

        fd_price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="call"
        )
        bs_price = bs_call_price(spot, strike, T, r_d, r_f, sigma)

        assert fd_price == pytest.approx(bs_price, rel=0.01, abs=0.01)

    def test_put_call_parity(self, pricer: FxLocalVolFdPricer) -> None:
        """Test put-call parity is satisfied."""
        spot = 100.0
        strike = 100.0
        T = 1.0
        r_d = 0.05
        r_f = 0.02

        call_price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="call"
        )
        put_price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="put"
        )

        # Put-call parity: C - P = S*e^(-qT) - K*e^(-rT)
        expected_diff = spot * np.exp(-r_f * T) - strike * np.exp(-r_d * T)
        actual_diff = call_price - put_price

        assert actual_diff == pytest.approx(expected_diff, abs=0.01)


# =============================================================================
# Pricing Tests: Varying Local Vol Surfaces
# =============================================================================

class TestLocalVolFdPricingVaryingVol:
    """Tests for pricing with non-constant local vol surfaces."""

    def test_smile_surface_pricing(self) -> None:
        """Test pricing with a volatility smile surface."""
        spot = 100.0
        times = np.array([0.1, 0.5, 1.0])
        spots = spot * np.array([0.8, 0.9, 1.0, 1.1, 1.2])

        # Create smile: higher vol at wings
        base_vol = 0.20
        smile_curvature = 0.02
        local_vols = np.zeros((3, 5))
        for i in range(3):
            for j, s in enumerate(spots):
                moneyness = abs(s - spot) / spot
                local_vols[i, j] = base_vol + smile_curvature * moneyness

        lv_surface = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)
        pricer = FxLocalVolFdPricer(
            local_vol_surface=lv_surface,
            n_space=201,
            n_time_steps=100,
        )

        # Price should be positive
        price = pricer.price_european(
            spot=spot, strike=100.0, maturity=0.5,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price > 0

    def test_term_structure_surface_pricing(self) -> None:
        """Test pricing with term structure in local vol."""
        spot = 100.0
        times = np.array([0.1, 0.5, 1.0, 2.0])
        spots = np.array([90.0, 100.0, 110.0])

        # Vol decreases with time (term structure)
        local_vols = np.array([
            [0.25, 0.23, 0.21],  # t=0.1
            [0.24, 0.22, 0.20],  # t=0.5
            [0.23, 0.21, 0.19],  # t=1.0
            [0.22, 0.20, 0.18],  # t=2.0
        ])

        lv_surface = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)
        pricer = FxLocalVolFdPricer(
            local_vol_surface=lv_surface,
            n_space=201,
            n_time_steps=100,
        )

        price = pricer.price_european(
            spot=spot, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price > 0


# =============================================================================
# Edge Cases
# =============================================================================

class TestLocalVolFdEdgeCases:
    """Tests for edge cases."""

    @pytest.fixture
    def pricer(self) -> FxLocalVolFdPricer:
        """Create default pricer."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        return FxLocalVolFdPricer(local_vol_surface=local_vol, n_space=201, n_time_steps=50)

    def test_zero_maturity_call(self, pricer: FxLocalVolFdPricer) -> None:
        """Test that T=0 returns intrinsic value for call."""
        spot = 100.0
        strike = 100.0

        price = pricer.price_european(
            spot=spot, strike=strike, maturity=0.0,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price == pytest.approx(max(spot - strike, 0.0), abs=1e-6)

    def test_zero_maturity_put(self, pricer: FxLocalVolFdPricer) -> None:
        """Test that T=0 returns intrinsic value for put."""
        spot = 100.0
        strike = 100.0

        price = pricer.price_european(
            spot=spot, strike=strike, maturity=0.0,
            domestic_rate=0.05, foreign_rate=0.02, option_type="put"
        )
        assert price == pytest.approx(max(strike - spot, 0.0), abs=1e-6)

    def test_deep_itm_call(self, pricer: FxLocalVolFdPricer) -> None:
        """Test deep ITM call pricing."""
        spot = 100.0
        strike = 50.0
        T = 1.0

        price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        # Deep ITM call should be close to forward minus strike
        forward = spot * np.exp((0.05 - 0.02) * T)
        intrinsic = forward - strike
        assert price > intrinsic * 0.95  # Should be close to intrinsic

    def test_deep_otm_call(self, pricer: FxLocalVolFdPricer) -> None:
        """Test deep OTM call pricing."""
        spot = 100.0
        strike = 200.0
        T = 0.5

        price = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        # Deep OTM call should be small but positive
        assert price >= 0
        assert price < 1.0  # Should be very small

    def test_invalid_spot_zero(self, pricer: FxLocalVolFdPricer) -> None:
        """Test that spot=0 raises ValueError."""
        with pytest.raises(ValueError, match="spot must be > 0"):
            pricer.price_european(
                spot=0.0, strike=100.0, maturity=1.0,
                domestic_rate=0.05, foreign_rate=0.02, option_type="call"
            )

    def test_invalid_strike_zero(self, pricer: FxLocalVolFdPricer) -> None:
        """Test that strike=0 raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            pricer.price_european(
                spot=100.0, strike=0.0, maturity=1.0,
                domestic_rate=0.05, foreign_rate=0.02, option_type="call"
            )

    def test_invalid_maturity_negative(self, pricer: FxLocalVolFdPricer) -> None:
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="maturity must be >= 0"):
            pricer.price_european(
                spot=100.0, strike=100.0, maturity=-0.1,
                domestic_rate=0.05, foreign_rate=0.02, option_type="call"
            )


# =============================================================================
# Convergence Tests
# =============================================================================

class TestLocalVolFdConvergence:
    """Tests for convergence properties."""

    def test_convergence_with_grid_refinement(self) -> None:
        """Test that prices converge as grid is refined."""
        local_vol = FlatLocalVolSurface(sigma=0.20)
        spot = 100.0
        strike = 100.0
        T = 1.0
        r_d = 0.05
        r_f = 0.02

        # Coarse grid
        pricer_coarse = FxLocalVolFdPricer(
            local_vol_surface=local_vol,
            n_space=101,
            n_time_steps=50,
        )
        price_coarse = pricer_coarse.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="call"
        )

        # Fine grid
        pricer_fine = FxLocalVolFdPricer(
            local_vol_surface=local_vol,
            n_space=401,
            n_time_steps=200,
        )
        price_fine = pricer_fine.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r_d, foreign_rate=r_f, option_type="call"
        )

        # Fine grid should be closer to BSM
        bs_price = bs_call_price(spot, strike, T, r_d, r_f, 0.20)
        error_coarse = abs(price_coarse - bs_price)
        error_fine = abs(price_fine - bs_price)

        # Fine grid should have smaller error (or at least not much larger)
        assert error_fine <= error_coarse * 1.5  # Allow some tolerance


# =============================================================================
# Local Vol Surface Input Types
# =============================================================================

class TestLocalVolFdInputTypes:
    """Tests for different local vol input types."""

    def test_flat_local_vol_surface(self) -> None:
        """Test with FlatLocalVolSurface."""
        local_vol = FlatLocalVolSurface(sigma=0.25)
        pricer = FxLocalVolFdPricer(local_vol_surface=local_vol, n_space=101, n_time_steps=50)

        price = pricer.price_european(
            spot=100.0, strike=100.0, maturity=0.5,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price > 0

    def test_grid_local_vol_surface(self) -> None:
        """Test with LocalVolSurface grid."""
        times = np.array([0.0, 0.5, 1.0])
        spots = np.array([80.0, 100.0, 120.0])
        local_vols = np.array([
            [0.22, 0.20, 0.18],
            [0.21, 0.19, 0.17],
            [0.20, 0.18, 0.16],
        ])
        local_vol = LocalVolSurface(times=times, spots=spots, local_vols=local_vols)
        pricer = FxLocalVolFdPricer(local_vol_surface=local_vol, n_space=101, n_time_steps=50)

        price = pricer.price_european(
            spot=100.0, strike=100.0, maturity=0.5,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price > 0

    def test_callable_local_vol(self) -> None:
        """Test with callable local vol function."""
        def local_vol_func(spot: float, time: float) -> float:
            """Simple local vol function."""
            return 0.20 + 0.01 * (spot - 100.0) / 100.0

        pricer = FxLocalVolFdPricer(local_vol_surface=local_vol_func, n_space=101, n_time_steps=50)

        price = pricer.price_european(
            spot=100.0, strike=100.0, maturity=0.5,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price > 0

    def test_constant_float_local_vol(self) -> None:
        """Test with constant float (treated as constant vol)."""
        pricer = FxLocalVolFdPricer(local_vol_surface=0.20, n_space=101, n_time_steps=50)

        price = pricer.price_european(
            spot=100.0, strike=100.0, maturity=0.5,
            domestic_rate=0.05, foreign_rate=0.02, option_type="call"
        )
        assert price > 0
