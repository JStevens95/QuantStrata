"""
Unit tests for Heston Monte Carlo Pricer.

Tests cover:
1. FxHestonMcPricer configuration validation
2. European option pricing under Heston
3. Convergence to Black-Scholes in constant vol limit
4. Put-call parity verification
5. Greeks computation
6. Standard error and confidence intervals
"""

import numpy as np
import pytest
from scipy.stats import norm

from src.pricers.fx.heston_mc import (
    FxHestonMcPricer,
    HestonMcResult,
    price_heston_european,
)
from src.models.stochastic_volatility.heston import HestonParameters


# =============================================================================
# Helper: Black-Scholes price for comparison
# =============================================================================

def bs_call_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes call price."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes put price."""
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# =============================================================================
# FxHestonMcPricer Configuration Tests
# =============================================================================

class TestFxHestonMcPricerConfig:
    """Tests for pricer configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        pricer = FxHestonMcPricer()
        assert pricer.n_paths == 100_000
        assert pricer.n_steps == 252
        assert pricer.scheme == "full_truncation"
        assert pricer.antithetic is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        pricer = FxHestonMcPricer(
            n_paths=50000,
            n_steps=100,
            scheme="qe",
            seed=42,
            antithetic=False,
        )
        assert pricer.n_paths == 50000
        assert pricer.n_steps == 100
        assert pricer.scheme == "qe"
        assert pricer.seed == 42
        assert pricer.antithetic is False

    def test_invalid_n_paths_zero(self) -> None:
        """Test that n_paths=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_paths must be > 0"):
            FxHestonMcPricer(n_paths=0)

    def test_invalid_n_paths_negative(self) -> None:
        """Test that negative n_paths raises ValueError."""
        with pytest.raises(ValueError, match="n_paths must be > 0"):
            FxHestonMcPricer(n_paths=-1000)

    def test_invalid_n_steps_zero(self) -> None:
        """Test that n_steps=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_steps must be > 0"):
            FxHestonMcPricer(n_steps=0)


# =============================================================================
# European Option Pricing Tests
# =============================================================================

class TestHestonEuropeanPricing:
    """Tests for European option pricing under Heston."""

    @pytest.fixture
    def default_params(self) -> HestonParameters:
        """Create default Heston parameters."""
        return HestonParameters(
            kappa=2.0,
            theta=0.04,  # 20% long-term vol.
            xi=0.3,
            v0=0.04,     # 20% initial vol.
            rho=-0.7,
        )

    @pytest.fixture
    def pricer(self) -> FxHestonMcPricer:
        """Create pricer with fixed seed for reproducibility."""
        return FxHestonMcPricer(n_paths=50000, n_steps=100, seed=42)

    def test_call_price_positive(
        self, pricer: FxHestonMcPricer, default_params: HestonParameters
    ) -> None:
        """Test that call price is positive."""
        result = pricer.price_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=default_params, option_type="call"
        )
        assert result.price > 0

    def test_put_price_positive(
        self, pricer: FxHestonMcPricer, default_params: HestonParameters
    ) -> None:
        """Test that put price is positive."""
        result = pricer.price_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=default_params, option_type="put"
        )
        assert result.price > 0

    def test_result_contains_expected_fields(
        self, pricer: FxHestonMcPricer, default_params: HestonParameters
    ) -> None:
        """Test that result contains all expected fields."""
        result = pricer.price_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=default_params, option_type="call"
        )

        assert isinstance(result, HestonMcResult)
        assert hasattr(result, "price")
        assert hasattr(result, "std_error")
        assert hasattr(result, "n_paths")
        assert hasattr(result, "n_steps")
        assert hasattr(result, "mean_terminal_spot")
        assert hasattr(result, "mean_terminal_vol")
        assert hasattr(result, "simulation")

    def test_std_error_positive(
        self, pricer: FxHestonMcPricer, default_params: HestonParameters
    ) -> None:
        """Test that standard error is positive."""
        result = pricer.price_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=default_params, option_type="call"
        )
        assert result.std_error > 0

    def test_confidence_interval(
        self, pricer: FxHestonMcPricer, default_params: HestonParameters
    ) -> None:
        """Test 95% confidence interval calculation."""
        result = pricer.price_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=default_params, option_type="call"
        )

        ci_low, ci_high = result.confidence_interval_95
        assert ci_low < result.price < ci_high
        assert ci_high - ci_low == pytest.approx(2 * 1.96 * result.std_error)


# =============================================================================
# Convergence Tests
# =============================================================================

class TestHestonConvergence:
    """Tests for convergence properties."""

    def test_convergence_to_bs_low_vol_of_vol(self) -> None:
        """
        Test: As ξ → 0, Heston should converge to Black-Scholes.

        When vol of vol is very small, variance stays near constant,
        so Heston ≈ Black-Scholes with vol = √θ.
        """
        # Heston parameters with very low vol-of-vol.
        params = HestonParameters(
            kappa=5.0,
            theta=0.04,
            xi=0.001,  # Very small vol-of-vol.
            v0=0.04,
            rho=0.0,
        )

        # Market parameters.
        spot, strike, T = 100.0, 100.0, 1.0
        r, q = 0.05, 0.02
        sigma_bs = np.sqrt(params.theta)  # Should be ~0.20.

        # Heston price.
        pricer = FxHestonMcPricer(n_paths=100000, n_steps=200, seed=42)
        heston_result = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r, foreign_rate=q,
            heston_params=params, option_type="call"
        )

        # Black-Scholes price.
        bs_price = bs_call_price(spot, strike, T, r, q, sigma_bs)

        # Should be close (within ~5%).
        assert heston_result.price == pytest.approx(bs_price, rel=0.05)


# =============================================================================
# Put-Call Parity Tests
# =============================================================================

class TestHestonPutCallParity:
    """Tests for put-call parity under Heston model."""

    def test_put_call_parity(self) -> None:
        """
        Test put-call parity: C - P = S*e^(-qT) - K*e^(-rT).

        This identity holds for any arbitrage-free model.
        """
        params = HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7
        )

        spot, strike, T = 100.0, 100.0, 1.0
        r, q = 0.05, 0.02

        pricer = FxHestonMcPricer(n_paths=100000, n_steps=200, seed=42)

        call_result = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r, foreign_rate=q,
            heston_params=params, option_type="call"
        )
        put_result = pricer.price_european(
            spot=spot, strike=strike, maturity=T,
            domestic_rate=r, foreign_rate=q,
            heston_params=params, option_type="put"
        )

        # Expected difference from parity.
        expected_diff = spot * np.exp(-q * T) - strike * np.exp(-r * T)
        actual_diff = call_result.price - put_result.price

        # Should be close (accounting for MC error).
        combined_se = np.sqrt(call_result.std_error**2 + put_result.std_error**2)
        assert actual_diff == pytest.approx(expected_diff, abs=3 * combined_se)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestPriceHestonEuropean:
    """Tests for the convenience pricing function."""

    def test_basic_call(self) -> None:
        """Test basic call pricing via convenience function."""
        price = price_heston_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7,
            option_type="call",
            n_paths=10000, seed=42
        )
        assert price > 0

    def test_basic_put(self) -> None:
        """Test basic put pricing via convenience function."""
        price = price_heston_european(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7,
            option_type="put",
            n_paths=10000, seed=42
        )
        assert price > 0


# =============================================================================
# Greeks Tests
# =============================================================================

class TestHestonGreeks:
    """Tests for Greeks computation under Heston."""

    @pytest.fixture
    def params(self) -> HestonParameters:
        """Create Heston parameters."""
        return HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7
        )

    def test_greeks_computation(self, params: HestonParameters) -> None:
        """Test that Greeks are computed without error."""
        pricer = FxHestonMcPricer(n_paths=10000, n_steps=50, seed=42)

        greeks = pricer.price_with_greeks(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=params, option_type="call"
        )

        # Check all Greeks are present.
        assert "price" in greeks
        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "rho" in greeks

    def test_call_delta_positive(self, params: HestonParameters) -> None:
        """Test that call delta is positive."""
        pricer = FxHestonMcPricer(n_paths=20000, n_steps=50, seed=42)

        greeks = pricer.price_with_greeks(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=params, option_type="call"
        )

        # Call delta should be positive.
        assert greeks["delta"] > 0

    def test_call_gamma_positive(self, params: HestonParameters) -> None:
        """Test that call gamma is positive."""
        pricer = FxHestonMcPricer(n_paths=20000, n_steps=50, seed=42)

        greeks = pricer.price_with_greeks(
            spot=100.0, strike=100.0, maturity=1.0,
            domestic_rate=0.05, foreign_rate=0.02,
            heston_params=params, option_type="call"
        )

        # Gamma should be positive for calls and puts.
        assert greeks["gamma"] > 0


# =============================================================================
# Validation Tests
# =============================================================================

class TestHestonPricerValidation:
    """Tests for input validation in the pricer."""

    @pytest.fixture
    def pricer(self) -> FxHestonMcPricer:
        """Create default pricer."""
        return FxHestonMcPricer(n_paths=1000, n_steps=50, seed=42)

    @pytest.fixture
    def params(self) -> HestonParameters:
        """Create Heston parameters."""
        return HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7
        )

    def test_invalid_spot_zero(
        self, pricer: FxHestonMcPricer, params: HestonParameters
    ) -> None:
        """Test that spot=0 raises ValueError."""
        with pytest.raises(ValueError, match="spot must be > 0"):
            pricer.price_european(
                spot=0.0, strike=100.0, maturity=1.0,
                domestic_rate=0.05, foreign_rate=0.02,
                heston_params=params, option_type="call"
            )

    def test_invalid_strike_zero(
        self, pricer: FxHestonMcPricer, params: HestonParameters
    ) -> None:
        """Test that strike=0 raises ValueError."""
        with pytest.raises(ValueError, match="strike must be > 0"):
            pricer.price_european(
                spot=100.0, strike=0.0, maturity=1.0,
                domestic_rate=0.05, foreign_rate=0.02,
                heston_params=params, option_type="call"
            )

    def test_invalid_maturity_zero(
        self, pricer: FxHestonMcPricer, params: HestonParameters
    ) -> None:
        """Test that maturity=0 raises ValueError."""
        with pytest.raises(ValueError, match="maturity must be > 0"):
            pricer.price_european(
                spot=100.0, strike=100.0, maturity=0.0,
                domestic_rate=0.05, foreign_rate=0.02,
                heston_params=params, option_type="call"
            )
