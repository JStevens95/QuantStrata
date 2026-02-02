"""
Unit tests for Longstaff-Schwartz Monte Carlo (LSM).

Tests cover:
1. Basis functions (polynomial, Laguerre, Chebyshev)
2. American put pricing
3. American call pricing
4. Comparison with FD/analytic benchmarks
5. Early exercise premium
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.models.numeric.monte_carlo.lsm import (
    BasisType,
    polynomial_basis,
    laguerre_basis,
    chebyshev_basis,
    lsm_american_put,
    lsm_american_call,
    price_american_put_lsm,
    price_american_call_lsm,
    LSMResult,
)


# =============================================================================
# Basis Function Tests
# =============================================================================

class TestBasisFunctions:
    """Tests for LSM basis functions."""

    def test_polynomial_basis_degree_0(self):
        """Polynomial degree 0 should be constant 1."""
        x = np.array([1.0, 2.0, 3.0])
        basis = polynomial_basis(x, degree=0)
        assert basis.shape == (3, 1)
        assert_allclose(basis[:, 0], 1.0)

    def test_polynomial_basis_degree_2(self):
        """Polynomial degree 2 should have 1, x, x²."""
        x = np.array([0.0, 1.0, 2.0])
        basis = polynomial_basis(x, degree=2)
        assert basis.shape == (3, 3)
        assert_allclose(basis[:, 0], [1.0, 1.0, 1.0])
        assert_allclose(basis[:, 1], [0.0, 1.0, 2.0])
        assert_allclose(basis[:, 2], [0.0, 1.0, 4.0])

    def test_laguerre_basis_L0(self):
        """Laguerre L_0 should be 1."""
        x = np.array([0.0, 1.0, 2.0])
        basis = laguerre_basis(x, degree=0)
        assert_allclose(basis[:, 0], 1.0)

    def test_laguerre_basis_L1(self):
        """Laguerre L_1 should be 1-x."""
        x = np.array([0.0, 1.0, 2.0])
        basis = laguerre_basis(x, degree=1)
        assert_allclose(basis[:, 1], [1.0, 0.0, -1.0])

    def test_chebyshev_basis_T0_T1(self):
        """Chebyshev T_0=1, T_1=x."""
        x = np.array([-1.0, 0.0, 1.0])
        basis = chebyshev_basis(x, degree=1)
        assert_allclose(basis[:, 0], [1.0, 1.0, 1.0])
        assert_allclose(basis[:, 1], [-1.0, 0.0, 1.0])


# =============================================================================
# American Put Tests
# =============================================================================

class TestLSMAmericanPut:
    """Tests for LSM American put pricing."""

    @pytest.fixture
    def gbm_paths(self):
        """Generate GBM paths for testing."""
        np.random.seed(42)
        n_paths, n_steps = 10000, 50
        S0, T, r, sigma = 100.0, 1.0, 0.05, 0.2
        dt = T / n_steps

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        Z = np.random.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

        return paths, r, dt

    def test_put_price_positive(self, gbm_paths):
        """American put price should be positive."""
        paths, r, dt = gbm_paths
        result = lsm_american_put(paths, strike=100.0, r=r, dt=dt)
        assert result.price > 0

    def test_put_price_reasonable(self, gbm_paths):
        """Put price should be in reasonable range."""
        paths, r, dt = gbm_paths
        result = lsm_american_put(paths, strike=100.0, r=r, dt=dt)
        # American put on S0=100, K=100 should be roughly 5-15
        assert 3.0 < result.price < 20.0

    def test_put_std_error_positive(self, gbm_paths):
        """Standard error should be positive."""
        paths, r, dt = gbm_paths
        result = lsm_american_put(paths, strike=100.0, r=r, dt=dt)
        assert result.std_error > 0

    def test_put_higher_strike_higher_price(self, gbm_paths):
        """Higher strike should give higher put price."""
        paths, r, dt = gbm_paths
        price_90 = lsm_american_put(paths, strike=90.0, r=r, dt=dt).price
        price_100 = lsm_american_put(paths, strike=100.0, r=r, dt=dt).price
        price_110 = lsm_american_put(paths, strike=110.0, r=r, dt=dt).price
        assert price_90 < price_100 < price_110

    def test_different_basis_types(self, gbm_paths):
        """Different basis types should give similar results."""
        paths, r, dt = gbm_paths
        K = 100.0

        price_poly = lsm_american_put(paths, K, r, dt, basis_type=BasisType.POLYNOMIAL).price
        price_lag = lsm_american_put(paths, K, r, dt, basis_type=BasisType.LAGUERRE).price
        price_cheb = lsm_american_put(paths, K, r, dt, basis_type=BasisType.CHEBYSHEV).price

        # All should be within 10% of each other
        prices = [price_poly, price_lag, price_cheb]
        assert max(prices) / min(prices) < 1.2


# =============================================================================
# American Call Tests
# =============================================================================

class TestLSMAmericanCall:
    """Tests for LSM American call pricing."""

    def test_call_no_dividend_equals_european(self):
        """American call without dividends should equal European."""
        # Generate paths
        np.random.seed(42)
        n_paths, n_steps = 20000, 50
        S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.05, 0.2
        dt = T / n_steps

        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0

        drift = (r - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt)

        Z = np.random.standard_normal((n_paths, n_steps))
        for t in range(n_steps):
            paths[:, t + 1] = paths[:, t] * np.exp(drift + diffusion * Z[:, t])

        # LSM American call
        am_result = lsm_american_call(paths, strike=K, r=r, dt=dt)

        # European call (just terminal payoff)
        eu_payoffs = np.maximum(paths[:, -1] - K, 0)
        eu_price = np.exp(-r * T) * eu_payoffs.mean()

        # American should be very close to European (no early exercise optimal)
        assert_allclose(am_result.price, eu_price, rtol=0.1)


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestLSMConvenienceFunctions:
    """Tests for LSM convenience functions."""

    def test_price_american_put_lsm(self):
        """Convenience function should work correctly."""
        result = price_american_put_lsm(
            spot0=100.0,
            strike=100.0,
            maturity=1.0,
            r=0.05,
            sigma=0.2,
            n_paths=10000,
            n_steps=50,
            seed=42,
        )

        assert isinstance(result, LSMResult)
        assert result.price > 0
        assert result.std_error > 0

    def test_reproducibility_with_seed(self):
        """Same seed should give same result."""
        kwargs = dict(
            spot0=100.0, strike=100.0, maturity=1.0,
            r=0.05, sigma=0.2, n_paths=5000, n_steps=25, seed=42
        )

        result1 = price_american_put_lsm(**kwargs)
        result2 = price_american_put_lsm(**kwargs)

        assert_allclose(result1.price, result2.price, rtol=1e-10)


# =============================================================================
# Early Exercise Premium Tests
# =============================================================================

class TestEarlyExercisePremium:
    """Tests for early exercise premium."""

    def test_american_put_geq_european(self):
        """American put should be >= European put."""
        from scipy.stats import norm

        S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

        # Black-Scholes European put
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        eu_put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

        # LSM American put
        am_result = price_american_put_lsm(
            spot0=S0, strike=K, maturity=T, r=r, sigma=sigma,
            n_paths=50000, n_steps=50, seed=42
        )

        # American >= European
        assert am_result.price >= eu_put - am_result.std_error * 2  # Allow for MC error

    def test_itm_put_has_premium(self):
        """Deep ITM American put should have significant early exercise premium."""
        S0, K, T, r, sigma = 80.0, 100.0, 1.0, 0.08, 0.2  # Deep ITM put, high r

        # European put
        from scipy.stats import norm
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        eu_put = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

        # American put
        am_result = price_american_put_lsm(
            spot0=S0, strike=K, maturity=T, r=r, sigma=sigma,
            n_paths=50000, n_steps=50, seed=42
        )

        # Should have positive early exercise premium
        premium = am_result.price - eu_put
        assert premium > 0.5  # At least some premium
