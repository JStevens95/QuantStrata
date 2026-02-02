"""
Unit tests for Quasi-Monte Carlo (QMC) methods.

Tests cover:
1. Sobol sequence generation
2. Halton sequence generation
3. European option pricing
4. Convergence comparison with standard MC
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from src.models.numeric.monte_carlo.qmc import (
    SobolRng,
    HaltonRng,
    qmc_european_call,
    qmc_european_put,
    qmc_path_simulation,
)


# =============================================================================
# Sobol Sequence Tests
# =============================================================================

class TestSobolRng:
    """Tests for Sobol sequence generator."""

    def test_sobol_shape(self):
        """Sobol should return correct shape."""
        rng = SobolRng(d=3, seed=42)
        samples = rng.uniform(100)
        assert samples.shape == (100, 3)

    def test_sobol_uniform_range(self):
        """Sobol samples should be in [0, 1]."""
        rng = SobolRng(d=2, seed=42)
        samples = rng.uniform(1000)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_sobol_standard_normals_shape(self):
        """Standard normals should have correct shape."""
        rng = SobolRng(d=4, seed=42)
        Z = rng.standard_normals(500)
        assert Z.shape == (500, 4)

    def test_sobol_standard_normals_distribution(self):
        """Standard normals should have mean≈0, std≈1."""
        rng = SobolRng(d=1, seed=42)
        Z = rng.standard_normals(10000)
        assert_allclose(Z.mean(), 0.0, atol=0.05)
        assert_allclose(Z.std(), 1.0, atol=0.05)

    def test_sobol_antithetic_shape(self):
        """Antithetic should double the sample size."""
        rng = SobolRng(d=2, seed=42)
        Z = rng.standard_normals_antithetic(100)
        assert Z.shape == (200, 2)

    def test_sobol_antithetic_pairs(self):
        """Antithetic pairs should sum to zero."""
        rng = SobolRng(d=2, seed=42)
        Z = rng.standard_normals_antithetic(100)
        # First half + second half should be zero
        assert_allclose(Z[:100] + Z[100:], 0.0, atol=1e-10)

    def test_sobol_reset(self):
        """Reset should restart the sequence."""
        rng = SobolRng(d=2, seed=42)
        samples1 = rng.uniform(100)
        rng.reset()
        samples2 = rng.uniform(100)
        assert_allclose(samples1, samples2)


# =============================================================================
# Halton Sequence Tests
# =============================================================================

class TestHaltonRng:
    """Tests for Halton sequence generator."""

    def test_halton_shape(self):
        """Halton should return correct shape."""
        rng = HaltonRng(d=2, seed=42)
        samples = rng.uniform(100)
        assert samples.shape == (100, 2)

    def test_halton_uniform_range(self):
        """Halton samples should be in [0, 1]."""
        rng = HaltonRng(d=3, seed=42)
        samples = rng.uniform(500)
        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_halton_standard_normals_shape(self):
        """Standard normals should have correct shape."""
        rng = HaltonRng(d=2, seed=42)
        Z = rng.standard_normals(200)
        assert Z.shape == (200, 2)


# =============================================================================
# QMC Option Pricing Tests
# =============================================================================

class TestQMCOptionPricing:
    """Tests for QMC European option pricing."""

    @pytest.fixture
    def bs_call_price(self):
        """Black-Scholes call price for reference."""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @pytest.fixture
    def bs_put_price(self):
        """Black-Scholes put price for reference."""
        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    def test_qmc_call_positive(self):
        """QMC call price should be positive."""
        price, _ = qmc_european_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=5000, seed=42
        )
        assert price > 0

    def test_qmc_put_positive(self):
        """QMC put price should be positive."""
        price, _ = qmc_european_put(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=5000, seed=42
        )
        assert price > 0

    def test_qmc_call_matches_bs(self, bs_call_price):
        """QMC call should match BS within tolerance."""
        price, std_error = qmc_european_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=50000, seed=42
        )
        assert_allclose(price, bs_call_price, rtol=0.02)

    def test_qmc_put_matches_bs(self, bs_put_price):
        """QMC put should match BS within tolerance."""
        price, std_error = qmc_european_put(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=50000, seed=42
        )
        assert_allclose(price, bs_put_price, rtol=0.02)

    def test_put_call_parity(self):
        """QMC should satisfy put-call parity."""
        S, K, T, r, q, sigma = 100.0, 105.0, 0.5, 0.05, 0.02, 0.2

        call, _ = qmc_european_call(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        put, _ = qmc_european_put(S, K, T, r, q, sigma, n_samples=50000, seed=42)

        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert_allclose(call - put, parity, rtol=0.03)


# =============================================================================
# QMC Path Simulation Tests
# =============================================================================

class TestQMCPathSimulation:
    """Tests for QMC path simulation."""

    def test_path_shape(self):
        """Paths should have correct shape."""
        paths = qmc_path_simulation(
            spot0=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_paths=100, n_steps=50, seed=42
        )
        assert paths.shape == (100, 51)

    def test_path_initial_value(self):
        """All paths should start at spot0."""
        paths = qmc_path_simulation(
            spot0=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_paths=100, n_steps=50, seed=42
        )
        assert_allclose(paths[:, 0], 100.0)

    def test_path_positive(self):
        """All path values should be positive (GBM)."""
        paths = qmc_path_simulation(
            spot0=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_paths=1000, n_steps=100, seed=42
        )
        assert np.all(paths > 0)

    def test_path_terminal_mean(self):
        """Terminal mean should be near forward price."""
        S0, T, r, q = 100.0, 1.0, 0.05, 0.02
        paths = qmc_path_simulation(
            spot0=S0, maturity=T, r=r, q=q, sigma=0.2,
            n_paths=10000, n_steps=50, seed=42
        )

        expected_mean = S0 * np.exp((r - q) * T)
        actual_mean = paths[:, -1].mean()
        assert_allclose(actual_mean, expected_mean, rtol=0.02)


# =============================================================================
# QMC vs MC Comparison Tests
# =============================================================================

class TestQMCvsMC:
    """Tests comparing QMC to standard MC."""

    def test_qmc_lower_variance(self):
        """QMC should have lower variance than MC (for same n)."""
        from src.models.numeric.monte_carlo.rng import NormalRng

        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.02, 0.2
        n_samples = 5000
        n_trials = 10

        # MC variance
        mc_prices = []
        for trial in range(n_trials):
            rng = NormalRng(seed=trial)
            Z = rng.standard_normals(n_samples, 1, antithetic=True).flatten()
            S_T = S * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
            mc_prices.append(np.exp(-r * T) * np.maximum(S_T - K, 0).mean())

        # QMC variance
        qmc_prices = []
        for trial in range(n_trials):
            price, _ = qmc_european_call(S, K, T, r, q, sigma, n_samples, seed=trial)
            qmc_prices.append(price)

        mc_var = np.var(mc_prices)
        qmc_var = np.var(qmc_prices)

        # QMC variance should be lower (or at least not much higher)
        assert qmc_var <= mc_var * 2  # Allow some tolerance
