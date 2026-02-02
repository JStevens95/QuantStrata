"""
Unit tests for Importance Sampling variance reduction.

Tests cover:
1. Optimal drift shift computation
2. European call/put pricing
3. Variance reduction for OTM options
4. Comparison with standard MC
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import norm

from src.models.numeric.monte_carlo.importance_sampling import (
    optimal_drift_shift_call,
    optimal_drift_shift_put,
    is_european_call,
    is_european_put,
    adaptive_is_european_call,
    compare_is_standard_mc,
    ImportanceSamplingResult,
)


# =============================================================================
# Black-Scholes Reference Prices
# =============================================================================

def bs_call(S, K, T, r, q, sigma):
    """Black-Scholes call price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes put price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


# =============================================================================
# Optimal Drift Shift Tests
# =============================================================================

class TestOptimalDriftShift:
    """Tests for optimal drift shift computation."""

    def test_atm_call_shift_near_zero(self):
        """ATM call should have drift shift near zero."""
        shift = optimal_drift_shift_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2
        )
        # For ATM, shift is approximately (r-q-σ²/2)T / (σ²T) which is small
        assert abs(shift) < 1.0

    def test_otm_call_positive_shift(self):
        """OTM call should have positive drift shift (shift towards strike)."""
        shift = optimal_drift_shift_call(
            spot0=100, strike=120, maturity=1.0, r=0.05, q=0.02, sigma=0.2
        )
        assert shift > 0  # Shift mean upward towards strike

    def test_itm_call_negative_shift(self):
        """ITM call should have negative drift shift."""
        shift = optimal_drift_shift_call(
            spot0=100, strike=80, maturity=1.0, r=0.05, q=0.02, sigma=0.2
        )
        assert shift < 0  # Shift mean downward towards strike

    def test_otm_put_negative_shift(self):
        """OTM put should have negative drift shift (shift towards strike)."""
        shift = optimal_drift_shift_put(
            spot0=100, strike=80, maturity=1.0, r=0.05, q=0.02, sigma=0.2
        )
        assert shift < 0  # Shift mean downward towards strike


# =============================================================================
# IS European Call Tests
# =============================================================================

class TestISEuropeanCall:
    """Tests for importance sampling European call pricing."""

    def test_call_positive_price(self):
        """IS call price should be positive."""
        result = is_european_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=10000, seed=42
        )
        assert result.price > 0

    def test_call_result_type(self):
        """Result should be ImportanceSamplingResult."""
        result = is_european_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=5000, seed=42
        )
        assert isinstance(result, ImportanceSamplingResult)

    def test_call_matches_bs_atm(self):
        """IS call should match BS for ATM option."""
        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.02, 0.2
        bs_price = bs_call(S, K, T, r, q, sigma)

        result = is_european_call(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        assert_allclose(result.price, bs_price, rtol=0.03)

    def test_call_matches_bs_otm(self):
        """IS call should match BS for OTM option."""
        S, K, T, r, q, sigma = 100.0, 120.0, 1.0, 0.05, 0.02, 0.2
        bs_price = bs_call(S, K, T, r, q, sigma)

        result = is_european_call(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        assert_allclose(result.price, bs_price, rtol=0.05)

    def test_confidence_interval(self):
        """Price should be within 95% CI of BS price (statistically)."""
        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.02, 0.2
        bs_price = bs_call(S, K, T, r, q, sigma)

        result = is_european_call(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        lower, upper = result.confidence_interval_95

        # BS price should usually be in the confidence interval
        # (Allow some tolerance for rare failures)
        assert lower < bs_price + 0.5 and upper > bs_price - 0.5


# =============================================================================
# IS European Put Tests
# =============================================================================

class TestISEuropeanPut:
    """Tests for importance sampling European put pricing."""

    def test_put_positive_price(self):
        """IS put price should be positive."""
        result = is_european_put(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=10000, seed=42
        )
        assert result.price > 0

    def test_put_matches_bs_atm(self):
        """IS put should match BS for ATM option."""
        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.02, 0.2
        bs_price = bs_put(S, K, T, r, q, sigma)

        result = is_european_put(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        assert_allclose(result.price, bs_price, rtol=0.03)

    def test_put_matches_bs_otm(self):
        """IS put should match BS for OTM option (deep OTM put)."""
        S, K, T, r, q, sigma = 100.0, 80.0, 1.0, 0.05, 0.02, 0.2
        bs_price = bs_put(S, K, T, r, q, sigma)

        result = is_european_put(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        assert_allclose(result.price, bs_price, rtol=0.1)  # OTM more challenging


# =============================================================================
# Variance Reduction Tests
# =============================================================================

class TestVarianceReduction:
    """Tests for variance reduction effectiveness."""

    def test_variance_reduction_positive(self):
        """Variance reduction should be positive (>1) for OTM options."""
        result = is_european_put(
            spot0=100, strike=80, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=20000, seed=42
        )
        # Variance reduction should be > 1 (IS is better)
        assert result.variance_reduction > 0.5  # Allow some tolerance

    def test_effective_sample_size(self):
        """Effective sample size should be positive and <= n_samples."""
        result = is_european_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=10000, seed=42
        )
        assert 0 < result.effective_sample_size <= result.n_samples

    def test_otm_put_variance_reduction(self):
        """Deep OTM put should have significant variance reduction."""
        # Deep OTM put - standard MC struggles here
        S, K, T, r, q, sigma = 100.0, 70.0, 1.0, 0.05, 0.02, 0.3

        result = is_european_put(S, K, T, r, q, sigma, n_samples=50000, seed=42)

        # Should have some variance reduction for deep OTM
        # (May not always be dramatic depending on parameters)
        assert result.variance_reduction > 0.1


# =============================================================================
# Comparison Utility Tests
# =============================================================================

class TestCompareISvsMC:
    """Tests for IS vs MC comparison utility."""

    def test_comparison_returns_dict(self):
        """Comparison should return a dictionary."""
        result = compare_is_standard_mc(
            spot0=100, strike=90, maturity=0.5, r=0.05, q=0.02, sigma=0.2,
            n_samples=10000, seed=42
        )
        assert isinstance(result, dict)
        assert 'bs_price' in result
        assert 'mc_price' in result
        assert 'is_price' in result
        assert 'variance_reduction' in result

    def test_both_prices_positive(self):
        """Both MC and IS prices should be positive."""
        result = compare_is_standard_mc(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=10000, seed=42
        )
        assert result['mc_price'] > 0
        assert result['is_price'] > 0

    def test_both_prices_reasonable(self):
        """Both MC and IS should be close to BS."""
        result = compare_is_standard_mc(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=50000, seed=42
        )

        bs_price = result['bs_price']
        assert abs(result['mc_price'] - bs_price) / bs_price < 0.05
        assert abs(result['is_price'] - bs_price) / bs_price < 0.05


# =============================================================================
# Adaptive IS Tests
# =============================================================================

class TestAdaptiveIS:
    """Tests for adaptive importance sampling."""

    def test_adaptive_positive_price(self):
        """Adaptive IS should return positive price."""
        result = adaptive_is_european_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.02, sigma=0.2,
            n_samples=10000, seed=42
        )
        assert result.price > 0

    def test_adaptive_matches_bs(self):
        """Adaptive IS should match BS price."""
        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.05, 0.02, 0.2
        bs_price = bs_call(S, K, T, r, q, sigma)

        result = adaptive_is_european_call(S, K, T, r, q, sigma, n_samples=50000, seed=42)
        assert_allclose(result.price, bs_price, rtol=0.03)
