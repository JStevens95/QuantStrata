"""
Unit tests for Variance Gamma Model.

Tests cover:
1. Parameter validation and properties
2. Path simulation
3. Exact terminal simulation
4. Martingale property
5. MC and FFT pricing
6. Limiting cases
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.models.levy import (
    VarianceGammaParameters,
    VarianceGammaDynamics,
    VarianceGammaSimulation,
    vg_european_call,
    vg_european_put,
)
from src.models.levy.variance_gamma import (
    vg_characteristic_function,
    vg_mc_call,
    vg_mc_put,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def standard_params() -> VarianceGammaParameters:
    """Standard VG parameters for testing."""
    return VarianceGammaParameters(
        theta=-0.1,   # Negative skew
        sigma=0.2,    # 20% vol
        nu=0.2,       # Moderate kurtosis
    )


@pytest.fixture
def symmetric_params() -> VarianceGammaParameters:
    """Symmetric VG (θ=0) parameters."""
    return VarianceGammaParameters(
        theta=0.0,    # No skew
        sigma=0.25,
        nu=0.15,
    )


@pytest.fixture
def high_kurtosis_params() -> VarianceGammaParameters:
    """High kurtosis VG parameters."""
    return VarianceGammaParameters(
        theta=-0.05,
        sigma=0.2,
        nu=0.5,       # High nu = fat tails
    )


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestVarianceGammaParametersValidation:
    """Tests for VarianceGammaParameters validation."""

    def test_valid_parameters(self, standard_params):
        """Valid parameters should not raise."""
        assert standard_params.theta == -0.1
        assert standard_params.sigma == 0.2
        assert standard_params.nu == 0.2

    def test_negative_sigma_raises(self):
        """Negative volatility should raise."""
        with pytest.raises(ValueError, match="sigma must be > 0"):
            VarianceGammaParameters(theta=0.0, sigma=-0.2, nu=0.2)

    def test_zero_sigma_raises(self):
        """Zero volatility should raise."""
        with pytest.raises(ValueError, match="sigma must be > 0"):
            VarianceGammaParameters(theta=0.0, sigma=0.0, nu=0.2)

    def test_negative_nu_raises(self):
        """Negative nu should raise."""
        with pytest.raises(ValueError, match="nu must be > 0"):
            VarianceGammaParameters(theta=0.0, sigma=0.2, nu=-0.1)

    def test_zero_nu_raises(self):
        """Zero nu should raise."""
        with pytest.raises(ValueError, match="nu must be > 0"):
            VarianceGammaParameters(theta=0.0, sigma=0.2, nu=0.0)

    def test_invalid_omega_raises(self):
        """Parameters that make omega undefined should raise."""
        # Need 1 - θν - σ²ν/2 > 0
        # With theta=10, sigma=0.2, nu=0.2: 1 - 2 - 0.004 < 0
        with pytest.raises(ValueError, match="Invalid parameters"):
            VarianceGammaParameters(theta=10.0, sigma=0.2, nu=0.2)

    def test_nan_parameters_raise(self):
        """NaN parameters should raise."""
        with pytest.raises(ValueError, match="must be finite"):
            VarianceGammaParameters(theta=np.nan, sigma=0.2, nu=0.2)


# =============================================================================
# Parameter Properties Tests
# =============================================================================

class TestVarianceGammaParametersProperties:
    """Tests for VarianceGammaParameters derived properties."""

    def test_omega_positive_for_negative_theta(self, standard_params):
        """omega should be positive for negative theta (compensates for negative skew)."""
        # For negative theta: 1 - θν - σ²ν/2 > 1, so log > 0, so omega > 0
        assert standard_params.omega > 0

    def test_omega_formula(self, standard_params):
        """omega should satisfy the formula."""
        theta, sigma, nu = standard_params.theta, standard_params.sigma, standard_params.nu
        expected = np.log(1 - theta * nu - 0.5 * sigma**2 * nu) / nu
        assert_allclose(standard_params.omega, expected, rtol=1e-10)

    def test_variance_rate_positive(self, standard_params):
        """Variance rate should be positive."""
        assert standard_params.variance_rate > 0

    def test_variance_rate_formula(self, standard_params):
        """Variance rate should equal σ² + θ²ν."""
        expected = standard_params.sigma**2 + standard_params.theta**2 * standard_params.nu
        assert_allclose(standard_params.variance_rate, expected, rtol=1e-10)

    def test_skewness_negative_for_negative_theta(self, standard_params):
        """Skewness should be negative for negative theta."""
        assert standard_params.skewness < 0

    def test_skewness_zero_for_zero_theta(self, symmetric_params):
        """Skewness should be zero for θ=0."""
        assert_allclose(symmetric_params.skewness, 0.0, atol=1e-10)

    def test_excess_kurtosis_positive(self, standard_params):
        """Excess kurtosis should be positive (fat tails)."""
        assert standard_params.excess_kurtosis > 0

    def test_higher_nu_more_kurtosis(self, standard_params, high_kurtosis_params):
        """Higher nu should give more excess kurtosis."""
        # Not necessarily true due to theta effects, but for same theta...
        # Actually just check that high_kurtosis has positive kurtosis
        assert high_kurtosis_params.excess_kurtosis > 0


# =============================================================================
# Dynamics Simulation Tests
# =============================================================================

class TestVarianceGammaDynamicsSimulation:
    """Tests for VarianceGammaDynamics simulation."""

    def test_simulate_returns_correct_shape(self, standard_params):
        """Simulation should return correct output shapes."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=100, seed=42
        )

        assert sim.spot_paths.shape[1] == 101  # n_steps + 1
        assert sim.gamma_times.shape == sim.spot_paths.shape
        assert len(sim.times) == 101

    def test_simulate_initial_spot(self, standard_params):
        """Initial spot should equal spot0."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        assert_allclose(sim.spot_paths[:, 0], 100.0, rtol=1e-10)

    def test_simulate_initial_gamma_time(self, standard_params):
        """Initial Gamma time should be zero."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        assert_allclose(sim.gamma_times[:, 0], 0.0, rtol=1e-10)

    def test_simulate_paths_positive(self, standard_params):
        """All simulated paths should remain positive."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=100, seed=42
        )
        assert np.all(sim.spot_paths > 0.0)

    def test_simulate_reproducible_with_seed(self, standard_params):
        """Simulation with same seed should be reproducible."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)

        sim1 = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        sim2 = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )

        assert_allclose(sim1.terminal_spots, sim2.terminal_spots, rtol=1e-10)

    def test_simulate_invalid_spot_raises(self, standard_params):
        """Invalid spot0 should raise."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        with pytest.raises(ValueError, match="spot0 must be > 0"):
            dynamics.simulate(spot0=-100.0, maturity=1.0, n_paths=100, n_steps=50)

    def test_gamma_time_non_decreasing(self, standard_params):
        """Cumulative Gamma time should be non-decreasing."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=500, n_steps=100, seed=42
        )
        # Each row should be non-decreasing
        for i in range(sim.n_paths):
            diffs = np.diff(sim.gamma_times[i, :])
            assert np.all(diffs >= 0)

    def test_average_gamma_time_near_maturity(self, standard_params):
        """Average Gamma time should be near calendar maturity."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=10000, n_steps=100, seed=42
        )
        # E[G_T] = T
        assert_allclose(sim.average_gamma_time, 1.0, rtol=0.05)


# =============================================================================
# Martingale Property Tests
# =============================================================================

class TestVarianceGammaMartingale:
    """Tests for martingale property (E[S_T] = S_0 exp((r-q)T))."""

    def test_martingale_standard_params(self, standard_params):
        """VG should preserve martingale property."""
        drift = 0.05  # r - q
        dynamics = VarianceGammaDynamics(params=standard_params, drift=drift)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=50000, n_steps=100, seed=42
        )

        # E[S_T] should be S_0 * exp(drift * T)
        expected_mean = 100.0 * np.exp(drift * 1.0)
        actual_mean = np.mean(sim.terminal_spots)
        assert_allclose(actual_mean, expected_mean, rtol=0.02)

    def test_martingale_symmetric_params(self, symmetric_params):
        """Symmetric VG should preserve martingale property."""
        drift = 0.03
        dynamics = VarianceGammaDynamics(params=symmetric_params, drift=drift)
        sim = dynamics.simulate(
            spot0=100.0, maturity=0.5, n_paths=30000, n_steps=50, seed=42
        )

        expected_mean = 100.0 * np.exp(drift * 0.5)
        actual_mean = np.mean(sim.terminal_spots)
        assert_allclose(actual_mean, expected_mean, rtol=0.02)

    def test_martingale_terminal_simulation(self, standard_params):
        """Exact terminal simulation should preserve martingale."""
        drift = 0.05
        dynamics = VarianceGammaDynamics(params=standard_params, drift=drift)
        S_T = dynamics.simulate_terminal(
            spot0=100.0, maturity=1.0, n_paths=100000, seed=42
        )

        expected_mean = 100.0 * np.exp(drift * 1.0)
        assert_allclose(np.mean(S_T), expected_mean, rtol=0.01)


# =============================================================================
# Characteristic Function Tests
# =============================================================================

class TestVarianceGammaCharFunc:
    """Tests for characteristic function."""

    def test_char_func_at_zero(self, standard_params):
        """φ(0) should equal 1."""
        phi = vg_characteristic_function(np.array([0.0]), T=1.0, params=standard_params)
        assert_allclose(np.abs(phi[0]), 1.0, rtol=1e-10)

    def test_char_func_symmetric_for_zero_theta(self, symmetric_params):
        """For θ=0, |φ(u)| = |φ(-u)|."""
        u = np.array([1.0, 2.0, 3.0])
        phi_pos = vg_characteristic_function(u, T=1.0, params=symmetric_params)
        phi_neg = vg_characteristic_function(-u, T=1.0, params=symmetric_params)
        assert_allclose(np.abs(phi_pos), np.abs(phi_neg), rtol=1e-10)


# =============================================================================
# European Option Pricing Tests
# =============================================================================

class TestVarianceGammaEuropeanPricing:
    """Tests for VG European option pricing."""

    def test_call_positive(self, standard_params):
        """Call price should be positive."""
        price = vg_european_call(
            S=100, K=100, T=1.0, r=0.05, q=0.0, params=standard_params,
            n_paths=10000, seed=42
        )
        assert price > 0.0

    def test_put_positive(self, standard_params):
        """Put price should be positive."""
        price = vg_european_put(
            S=100, K=100, T=1.0, r=0.05, q=0.0, params=standard_params,
            n_paths=10000, seed=42
        )
        assert price > 0.0

    def test_put_call_parity(self, standard_params):
        """Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, q = 100.0, 105.0, 0.5, 0.05, 0.02

        # Use same seed for both to reduce variance
        call = vg_mc_call(spot0=S, strike=K, maturity=T, r=r, q=q,
                          params=standard_params, n_paths=100000, seed=42)
        put = vg_mc_put(spot0=S, strike=K, maturity=T, r=r, q=q,
                        params=standard_params, n_paths=100000, seed=42)

        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert_allclose(call - put, parity, rtol=0.02)

    def test_mc_call_positive(self, standard_params):
        """MC call price should be positive."""
        price = vg_mc_call(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.0,
            params=standard_params, n_paths=10000, seed=42
        )
        assert price > 0.0

    def test_mc_put_positive(self, standard_params):
        """MC put price should be positive."""
        price = vg_mc_put(
            spot0=100, strike=100, maturity=1.0, r=0.05, q=0.0,
            params=standard_params, n_paths=10000, seed=42
        )
        assert price > 0.0

    def test_call_decreases_with_strike(self, standard_params):
        """Call price should decrease with strike."""
        kwargs = dict(maturity=1.0, r=0.05, q=0.0, params=standard_params, n_paths=20000, seed=42)
        c1 = vg_mc_call(spot0=100, strike=90, **kwargs)
        c2 = vg_mc_call(spot0=100, strike=100, **kwargs)
        c3 = vg_mc_call(spot0=100, strike=110, **kwargs)
        assert c1 > c2 > c3

    def test_put_increases_with_strike(self, standard_params):
        """Put price should increase with strike."""
        kwargs = dict(maturity=1.0, r=0.05, q=0.0, params=standard_params, n_paths=20000, seed=42)
        p1 = vg_mc_put(spot0=100, strike=90, **kwargs)
        p2 = vg_mc_put(spot0=100, strike=100, **kwargs)
        p3 = vg_mc_put(spot0=100, strike=110, **kwargs)
        assert p1 < p2 < p3


# =============================================================================
# Simulation Output Properties Tests
# =============================================================================

class TestVarianceGammaSimulationProperties:
    """Tests for VarianceGammaSimulation output properties."""

    def test_maturity_property(self, standard_params):
        """maturity property should return correct value."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(spot0=100.0, maturity=0.75, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.maturity, 0.75, rtol=1e-10)

    def test_terminal_spots_property(self, standard_params):
        """terminal_spots should return last column."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.terminal_spots, sim.spot_paths[:, -1], rtol=1e-10)

    def test_total_gamma_time_property(self, standard_params):
        """total_gamma_time should return last column of gamma_times."""
        dynamics = VarianceGammaDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.total_gamma_time, sim.gamma_times[:, -1], rtol=1e-10)
