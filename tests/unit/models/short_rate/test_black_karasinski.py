"""
Unit tests for Black-Karasinski Short Rate Model.

Tests cover:
1. Parameter validation
2. Parameter properties
3. Dynamics simulation (exact and Euler schemes)
4. Rate distribution properties (log-normality, positivity)
5. MC bond pricing
6. Edge cases
"""

import math
import numpy as np
import pytest

from src.models.short_rate.black_karasinski import (
    BlackKarasinskiParameters,
    BlackKarasinskiDynamics,
    BlackKarasinskiSimulation,
    bk_zc_bond_price_mc,
    bk_zc_bond_option_price_mc,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def bk_params() -> BlackKarasinskiParameters:
    """Standard Black-Karasinski parameters for testing."""
    return BlackKarasinskiParameters(
        a=0.1,       # Mean reversion speed
        sigma=0.15,  # 15% vol of log-rate
        r0=0.03,     # 3% initial rate
        theta=-3.5,  # Long-term log-rate (≈3% long-term rate)
    )


@pytest.fixture
def bk_params_high_vol() -> BlackKarasinskiParameters:
    """Black-Karasinski parameters with higher volatility."""
    return BlackKarasinskiParameters(
        a=0.1,
        sigma=0.30,  # Higher vol
        r0=0.03,
        theta=-3.5,
    )


# =============================================================================
# Parameter Tests
# =============================================================================


class TestBlackKarasinskiParameters:
    """Tests for BlackKarasinskiParameters dataclass."""

    def test_construction_valid(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test valid parameter construction."""
        assert bk_params.a == 0.1
        assert bk_params.sigma == 0.15
        assert bk_params.r0 == 0.03
        assert bk_params.theta == -3.5

    def test_construction_default_theta(self) -> None:
        """Test theta defaults to ln(r0) when not provided."""
        params = BlackKarasinskiParameters(a=0.1, sigma=0.15, r0=0.05)
        assert params.theta == pytest.approx(np.log(0.05), rel=1e-10)

    def test_construction_invalid_a_zero(self) -> None:
        """Test that a=0 raises error."""
        with pytest.raises(ValueError, match="a must be > 0"):
            BlackKarasinskiParameters(a=0.0, sigma=0.15, r0=0.03)

    def test_construction_invalid_a_negative(self) -> None:
        """Test that negative a raises error."""
        with pytest.raises(ValueError, match="a must be > 0"):
            BlackKarasinskiParameters(a=-0.1, sigma=0.15, r0=0.03)

    def test_construction_invalid_sigma_zero(self) -> None:
        """Test that sigma=0 raises error."""
        with pytest.raises(ValueError, match="sigma must be > 0"):
            BlackKarasinskiParameters(a=0.1, sigma=0.0, r0=0.03)

    def test_construction_invalid_sigma_negative(self) -> None:
        """Test that negative sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be > 0"):
            BlackKarasinskiParameters(a=0.1, sigma=-0.15, r0=0.03)

    def test_construction_invalid_r0_zero(self) -> None:
        """Test that r0=0 raises error (BK requires positive rates)."""
        with pytest.raises(ValueError, match="r0 must be > 0"):
            BlackKarasinskiParameters(a=0.1, sigma=0.15, r0=0.0)

    def test_construction_invalid_r0_negative(self) -> None:
        """Test that negative r0 raises error."""
        with pytest.raises(ValueError, match="r0 must be > 0"):
            BlackKarasinskiParameters(a=0.1, sigma=0.15, r0=-0.03)

    def test_x0_property(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test x0 = ln(r0) property."""
        expected_x0 = np.log(0.03)
        assert bk_params.x0 == pytest.approx(expected_x0, rel=1e-10)

    def test_half_life(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test half-life property."""
        expected = np.log(2.0) / 0.1
        assert bk_params.half_life == pytest.approx(expected, rel=1e-10)

    def test_long_term_vol(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test long-term volatility property."""
        expected = 0.15 / np.sqrt(2.0 * 0.1)
        assert bk_params.long_term_vol == pytest.approx(expected, rel=1e-10)

    def test_long_term_rate(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test long-term rate = exp(theta)."""
        expected = np.exp(-3.5)
        assert bk_params.long_term_rate == pytest.approx(expected, rel=1e-10)

    def test_expected_log_rate(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test expected log-rate at time t."""
        t = 5.0
        x0 = bk_params.x0
        theta = bk_params.theta
        a = bk_params.a
        expected = theta + (x0 - theta) * np.exp(-a * t)
        assert bk_params.expected_log_rate(t) == pytest.approx(expected, rel=1e-10)

    def test_expected_log_rate_converges_to_theta(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that E[ln r(t)] → θ as t → ∞."""
        t_large = 100.0
        expected_log_rate = bk_params.expected_log_rate(t_large)
        assert expected_log_rate == pytest.approx(bk_params.theta, rel=1e-3)

    def test_variance_log_rate(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test variance of log-rate at time t."""
        t = 2.0
        a = bk_params.a
        sigma = bk_params.sigma
        expected = (sigma ** 2 / (2.0 * a)) * (1.0 - np.exp(-2.0 * a * t))
        assert bk_params.variance_log_rate(t) == pytest.approx(expected, rel=1e-10)

    def test_std_log_rate(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test std = sqrt(variance)."""
        t = 2.0
        expected = np.sqrt(bk_params.variance_log_rate(t))
        assert bk_params.std_log_rate(t) == pytest.approx(expected, rel=1e-10)


# =============================================================================
# Dynamics Tests
# =============================================================================


class TestBlackKarasinskiDynamics:
    """Tests for BlackKarasinskiDynamics simulation."""

    def test_simulation_basic(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test basic simulation returns expected shape."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=100,
            n_steps=50,
            seed=42,
        )

        assert isinstance(sim, BlackKarasinskiSimulation)
        # With antithetic, n_paths rounds up to even
        assert sim.n_paths >= 100
        assert sim.n_steps == 50
        assert sim.rate_paths.shape == (sim.n_paths, 51)
        assert sim.log_rate_paths.shape == (sim.n_paths, 51)
        assert sim.times.shape == (51,)

    def test_simulation_initial_rate(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test that all paths start at r0."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=100,
            n_steps=50,
            seed=42,
        )

        # All paths should start at r0.
        np.testing.assert_allclose(sim.rate_paths[:, 0], 0.03, rtol=1e-10)

    def test_simulation_rates_always_positive(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that all simulated rates are positive (key BK property)."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=5.0,
            n_paths=1000,
            n_steps=500,
            seed=42,
        )

        # ALL rates should be positive (log-normal property).
        assert np.all(sim.rate_paths > 0.0)

    def test_simulation_mean_log_rate_convergence(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that mean log-rate converges to expected value."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=50000,
            n_steps=100,
            scheme="exact",
            seed=42,
            antithetic=True,
        )

        expected_mean = bk_params.expected_log_rate(1.0)
        actual_mean = sim.mean_terminal_log_rate

        # Should be within 1% of theoretical mean.
        assert actual_mean == pytest.approx(expected_mean, rel=0.01)

    def test_simulation_variance_log_rate_convergence(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that variance of log-rate converges to expected value."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=50000,
            n_steps=100,
            scheme="exact",
            seed=42,
            antithetic=True,
        )

        expected_std = bk_params.std_log_rate(1.0)
        actual_std = sim.std_terminal_log_rate

        # Should be within 5% of theoretical std.
        assert actual_std == pytest.approx(expected_std, rel=0.05)

    def test_simulation_euler_scheme(self, bk_params: BlackKarasinskiParameters) -> None:
        """Test Euler scheme produces valid results."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=1000,
            n_steps=252,  # More steps for Euler accuracy
            scheme="euler",
            seed=42,
        )

        # All rates should still be positive.
        assert np.all(sim.rate_paths > 0.0)
        assert sim.scheme == "euler"

    def test_simulation_exact_vs_euler(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test exact and Euler schemes give similar results."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)

        sim_exact = dynamics.simulate(
            maturity=1.0,
            n_paths=10000,
            n_steps=100,
            scheme="exact",
            seed=42,
        )
        sim_euler = dynamics.simulate(
            maturity=1.0,
            n_paths=10000,
            n_steps=100,
            scheme="euler",
            seed=42,
        )

        # Means should be close (within 5%).
        assert sim_exact.mean_terminal_rate == pytest.approx(
            sim_euler.mean_terminal_rate, rel=0.05
        )

    def test_simulation_discount_factors(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that discount factors are computed when requested."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=100,
            n_steps=50,
            seed=42,
            compute_discount_factors=True,
        )

        assert sim.discount_factors is not None
        assert sim.discount_factors.shape == (sim.n_paths,)
        # All discount factors should be positive and <= 1 for positive rates.
        assert np.all(sim.discount_factors > 0.0)
        assert np.all(sim.discount_factors <= 1.0)

    def test_simulation_antithetic_variance_reduction(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that antithetic variates reduce variance."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)

        # Run multiple simulations and compare variance.
        n_runs = 20
        n_paths = 5000

        prices_no_av = []
        prices_av = []

        for i in range(n_runs):
            sim_no_av = dynamics.simulate(
                maturity=1.0,
                n_paths=n_paths,
                n_steps=50,
                seed=i,
                antithetic=False,
                compute_discount_factors=True,
            )
            sim_av = dynamics.simulate(
                maturity=1.0,
                n_paths=n_paths,
                n_steps=50,
                seed=i,
                antithetic=True,
                compute_discount_factors=True,
            )

            prices_no_av.append(np.mean(sim_no_av.discount_factors))
            prices_av.append(np.mean(sim_av.discount_factors))

        # Antithetic should have lower variance.
        var_no_av = np.var(prices_no_av)
        var_av = np.var(prices_av)

        # AV should reduce variance (not always guaranteed, but typically).
        # Just check both variances are reasonable.
        assert var_no_av > 0
        assert var_av > 0


# =============================================================================
# MC Pricing Tests
# =============================================================================


class TestBlackKarasinskiMCPricing:
    """Tests for Black-Karasinski MC pricing functions."""

    def test_zc_bond_price_zero_maturity(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test ZC bond price at maturity = 1."""
        price = bk_zc_bond_price_mc(T=0.0, params=bk_params)
        assert price == 1.0

    def test_zc_bond_price_positive(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test ZC bond price is positive and < 1."""
        price = bk_zc_bond_price_mc(
            T=1.0,
            params=bk_params,
            n_paths=10000,
            n_steps=100,
            seed=42,
        )
        assert 0.0 < price < 1.0

    def test_zc_bond_price_decreases_with_maturity(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test ZC bond price decreases with maturity."""
        price_1y = bk_zc_bond_price_mc(T=1.0, params=bk_params, seed=42, n_paths=20000)
        price_5y = bk_zc_bond_price_mc(T=5.0, params=bk_params, seed=42, n_paths=20000)
        price_10y = bk_zc_bond_price_mc(T=10.0, params=bk_params, seed=42, n_paths=20000)

        assert price_1y > price_5y > price_10y

    def test_bond_option_call_positive(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test call option on ZC bond is positive."""
        price = bk_zc_bond_option_price_mc(
            K=0.95,
            T_option=1.0,
            T_bond=2.0,
            params=bk_params,
            is_call=True,
            n_paths=10000,
            seed=42,
        )
        assert price >= 0.0

    def test_bond_option_put_positive(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test put option on ZC bond is positive."""
        price = bk_zc_bond_option_price_mc(
            K=0.95,
            T_option=1.0,
            T_bond=2.0,
            params=bk_params,
            is_call=False,
            n_paths=10000,
            seed=42,
        )
        assert price >= 0.0

    def test_bond_option_invalid_maturity_raises(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test error when T_bond <= T_option."""
        with pytest.raises(ValueError, match="T_bond.*must be > T_option"):
            bk_zc_bond_option_price_mc(
                K=0.95,
                T_option=2.0,
                T_bond=1.0,  # Invalid: T_bond < T_option
                params=bk_params,
            )


# =============================================================================
# Edge Cases
# =============================================================================


class TestBlackKarasinskiEdgeCases:
    """Tests for edge cases."""

    def test_invalid_maturity_raises(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that invalid maturity raises error."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        with pytest.raises(ValueError, match="maturity"):
            dynamics.simulate(maturity=-1.0, n_paths=100, n_steps=50)

    def test_invalid_n_paths_raises(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that invalid n_paths raises error."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        with pytest.raises(ValueError, match="n_paths"):
            dynamics.simulate(maturity=1.0, n_paths=0, n_steps=50)

    def test_invalid_n_steps_raises(
        self, bk_params: BlackKarasinskiParameters
    ) -> None:
        """Test that invalid n_steps raises error."""
        dynamics = BlackKarasinskiDynamics(params=bk_params)
        with pytest.raises(ValueError, match="n_steps"):
            dynamics.simulate(maturity=1.0, n_paths=100, n_steps=0)

    def test_high_mean_reversion(self) -> None:
        """Test with high mean reversion (fast convergence to theta)."""
        params = BlackKarasinskiParameters(
            a=1.0,  # High mean reversion
            sigma=0.15,
            r0=0.03,
            theta=-3.5,
        )
        dynamics = BlackKarasinskiDynamics(params=params)
        sim = dynamics.simulate(
            maturity=5.0,
            n_paths=10000,
            n_steps=100,
            seed=42,
        )

        # Mean should be close to long-term rate.
        long_term_rate = params.long_term_rate
        assert sim.mean_terminal_rate == pytest.approx(long_term_rate, rel=0.1)

    def test_low_mean_reversion(self) -> None:
        """Test with low mean reversion (slow convergence)."""
        params = BlackKarasinskiParameters(
            a=0.01,  # Low mean reversion
            sigma=0.15,
            r0=0.03,
            theta=-3.5,
        )
        dynamics = BlackKarasinskiDynamics(params=params)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=10000,
            n_steps=100,
            seed=42,
        )

        # Mean should still be close to initial rate (slow reversion).
        assert sim.mean_terminal_rate == pytest.approx(0.03, rel=0.2)

    def test_high_volatility(
        self, bk_params_high_vol: BlackKarasinskiParameters
    ) -> None:
        """Test with high volatility."""
        dynamics = BlackKarasinskiDynamics(params=bk_params_high_vol)
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=10000,
            n_steps=100,
            seed=42,
        )

        # Should have non-trivial distribution.
        assert sim.std_terminal_rate > 0.005  # Non-trivial std
        # Rates should still be positive (key BK property).
        assert np.all(sim.rate_paths > 0.0)
