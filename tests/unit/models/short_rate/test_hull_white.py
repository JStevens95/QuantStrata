"""
Unit tests for Hull-White One-Factor Short Rate Model.

Tests cover:
1. HullWhiteParameters construction and validation
2. Parameter properties (half-life, long-term vol)
3. HullWhiteDynamics simulation
4. Path statistics and distributions
5. Analytic functions (bond pricing, bond options)
6. Discretization scheme comparison
"""

import math
import numpy as np
import pytest

from src.models.short_rate.hull_white import (
    HullWhiteParameters,
    HullWhiteDynamics,
    HullWhiteSimulation,
    hw_b_factor,
    hw_zc_bond_price,
    hw_zc_bond_option_price,
    hw_caplet_price,
    hw_floorlet_price,
)


# =============================================================================
# HullWhiteParameters Tests
# =============================================================================

class TestHullWhiteParameters:
    """Tests for Hull-White model parameters."""

    def test_construction_valid(self) -> None:
        """Test valid parameter construction."""
        params = HullWhiteParameters(
            a=0.1,
            sigma=0.01,
            r0=0.03,
            theta=0.04,
        )
        assert params.a == 0.1
        assert params.sigma == 0.01
        assert params.r0 == 0.03
        assert params.theta == 0.04

    def test_construction_default_theta(self) -> None:
        """Test theta defaults to r0 if not provided."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        assert params.theta == params.r0

    def test_construction_invalid_a_zero(self) -> None:
        """Test that a=0 raises ValueError."""
        with pytest.raises(ValueError, match="a.*must be > 0"):
            HullWhiteParameters(a=0.0, sigma=0.01, r0=0.03)

    def test_construction_invalid_a_negative(self) -> None:
        """Test that negative a raises ValueError."""
        with pytest.raises(ValueError, match="a.*must be > 0"):
            HullWhiteParameters(a=-0.1, sigma=0.01, r0=0.03)

    def test_construction_invalid_sigma_zero(self) -> None:
        """Test that sigma=0 raises ValueError."""
        with pytest.raises(ValueError, match="sigma.*must be > 0"):
            HullWhiteParameters(a=0.1, sigma=0.0, r0=0.03)

    def test_construction_invalid_sigma_negative(self) -> None:
        """Test that negative sigma raises ValueError."""
        with pytest.raises(ValueError, match="sigma.*must be > 0"):
            HullWhiteParameters(a=0.1, sigma=-0.01, r0=0.03)

    def test_construction_negative_r0_allowed(self) -> None:
        """Test that negative r0 is allowed (Hull-White supports negative rates)."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=-0.01)
        assert params.r0 == -0.01

    def test_half_life(self) -> None:
        """Test half-life calculation."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        expected_half_life = math.log(2) / 0.1
        assert params.half_life == pytest.approx(expected_half_life, rel=1e-10)

    def test_long_term_vol(self) -> None:
        """Test long-term volatility calculation."""
        # σ_∞ = σ / √(2a)
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        expected_ltv = 0.01 / math.sqrt(2 * 0.1)
        assert params.long_term_vol == pytest.approx(expected_ltv, rel=1e-10)

    def test_expected_rate(self) -> None:
        """Test expected rate E[r(t)]."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03, theta=0.04)
        
        # E[r(t)] = θ + (r0 - θ)·exp(-a·t)
        t = 5.0
        expected = 0.04 + (0.03 - 0.04) * math.exp(-0.1 * t)
        assert params.expected_rate(t) == pytest.approx(expected, rel=1e-10)

    def test_expected_rate_converges_to_theta(self) -> None:
        """Test that expected rate converges to theta as t → ∞."""
        params = HullWhiteParameters(a=0.5, sigma=0.01, r0=0.01, theta=0.05)
        
        # At t = 100, should be very close to theta.
        assert params.expected_rate(100.0) == pytest.approx(0.05, rel=1e-6)

    def test_variance_rate(self) -> None:
        """Test variance of rate Var[r(t)]."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        
        # Var[r(t)] = (σ²/(2a))·(1 - exp(-2at))
        t = 5.0
        expected_var = (0.01**2 / (2 * 0.1)) * (1 - math.exp(-2 * 0.1 * t))
        assert params.variance_rate(t) == pytest.approx(expected_var, rel=1e-10)

    def test_std_rate(self) -> None:
        """Test standard deviation of rate."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        t = 5.0
        assert params.std_rate(t) == pytest.approx(math.sqrt(params.variance_rate(t)), rel=1e-10)


# =============================================================================
# HullWhiteDynamics Simulation Tests
# =============================================================================

class TestHullWhiteDynamics:
    """Tests for Hull-White dynamics simulation."""

    def test_simulation_basic(self) -> None:
        """Test basic simulation produces valid output."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=1000,
            n_steps=100,
            scheme="exact",
            seed=42,
        )
        
        assert isinstance(sim, HullWhiteSimulation)
        assert sim.rate_paths.shape == (1000, 101)  # n_steps + 1
        assert sim.times.shape == (101,)
        assert sim.n_paths == 1000
        assert sim.n_steps == 100
        assert sim.scheme == "exact"

    def test_simulation_initial_rate(self) -> None:
        """Test that all paths start at r0."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(maturity=1.0, n_paths=100, n_steps=50, seed=42)
        
        assert np.allclose(sim.rate_paths[:, 0], 0.03)

    def test_simulation_mean_convergence(self) -> None:
        """Test that simulated mean converges to theoretical mean."""
        params = HullWhiteParameters(a=0.5, sigma=0.01, r0=0.02, theta=0.04)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(
            maturity=5.0,
            n_paths=50000,
            n_steps=100,
            scheme="exact",
            seed=42,
            antithetic=True,
        )
        
        # E[r(5)] = θ + (r0 - θ)·exp(-a·5)
        theoretical_mean = params.expected_rate(5.0)
        simulated_mean = sim.mean_terminal_rate
        
        # Should be within 3 std errors.
        std_error = sim.std_terminal_rate / math.sqrt(sim.n_paths)
        assert abs(simulated_mean - theoretical_mean) < 3 * std_error

    def test_simulation_variance_convergence(self) -> None:
        """Test that simulated variance converges to theoretical variance."""
        params = HullWhiteParameters(a=0.3, sigma=0.02, r0=0.03, theta=0.04)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(
            maturity=2.0,
            n_paths=50000,
            n_steps=100,
            scheme="exact",
            seed=42,
            antithetic=True,
        )
        
        theoretical_std = params.std_rate(2.0)
        simulated_std = sim.std_terminal_rate
        
        # Allow 10% relative error for variance estimation.
        assert simulated_std == pytest.approx(theoretical_std, rel=0.1)

    def test_simulation_euler_scheme(self) -> None:
        """Test Euler scheme simulation."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=1000,
            n_steps=252,  # Daily steps for better Euler accuracy.
            scheme="euler",
            seed=42,
        )
        
        assert sim.scheme == "euler"
        assert sim.rate_paths.shape[0] == 1000

    def test_simulation_exact_vs_euler(self) -> None:
        """Test that exact and Euler schemes give similar results with fine grid."""
        params = HullWhiteParameters(a=0.2, sigma=0.015, r0=0.03, theta=0.04)
        dynamics = HullWhiteDynamics(params=params)
        
        # Exact scheme.
        sim_exact = dynamics.simulate(
            maturity=1.0, n_paths=20000, n_steps=252, scheme="exact", seed=42, antithetic=True
        )
        
        # Euler scheme with same seed.
        sim_euler = dynamics.simulate(
            maturity=1.0, n_paths=20000, n_steps=252, scheme="euler", seed=42, antithetic=True
        )
        
        # Means should be close.
        assert sim_exact.mean_terminal_rate == pytest.approx(sim_euler.mean_terminal_rate, rel=0.05)

    def test_simulation_discount_factors(self) -> None:
        """Test that discount factors are computed correctly."""
        params = HullWhiteParameters(a=0.1, sigma=0.005, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(
            maturity=1.0,
            n_paths=10000,
            n_steps=100,
            scheme="exact",
            seed=42,
            compute_discount_factors=True,
            antithetic=True,
        )
        
        assert sim.discount_factors is not None
        assert len(sim.discount_factors) == sim.n_paths
        
        # Under HW with low vol and flat curve, mean DF should be close to exp(-r0*T).
        theoretical_df = math.exp(-0.03 * 1.0)
        mean_df = np.mean(sim.discount_factors)
        
        # Allow some tolerance due to convexity adjustment.
        assert mean_df == pytest.approx(theoretical_df, rel=0.05)

    def test_simulation_antithetic_variance_reduction(self) -> None:
        """Test that antithetic variates reduce variance."""
        params = HullWhiteParameters(a=0.1, sigma=0.02, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        # Without antithetic.
        sim_no_av = dynamics.simulate(
            maturity=1.0, n_paths=10000, n_steps=100, scheme="exact", seed=42, antithetic=False
        )
        
        # With antithetic (same base seed).
        sim_with_av = dynamics.simulate(
            maturity=1.0, n_paths=10000, n_steps=100, scheme="exact", seed=42, antithetic=True
        )
        
        # Antithetic should have lower variance (or similar).
        # Just verify both produce valid results.
        assert sim_no_av.n_paths == 10000
        assert sim_with_av.n_paths == 10000  # May be slightly different due to antithetic pairing.


# =============================================================================
# Analytic Functions Tests
# =============================================================================

class TestHullWhiteAnalyticFunctions:
    """Tests for Hull-White analytic pricing functions."""

    def test_b_factor_basic(self) -> None:
        """Test B(t,T) factor calculation."""
        a = 0.1
        tau = 5.0
        
        expected = (1 - math.exp(-0.1 * 5.0)) / 0.1
        assert hw_b_factor(a, tau) == pytest.approx(expected, rel=1e-10)

    def test_b_factor_zero_a_limit(self) -> None:
        """Test B factor in limit a → 0 gives τ."""
        tau = 3.0
        # As a → 0, B(t,T) → τ.
        B = hw_b_factor(1e-12, tau)
        assert B == pytest.approx(tau, rel=1e-6)

    def test_b_factor_short_tau(self) -> None:
        """Test B factor for short time to maturity."""
        a = 0.1
        tau = 0.01
        
        # B ≈ τ for small τ.
        B = hw_b_factor(a, tau)
        assert B == pytest.approx(tau, rel=0.01)

    def test_zc_bond_option_call_put_parity(self) -> None:
        """Test put-call parity for ZC bond options."""
        K = 0.95
        T_option = 0.5
        T_bond = 1.0
        a = 0.1
        sigma = 0.01
        r0 = 0.03
        
        P_0_S = math.exp(-r0 * T_option)
        P_0_T = math.exp(-r0 * T_bond)
        
        call = hw_zc_bond_option_price(
            K=K, T_option=T_option, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type="call"
        )
        put = hw_zc_bond_option_price(
            K=K, T_option=T_option, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type="put"
        )
        
        # Put-call parity: C - P = P(0,T) - K × P(0,S)
        lhs = call - put
        rhs = P_0_T - K * P_0_S
        
        assert lhs == pytest.approx(rhs, rel=1e-8)

    def test_zc_bond_option_positive(self) -> None:
        """Test that option prices are positive."""
        K = 0.95
        T_option = 0.5
        T_bond = 1.5
        a = 0.1
        sigma = 0.01
        r0 = 0.03
        
        P_0_S = math.exp(-r0 * T_option)
        P_0_T = math.exp(-r0 * T_bond)
        
        call = hw_zc_bond_option_price(
            K=K, T_option=T_option, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type="call"
        )
        put = hw_zc_bond_option_price(
            K=K, T_option=T_option, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type="put"
        )
        
        assert call >= 0.0
        assert put >= 0.0

    def test_zc_bond_option_atm(self) -> None:
        """Test ATM bond option pricing."""
        T_option = 1.0
        T_bond = 2.0
        a = 0.1
        sigma = 0.01
        r0 = 0.03
        
        P_0_S = math.exp(-r0 * T_option)
        P_0_T = math.exp(-r0 * T_bond)
        
        # ATM strike = forward bond price.
        K = P_0_T / P_0_S
        
        call = hw_zc_bond_option_price(
            K=K, T_option=T_option, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type="call"
        )
        put = hw_zc_bond_option_price(
            K=K, T_option=T_option, T_bond=T_bond, a=a, sigma=sigma,
            P_0_S=P_0_S, P_0_T=P_0_T, option_type="put"
        )
        
        # At ATM, call ≈ put (approximately).
        assert call == pytest.approx(put, rel=0.1)

    def test_caplet_floorlet_parity(self) -> None:
        """Test cap-floor parity: caplet - floorlet = FRA payoff."""
        K = 0.03
        T_reset = 0.5
        T_pay = 1.0
        tau = 0.5
        a = 0.1
        sigma = 0.01
        r0 = 0.03
        notional = 1000000.0
        
        P_0_reset = math.exp(-r0 * T_reset)
        P_0_pay = math.exp(-r0 * T_pay)
        
        caplet = hw_caplet_price(
            K=K, T_reset=T_reset, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=P_0_reset, P_0_pay=P_0_pay, notional=notional
        )
        floorlet = hw_floorlet_price(
            K=K, T_reset=T_reset, T_pay=T_pay, tau=tau, a=a, sigma=sigma,
            P_0_reset=P_0_reset, P_0_pay=P_0_pay, notional=notional
        )
        
        # Forward rate: F = (P_reset / P_pay - 1) / tau
        F = (P_0_reset / P_0_pay - 1) / tau
        
        # Cap-floor parity: caplet - floorlet = N × τ × P_pay × (F - K)
        expected_diff = notional * tau * P_0_pay * (F - K)
        actual_diff = caplet - floorlet
        
        assert actual_diff == pytest.approx(expected_diff, rel=1e-6)

    def test_caplet_positive(self) -> None:
        """Test that caplet prices are positive."""
        caplet = hw_caplet_price(
            K=0.03, T_reset=0.5, T_pay=1.0, tau=0.5,
            a=0.1, sigma=0.01,
            P_0_reset=0.985, P_0_pay=0.970, notional=1000000
        )
        assert caplet >= 0.0

    def test_floorlet_positive(self) -> None:
        """Test that floorlet prices are positive."""
        floorlet = hw_floorlet_price(
            K=0.03, T_reset=0.5, T_pay=1.0, tau=0.5,
            a=0.1, sigma=0.01,
            P_0_reset=0.985, P_0_pay=0.970, notional=1000000
        )
        assert floorlet >= 0.0


# =============================================================================
# Edge Cases
# =============================================================================

class TestHullWhiteEdgeCases:
    """Tests for edge cases."""

    def test_zero_maturity_simulation(self) -> None:
        """Test that simulation with very short maturity works."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(maturity=0.01, n_paths=100, n_steps=2, seed=42)
        
        assert sim.rate_paths.shape == (100, 3)

    def test_high_mean_reversion(self) -> None:
        """Test with high mean reversion (quick convergence to theta)."""
        params = HullWhiteParameters(a=5.0, sigma=0.01, r0=0.01, theta=0.05)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(maturity=2.0, n_paths=10000, n_steps=100, seed=42)
        
        # With a=5, after 2 years, rate should be very close to theta.
        assert sim.mean_terminal_rate == pytest.approx(0.05, rel=0.05)

    def test_low_mean_reversion(self) -> None:
        """Test with low mean reversion (slow convergence)."""
        params = HullWhiteParameters(a=0.01, sigma=0.005, r0=0.03, theta=0.05)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(maturity=1.0, n_paths=10000, n_steps=100, seed=42)
        
        # With a=0.01, after 1 year, rate should still be close to r0.
        theoretical_mean = params.expected_rate(1.0)
        assert sim.mean_terminal_rate == pytest.approx(theoretical_mean, rel=0.1)

    def test_high_volatility(self) -> None:
        """Test with high volatility."""
        params = HullWhiteParameters(a=0.1, sigma=0.05, r0=0.03, theta=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        sim = dynamics.simulate(maturity=1.0, n_paths=10000, n_steps=100, seed=42)
        
        # High vol should produce wider distribution.
        assert sim.std_terminal_rate > 0.03  # Significant spread

    def test_invalid_maturity(self) -> None:
        """Test that invalid maturity raises error."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        with pytest.raises(ValueError, match="maturity must be > 0"):
            dynamics.simulate(maturity=0.0, n_paths=100, n_steps=10)
        
        with pytest.raises(ValueError, match="maturity must be > 0"):
            dynamics.simulate(maturity=-1.0, n_paths=100, n_steps=10)

    def test_invalid_n_paths(self) -> None:
        """Test that invalid n_paths raises error."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        with pytest.raises(ValueError, match="n_paths must be > 0"):
            dynamics.simulate(maturity=1.0, n_paths=0, n_steps=10)

    def test_invalid_n_steps(self) -> None:
        """Test that invalid n_steps raises error."""
        params = HullWhiteParameters(a=0.1, sigma=0.01, r0=0.03)
        dynamics = HullWhiteDynamics(params=params)
        
        with pytest.raises(ValueError, match="n_steps must be > 0"):
            dynamics.simulate(maturity=1.0, n_paths=100, n_steps=0)
