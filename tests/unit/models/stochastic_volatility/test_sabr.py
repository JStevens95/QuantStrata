"""
Unit tests for SABR Stochastic Volatility Model Dynamics.

Tests cover:
1. Simulation output shapes and properties
2. Martingale property (forward has zero drift)
3. Different discretization schemes
4. MC pricing vs analytic approximation
5. Limiting cases
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.calibration.volatility_surface.sabr import (
    SabrParameters,
    sabr_implied_vol,
)
from src.models.stochastic_volatility.sabr import (
    SabrDynamics,
    SabrSimulation,
    sabr_mc_call,
    sabr_mc_put,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def lognormal_params() -> SabrParameters:
    """Log-normal SABR (β=1) parameters."""
    return SabrParameters(
        alpha=0.3,   # 30% initial vol
        beta=1.0,    # Log-normal
        rho=-0.5,    # Negative correlation
        nu=0.4,      # Vol of vol
    )


@pytest.fixture
def normal_params() -> SabrParameters:
    """Normal SABR (β=0) parameters."""
    return SabrParameters(
        alpha=0.02,  # Absolute vol (like rates)
        beta=0.0,    # Normal
        rho=-0.3,
        nu=0.5,
    )


@pytest.fixture
def cev_params() -> SabrParameters:
    """CIR-like SABR (β=0.5) parameters."""
    return SabrParameters(
        alpha=0.15,
        beta=0.5,
        rho=-0.4,
        nu=0.35,
    )


@pytest.fixture
def zero_volvol_params() -> SabrParameters:
    """Zero vol-of-vol (reduces to CEV)."""
    return SabrParameters(
        alpha=0.25,
        beta=1.0,
        rho=0.0,
        nu=0.0,  # No stochastic vol
    )


# =============================================================================
# Simulation Shape and Property Tests
# =============================================================================

class TestSabrSimulationShape:
    """Tests for SABR simulation output shapes."""

    def test_simulate_returns_correct_shape(self, lognormal_params):
        """Simulation should return correct output shapes."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=1000, n_steps=100, seed=42
        )

        # With antithetic, n_paths rounds up to even
        assert sim.forward_paths.shape[1] == 101  # n_steps + 1
        assert sim.vol_paths.shape == sim.forward_paths.shape
        assert len(sim.times) == 101

    def test_simulate_initial_values(self, lognormal_params):
        """Initial values should match inputs."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )

        assert_allclose(sim.forward_paths[:, 0], 100.0, rtol=1e-10)
        assert_allclose(sim.vol_paths[:, 0], lognormal_params.alpha, rtol=1e-10)

    def test_simulate_reproducible_with_seed(self, lognormal_params):
        """Simulation with same seed should be reproducible."""
        dynamics = SabrDynamics(params=lognormal_params)

        sim1 = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        sim2 = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )

        assert_allclose(sim1.terminal_forwards, sim2.terminal_forwards, rtol=1e-10)

    def test_simulate_invalid_forward_raises(self, lognormal_params):
        """Invalid forward0 should raise."""
        dynamics = SabrDynamics(params=lognormal_params)
        with pytest.raises(ValueError, match="forward0 must be > 0"):
            dynamics.simulate(forward0=-100.0, maturity=1.0, n_paths=100, n_steps=50)

    def test_simulate_invalid_maturity_raises(self, lognormal_params):
        """Invalid maturity should raise."""
        dynamics = SabrDynamics(params=lognormal_params)
        with pytest.raises(ValueError, match="maturity must be > 0"):
            dynamics.simulate(forward0=100.0, maturity=0.0, n_paths=100, n_steps=50)


# =============================================================================
# Martingale Property Tests
# =============================================================================

class TestSabrMartingale:
    """Tests for martingale property (E[F_T] = F_0)."""

    def test_lognormal_martingale(self, lognormal_params):
        """Log-normal SABR should preserve martingale property."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=50000, n_steps=100, seed=42
        )

        # E[F_T] should be approximately F_0
        mean_terminal = np.mean(sim.terminal_forwards)
        assert_allclose(mean_terminal, 100.0, rtol=0.02)

    def test_cev_martingale(self, cev_params):
        """CEV SABR should preserve martingale property."""
        dynamics = SabrDynamics(params=cev_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=50000, n_steps=100, seed=42
        )

        # E[F_T] should be approximately F_0
        mean_terminal = np.mean(sim.terminal_forwards)
        assert_allclose(mean_terminal, 100.0, rtol=0.03)

    def test_zero_volvol_martingale(self, zero_volvol_params):
        """Zero vol-of-vol SABR should still be martingale."""
        dynamics = SabrDynamics(params=zero_volvol_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=30000, n_steps=100, seed=42
        )

        mean_terminal = np.mean(sim.terminal_forwards)
        assert_allclose(mean_terminal, 100.0, rtol=0.02)


# =============================================================================
# Discretization Scheme Tests
# =============================================================================

class TestSabrSchemes:
    """Tests for different discretization schemes."""

    def test_euler_scheme(self, lognormal_params):
        """Euler scheme should work for β=1."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=0.5, n_paths=1000, n_steps=50,
            scheme="euler", seed=42
        )
        assert np.all(np.isfinite(sim.terminal_forwards))

    def test_log_euler_scheme(self, lognormal_params):
        """Log-Euler scheme should work and stay positive."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=1000, n_steps=100,
            scheme="log_euler", seed=42
        )
        assert np.all(sim.forward_paths > 0)

    def test_absorbing_scheme(self, cev_params):
        """Absorbing scheme should handle β < 1 without negative values."""
        dynamics = SabrDynamics(params=cev_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=1000, n_steps=100,
            scheme="absorbing", seed=42
        )
        assert np.all(sim.forward_paths >= 0)

    def test_invalid_scheme_raises(self, lognormal_params):
        """Invalid scheme should raise."""
        dynamics = SabrDynamics(params=lognormal_params)
        with pytest.raises(ValueError, match="Unknown scheme"):
            dynamics.simulate(
                forward0=100.0, maturity=1.0, n_paths=100, n_steps=50,
                scheme="invalid"
            )


# =============================================================================
# Volatility Process Tests
# =============================================================================

class TestSabrVolatilityProcess:
    """Tests for stochastic volatility process."""

    def test_vol_always_positive(self, lognormal_params):
        """Volatility should always be positive (log-normal dynamics)."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=2.0, n_paths=5000, n_steps=200, seed=42
        )
        assert np.all(sim.vol_paths > 0)

    def test_vol_mean_reversion_like(self, lognormal_params):
        """Terminal vol mean should be near initial (no drift)."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=0.5, n_paths=20000, n_steps=50, seed=42
        )

        # Vol is a martingale: E[σ_T] ≈ σ_0
        # Actually, for GBM vol: E[σ_T] = σ_0 * exp(0) = σ_0
        # But due to log-normal, E[σ_T] = σ_0 * exp(ν²T/2) ≠ σ_0
        # The correct test is that the dynamics work properly
        mean_vol = np.mean(sim.terminal_vols)
        expected_mean = lognormal_params.alpha * np.exp(0.5 * lognormal_params.nu**2 * 0.5)
        assert_allclose(mean_vol, expected_mean, rtol=0.05)

    def test_zero_volvol_constant_vol(self, zero_volvol_params):
        """With ν=0, volatility should remain constant."""
        dynamics = SabrDynamics(params=zero_volvol_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )

        # All vol paths should equal initial alpha
        assert_allclose(sim.vol_paths, zero_volvol_params.alpha, rtol=1e-10)


# =============================================================================
# MC Pricing Tests
# =============================================================================

class TestSabrMCPricing:
    """Tests for SABR Monte Carlo pricing."""

    def test_call_positive(self, lognormal_params):
        """Call price should be positive."""
        price = sabr_mc_call(
            forward0=100.0, strike=100.0, maturity=1.0,
            discount_factor=0.95, params=lognormal_params,
            n_paths=10000, seed=42
        )
        assert price > 0.0

    def test_put_positive(self, lognormal_params):
        """Put price should be positive."""
        price = sabr_mc_put(
            forward0=100.0, strike=100.0, maturity=1.0,
            discount_factor=0.95, params=lognormal_params,
            n_paths=10000, seed=42
        )
        assert price > 0.0

    def test_put_call_parity(self, lognormal_params):
        """Put-call parity: C - P = DF * (F - K)."""
        F, K, DF = 100.0, 95.0, 0.95

        call = sabr_mc_call(
            forward0=F, strike=K, maturity=1.0,
            discount_factor=DF, params=lognormal_params,
            n_paths=50000, seed=42
        )
        put = sabr_mc_put(
            forward0=F, strike=K, maturity=1.0,
            discount_factor=DF, params=lognormal_params,
            n_paths=50000, seed=42
        )

        parity = DF * (F - K)
        assert_allclose(call - put, parity, rtol=0.02)

    def test_call_increases_with_strike(self):
        """Call should decrease with strike (inverse)."""
        params = SabrParameters(alpha=0.3, beta=1.0, rho=-0.3, nu=0.4)

        c_itm = sabr_mc_call(forward0=100, strike=90, maturity=0.5, discount_factor=0.98, params=params, n_paths=20000, seed=42)
        c_atm = sabr_mc_call(forward0=100, strike=100, maturity=0.5, discount_factor=0.98, params=params, n_paths=20000, seed=42)
        c_otm = sabr_mc_call(forward0=100, strike=110, maturity=0.5, discount_factor=0.98, params=params, n_paths=20000, seed=42)

        assert c_itm > c_atm > c_otm


# =============================================================================
# MC vs Analytic Comparison Tests
# =============================================================================

class TestSabrMCvsAnalytic:
    """Tests comparing MC to analytic Hagan approximation."""

    def test_atm_mc_vs_analytic(self, lognormal_params):
        """ATM MC price should be close to analytic Black price using SABR vol."""
        from scipy.stats import norm

        F, K, T, DF = 100.0, 100.0, 0.5, 0.975

        # Get SABR implied vol
        sigma = sabr_implied_vol(
            forward=F, strike=K, expiry=T, params=lognormal_params
        )

        # Black call price
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        black_call = DF * (F * norm.cdf(d1) - K * norm.cdf(d2))

        # MC call price
        mc_call = sabr_mc_call(
            forward0=F, strike=K, maturity=T, discount_factor=DF,
            params=lognormal_params, n_paths=100000, seed=42
        )

        # Should match within 3%
        assert_allclose(mc_call, black_call, rtol=0.03)

    def test_otm_mc_vs_analytic(self, lognormal_params):
        """OTM MC price should be close to analytic Black price using SABR vol."""
        from scipy.stats import norm

        F, K, T, DF = 100.0, 110.0, 0.5, 0.975

        # Get SABR implied vol at this strike
        sigma = sabr_implied_vol(
            forward=F, strike=K, expiry=T, params=lognormal_params
        )

        # Black call price
        d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        black_call = DF * (F * norm.cdf(d1) - K * norm.cdf(d2))

        # MC call price
        mc_call = sabr_mc_call(
            forward0=F, strike=K, maturity=T, discount_factor=DF,
            params=lognormal_params, n_paths=100000, seed=42
        )

        # Should match within 5% (OTM has more error)
        assert_allclose(mc_call, black_call, rtol=0.05)


# =============================================================================
# Simulation Properties Tests
# =============================================================================

class TestSabrSimulationProperties:
    """Tests for SabrSimulation output properties."""

    def test_maturity_property(self, lognormal_params):
        """maturity property should return correct value."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(forward0=100.0, maturity=0.75, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.maturity, 0.75, rtol=1e-10)

    def test_terminal_forwards_property(self, lognormal_params):
        """terminal_forwards should return last column."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(forward0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.terminal_forwards, sim.forward_paths[:, -1], rtol=1e-10)

    def test_terminal_vols_property(self, lognormal_params):
        """terminal_vols should return last column."""
        dynamics = SabrDynamics(params=lognormal_params)
        sim = dynamics.simulate(forward0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.terminal_vols, sim.vol_paths[:, -1], rtol=1e-10)

    def test_absorption_fraction_bounded(self, cev_params):
        """Absorption fraction should be between 0 and 1."""
        dynamics = SabrDynamics(params=cev_params)
        sim = dynamics.simulate(
            forward0=100.0, maturity=1.0, n_paths=1000, n_steps=50,
            scheme="absorbing", seed=42
        )
        assert 0.0 <= sim.absorption_fraction <= 1.0
