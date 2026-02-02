"""
Unit tests for Merton Jump-Diffusion Model.

Tests cover:
1. Parameter validation and properties
2. Path simulation (with and without antithetic variates)
3. Exact terminal simulation
4. European option pricing (semi-closed form)
5. Implied volatility computation
6. Limiting cases (zero jumps → GBM)
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from src.models.jump_diffusion import (
    MertonParameters,
    MertonDynamics,
    MertonSimulation,
)
from src.models.jump_diffusion.merton import (
    merton_european_call,
    merton_european_put,
    merton_implied_vol,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def standard_params() -> MertonParameters:
    """Standard Merton parameters for testing."""
    return MertonParameters(
        sigma=0.2,      # 20% diffusion vol
        lambda_=0.5,    # 0.5 jumps/year
        mu_j=-0.1,      # Negative mean (crash-like)
        sigma_j=0.2,    # 20% jump size uncertainty
    )


@pytest.fixture
def zero_jump_params() -> MertonParameters:
    """Parameters with zero jump intensity (reduces to GBM)."""
    return MertonParameters(
        sigma=0.2,
        lambda_=0.0,    # No jumps
        mu_j=0.0,
        sigma_j=0.0,
    )


@pytest.fixture
def high_jump_params() -> MertonParameters:
    """High jump intensity parameters."""
    return MertonParameters(
        sigma=0.15,
        lambda_=2.0,    # 2 jumps/year expected
        mu_j=-0.05,
        sigma_j=0.15,
    )


# =============================================================================
# Parameter Validation Tests
# =============================================================================

class TestMertonParametersValidation:
    """Tests for MertonParameters validation."""

    def test_valid_parameters(self, standard_params):
        """Valid parameters should not raise."""
        assert standard_params.sigma == 0.2
        assert standard_params.lambda_ == 0.5
        assert standard_params.mu_j == -0.1
        assert standard_params.sigma_j == 0.2

    def test_negative_sigma_raises(self):
        """Negative diffusion volatility should raise."""
        with pytest.raises(ValueError, match="sigma must be >= 0"):
            MertonParameters(sigma=-0.1, lambda_=0.5, mu_j=0.0, sigma_j=0.2)

    def test_negative_lambda_raises(self):
        """Negative jump intensity should raise."""
        with pytest.raises(ValueError, match="lambda_ must be >= 0"):
            MertonParameters(sigma=0.2, lambda_=-0.5, mu_j=0.0, sigma_j=0.2)

    def test_negative_sigma_j_raises(self):
        """Negative jump size std should raise."""
        with pytest.raises(ValueError, match="sigma_j must be >= 0"):
            MertonParameters(sigma=0.2, lambda_=0.5, mu_j=0.0, sigma_j=-0.2)

    def test_nan_parameters_raise(self):
        """NaN parameters should raise."""
        with pytest.raises(ValueError, match="must be finite"):
            MertonParameters(sigma=np.nan, lambda_=0.5, mu_j=0.0, sigma_j=0.2)

    def test_inf_parameters_raise(self):
        """Infinite parameters should raise."""
        with pytest.raises(ValueError, match="must be finite"):
            MertonParameters(sigma=np.inf, lambda_=0.5, mu_j=0.0, sigma_j=0.2)

    def test_zero_sigma_allowed(self):
        """Zero diffusion volatility is allowed (pure jump process)."""
        params = MertonParameters(sigma=0.0, lambda_=0.5, mu_j=0.0, sigma_j=0.2)
        assert params.sigma == 0.0

    def test_zero_lambda_allowed(self, zero_jump_params):
        """Zero jump intensity is allowed (pure diffusion)."""
        assert zero_jump_params.lambda_ == 0.0


# =============================================================================
# Parameter Properties Tests
# =============================================================================

class TestMertonParametersProperties:
    """Tests for MertonParameters derived properties."""

    def test_expected_jump_positive_mu(self):
        """Expected jump should be > 1 for positive mu_j."""
        params = MertonParameters(sigma=0.2, lambda_=0.5, mu_j=0.1, sigma_j=0.2)
        # E[J] = exp(0.1 + 0.5*0.04) = exp(0.12) ≈ 1.127
        assert params.expected_jump > 1.0
        assert_allclose(params.expected_jump, np.exp(0.1 + 0.02), rtol=1e-10)

    def test_expected_jump_negative_mu(self, standard_params):
        """Expected jump should be < 1 for negative mu_j."""
        # E[J] = exp(-0.1 + 0.5*0.04) = exp(-0.08) ≈ 0.923
        assert standard_params.expected_jump < 1.0
        assert_allclose(standard_params.expected_jump, np.exp(-0.1 + 0.02), rtol=1e-10)

    def test_kappa_matches_expected_jump(self, standard_params):
        """kappa should equal E[J] - 1."""
        assert_allclose(standard_params.kappa, standard_params.expected_jump - 1.0, rtol=1e-10)

    def test_total_variance_rate_positive(self, standard_params):
        """Total variance rate should be positive."""
        assert standard_params.total_variance_rate > 0.0

    def test_equivalent_bs_vol_positive(self, standard_params):
        """Equivalent BS vol should be positive."""
        assert standard_params.equivalent_bs_vol > 0.0

    def test_expected_num_jumps(self, standard_params):
        """Expected number of jumps should scale with time."""
        assert_allclose(standard_params.expected_num_jumps(1.0), 0.5, rtol=1e-10)
        assert_allclose(standard_params.expected_num_jumps(2.0), 1.0, rtol=1e-10)

    def test_jump_variance_positive(self, standard_params):
        """Jump variance should be positive."""
        assert standard_params.jump_variance > 0.0


# =============================================================================
# Dynamics Simulation Tests
# =============================================================================

class TestMertonDynamicsSimulation:
    """Tests for MertonDynamics simulation."""

    def test_simulate_returns_correct_shape(self, standard_params):
        """Simulation should return correct output shapes."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=252, seed=42
        )

        # With antithetic, n_paths may round up to even
        assert sim.spot_paths.shape[1] == 253  # n_steps + 1
        assert sim.jump_counts.shape == sim.spot_paths.shape
        assert len(sim.times) == 253

    def test_simulate_initial_spot(self, standard_params):
        """Initial spot should equal spot0."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        assert_allclose(sim.spot_paths[:, 0], 100.0, rtol=1e-10)

    def test_simulate_paths_positive(self, standard_params):
        """All simulated paths should remain positive."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=252, seed=42
        )
        assert np.all(sim.spot_paths > 0.0)

    def test_simulate_reproducible_with_seed(self, standard_params):
        """Simulation with same seed should be reproducible."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)

        sim1 = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        sim2 = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )

        assert_allclose(sim1.terminal_spots, sim2.terminal_spots, rtol=1e-10)

    def test_simulate_invalid_spot_raises(self, standard_params):
        """Invalid spot0 should raise."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        with pytest.raises(ValueError, match="spot0 must be > 0"):
            dynamics.simulate(spot0=-100.0, maturity=1.0, n_paths=100, n_steps=50)

    def test_simulate_invalid_maturity_raises(self, standard_params):
        """Invalid maturity should raise."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        with pytest.raises(ValueError, match="maturity must be > 0"):
            dynamics.simulate(spot0=100.0, maturity=0.0, n_paths=100, n_steps=50)

    def test_jump_counts_non_negative(self, standard_params):
        """Jump counts should be non-negative integers."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=500, n_steps=100, seed=42
        )
        assert np.all(sim.jump_counts >= 0)
        assert sim.jump_counts.dtype == np.int32

    def test_jump_counts_non_decreasing(self, standard_params):
        """Cumulative jump counts should be non-decreasing."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=500, n_steps=100, seed=42
        )
        # Each row should be non-decreasing
        for i in range(sim.n_paths):
            diffs = np.diff(sim.jump_counts[i, :])
            assert np.all(diffs >= 0)

    def test_average_jumps_near_expected(self, standard_params):
        """Average number of jumps should be near λT."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=10000, n_steps=100, seed=42
        )
        # Expected: λT = 0.5 * 1.0 = 0.5 jumps per path
        expected_jumps = standard_params.lambda_ * 1.0
        assert_allclose(sim.average_jumps_per_path, expected_jumps, rtol=0.1)


# =============================================================================
# Zero Jump (GBM Limit) Tests
# =============================================================================

class TestMertonGBMLimit:
    """Tests for Merton with zero jumps (should reduce to GBM)."""

    def test_zero_jump_mean_matches_gbm(self, zero_jump_params):
        """With zero jumps, terminal mean should match GBM."""
        drift = 0.05
        dynamics = MertonDynamics(params=zero_jump_params, drift=drift)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=50000, n_steps=100, seed=42
        )

        # GBM expected terminal: S_0 * exp(μT) = 100 * exp(0.05) ≈ 105.13
        expected_mean = 100.0 * np.exp(drift * 1.0)
        actual_mean = np.mean(sim.terminal_spots)
        assert_allclose(actual_mean, expected_mean, rtol=0.02)

    def test_zero_jump_no_jumps_occur(self, zero_jump_params):
        """With λ=0, no jumps should occur."""
        dynamics = MertonDynamics(params=zero_jump_params, drift=0.05)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=100, seed=42
        )
        assert sim.average_jumps_per_path == 0.0
        assert sim.paths_with_jumps == 0


# =============================================================================
# Exact Terminal Simulation Tests
# =============================================================================

class TestMertonExactSimulation:
    """Tests for exact terminal spot simulation."""

    def test_exact_simulation_shape(self, standard_params):
        """Exact simulation should return correct shape."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        S_T = dynamics.simulate_exact(spot0=100.0, maturity=1.0, n_paths=1000, seed=42)
        assert S_T.shape == (1000,)

    def test_exact_simulation_positive(self, standard_params):
        """Exact terminal spots should be positive."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        S_T = dynamics.simulate_exact(spot0=100.0, maturity=1.0, n_paths=5000, seed=42)
        assert np.all(S_T > 0.0)

    def test_exact_simulation_reproducible(self, standard_params):
        """Exact simulation with same seed should be reproducible."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)

        S_T_1 = dynamics.simulate_exact(spot0=100.0, maturity=1.0, n_paths=1000, seed=42)
        S_T_2 = dynamics.simulate_exact(spot0=100.0, maturity=1.0, n_paths=1000, seed=42)

        assert_allclose(S_T_1, S_T_2, rtol=1e-10)

    def test_exact_vs_path_terminal_mean(self, standard_params):
        """Exact and path simulation should give similar terminal means."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)

        S_T_exact = dynamics.simulate_exact(spot0=100.0, maturity=1.0, n_paths=50000, seed=42)

        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=50000, n_steps=100, seed=123
        )

        # Means should be close (both are unbiased estimators)
        assert_allclose(np.mean(S_T_exact), np.mean(sim.terminal_spots), rtol=0.02)


# =============================================================================
# European Option Pricing Tests
# =============================================================================

class TestMertonEuropeanPricing:
    """Tests for Merton European option pricing."""

    def test_call_positive(self):
        """Call price should be positive."""
        price = merton_european_call(
            S=100, K=100, T=1.0, r=0.05, q=0.0,
            sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2
        )
        assert price > 0.0

    def test_put_positive(self):
        """Put price should be positive."""
        price = merton_european_put(
            S=100, K=100, T=1.0, r=0.05, q=0.0,
            sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2
        )
        assert price > 0.0

    def test_put_call_parity(self):
        """Put-call parity: C - P = S*exp(-qT) - K*exp(-rT)."""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02
        sigma, lambda_, mu_j, sigma_j = 0.2, 0.5, -0.1, 0.2

        call = merton_european_call(S, K, T, r, q, sigma, lambda_, mu_j, sigma_j)
        put = merton_european_put(S, K, T, r, q, sigma, lambda_, mu_j, sigma_j)

        parity = S * np.exp(-q * T) - K * np.exp(-r * T)
        assert_allclose(call - put, parity, rtol=1e-6)

    def test_zero_lambda_equals_black_scholes(self):
        """With λ=0, Merton should equal Black-Scholes."""
        from scipy.stats import norm

        S, K, T, r, q, sigma = 100.0, 100.0, 1.0, 0.05, 0.02, 0.2

        # Merton with zero jumps
        merton_call = merton_european_call(S, K, T, r, q, sigma, lambda_=0.0, mu_j=0.0, sigma_j=0.0)

        # Black-Scholes
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        bs_call = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

        assert_allclose(merton_call, bs_call, rtol=1e-6)

    def test_call_increases_with_spot(self):
        """Call price should increase with spot."""
        kwargs = dict(K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2)
        c1 = merton_european_call(S=90, **kwargs)
        c2 = merton_european_call(S=100, **kwargs)
        c3 = merton_european_call(S=110, **kwargs)
        assert c1 < c2 < c3

    def test_call_decreases_with_strike(self):
        """Call price should decrease with strike."""
        kwargs = dict(S=100, T=1.0, r=0.05, q=0.0, sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2)
        c1 = merton_european_call(K=90, **kwargs)
        c2 = merton_european_call(K=100, **kwargs)
        c3 = merton_european_call(K=110, **kwargs)
        assert c1 > c2 > c3

    def test_call_increases_with_volatility(self):
        """Call price should increase with volatility."""
        kwargs = dict(S=100, K=100, T=1.0, r=0.05, q=0.0, lambda_=0.5, mu_j=-0.1, sigma_j=0.2)
        c1 = merton_european_call(sigma=0.15, **kwargs)
        c2 = merton_european_call(sigma=0.25, **kwargs)
        c3 = merton_european_call(sigma=0.35, **kwargs)
        assert c1 < c2 < c3

    def test_negative_jump_increases_put(self):
        """Negative jump mean should increase put prices (crash protection)."""
        kwargs = dict(S=100, K=100, T=1.0, r=0.05, q=0.0, sigma=0.2, lambda_=0.5, sigma_j=0.2)
        put_no_jump = merton_european_put(mu_j=0.0, **kwargs)
        put_neg_jump = merton_european_put(mu_j=-0.2, **kwargs)
        assert put_neg_jump > put_no_jump


# =============================================================================
# Implied Volatility Tests
# =============================================================================

class TestMertonImpliedVol:
    """Tests for Merton implied volatility."""

    def test_implied_vol_positive(self):
        """Implied vol should be positive."""
        iv = merton_implied_vol(
            S=100, K=100, T=1.0, r=0.05, q=0.0,
            sigma=0.2, lambda_=0.5, mu_j=-0.1, sigma_j=0.2
        )
        assert iv > 0.0

    def test_implied_vol_zero_lambda_equals_sigma(self):
        """With λ=0, implied vol should equal σ."""
        sigma = 0.25
        iv = merton_implied_vol(
            S=100, K=100, T=1.0, r=0.05, q=0.0,
            sigma=sigma, lambda_=0.0, mu_j=0.0, sigma_j=0.0
        )
        assert_allclose(iv, sigma, rtol=1e-4)

    def test_implied_vol_smile_negative_jumps(self):
        """Negative jump mean should create downside skew (higher OTM put vol)."""
        kwargs = dict(S=100, T=1.0, r=0.05, q=0.0, sigma=0.2, lambda_=0.5, mu_j=-0.15, sigma_j=0.2)

        iv_otm_put = merton_implied_vol(K=90, **kwargs)   # OTM put
        iv_atm = merton_implied_vol(K=100, **kwargs)       # ATM
        iv_otm_call = merton_implied_vol(K=110, **kwargs)  # OTM call

        # With negative jumps, OTM puts should have higher implied vol
        assert iv_otm_put > iv_atm


# =============================================================================
# Monte Carlo vs Analytic Tests
# =============================================================================

class TestMertonMCvsAnalytic:
    """Tests comparing Monte Carlo to analytic pricing."""

    def test_mc_call_matches_analytic(self, standard_params):
        """MC call price should match analytic within tolerance."""
        S, K, T, r, q = 100.0, 100.0, 1.0, 0.05, 0.02

        # Analytic price
        analytic = merton_european_call(
            S, K, T, r, q,
            standard_params.sigma, standard_params.lambda_,
            standard_params.mu_j, standard_params.sigma_j
        )

        # MC price
        dynamics = MertonDynamics(params=standard_params, drift=r - q)
        S_T = dynamics.simulate_exact(spot0=S, maturity=T, n_paths=100000, seed=42)
        payoffs = np.maximum(S_T - K, 0)
        mc_price = np.exp(-r * T) * np.mean(payoffs)

        # Should match within 2%
        assert_allclose(mc_price, analytic, rtol=0.02)

    def test_mc_put_matches_analytic(self, standard_params):
        """MC put price should match analytic within tolerance."""
        S, K, T, r, q = 100.0, 105.0, 0.5, 0.05, 0.02

        # Analytic price
        analytic = merton_european_put(
            S, K, T, r, q,
            standard_params.sigma, standard_params.lambda_,
            standard_params.mu_j, standard_params.sigma_j
        )

        # MC price
        dynamics = MertonDynamics(params=standard_params, drift=r - q)
        S_T = dynamics.simulate_exact(spot0=S, maturity=T, n_paths=100000, seed=42)
        payoffs = np.maximum(K - S_T, 0)
        mc_price = np.exp(-r * T) * np.mean(payoffs)

        # Should match within 2%
        assert_allclose(mc_price, analytic, rtol=0.02)


# =============================================================================
# Simulation Output Properties Tests
# =============================================================================

class TestMertonSimulationProperties:
    """Tests for MertonSimulation output properties."""

    def test_maturity_property(self, standard_params):
        """maturity property should return correct value."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(spot0=100.0, maturity=0.75, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.maturity, 0.75, rtol=1e-10)

    def test_terminal_spots_property(self, standard_params):
        """terminal_spots should return last column."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        assert_allclose(sim.terminal_spots, sim.spot_paths[:, -1], rtol=1e-10)

    def test_jump_fraction_bounded(self, standard_params):
        """Jump fraction should be between 0 and 1."""
        dynamics = MertonDynamics(params=standard_params, drift=0.05)
        sim = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=1000, n_steps=50, seed=42)
        assert 0.0 <= sim.jump_fraction <= 1.0
