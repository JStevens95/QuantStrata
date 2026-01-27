"""
Unit tests for Heston Stochastic Volatility Model.

Tests cover:
1. HestonParameters construction and validation
2. Feller condition checking
3. HestonDynamics simulation
4. Path statistics and distributions
5. Discretization schemes comparison
"""

import numpy as np
import pytest

from src.models.stochastic_volatility.heston import (
    HestonParameters,
    HestonDynamics,
    HestonSimulation,
)


# =============================================================================
# HestonParameters Tests
# =============================================================================

class TestHestonParameters:
    """Tests for Heston model parameters."""

    def test_construction_valid(self) -> None:
        """Test valid parameter construction."""
        params = HestonParameters(
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            v0=0.04,
            rho=-0.7,
        )
        assert params.kappa == 2.0
        assert params.theta == 0.04
        assert params.xi == 0.3
        assert params.v0 == 0.04
        assert params.rho == -0.7

    def test_construction_invalid_kappa_zero(self) -> None:
        """Test that kappa=0 raises ValueError."""
        with pytest.raises(ValueError, match="kappa must be > 0"):
            HestonParameters(kappa=0.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)

    def test_construction_invalid_kappa_negative(self) -> None:
        """Test that negative kappa raises ValueError."""
        with pytest.raises(ValueError, match="kappa must be > 0"):
            HestonParameters(kappa=-1.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)

    def test_construction_invalid_theta_zero(self) -> None:
        """Test that theta=0 raises ValueError."""
        with pytest.raises(ValueError, match="theta must be > 0"):
            HestonParameters(kappa=2.0, theta=0.0, xi=0.3, v0=0.04, rho=-0.7)

    def test_construction_invalid_xi_zero(self) -> None:
        """Test that xi=0 raises ValueError."""
        with pytest.raises(ValueError, match="xi must be > 0"):
            HestonParameters(kappa=2.0, theta=0.04, xi=0.0, v0=0.04, rho=-0.7)

    def test_construction_invalid_v0_zero(self) -> None:
        """Test that v0=0 raises ValueError."""
        with pytest.raises(ValueError, match="v0 must be > 0"):
            HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.0, rho=-0.7)

    def test_construction_invalid_rho_out_of_bounds(self) -> None:
        """Test that rho outside (-1, 1) raises ValueError."""
        with pytest.raises(ValueError, match="rho must be in"):
            HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-1.0)

        with pytest.raises(ValueError, match="rho must be in"):
            HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=1.0)

    def test_feller_condition_satisfied(self) -> None:
        """Test Feller condition detection when satisfied."""
        # 2κθ/ξ² = 2*2*0.04/0.09 = 1.78 > 1.
        params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
        assert params.feller_satisfied is True
        assert params.feller_ratio > 1.0

    def test_feller_condition_not_satisfied(self) -> None:
        """Test Feller condition detection when not satisfied."""
        # 2κθ/ξ² = 2*0.5*0.04/1.0 = 0.04 < 1.
        params = HestonParameters(kappa=0.5, theta=0.04, xi=1.0, v0=0.04, rho=-0.7)
        assert params.feller_satisfied is False
        assert params.feller_ratio < 1.0

    def test_long_term_vol(self) -> None:
        """Test long-term volatility calculation."""
        params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
        assert params.long_term_vol == pytest.approx(0.20, rel=1e-6)

    def test_initial_vol(self) -> None:
        """Test initial volatility calculation."""
        params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.09, rho=-0.7)
        assert params.initial_vol == pytest.approx(0.30, rel=1e-6)

    def test_expected_variance(self) -> None:
        """Test expected variance at time t."""
        params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.09, rho=-0.7)

        # At t=0, E[V_0] = v0.
        assert params.expected_variance(0.0) == pytest.approx(0.09, rel=1e-6)

        # At t=∞, E[V_∞] → θ.
        assert params.expected_variance(100.0) == pytest.approx(0.04, rel=1e-3)

        # At intermediate t, check formula.
        t = 1.0
        expected = 0.04 + (0.09 - 0.04) * np.exp(-2.0 * 1.0)
        assert params.expected_variance(t) == pytest.approx(expected, rel=1e-6)


# =============================================================================
# HestonDynamics Tests
# =============================================================================

class TestHestonDynamics:
    """Tests for Heston dynamics simulation."""

    @pytest.fixture
    def default_params(self) -> HestonParameters:
        """Create default Heston parameters."""
        return HestonParameters(
            kappa=2.0,
            theta=0.04,
            xi=0.3,
            v0=0.04,
            rho=-0.7,
        )

    @pytest.fixture
    def dynamics(self, default_params: HestonParameters) -> HestonDynamics:
        """Create Heston dynamics with default parameters."""
        return HestonDynamics(params=default_params, drift=0.03)

    def test_construction(self, default_params: HestonParameters) -> None:
        """Test dynamics construction."""
        dynamics = HestonDynamics(params=default_params, drift=0.05)
        assert dynamics.params == default_params
        assert dynamics.drift == 0.05

    def test_simulation_returns_simulation_object(self, dynamics: HestonDynamics) -> None:
        """Test that simulate returns HestonSimulation."""
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )
        assert isinstance(sim, HestonSimulation)

    def test_simulation_shape(self, dynamics: HestonDynamics) -> None:
        """Test simulation output shapes."""
        n_paths = 1000
        n_steps = 100
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=n_paths, n_steps=n_steps,
            seed=42, antithetic=False
        )

        # Without antithetic, should have n_paths paths.
        assert sim.spot_paths.shape == (n_paths, n_steps + 1)
        assert sim.variance_paths.shape == (n_paths, n_steps + 1)
        assert sim.times.shape == (n_steps + 1,)

    def test_simulation_shape_with_antithetic(self, dynamics: HestonDynamics) -> None:
        """Test simulation shapes with antithetic variates."""
        n_paths = 1000
        n_steps = 100
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=n_paths, n_steps=n_steps,
            seed=42, antithetic=True
        )

        # With antithetic, should have 2 * ceil(n_paths/2) paths.
        assert sim.spot_paths.shape[0] >= n_paths
        assert sim.spot_paths.shape[1] == n_steps + 1

    def test_simulation_initial_values(self, dynamics: HestonDynamics) -> None:
        """Test that simulation starts at correct initial values."""
        spot0 = 100.0
        sim = dynamics.simulate(
            spot0=spot0, maturity=1.0, n_paths=100, n_steps=50, seed=42
        )

        # All paths start at spot0.
        np.testing.assert_array_equal(sim.spot_paths[:, 0], spot0)

        # All paths start at v0.
        np.testing.assert_array_equal(sim.variance_paths[:, 0], dynamics.params.v0)

    def test_simulation_positive_spots(self, dynamics: HestonDynamics) -> None:
        """Test that simulated spots are always positive."""
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=100, seed=42
        )
        assert np.all(sim.spot_paths > 0)

    def test_simulation_non_negative_variance(self, dynamics: HestonDynamics) -> None:
        """Test that simulated variances are non-negative with truncation."""
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=1000, n_steps=100,
            scheme="full_truncation", seed=42
        )
        assert np.all(sim.variance_paths >= 0)

    def test_simulation_reproducibility(self, dynamics: HestonDynamics) -> None:
        """Test that same seed gives same results."""
        sim1 = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        sim2 = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)

        np.testing.assert_array_almost_equal(sim1.spot_paths, sim2.spot_paths)
        np.testing.assert_array_almost_equal(sim1.variance_paths, sim2.variance_paths)

    def test_simulation_different_seeds(self, dynamics: HestonDynamics) -> None:
        """Test that different seeds give different results."""
        sim1 = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=42)
        sim2 = dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=50, seed=43)

        # Should be different (statistically almost certain).
        assert not np.allclose(sim1.spot_paths, sim2.spot_paths)

    def test_mean_terminal_spot(self, dynamics: HestonDynamics) -> None:
        """Test that mean terminal spot is approximately E[S_T]."""
        spot0 = 100.0
        T = 1.0
        drift = dynamics.drift

        # E[S_T] = S_0 * exp(drift * T) under risk-neutral measure.
        expected_mean = spot0 * np.exp(drift * T)

        # Simulate many paths for accuracy.
        sim = dynamics.simulate(
            spot0=spot0, maturity=T, n_paths=100000, n_steps=252,
            seed=42, antithetic=True
        )

        actual_mean = np.mean(sim.terminal_spots)

        # Should be close (within 2% due to MC noise).
        assert actual_mean == pytest.approx(expected_mean, rel=0.02)

    def test_variance_mean_reversion(self, default_params: HestonParameters) -> None:
        """Test that variance mean reverts toward theta."""
        # Start with v0 much higher than theta.
        params = HestonParameters(
            kappa=5.0,  # Fast mean reversion.
            theta=0.04,
            xi=0.2,
            v0=0.16,  # Start at 40% vol, should revert to 20%.
            rho=-0.5,
        )
        dynamics = HestonDynamics(params=params, drift=0.0)

        sim = dynamics.simulate(
            spot0=100.0, maturity=2.0, n_paths=10000, n_steps=200,
            seed=42, antithetic=True
        )

        # Mean terminal variance should be close to E[V_T].
        expected_mean_var = params.expected_variance(2.0)
        actual_mean_var = np.mean(sim.terminal_variances)

        assert actual_mean_var == pytest.approx(expected_mean_var, rel=0.05)

    def test_correlation_effect(self) -> None:
        """Test that negative rho gives negative spot-vol correlation."""
        params_neg_rho = HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.8
        )
        params_pos_rho = HestonParameters(
            kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=0.8
        )

        dyn_neg = HestonDynamics(params=params_neg_rho, drift=0.0)
        dyn_pos = HestonDynamics(params=params_pos_rho, drift=0.0)

        sim_neg = dyn_neg.simulate(spot0=100.0, maturity=1.0, n_paths=10000, n_steps=100, seed=42)
        sim_pos = dyn_pos.simulate(spot0=100.0, maturity=1.0, n_paths=10000, n_steps=100, seed=42)

        # Compute correlation of terminal spots with terminal variance.
        corr_neg = np.corrcoef(sim_neg.terminal_spots, sim_neg.terminal_variances)[0, 1]
        corr_pos = np.corrcoef(sim_pos.terminal_spots, sim_pos.terminal_variances)[0, 1]

        # Negative rho should give negative correlation.
        assert corr_neg < 0
        # Positive rho should give positive correlation.
        assert corr_pos > 0


# =============================================================================
# Discretization Scheme Tests
# =============================================================================

class TestHestonSchemes:
    """Tests comparing different discretization schemes."""

    @pytest.fixture
    def params(self) -> HestonParameters:
        """Parameters that may cause negative variance without care."""
        return HestonParameters(
            kappa=1.0,
            theta=0.04,
            xi=0.5,  # High vol of vol.
            v0=0.04,
            rho=-0.7,
        )

    def test_full_truncation_non_negative(self, params: HestonParameters) -> None:
        """Test full truncation keeps variance non-negative."""
        dynamics = HestonDynamics(params=params, drift=0.03)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=5000, n_steps=100,
            scheme="full_truncation", seed=42
        )
        assert np.all(sim.variance_paths >= 0)

    def test_reflection_non_negative(self, params: HestonParameters) -> None:
        """Test reflection scheme keeps variance non-negative."""
        dynamics = HestonDynamics(params=params, drift=0.03)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=5000, n_steps=100,
            scheme="reflection", seed=42
        )
        assert np.all(sim.variance_paths >= 0)

    def test_qe_non_negative(self, params: HestonParameters) -> None:
        """Test QE scheme keeps variance non-negative."""
        dynamics = HestonDynamics(params=params, drift=0.03)
        sim = dynamics.simulate(
            spot0=100.0, maturity=1.0, n_paths=5000, n_steps=100,
            scheme="qe", seed=42
        )
        assert np.all(sim.variance_paths >= 0)

    def test_schemes_give_similar_prices(self, params: HestonParameters) -> None:
        """Test that all schemes give similar European option prices."""
        dynamics = HestonDynamics(params=params, drift=0.03)
        spot0 = 100.0
        strike = 100.0
        T = 1.0
        r = 0.05

        prices = {}
        for scheme in ["full_truncation", "reflection", "qe"]:
            sim = dynamics.simulate(
                spot0=spot0, maturity=T, n_paths=50000, n_steps=200,
                scheme=scheme, seed=42, antithetic=True  # type: ignore
            )
            payoffs = np.maximum(sim.terminal_spots - strike, 0)
            prices[scheme] = np.exp(-r * T) * np.mean(payoffs)

        # All schemes should give similar prices (within 5% of each other).
        price_list = list(prices.values())
        mean_price = np.mean(price_list)
        for scheme, price in prices.items():
            assert price == pytest.approx(mean_price, rel=0.05), f"Scheme {scheme} diverges"


# =============================================================================
# Validation Tests
# =============================================================================

class TestHestonValidation:
    """Tests for input validation in Heston dynamics."""

    @pytest.fixture
    def dynamics(self) -> HestonDynamics:
        """Create default dynamics."""
        params = HestonParameters(kappa=2.0, theta=0.04, xi=0.3, v0=0.04, rho=-0.7)
        return HestonDynamics(params=params, drift=0.03)

    def test_invalid_spot_zero(self, dynamics: HestonDynamics) -> None:
        """Test that spot=0 raises ValueError."""
        with pytest.raises(ValueError, match="spot0 must be > 0"):
            dynamics.simulate(spot0=0.0, maturity=1.0, n_paths=100, n_steps=50)

    def test_invalid_spot_negative(self, dynamics: HestonDynamics) -> None:
        """Test that negative spot raises ValueError."""
        with pytest.raises(ValueError, match="spot0 must be > 0"):
            dynamics.simulate(spot0=-100.0, maturity=1.0, n_paths=100, n_steps=50)

    def test_invalid_maturity_zero(self, dynamics: HestonDynamics) -> None:
        """Test that maturity=0 raises ValueError."""
        with pytest.raises(ValueError, match="maturity must be > 0"):
            dynamics.simulate(spot0=100.0, maturity=0.0, n_paths=100, n_steps=50)

    def test_invalid_maturity_negative(self, dynamics: HestonDynamics) -> None:
        """Test that negative maturity raises ValueError."""
        with pytest.raises(ValueError, match="maturity must be > 0"):
            dynamics.simulate(spot0=100.0, maturity=-1.0, n_paths=100, n_steps=50)

    def test_invalid_n_paths_zero(self, dynamics: HestonDynamics) -> None:
        """Test that n_paths=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_paths must be > 0"):
            dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=0, n_steps=50)

    def test_invalid_n_steps_zero(self, dynamics: HestonDynamics) -> None:
        """Test that n_steps=0 raises ValueError."""
        with pytest.raises(ValueError, match="n_steps must be > 0"):
            dynamics.simulate(spot0=100.0, maturity=1.0, n_paths=100, n_steps=0)
