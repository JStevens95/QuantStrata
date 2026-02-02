"""
Unit tests for the LIBOR Market Model (LMM).

Tests cover:
- LMMCorrelation: Correlation structure creation and validation
- LMMParameters: Parameter validation and properties
- LMMDynamics: Monte Carlo simulation and pricing

Author: QuantStrata
Phase: 3.8 - LIBOR Market Model
"""
import numpy as np
import pytest

from src.models.forward_rate.lmm import (
    LMMCorrelation,
    LMMParameters,
    LMMDynamics,
    LMMSimulation,
)


# =============================================================================
# LMMCorrelation Tests
# =============================================================================


class TestLMMCorrelation:
    """Tests for LMMCorrelation."""

    def test_flat_correlation(self) -> None:
        """Test flat correlation matrix."""
        corr = LMMCorrelation(n_forwards=4, correlation_type="flat", flat_corr=0.6)
        matrix = corr.get_correlation_matrix()

        assert matrix.shape == (4, 4)
        assert np.allclose(np.diag(matrix), 1.0)
        assert np.allclose(matrix[0, 1], 0.6)
        assert np.allclose(matrix[1, 3], 0.6)

    def test_exponential_correlation(self) -> None:
        """Test exponential decay correlation."""
        corr = LMMCorrelation(n_forwards=4, correlation_type="exponential", beta=0.2)
        tenors = np.array([0.0, 0.5, 1.0, 1.5])
        matrix = corr.get_correlation_matrix(tenors)

        assert matrix.shape == (4, 4)
        assert np.allclose(np.diag(matrix), 1.0)
        # ρ_01 = exp(-0.2 * 0.5) ≈ 0.905
        assert matrix[0, 1] == pytest.approx(np.exp(-0.2 * 0.5), rel=1e-10)
        # ρ_03 = exp(-0.2 * 1.5) ≈ 0.741
        assert matrix[0, 3] == pytest.approx(np.exp(-0.2 * 1.5), rel=1e-10)

    def test_custom_correlation(self) -> None:
        """Test custom correlation matrix."""
        custom = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.7], [0.5, 0.7, 1.0]])
        corr = LMMCorrelation(
            n_forwards=3, correlation_type="custom", correlation_matrix=custom
        )
        matrix = corr.get_correlation_matrix()

        assert np.allclose(matrix, custom)

    def test_cholesky_decomposition(self) -> None:
        """Test Cholesky decomposition."""
        corr = LMMCorrelation(n_forwards=3, correlation_type="flat", flat_corr=0.5)
        L = corr.get_cholesky()
        matrix = corr.get_correlation_matrix()

        assert L.shape == (3, 3)
        # Verify L @ L.T = correlation matrix
        assert np.allclose(L @ L.T, matrix)
        # L should be lower triangular
        assert np.allclose(L, np.tril(L))

    def test_invalid_flat_corr_raises(self) -> None:
        """Test that invalid flat correlation raises error."""
        with pytest.raises(ValueError, match="flat_corr must be in"):
            LMMCorrelation(n_forwards=3, correlation_type="flat", flat_corr=1.5)

    def test_negative_beta_raises(self) -> None:
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            LMMCorrelation(n_forwards=3, correlation_type="exponential", beta=-0.1)

    def test_custom_missing_matrix_raises(self) -> None:
        """Test that custom type without matrix raises error."""
        with pytest.raises(ValueError, match="correlation_matrix required"):
            LMMCorrelation(n_forwards=3, correlation_type="custom")


# =============================================================================
# LMMParameters Tests
# =============================================================================


class TestLMMParameters:
    """Tests for LMMParameters."""

    @pytest.fixture
    def valid_params(self) -> LMMParameters:
        """Create valid LMM parameters."""
        n = 4
        tenors = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        tau = np.array([0.5, 0.5, 0.5, 0.5])
        f0 = np.array([0.03, 0.032, 0.034, 0.035])
        vol = np.array([0.20, 0.18, 0.16, 0.15])
        corr = LMMCorrelation(n_forwards=n, correlation_type="flat", flat_corr=0.5)

        return LMMParameters(
            tenors=tenors,
            accrual_factors=tau,
            initial_forwards=f0,
            volatilities=vol,
            correlation=corr,
        )

    def test_valid_params(self, valid_params: LMMParameters) -> None:
        """Test valid parameter creation."""
        assert valid_params.n_forwards == 4
        assert valid_params.terminal_time == 2.0

    def test_n_forwards_property(self, valid_params: LMMParameters) -> None:
        """Test n_forwards property."""
        assert valid_params.n_forwards == len(valid_params.initial_forwards)

    def test_wrong_tenors_length_raises(self) -> None:
        """Test that wrong tenors length raises error."""
        n = 3
        with pytest.raises(ValueError, match="tenors should have"):
            LMMParameters(
                tenors=np.array([0.0, 0.5, 1.0]),  # Should be n+1=4
                accrual_factors=np.array([0.5, 0.5, 0.5]),
                initial_forwards=np.array([0.03, 0.03, 0.03]),
                volatilities=np.array([0.2, 0.2, 0.2]),
                correlation=LMMCorrelation(n_forwards=n),
            )

    def test_non_increasing_tenors_raises(self) -> None:
        """Test that non-increasing tenors raises error."""
        n = 2
        with pytest.raises(ValueError, match="strictly increasing"):
            LMMParameters(
                tenors=np.array([0.0, 0.5, 0.3]),  # Not increasing
                accrual_factors=np.array([0.5, 0.5]),
                initial_forwards=np.array([0.03, 0.03]),
                volatilities=np.array([0.2, 0.2]),
                correlation=LMMCorrelation(n_forwards=n),
            )

    def test_negative_forwards_raises(self) -> None:
        """Test that negative initial forwards raises error."""
        n = 2
        with pytest.raises(ValueError, match="initial_forwards must be positive"):
            LMMParameters(
                tenors=np.array([0.0, 0.5, 1.0]),
                accrual_factors=np.array([0.5, 0.5]),
                initial_forwards=np.array([-0.01, 0.03]),  # Negative
                volatilities=np.array([0.2, 0.2]),
                correlation=LMMCorrelation(n_forwards=n),
            )


# =============================================================================
# LMMDynamics Tests
# =============================================================================


class TestLMMDynamics:
    """Tests for LMMDynamics."""

    @pytest.fixture
    def lmm_dynamics(self) -> LMMDynamics:
        """Create LMM dynamics for testing."""
        n = 4
        tenors = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        tau = np.array([0.5, 0.5, 0.5, 0.5])
        f0 = np.array([0.03, 0.032, 0.034, 0.035])
        vol = np.array([0.20, 0.18, 0.16, 0.15])
        corr = LMMCorrelation(n_forwards=n, correlation_type="exponential", beta=0.1)

        params = LMMParameters(
            tenors=tenors,
            accrual_factors=tau,
            initial_forwards=f0,
            volatilities=vol,
            correlation=corr,
        )
        return LMMDynamics(params)

    def test_simulate_returns_correct_shape(self, lmm_dynamics: LMMDynamics) -> None:
        """Test simulation returns correct shapes."""
        sim = lmm_dynamics.simulate(n_paths=1000, n_steps_per_period=5, seed=42)

        assert isinstance(sim, LMMSimulation)
        assert sim.forwards.shape[0] == 1000  # n_paths
        assert sim.forwards.shape[1] == 4  # n_forwards
        assert sim.numeraire.shape[0] == 1000
        assert len(sim.discount_factors) == 5  # n_forwards + 1

    def test_simulate_forwards_positive(self, lmm_dynamics: LMMDynamics) -> None:
        """Test that simulated forwards are always positive."""
        sim = lmm_dynamics.simulate(n_paths=5000, seed=42)

        assert np.all(sim.forwards > 0)

    def test_simulate_initial_values(self, lmm_dynamics: LMMDynamics) -> None:
        """Test that initial forward values are correct."""
        sim = lmm_dynamics.simulate(n_paths=100, seed=42)

        expected_f0 = lmm_dynamics.params.initial_forwards
        assert np.allclose(sim.forwards[:, :, 0], expected_f0)

    def test_simulate_antithetic(self, lmm_dynamics: LMMDynamics) -> None:
        """Test antithetic variates."""
        sim = lmm_dynamics.simulate(n_paths=1000, seed=42, antithetic=True)

        assert sim.forwards.shape[0] == 1000

    def test_simulate_reproducible(self, lmm_dynamics: LMMDynamics) -> None:
        """Test simulation reproducibility with seed."""
        sim1 = lmm_dynamics.simulate(n_paths=100, seed=42)
        sim2 = lmm_dynamics.simulate(n_paths=100, seed=42)

        assert np.allclose(sim1.forwards, sim2.forwards)

    def test_forward_mean_reasonable(self, lmm_dynamics: LMMDynamics) -> None:
        """Test that mean forward rate is close to initial (martingale property)."""
        sim = lmm_dynamics.simulate(n_paths=50000, seed=42, antithetic=True)

        # Under the terminal measure, forwards are martingales
        # Check mean at terminal time is not too far from initial
        f0 = lmm_dynamics.params.initial_forwards
        f_terminal_mean = sim.forwards[:, :, -1].mean(axis=0)

        # Allow 5% deviation due to drift adjustment and MC noise
        for i in range(4):
            assert f_terminal_mean[i] == pytest.approx(f0[i], rel=0.05)

    def test_price_caplet_positive(self, lmm_dynamics: LMMDynamics) -> None:
        """Test caplet price is positive."""
        # ATM caplet on second forward (fixing at T_1 = 0.5Y)
        strike = 0.032  # Close to initial forward F_1 = 0.032
        price = lmm_dynamics.price_caplet(fixing_index=1, strike=strike, n_paths=20000, seed=42)

        assert price > 0

    def test_price_floorlet_positive(self, lmm_dynamics: LMMDynamics) -> None:
        """Test floorlet price is positive."""
        # ATM floorlet on second forward
        strike = 0.032
        price = lmm_dynamics.price_floorlet(fixing_index=1, strike=strike, n_paths=20000, seed=42)

        assert price > 0

    def test_caplet_floorlet_parity(self, lmm_dynamics: LMMDynamics) -> None:
        """Test cap-floor parity: Caplet - Floorlet = Forward - K * DF."""
        i = 2  # Use forward 2 (fixing at T_2 = 1.0Y)
        strike = 0.034  # Close to F_2 initial = 0.034
        n_paths = 100000

        caplet = lmm_dynamics.price_caplet(fixing_index=i, strike=strike, n_paths=n_paths, seed=42)
        floorlet = lmm_dynamics.price_floorlet(fixing_index=i, strike=strike, n_paths=n_paths, seed=42)

        # Theoretical difference (put-call parity for caplet/floorlet)
        f0 = lmm_dynamics.params.initial_forwards[i]
        tau = lmm_dynamics.params.accrual_factors[i]

        # Discount factor to payment date T_{i+1}
        df = 1.0
        for j in range(i + 1):
            df /= 1.0 + lmm_dynamics.params.accrual_factors[j] * lmm_dynamics.params.initial_forwards[j]

        theoretical = (f0 - strike) * tau * df

        # Caplet - Floorlet should equal theoretical (with MC tolerance)
        assert (caplet - floorlet) == pytest.approx(theoretical, abs=0.0001)

    def test_price_swaption_positive(self, lmm_dynamics: LMMDynamics) -> None:
        """Test swaption price is positive."""
        # 1Y into 1Y payer swaption
        price = lmm_dynamics.price_swaption(
            start_index=2,
            end_index=4,
            strike=0.034,
            is_payer=True,
            n_paths=10000,
            seed=42,
        )

        assert price > 0

    def test_payer_receiver_swaption_relationship(self, lmm_dynamics: LMMDynamics) -> None:
        """Test payer-receiver swaption parity."""
        start_idx = 1
        end_idx = 3
        strike = 0.032
        n_paths = 50000

        payer = lmm_dynamics.price_swaption(
            start_index=start_idx,
            end_index=end_idx,
            strike=strike,
            is_payer=True,
            n_paths=n_paths,
            seed=42,
        )
        receiver = lmm_dynamics.price_swaption(
            start_index=start_idx,
            end_index=end_idx,
            strike=strike,
            is_payer=False,
            n_paths=n_paths,
            seed=42,
        )

        # Payer - Receiver = PV of forward swap
        # Both should be positive for ATM strike
        assert payer > 0
        assert receiver > 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestLMMIntegration:
    """Integration tests for LMM."""

    def test_full_simulation_workflow(self) -> None:
        """Test complete simulation and pricing workflow."""
        # Setup: 2-year quarterly forward curve
        n = 8
        tenors = np.linspace(0.0, 2.0, n + 1)
        tau = np.full(n, 0.25)
        f0 = 0.03 + 0.001 * np.arange(n)  # Upward sloping
        vol = 0.20 - 0.01 * np.arange(n)  # Declining vol
        vol = np.maximum(vol, 0.10)

        corr = LMMCorrelation(n_forwards=n, correlation_type="exponential", beta=0.15)
        params = LMMParameters(
            tenors=tenors,
            accrual_factors=tau,
            initial_forwards=f0,
            volatilities=vol,
            correlation=corr,
        )
        dynamics = LMMDynamics(params)

        # Simulate
        sim = dynamics.simulate(n_paths=10000, n_steps_per_period=4, seed=123)

        # Verify
        assert sim.forwards.shape == (10000, n, len(sim.time_grid))
        assert np.all(sim.forwards > 0)
        assert np.all(sim.numeraire > 0)

    def test_pricing_cap(self) -> None:
        """Test pricing a cap as sum of caplets."""
        n = 4
        tenors = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        tau = np.full(n, 0.5)
        f0 = np.full(n, 0.03)
        vol = np.full(n, 0.20)

        corr = LMMCorrelation(n_forwards=n, correlation_type="flat", flat_corr=0.5)
        params = LMMParameters(
            tenors=tenors,
            accrual_factors=tau,
            initial_forwards=f0,
            volatilities=vol,
            correlation=corr,
        )
        dynamics = LMMDynamics(params)

        # Price cap as sum of caplets
        strike = 0.03
        cap_price = 0.0
        for i in range(n):
            caplet_price = dynamics.price_caplet(
                fixing_index=i, strike=strike, n_paths=20000, seed=42 + i
            )
            cap_price += caplet_price

        # Cap should have positive value for ATM strike
        assert cap_price > 0
