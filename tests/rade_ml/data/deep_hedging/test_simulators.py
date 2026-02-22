"""Unit tests for rade_ml.data.deep_hedging.simulators -- GBMSimulator, HestonSimulator."""
import numpy as np
import pytest

from src.rade_ml.data.deep_hedging.simulators import GBMSimulator, HestonSimulator, SimulationResult


NUM_PATHS = 500
NUM_STEPS = 50


class TestGBMSimulator:
    @pytest.fixture
    def simulator(self):
        return GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)

    def test_returns_simulation_result(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert isinstance(res, SimulationResult)

    def test_spot_path_shape(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert res.spot_paths.shape == (NUM_PATHS, NUM_STEPS + 1)

    def test_initial_spot(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        np.testing.assert_allclose(res.spot_paths[:, 0], 100.0, rtol=1e-5)

    def test_all_prices_positive(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert np.all(res.spot_paths > 0)

    def test_times_grid(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert res.times.shape == (NUM_STEPS + 1,)
        np.testing.assert_allclose(res.times[0], 0.0)
        np.testing.assert_allclose(res.times[-1], 0.25, rtol=1e-5)

    def test_dt_correct(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        np.testing.assert_allclose(res.dt, 0.25 / NUM_STEPS, rtol=1e-5)

    def test_seed_reproducibility(self, simulator):
        r1 = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        r2 = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        np.testing.assert_array_equal(r1.spot_paths, r2.spot_paths)

    def test_different_seeds_differ(self, simulator):
        r1 = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        r2 = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=99)
        assert not np.allclose(r1.spot_paths, r2.spot_paths)

    def test_no_payoffs_without_strike(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert res.payoffs is None
        assert res.bs_deltas is None


class TestGBMPayoffs:
    @pytest.fixture
    def simulator(self):
        return GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)

    def test_call_payoffs_shape(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        assert res.payoffs.shape == (NUM_PATHS,)

    def test_call_payoffs_non_negative(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        assert np.all(res.payoffs >= 0)

    def test_call_payoff_formula(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        expected = np.maximum(res.spot_paths[:, -1] - 100.0, 0.0)
        np.testing.assert_allclose(res.payoffs, expected, rtol=1e-5)

    def test_put_payoffs_non_negative(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="put",
        )
        assert np.all(res.payoffs >= 0)

    def test_put_payoff_formula(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="put",
        )
        expected = np.maximum(100.0 - res.spot_paths[:, -1], 0.0)
        np.testing.assert_allclose(res.payoffs, expected, rtol=1e-5)


class TestGBMBSDeltas:
    @pytest.fixture
    def simulator(self):
        return GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)

    def test_bs_deltas_shape(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        assert res.bs_deltas.shape == (NUM_PATHS, NUM_STEPS + 1)

    def test_call_deltas_bounded(self, simulator):
        """Call deltas should be in [0, 1]."""
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        assert np.all(res.bs_deltas >= -0.01)
        assert np.all(res.bs_deltas <= 1.01)

    def test_put_deltas_bounded(self, simulator):
        """Put deltas should be in [-1, 0]."""
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="put",
        )
        assert np.all(res.bs_deltas >= -1.01)
        assert np.all(res.bs_deltas <= 0.01)

    def test_deep_itm_call_delta_near_one(self, simulator):
        """A very deep ITM call should have delta close to 1."""
        sim = GBMSimulator(spot_0=200.0, risk_free_rate=0.05, volatility=0.01)
        res = sim.simulate(
            maturity=0.25, num_steps=10, num_paths=100,
            seed=42, strike=100.0, option_type="call",
        )
        assert np.mean(res.bs_deltas[:, 0]) > 0.95


class TestGBMStatisticalProperties:
    def test_mean_return_reasonable(self):
        """With many paths the mean terminal spot should approximate E[S_T]."""
        sim = GBMSimulator(spot_0=100.0, risk_free_rate=0.05, volatility=0.2)
        res = sim.simulate(maturity=1.0, num_steps=252, num_paths=50_000, seed=42)
        expected_mean = 100.0 * np.exp(0.05)  # risk-neutral E[S_T]
        actual_mean = res.spot_paths[:, -1].mean()
        np.testing.assert_allclose(actual_mean, expected_mean, rtol=0.02)


class TestHestonSimulator:
    @pytest.fixture
    def simulator(self):
        return HestonSimulator(
            spot_0=100.0, v0=0.04, risk_free_rate=0.05,
            kappa=1.5, theta=0.04, xi=0.3, rho=-0.7,
        )

    def test_returns_simulation_result(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert isinstance(res, SimulationResult)

    def test_spot_path_shape(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert res.spot_paths.shape == (NUM_PATHS, NUM_STEPS + 1)

    def test_vol_path_shape(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert res.vol_paths.shape == (NUM_PATHS, NUM_STEPS + 1)

    def test_vol_paths_non_negative(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert np.all(res.vol_paths >= 0)

    def test_initial_spot(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        np.testing.assert_allclose(res.spot_paths[:, 0], 100.0, rtol=1e-5)

    def test_initial_vol(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        np.testing.assert_allclose(res.vol_paths[:, 0], 0.04, rtol=1e-5)

    def test_all_prices_positive(self, simulator):
        res = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        assert np.all(res.spot_paths > 0)

    def test_call_payoffs(self, simulator):
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        expected = np.maximum(res.spot_paths[:, -1] - 100.0, 0.0)
        np.testing.assert_allclose(res.payoffs, expected, rtol=1e-5)

    def test_seed_reproducibility(self, simulator):
        r1 = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        r2 = simulator.simulate(maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS, seed=42)
        np.testing.assert_array_equal(r1.spot_paths, r2.spot_paths)

    def test_no_bs_deltas(self, simulator):
        """Heston simulator does not compute BS deltas."""
        res = simulator.simulate(
            maturity=0.25, num_steps=NUM_STEPS, num_paths=NUM_PATHS,
            seed=42, strike=100.0, option_type="call",
        )
        assert res.bs_deltas is None
