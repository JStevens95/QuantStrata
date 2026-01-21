from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.scenarios.generator import ScenarioDriver, ScenarioSpec


def test_scenario_driver_validates_correlation_shape_and_spd() -> None:
    """Driver should reject invalid correlation matrices in a desk-safe way."""
    with pytest.raises(ValueError):
        ScenarioDriver(ScenarioSpec(factor_names=("A", "B"), correlation=np.eye(3)))  # shape mismatch

    with pytest.raises(ValueError):
        ScenarioDriver(ScenarioSpec(factor_names=("A", "B"), correlation=np.array([[1.0, 2.0], [0.0, 1.0]])))  # not symmetric

    # Not SPD (Cholesky fails): diag ok but matrix singular / indefinite.
    with pytest.raises(ValueError):
        ScenarioDriver(ScenarioSpec(factor_names=("A", "B"), correlation=np.array([[1.0, 1.0], [1.0, 1.0]])))


def test_scenario_driver_sample_shocks_shape_and_determinism() -> None:
    """sample_shocks must be deterministic for same seed + inputs."""
    spec = ScenarioSpec(
        factor_names=("A", "B"),
        correlation=np.array([[1.0, 0.25], [0.25, 1.0]], dtype=float),
        dt=1.0 / 252.0,
    )
    driver = ScenarioDriver(spec=spec)

    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)

    z1 = driver.sample_shocks(rng=rng1, n_time=4, n_scenarios=5)
    z2 = driver.sample_shocks(rng=rng2, n_time=4, n_scenarios=5)

    assert z1.shape == (4, 5, 2)
    assert np.array_equal(z1, z2)