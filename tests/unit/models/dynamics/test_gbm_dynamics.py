from __future__ import annotations

import math
import pytest
import numpy as np

from src.models.dynamics.gbm_dynamics import GbmDynamicsSimulator
from src.models.numeric.monte_carlo.rng import NormalRng


def _make_normals(*, seed: int, n_paths: int, n_steps: int) -> np.ndarray:
    """
    Helper: generate deterministic standard normals in the exact shape GBM expects.
    """
    rng = NormalRng(seed=seed)
    z = rng.standard_normals(n=n_paths, d=n_steps, antithetic=False, dtype=np.float64)
    assert z.shape == (n_paths, n_steps)
    return z


def test_simulate_paths_shape_and_initial_column() -> None:
    dynamics = GbmDynamicsSimulator(drift=0.02, vol=0.15)

    n_paths = 50
    n_steps = 10
    spot0 = 1.25
    maturity = 1.0

    normals = _make_normals(seed=123, n_paths=n_paths, n_steps=n_steps)
    paths = dynamics.simulate_paths(
        spot0=spot0,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        normals=normals,
        scheme="exact",
    )

    assert paths.shape == (n_paths, n_steps + 1)
    assert np.allclose(paths[:, 0], spot0)


def test_exact_gbm_zero_vol_is_deterministic_growth() -> None:
    """
    If sigma=0, GBM becomes deterministic:
      S_T = S0 * exp(mu * T)
    """
    drift = 0.03
    vol = 0.0
    dynamics = GbmDynamicsSimulator(drift=drift, vol=vol)

    n_paths = 100
    n_steps = 20
    spot0 = 2.0
    maturity = 1.5

    normals = _make_normals(seed=1, n_paths=n_paths, n_steps=n_steps)
    paths = dynamics.simulate_paths(
        spot0=spot0,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        normals=normals,
        scheme="exact",
    )

    expected_ST = spot0 * math.exp(drift * maturity)

    # All paths identical (since vol=0 ignores normals).
    assert np.allclose(paths[:, -1], expected_ST, rtol=0.0, atol=1e-12)


def test_exact_gbm_mean_matches_theory_reasonably() -> None:
    """
    For exact GBM under drift mu:
      E[S_T] = S0 * exp(mu*T)
    With finite samples, we allow a loose tolerance.
    """
    drift = 0.01
    vol = 0.2
    dynamics = GbmDynamicsSimulator(drift=drift, vol=vol)

    n_paths = 200_000  # stable mean without being too heavy
    n_steps = 1
    spot0 = 1.0
    maturity = 1.0

    normals = _make_normals(seed=123, n_paths=n_paths, n_steps=n_steps)
    paths = dynamics.simulate_paths(
        spot0=spot0,
        maturity=maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        normals=normals,
        scheme="exact",
    )

    sample_mean = float(paths[:, -1].mean())
    theoretical_mean = spot0 * math.exp(drift * maturity)

    # Monte Carlo error ~ O(1/sqrt(n)); allow a small band.
    assert sample_mean == pytest.approx(theoretical_mean, rel=5e-3, abs=0.0)


def test_invalid_normals_shape_raises() -> None:
    dynamics = GbmDynamicsSimulator(drift=0.01, vol=0.2)

    n_paths = 10
    n_steps = 5
    spot0 = 1.0
    maturity = 1.0

    # Wrong shape on purpose: (n_paths, n_steps+1)
    normals = _make_normals(seed=0, n_paths=n_paths, n_steps=n_steps + 1)

    with pytest.raises(ValueError, match="normals must have shape"):
        dynamics.simulate_paths(
            spot0=spot0,
            maturity=maturity,
            n_steps=n_steps,
            n_paths=n_paths,
            normals=normals,
            scheme="exact",
        )


def test_unknown_scheme_raises() -> None:
    dynamics = GbmDynamicsSimulator(drift=0.01, vol=0.2)

    n_paths = 10
    n_steps = 5
    normals = _make_normals(seed=0, n_paths=n_paths, n_steps=n_steps)

    with pytest.raises(ValueError, match="Unknown scheme"):
        dynamics.simulate_paths(
            spot0=1.0,
            maturity=1.0,
            n_steps=n_steps,
            n_paths=n_paths,
            normals=normals,
            scheme="does_not_exist",  # type: ignore[arg-type]
        )