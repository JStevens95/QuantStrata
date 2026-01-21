from __future__ import annotations

import numpy as np

from src.marketdata.synthetic.specs import CurveZeroSpec, SpotGbmSpec, VolGridSmileSpec


def test_spot_gbm_spec_can_be_constructed() -> None:
    """
    Smoke test: SpotGbmSpec is constructible and stores values.

    We keep this light because specs are dataclasses; heavy validation is optional.
    """
    spec = SpotGbmSpec(initial_level=1.10)

    assert float(spec.initial_level) == 1.10
    assert float(spec.dt) > 0.0


def test_curve_zero_spec_stores_tenors() -> None:
    """
    CurveZeroSpec should store a tenor grid.
    """
    tenors = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)
    spec = CurveZeroSpec(tenors=tenors)

    # Ensure the array round-trips as expected.
    assert np.asarray(spec.tenors, dtype=float).shape == (4,)
    assert float(np.asarray(spec.tenors, dtype=float)[0]) == 0.25


def test_vol_grid_smile_spec_stores_grid_axes() -> None:
    """
    VolGridSmileSpec should store expiries and strikes arrays.
    """
    expiries = np.array([0.25, 0.5, 1.0], dtype=float)
    strikes = np.array([0.9, 1.0, 1.1], dtype=float)

    spec = VolGridSmileSpec(expiries=expiries, strikes=strikes)

    assert np.asarray(spec.expiries, dtype=float).shape == (3,)
    assert np.asarray(spec.strikes, dtype=float).shape == (3,)