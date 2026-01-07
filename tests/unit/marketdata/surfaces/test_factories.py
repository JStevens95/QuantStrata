from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.surfaces.factories import FlatVolFactory, GridVolFactory
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface


def test_flat_vol_factory_accepts_scalar_blocks() -> None:
    f = FlatVolFactory()

    s1 = f.build(np.array(0.12))
    assert isinstance(s1, FlatVolSurface)
    assert abs(s1.sigma - 0.12) < 1e-15

    s2 = f.build(np.array([0.34], dtype=float))
    assert abs(s2.sigma - 0.34) < 1e-15

    with pytest.raises(ValueError, match="Expected scalar params"):
        _ = f.build(np.array([0.1, 0.2], dtype=float))


def test_grid_vol_factory_builds_from_2d_or_flattened() -> None:
    expiries = np.array([0.5, 1.0], dtype=float)
    strikes = np.array([90.0, 100.0, 110.0], dtype=float)

    g = GridVolFactory(expiries=expiries, strikes=strikes, extrapolation="flat")

    vols2d = np.full((2, 3), 0.2, dtype=float)
    s1 = g.build(vols2d)
    assert isinstance(s1, GridVolSurface)
    assert s1.implied_vols.shape == (2, 3)

    flat = vols2d.reshape(-1, order="C")
    s2 = g.build(flat)
    assert np.allclose(s2.implied_vols, vols2d)

    with pytest.raises(ValueError, match="expected params shape"):
        _ = g.build(np.full((3, 2), 0.2, dtype=float))

    with pytest.raises(ValueError, match="expected .* params"):
        _ = g.build(np.array([0.2, 0.2], dtype=float))