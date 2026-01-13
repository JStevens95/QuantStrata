from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface


def test_flat_vol_surface_validates_and_returns_constant() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        _ = FlatVolSurface(sigma=0.0)

    s = FlatVolSurface(sigma=0.123)
    assert abs(s.implied_vol(0.0, 100.0) - 0.123) < 1e-15
    assert abs(s.implied_vol(2.0, 50.0) - 0.123) < 1e-15
    assert abs(s.vol(1.0, 100.0) - 0.123) < 1e-15


def test_grid_vol_surface_validates_inputs() -> None:
    exp = np.array([0.5, 1.0], dtype=float)
    k = np.array([90.0, 100.0, 110.0], dtype=float)
    vols = np.full((2, 3), 0.2, dtype=float)

    _ = GridVolSurface(expiries=exp, strikes=k, implied_vols=vols)

    with pytest.raises(ValueError, match="strictly increasing"):
        _ = GridVolSurface(expiries=np.array([1.0, 1.0], float), strikes=k, implied_vols=vols)

    with pytest.raises(ValueError, match="strictly increasing"):
        _ = GridVolSurface(expiries=exp, strikes=np.array([100.0, 100.0], float), implied_vols=np.full((2, 2), 0.2))

    with pytest.raises(ValueError, match="must be a 2D"):
        _ = GridVolSurface(expiries=exp, strikes=k, implied_vols=np.array([0.2, 0.2], float))

    with pytest.raises(ValueError, match="must be strictly positive"):
        bad = vols.copy()
        bad[0, 0] = 0.0
        _ = GridVolSurface(expiries=exp, strikes=k, implied_vols=bad)


def test_grid_vol_surface_interp_and_flat_extrapolation() -> None:
    exp = np.array([0.5, 1.0], dtype=float)
    k = np.array([90.0, 110.0], dtype=float)

    # a simple plane so bilinear interpolation is predictable
    # z(T,K):
    # T=0.5: [0.10, 0.20]
    # T=1.0: [0.20, 0.30]
    vols = np.array([[0.10, 0.20], [0.20, 0.30]], dtype=float)

    s = GridVolSurface(expiries=exp, strikes=k, implied_vols=vols, extrapolation="flat")

    # midpoint in both dims should be average of corners = 0.20
    assert abs(s.implied_vol(0.75, 100.0) - 0.20) < 1e-12

    # flat extrapolation clamps
    assert abs(s.implied_vol(0.10, 100.0) - s.implied_vol(0.5, 100.0)) < 1e-15
    assert abs(s.implied_vol(2.00, 100.0) - s.implied_vol(1.0, 100.0)) < 1e-15
    assert abs(s.implied_vol(0.75, 50.0) - s.implied_vol(0.75, 90.0)) < 1e-15
    assert abs(s.implied_vol(0.75, 200.0) - s.implied_vol(0.75, 110.0)) < 1e-15


def test_grid_vol_surface_error_extrapolation_raises() -> None:
    exp = np.array([0.5, 1.0], dtype=float)
    k = np.array([90.0, 110.0], dtype=float)
    vols = np.full((2, 2), 0.2, dtype=float)

    s = GridVolSurface(expiries=exp, strikes=k, implied_vols=vols, extrapolation="error")

    with pytest.raises(ValueError, match="outside grid"):
        _ = s.implied_vol(0.1, 100.0)

    with pytest.raises(ValueError, match="outside grid"):
        _ = s.implied_vol(0.75, 50.0)