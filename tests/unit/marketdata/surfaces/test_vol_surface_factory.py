from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.surfaces.factories import FlatVolFactory
from src.marketdata.surfaces.vol_surface import FlatVolSurface


def test_flat_vol_surface_returns_constant_vol() -> None:
    surface = FlatVolSurface(implied_vol=0.12)

    assert surface.vol(expiry=0.0, strike=1.0) == pytest.approx(0.12)
    assert surface.vol(expiry=1.0, strike=0.8) == pytest.approx(0.12)
    assert surface.vol(expiry=2.0, strike=1.5) == pytest.approx(0.12)

    with pytest.raises(ValueError):
        surface.vol(expiry=-0.1, strike=1.0)


def test_flat_vol_surface_validation() -> None:
    with pytest.raises(ValueError):
        FlatVolSurface(implied_vol=0.0)
    with pytest.raises(ValueError):
        FlatVolSurface(implied_vol=-0.2)
    with pytest.raises(ValueError):
        FlatVolSurface(implied_vol=float("nan"))


def test_flat_vol_factory_accepts_scalar_formats() -> None:
    factory = FlatVolFactory()

    s1 = factory.build(np.asarray(0.25))
    assert s1.vol(expiry=1.0, strike=1.0) == pytest.approx(0.25)

    s2 = factory.build(np.asarray([0.18]))
    assert s2.vol(expiry=3.0, strike=2.0) == pytest.approx(0.18)

    with pytest.raises(ValueError):
        factory.build(np.asarray([0.1, 0.2]))