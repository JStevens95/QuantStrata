"""Unit tests for the market-data layer."""
import numpy as np
import pytest

from src.rade_static_replication.domain.errors import MarketDataError
from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.fx.instruments import VolSurface


def test_discount_curve_df_monotone():
    c = DiscountCurve("USD", np.array([0.5, 1.0, 2.0, 5.0]), np.array([0.04, 0.042, 0.044, 0.045]))
    assert c.df(0.0) == 1.0
    dfs = [c.df(t) for t in (0.5, 1.0, 2.0, 5.0)]
    assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1))
    # zero/df round-trip
    assert c.df(1.0) == pytest.approx(np.exp(-c.zero(1.0) * 1.0), rel=1e-10)


def test_curve_rejects_unsorted():
    with pytest.raises(MarketDataError):
        DiscountCurve("USD", np.array([1.0, 0.5]), np.array([0.04, 0.04]))


def test_vol_surface_interp_in_range():
    exp = np.array([0.25, 1.0, 2.0])
    k = np.array([0.9, 1.0, 1.1])
    vols = np.array([[0.12, 0.10, 0.11], [0.13, 0.11, 0.12], [0.14, 0.12, 0.13]])
    s = VolSurface(exp, k, vols)
    v = s.vol(1.0, 0.5)
    assert 0.09 < v < 0.15
