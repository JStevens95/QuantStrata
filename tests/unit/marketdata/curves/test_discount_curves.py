from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.curves.discount import FlatDiscountCurve, ZeroRateDiscountCurve


def test_flat_discount_curve_df_properties() -> None:
    curve = FlatDiscountCurve(continuously_compounded_rate=0.05)

    # df(0) should be 1.0 and df(t) should be decreasing for positive t when r>0
    assert curve.df(0.0) == pytest.approx(1.0)
    assert curve.df(1.0) < curve.df(0.5) < curve.df(0.0)

    # zero rate is constant
    assert curve.zero_rate(0.1) == pytest.approx(0.05)
    assert curve.zero_rate(10.0) == pytest.approx(0.05)

    # forward is constant
    assert curve.fwd_rate(0.5, 1.0) == pytest.approx(0.05)


def test_zero_rate_curve_interpolation_and_df() -> None:
    tenors = np.array([0.5, 1.0, 2.0, 5.0], dtype=float)
    rates = np.array([0.02, 0.025, 0.03, 0.035], dtype=float)
    curve = ZeroRateDiscountCurve(tenors=tenors, zero_rates=rates, extrapolation="flat")

    # df(0) = 1
    assert curve.df(0.0) == pytest.approx(1.0)

    # inside-grid interpolation sanity (monotonic increasing rates in this example)
    r_1y = curve.zero_rate(1.0)
    assert r_1y == pytest.approx(0.025)

    r_15y = curve.zero_rate(1.5)
    assert 0.025 < r_15y < 0.03

    # df should be decreasing for positive t in this positive-rate setup
    assert curve.df(2.0) < curve.df(1.0) < curve.df(0.5)

    # forward rates should be sensible (positive here)
    f_1y_2y = curve.fwd_rate(1.0, 2.0)
    assert f_1y_2y > 0.0


def test_zero_rate_curve_validation() -> None:
    # tenors must be increasing
    with pytest.raises(ValueError):
        ZeroRateDiscountCurve(tenors=np.array([1.0, 0.5]), zero_rates=np.array([0.02, 0.03]))

    # length mismatch
    with pytest.raises(ValueError):
        ZeroRateDiscountCurve(tenors=np.array([0.5, 1.0]), zero_rates=np.array([0.02]))