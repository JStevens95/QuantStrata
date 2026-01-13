from __future__ import annotations

import numpy as np
import pytest

from src.marketdata.curves.term_structure import FlatZeroRateCurve, ZeroRateCurve


def test_flat_discount_curve_df_zero_forward() -> None:
    curve = FlatZeroRateCurve(continuously_compounded_rate=0.05)

    assert curve.df(0.0) == 1.0
    assert curve.df(-1.0) == 1.0

    df_2y = curve.df(2.0)
    assert np.isfinite(df_2y)
    assert 0.0 < df_2y < 1.0

    assert curve.zero_rate(0.0) == 0.05
    assert curve.forward_rate(0.25, 1.25) == 0.05


def test_zero_rate_discount_curve_validates_inputs() -> None:
    with pytest.raises(ValueError, match="tenors must not be empty"):
        _ = ZeroRateCurve(tenors=np.array([], float), zero_rates=np.array([], float))

    with pytest.raises(ValueError, match="same length"):
        _ = ZeroRateCurve(tenors=np.array([0.5, 1.0], float), zero_rates=np.array([0.01], float))

    with pytest.raises(ValueError, match="strictly increasing"):
        _ = ZeroRateCurve(tenors=np.array([1.0, 1.0], float), zero_rates=np.array([0.01, 0.02], float))

    with pytest.raises(ValueError, match="extrapolation"):
        _ = ZeroRateCurve(
            tenors=np.array([0.5, 1.0], float),
            zero_rates=np.array([0.01, 0.02], float),
            extrapolation="nonsense",  # type: ignore[arg-type]
        )


def test_zero_rate_discount_curve_interp_and_flat_extrapolation() -> None:
    tenors = np.array([0.5, 1.0, 2.0], float)
    zeros = np.array([0.01, 0.02, 0.03], float)

    curve = ZeroRateCurve(tenors=tenors, zero_rates=zeros, extrapolation="flat")

    # inside grid -> linear interpolation
    r_075 = curve.zero_rate(0.75)
    assert np.isclose(r_075, 0.015)

    # flat extrap below/above
    assert curve.zero_rate(0.1) == pytest.approx(0.01)
    assert curve.zero_rate(10.0) == pytest.approx(0.03)

    # df(t) = exp(-r(t)*t)
    df_075 = curve.df(0.75)
    assert np.isfinite(df_075)
    assert 0.0 < df_075 < 1.0


def test_zero_rate_discount_curve_forward_rate_matches_df_definition() -> None:
    tenors = np.array([0.5, 1.0, 2.0], float)
    zeros = np.array([0.02, 0.02, 0.02], float)
    curve = ZeroRateCurve(tenors=tenors, zero_rates=zeros, extrapolation="flat")

    f = curve.forward_rate(0.5, 2.0)
    df1 = curve.df(0.5)
    df2 = curve.df(2.0)
    f_expected = -np.log(df2 / df1) / (2.0 - 0.5)

    assert np.isclose(f, f_expected)