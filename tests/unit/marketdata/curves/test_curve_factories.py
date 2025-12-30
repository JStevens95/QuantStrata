from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.curves.factories import FlatCurveFactory, ZeroCurveFactory


def test_flat_curve_factory_accepts_scalar_formats() -> None:
    factory = FlatCurveFactory()

    c1 = factory.build(np.asarray(0.02))
    assert c1.zero_rate(1.0) == pytest.approx(0.02)

    c2 = factory.build(np.asarray([0.03]))
    assert c2.zero_rate(5.0) == pytest.approx(0.03)

    with pytest.raises(ValueError):
        factory.build(np.asarray([0.01, 0.02]))


def test_zero_curve_factory_accepts_k_by_2_grid() -> None:
    factory = ZeroCurveFactory(extrapolation="flat")

    # (K,2): [tenor, rate]
    params = np.asarray(
        [
            [0.5, 0.02],
            [1.0, 0.025],
            [2.0, 0.03],
        ],
        dtype=float,
    )
    curve = factory.build(params)
    assert curve.zero_rate(1.0) == pytest.approx(0.025)


def test_zero_curve_factory_accepts_2_by_k_grid() -> None:
    factory = ZeroCurveFactory(extrapolation="flat")

    # (2,K): row0 tenors, row1 rates
    params = np.asarray(
        [
            [0.5, 1.0, 2.0],
            [0.02, 0.025, 0.03],
        ],
        dtype=float,
    )
    curve = factory.build(params)
    assert curve.zero_rate(2.0) == pytest.approx(0.03)

    with pytest.raises(ValueError):
        factory.build(np.asarray([0.5, 0.02]))  # invalid (not 2D)