import pytest
import numpy as np

from src.marketdata.curves.term_structure import ZeroRateCurve
from src.marketdata.curves.factories import ZeroRateCurveFactory


def test_zero_curve_factory_builds_curve() -> None:
    params = np.array(
        [
            [0.5, 0.01],
            [1.0, 0.02],
            [2.0, 0.03],
        ],
        dtype=float,
    )

    factory = ZeroRateCurveFactory(extrapolation="flat")
    curve = factory.build(params)

    assert isinstance(curve, ZeroRateCurve)
    assert curve.tenors.shape == (3,)
    assert curve.zero_rates.shape == (3,)
    assert curve.extrapolation == "flat"

    # invalid: 1D input
    with pytest.raises(ValueError, match=r"must be 2D"):
        _ = factory.build(np.array([0.5, 0.01], dtype=float))