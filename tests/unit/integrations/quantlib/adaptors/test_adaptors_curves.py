from __future__ import annotations

import pytest
import numpy as np

from src.marketdata.integration.quantlib.adaptors.curves import curve_to_yts_handle
from src.marketdata.integration.quantlib.context import QlContext, require_quantlib
from src.marketdata.curves.term_structure import FlatZeroRateCurve, ZeroRateCurve


def _has_quantlib() -> bool:
    try:
        require_quantlib()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_quantlib(), reason="QuantLib not installed")


def test_flat_curve_to_yts_handle_discount_factor_is_reasonable() -> None:
    ql = require_quantlib()
    ctx = QlContext(asof="2025-12-29").with_defaults()

    curve = FlatZeroRateCurve(continuously_compounded_rate=0.02)
    h = curve_to_yts_handle(curve, ctx=ctx)

    # check discount factor at ~1y > 0 and < 1
    d0 = ql.Settings.instance().evaluationDate
    d1 = d0 + ql.Period(365, ql.Days)
    df_1y = h.discount(d1)
    assert 0.0 < df_1y < 1.0


def test_zero_rate_curve_to_yts_handle_discount_factor_is_reasonable() -> None:
    ql = require_quantlib()
    ctx = QlContext(asof="2025-12-29").with_defaults()

    tenors = np.array([0.5, 1.0, 2.0, 5.0])
    zeros = np.array([0.02, 0.021, 0.023, 0.025])
    curve = ZeroRateCurve(tenors=tenors, zero_rates=zeros, extrapolation="flat")

    h = curve_to_yts_handle(curve, ctx=ctx)

    d0 = ql.Settings.instance().evaluationDate
    d2 = d0 + ql.Period(2 * 365, ql.Days)
    df_2y = h.discount(d2)
    assert 0.0 < df_2y < 1.0