from __future__ import annotations

import pytest
import numpy as np

from src.integrations.quantlib.ql_utils import QlContext, require_quantlib
from src.integrations.quantlib.ql_vol import vol_surface_to_black_vol_handle
from src.marketdata.surfaces.vol_surface import FlatVolSurface, GridVolSurface


def _has_quantlib() -> bool:
    try:
        require_quantlib()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_quantlib(), reason="QuantLib not installed")


def test_flat_vol_to_black_vol_handle() -> None:
    ql = require_quantlib()
    ctx = QlContext(asof="2025-12-29").with_defaults()

    surf = FlatVolSurface(sigma=0.12)
    h = vol_surface_to_black_vol_handle(surf, ctx=ctx)

    d0 = ql.Settings.instance().evaluationDate
    d1 = d0 + ql.Period(365, ql.Days)

    vol = h.blackVol(d1, 1.0)
    assert vol == pytest.approx(0.12, rel=0.0, abs=1e-12)


def test_grid_vol_to_black_vol_handle_smoke() -> None:
    ql = require_quantlib()
    ctx = QlContext(asof="2025-12-29").with_defaults()

    expiries = np.array([0.5, 1.0])
    strikes = np.array([0.9, 1.0, 1.1])
    vols = np.array([
        [0.13, 0.12, 0.13],
        [0.14, 0.125, 0.14],
    ])

    surf = GridVolSurface(
        expiries=expiries,
        strikes=strikes,
        implied_vols=vols,
        strike_space="absolute",
        extrapolation="flat",
    )

    h = vol_surface_to_black_vol_handle(surf, ctx=ctx)

    d0 = ql.Settings.instance().evaluationDate
    d = d0 + ql.Period(int(round(0.5 * 365)), ql.Days)

    # should interpolate without throwing
    v = h.blackVol(d, 1.0)
    assert v > 0.0