from __future__ import annotations

import pytest
from dataclasses import dataclass

from src.marketdata.integration.quantlib.adaptors.market import build_ql_fx_market_view
from src.marketdata.integration.quantlib.context import QlContext, require_quantlib
from src.marketdata.curves.term_structure import FlatZeroRateCurve
from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.surfaces.vol_surface import FlatVolSurface


def _has_quantlib() -> bool:
    try:
        require_quantlib()
        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(not _has_quantlib(), reason="QuantLib not installed")


@dataclass(frozen=True, slots=True)
class _Quote:
    """
    Minimal Quote-like object for runtime tests.

    Works whether src.marketdata.interfaces.Quote is a Protocol or a concrete type,
    because Market only requires `.value` at runtime.
    """
    value: float


def test_build_ql_fx_market_view_smoke() -> None:
    ql = require_quantlib()

    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    dom_id = MarketId("IR", "CURVE", "USD.OIS")
    for_id = MarketId("IR", "CURVE", "EUR.OIS")

    market = Market(
        asof="2025-12-29",
        quotes={spot_id: _Quote(value=1.10)},
        curves={
            dom_id: FlatZeroRateCurve(continuously_compounded_rate=0.02),
            for_id: FlatZeroRateCurve(continuously_compounded_rate=0.01),
        },
        vols={vol_id: FlatVolSurface(sigma=0.12)},
        meta=None,
    )

    ctx = QlContext(asof=market.asof)

    view = build_ql_fx_market_view(
        market=market,
        ctx=ctx,
        spot_id=spot_id,
        domestic_curve_id=dom_id,
        foreign_curve_id=for_id,
        vol_id=vol_id,
    )

    # Handles exist
    assert view.spot is not None
    assert view.dom_yts is not None
    assert view.for_yts is not None
    assert view.black_vol is not None

    # Spot round-trip sanity
    assert float(view.spot.value()) == pytest.approx(1.10)

    # Global evaluation date set correctly
    assert ql.Settings.instance().evaluationDate == ql.Date(29, 12, 2025)