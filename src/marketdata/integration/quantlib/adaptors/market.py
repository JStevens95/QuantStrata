from __future__ import annotations

from typing import Any
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market

from src.marketdata.integration.quantlib.adaptors.curves import curve_to_yts_handle
from src.marketdata.integration.quantlib.context import QlContext, require_quantlib
from src.marketdata.integration.quantlib.adaptors.vols import vol_surface_to_black_vol_handle


@dataclass(frozen=True, slots=True)
class QlFxMarketView:
    """
    QuantLib-ready FX market view built from your canonical `Market`.
    """
    spot: Any       # ql.QuoteHandle
    dom_yts: Any    # ql.YieldTermStructureHandle
    for_yts: Any    # ql.YieldTermStructureHandle
    black_vol: Any  # ql.BlackVolTermStructureHandle


def build_ql_fx_market_view(
    *,
    market: Market,
    ctx: QlContext,
    spot_id: MarketId,
    domestic_curve_id: MarketId,
    foreign_curve_id: MarketId,
    vol_id: MarketId,
) -> QlFxMarketView:
    """
    Convert your Market snapshot into QuantLib handles for FX vanilla pricing.
    """
    ql = require_quantlib()

    # Fill defaults and set global evaluation date ONCE.
    ctx2 = ctx.with_defaults()
    ctx2.set_evaluation_date()

    spot_value = float(market.quote(spot_id))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_value))

    dom_handle = curve_to_yts_handle(market.curve(domestic_curve_id), ctx=ctx2)
    for_handle = curve_to_yts_handle(market.curve(foreign_curve_id), ctx=ctx2)
    vol_handle = vol_surface_to_black_vol_handle(market.vol_surface(vol_id), ctx=ctx2)

    return QlFxMarketView(
        spot=spot_handle,
        dom_yts=dom_handle,
        for_yts=for_handle,
        black_vol=vol_handle,
    )