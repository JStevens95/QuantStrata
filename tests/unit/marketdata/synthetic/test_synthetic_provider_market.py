from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider


def test_get_market_returns_market_with_expected_objects() -> None:
    """
    Ensure SyntheticProvider.get_market() returns a usable Market snapshot that contains:
    - scalar SPOT quote
    - VolSurface object for VOL id
    - Curve object for CURVE id
    """
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    provider = SyntheticProvider(seed=123)

    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id, vol_id, curve_id]),
        )
    )

    # SPOT should be a positive scalar quote
    spot = market.quote(spot_id)
    assert spot > 0.0

    # VOL should be reconstructed as a VolSurface object
    surface = market.vol_surface(vol_id)
    assert surface.vol(expiry=1.0, strike=1.0) > 0.0

    # CURVE should be reconstructed as a Curve object with df(0)=1
    curve = market.curve(curve_id)
    assert curve.df(0.0) == pytest.approx(1.0)
    assert curve.df(1.0) > 0.0