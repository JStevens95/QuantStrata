from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.providers.synthetic.provider import SyntheticProvider


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


def test_get_market_is_deterministic_for_same_request() -> None:
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD")
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    provider = SyntheticProvider(seed=123)

    req = MarketRequest(asof="2025-12-29", universe=Universe([spot_id, vol_id, curve_id]))

    m1 = provider.get_market(req)
    m2 = provider.get_market(req)

    assert m1.quote(spot_id) == pytest.approx(m2.quote(spot_id))
    assert m1.vol_surface(vol_id).vol(1.0, 1.0) == pytest.approx(m2.vol_surface(vol_id).vol(1.0, 1.0))
    assert m1.curve(curve_id).df(1.0) == pytest.approx(m2.curve(curve_id).df(1.0))


def test_get_market_scenario_changes_outputs() -> None:
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD")
    curve_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS")

    provider = SyntheticProvider(seed=123)

    m0 = provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe([vol_id, curve_id]), scenario=0))
    m1 = provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe([vol_id, curve_id]), scenario=1))

    # With n_time==1, SPOT is deterministic across scenarios by design,
    # but CURVE/VOL include scenario noise, so scenario slices should differ.
    assert m0.vol_surface(vol_id).vol(expiry=1.0, strike=1.0) != m1.vol_surface(vol_id).vol(expiry=1.0, strike=1.0)
    assert m0.curve(curve_id).df(1.0) != m1.curve(curve_id).df(1.0)


def test_unknown_kind_falls_back_to_scalar_quote_panel() -> None:
    weird_id = MarketId(asset_class="EQ", mkt_type="DIV", name="AAPL")  # not handled explicitly
    provider = SyntheticProvider(seed=123)

    market = provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe([weird_id])))

    # Default fallback is a constant scalar quote panel of 1.0
    assert market.quote(weird_id) == pytest.approx(1.0)