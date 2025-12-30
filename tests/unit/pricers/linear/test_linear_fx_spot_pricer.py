from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId
from src.instruments.fx.linear.spot import FxSpot
from src.marketdata.scenarios.shocks import SpotShock
from src.pricers.linear.spot import LinearFxSpotPricer
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider


@pytest.fixture(scope="module")
def fx_spot_market():
    """
    Build a deterministic synthetic Market snapshot containing EURUSD spot.
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id]),
        )
    )
    return market, spot_id


def test_linear_fx_spot_pricer_prices_one_unit_correctly(fx_spot_market) -> None:
    """
    PV per 1 instrument unit should equal spot (times contract_multiplier).
    """
    market, spot_id = fx_spot_market

    pricer = LinearFxSpotPricer()
    instrument = FxSpot(spot_id=spot_id, contract_multiplier=1.0)

    pv = pricer.price(instrument, market)
    expected = float(market.quote(spot_id))

    assert pv == pytest.approx(expected, rel=1e-12, abs=1e-12)

    greeks = pricer.greeks(instrument, market)
    assert greeks["delta"] == pytest.approx(1.0)
    assert greeks["gamma"] == pytest.approx(0.0)
    assert greeks["vega"] == pytest.approx(0.0)


def test_linear_fx_spot_pricer_under_spot_shock(fx_spot_market) -> None:
    """
    Under a relative spot shock, PV should scale linearly with spot.
    """
    market, spot_id = fx_spot_market

    pricer = LinearFxSpotPricer()
    instrument = FxSpot(spot_id=spot_id, contract_multiplier=2.5)

    base_pv = pricer.price(instrument, market)

    bump = 0.01  # +1%
    shocked_market = SpotShock(
        name="eurusd_up_1pct",
        spot_id=spot_id,
        bump=bump,
        bump_mode="relative",
    ).apply(market)

    shocked_pv = pricer.price(instrument, shocked_market)

    # Since PV = multiplier * spot, and spot is bumped by +1%, PV should also bump by +1%.
    assert shocked_pv == pytest.approx(base_pv * (1.0 + bump), rel=1e-12, abs=1e-10)