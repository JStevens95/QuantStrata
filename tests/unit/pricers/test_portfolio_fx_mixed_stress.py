from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider

from src.marketdata.scenarios.shocks import SpotShock
from src.portfolio.core import Portfolio, Position
from src.portfolio.portfolio import PortfolioPricer

from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption

from src.pricers.registry import DefaultPricerRegistry


@pytest.fixture(scope="module")
def fx_market_and_ids():
    """
    Returns a synthetic FX market snapshot + the MarketIds needed by FX spot/vanilla pricing.
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD.VOL")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_id, vol_id, rd_id, rf_id]),
        )
    )
    return market, spot_id, vol_id, rd_id, rf_id


def _build_portfolio(*, market, spot_id, vol_id, rd_id, rf_id) -> Portfolio:
    """
    Long spot + long ATM call (large notional) -> portfolio should benefit from spot up moves.
    """
    s0 = float(market.quote(spot_id))

    return Portfolio(
        positions=[
            Position(
                position_id="SPOT",
                instrument=FxSpot(spot_id=spot_id, contract_multiplier=1.0),
                quantity=1.0,
            ),
            Position(
                position_id="CALL",
                instrument=EuropeanFxVanillaOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=s0,      # ATM for stable monotonic behaviour
                    expiry=1.0,
                    spot_id=spot_id,
                    vol_id=vol_id,
                    domestic_curve_id=rd_id,
                    foreign_curve_id=rf_id,
                ),
                quantity=1.0,
            ),
        ]
    )


def test_portfolio_spot_up_shock_increases_pv_for_long_spot_and_call(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    registry = DefaultPricerRegistry().build()
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    portfolio = _build_portfolio(market=market, spot_id=spot_id, vol_id=vol_id, rd_id=rd_id, rf_id=rf_id)

    pv_base = float(portfolio_pricer.price(portfolio, market).totals.pv)

    shocked_market = SpotShock(
        name="spot_up_1pct",
        spot_id=spot_id,
        bump=+0.01,
        bump_mode="relative",
    ).apply(market)

    pv_up = float(portfolio_pricer.price(portfolio, shocked_market).totals.pv)

    assert pv_up > pv_base


def test_portfolio_spot_down_shock_decreases_pv_for_long_spot_and_call(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    registry = DefaultPricerRegistry().build()
    portfolio_pricer = PortfolioPricer(pricer_registry=registry)

    portfolio = _build_portfolio(market=market, spot_id=spot_id, vol_id=vol_id, rd_id=rd_id, rf_id=rf_id)

    pv_base = float(portfolio_pricer.price(portfolio, market).totals.pv)

    shocked_market = SpotShock(
        name="spot_down_1pct",
        spot_id=spot_id,
        bump=-0.01,
        bump_mode="relative",
    ).apply(market)

    pv_down = float(portfolio_pricer.price(portfolio, shocked_market).totals.pv)

    assert pv_down < pv_base