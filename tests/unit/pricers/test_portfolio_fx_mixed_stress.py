from __future__ import annotations

import pytest

from src.instruments.fx.linear.spot import FxSpot
from src.instruments.fx.options.european import EuropeanFxOption
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.shocks import SpotShock
from src.marketdata.synthetic.provider import SyntheticProvider
from src.portfolio.core import Portfolio, Position
from src.pricers.portfolio import PortfolioPricer
from src.pricers.registry import DefaultPricerRegistry


@pytest.fixture(scope="module")
def fx_market_and_ids():
    """
    Deterministic synthetic market containing the minimum set of inputs for:
      - FX spot exposure
      - European FX option pricing (BS-style)
    """
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
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


def test_portfolio_spot_up_shock_increases_pv_for_long_spot_and_call(fx_market_and_ids) -> None:
    """
    Long spot + long call should benefit from an upward spot shock.
    """
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    portfolio_pricer = PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())

    S0 = float(market.quote(spot_id))

    portfolio = Portfolio(
        positions=[
            # Linear spot exposure (one-unit) scaled by quantity at portfolio level.
            Position(
                position_id="SPOT",
                instrument=FxSpot(spot_id=spot_id, contract_multiplier=1.0),
                quantity=1.0,
            ),
            # European call
            Position(
                position_id="CALL",
                instrument=EuropeanFxOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=S0,
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

    base = portfolio_pricer.price(portfolio, market)
    base_pv = base.totals.pv
    base_delta = base.totals.greeks.get("delta", 0.0)

    bump = 0.01  # +1% spot shock
    shocked_market = SpotShock(
        name="eurusd_up_1pct",
        spot_id=spot_id,
        bump=bump,
        bump_mode="relative",
    ).apply(market)

    shocked = portfolio_pricer.price(portfolio, shocked_market)
    shocked_pv = shocked.totals.pv

    pnl = shocked_pv - base_pv

    # Basic directional check
    assert shocked_pv > base_pv

    # Sanity check: for small bump, PnL should be at least roughly delta * dS (ignoring convexity).
    # This is not a strict replication bound, but it catches sign/scale errors.
    dS = S0 * bump
    assert pnl > 0.25 * base_delta * dS  # loose bound to remain robust to conventions/scaling


def test_portfolio_spot_down_shock_decreases_pv_for_long_spot_and_call(fx_market_and_ids) -> None:
    """
    Long spot + long call should lose value under a downward spot shock.
    """
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids

    portfolio_pricer = PortfolioPricer(pricer_registry=DefaultPricerRegistry().build())

    S0 = float(market.quote(spot_id))

    portfolio = Portfolio(
        positions=[
            Position(
                position_id="SPOT",
                instrument=FxSpot(spot_id=spot_id, contract_multiplier=1.0),
                quantity=1.0,
            ),
            Position(
                position_id="CALL",
                instrument=EuropeanFxOption(
                    option_type="call",
                    notional=1_000_000.0,
                    strike=S0,
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

    base_pv = portfolio_pricer.price(portfolio, market).totals.pv

    bump = -0.01  # -1%
    shocked_market = SpotShock(
        name="eurusd_down_1pct",
        spot_id=spot_id,
        bump=bump,
        bump_mode="relative",
    ).apply(market)

    shocked_pv = portfolio_pricer.price(portfolio, shocked_market).totals.pv

    assert shocked_pv < base_pv