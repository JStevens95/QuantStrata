from __future__ import annotations

import pytest

from src.marketdata.ids import MarketId
from src.portfolio.core import Portfolio, Position
from src.pricers.portfolio import PortfolioPricer
from src.instruments.fx.options.european import EuropeanFxOption
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.pricers.analytic.black_scholes import BlackScholesPricer


@pytest.fixture(scope="module")
def fx_market():
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


def test_portfolio_pv_equals_sum_of_position_pvs(fx_market) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market

    pricer_registry = {EuropeanFxOption: BlackScholesPricer()}
    portfolio_pricer = PortfolioPricer(pricer_registry=pricer_registry)

    S = float(market.quote(spot_id))

    # Build a small mixed set: different strikes/expiries, including a short.
    positions = [
        Position(
            position_id="POS_1",
            instrument=EuropeanFxOption("call", 1_000_000.0, strike=S, expiry=1.0, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id, foreign_curve_id=rf_id),
            quantity=1.0,
        ),
        Position(
            position_id="POS_2",
            instrument=EuropeanFxOption("put", 500_000.0, strike=0.98 * S, expiry=0.5, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id, foreign_curve_id=rf_id),
            quantity=2.0,
        ),
        Position(
            position_id="POS_3",
            instrument=EuropeanFxOption("call", 750_000.0, strike=1.02 * S, expiry=2.0, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id, foreign_curve_id=rf_id),
            quantity=-0.5,  # short half a unit
        ),
    ]

    portfolio = Portfolio(positions=positions)

    # Portfolio result
    result = portfolio_pricer.price(portfolio, market)

    # Manual sum (using the same routed pricer)
    bs = BlackScholesPricer()
    expected_total = 0.0
    for p in positions:
        expected_total += p.quantity * bs.price(p.instrument, market)

    assert result.totals.pv == pytest.approx(expected_total, rel=1e-12, abs=1e-6)


def test_portfolio_greeks_equal_sum_of_position_greeks(fx_market) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market

    pricer_registry = {EuropeanFxOption: BlackScholesPricer()}
    portfolio_pricer = PortfolioPricer(pricer_registry=pricer_registry)

    S = float(market.quote(spot_id))

    positions = [
        Position(
            position_id="POS_A",
            instrument=EuropeanFxOption(
                "call", 1_000_000.0, strike=S, expiry=1.0, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id,
                foreign_curve_id=rf_id
            ),
            quantity=1.0,
        ),
        Position(
            position_id="POS_B",
            instrument=EuropeanFxOption(
                "put", 1_000_000.0, strike=S, expiry=1.0, spot_id=spot_id, vol_id=vol_id, domestic_curve_id=rd_id,
                foreign_curve_id=rf_id
            ),
            quantity=1.0,
        ),
    ]

    portfolio = Portfolio(positions=positions)
    result = portfolio_pricer.price(portfolio, market)

    # Manual aggregation
    bs = BlackScholesPricer()

    expected = {"delta": 0.0, "gamma": 0.0, "vega": 0.0}
    for p in positions:
        g = bs.greeks(p.instrument, market)
        for k in expected.keys():
            expected[k] += p.quantity * g[k]

    # Compare each greek
    assert result.totals.greeks["delta"] == pytest.approx(expected["delta"], rel=1e-10, abs=1e-4)
    assert result.totals.greeks["gamma"] == pytest.approx(expected["gamma"], rel=1e-10, abs=1e-10)
    assert result.totals.greeks["vega"] == pytest.approx(expected["vega"], rel=1e-10, abs=1e-4)