from __future__ import annotations

import pytest

from src.instruments.fx.options_test import EuropeanFxOption
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.synthetic.provider import SyntheticProvider
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock
from src.pricers.analytic.black_scholes import BlackScholesPricer


@pytest.fixture(scope="module")
def fx_market_snapshot():
    """Create a deterministic Market snapshot for scenario tests."""
    spot_eurusd = MarketId("FX", "SPOT", "EURUSD")
    vol_eurusd = MarketId("FX", "VOL", "EURUSD")
    usd_curve = MarketId("IR", "CURVE", "USD.OIS")  # domestic
    eur_curve = MarketId("IR", "CURVE", "EUR.OIS")  # foreign

    provider = SyntheticProvider(seed=123)

    base_market = provider.get_market(
        MarketRequest(
            asof="2025-12-29",
            universe=Universe([spot_eurusd, vol_eurusd, usd_curve, eur_curve]),
        )
    )

    return base_market, spot_eurusd, vol_eurusd, usd_curve, eur_curve


def test_spot_up_increases_call_and_decreases_put(fx_market_snapshot) -> None:
    base_market, spot_id, vol_id, domestic_curve_id, foreign_curve_id = fx_market_snapshot
    pricer = BlackScholesPricer()

    base_spot = float(base_market.quote(spot_id))
    expiry_years = 1.0
    strike = base_spot  # ATM-ish for stable monotonic behavior
    notional = 1_000_000.0

    call = EuropeanFxOption("call", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)
    put = EuropeanFxOption("put", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)

    pv_call_base = pricer.price(call, base_market)
    pv_put_base = pricer.price(put, base_market)

    shocked_market = SpotShock(
        name="spot_up_1pct",
        spot_id=spot_id,
        bump=0.01,
        bump_mode="relative",
    ).apply(base_market)

    pv_call_shocked = pricer.price(call, shocked_market)
    pv_put_shocked = pricer.price(put, shocked_market)

    assert pv_call_shocked > pv_call_base
    assert pv_put_shocked < pv_put_base


def test_vol_up_increases_call_and_put(fx_market_snapshot) -> None:
    base_market, spot_id, vol_id, domestic_curve_id, foreign_curve_id = fx_market_snapshot
    pricer = BlackScholesPricer()

    base_spot = float(base_market.quote(spot_id))
    expiry_years = 1.0
    strike = base_spot
    notional = 1_000_000.0

    call = EuropeanFxOption("call", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)
    put = EuropeanFxOption("put", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)

    pv_call_base = pricer.price(call, base_market)
    pv_put_base = pricer.price(put, base_market)

    shocked_market = FlatVolShock(
        name="vol_up_1volpt",
        vol_id=vol_id,
        vol_bump=0.01,
    ).apply(base_market)

    pv_call_shocked = pricer.price(call, shocked_market)
    pv_put_shocked = pricer.price(put, shocked_market)

    assert pv_call_shocked > pv_call_base
    assert pv_put_shocked > pv_put_base


def test_domestic_rate_up_increases_call_and_decreases_put(fx_market_snapshot) -> None:
    """
    Vanilla FX under Garman–Kohlhagen: increasing domestic rates (holding foreign fixed)
    typically increases call PV and decreases put PV.
    """
    base_market, spot_id, vol_id, domestic_curve_id, foreign_curve_id = fx_market_snapshot
    pricer = BlackScholesPricer()

    base_spot = float(base_market.quote(spot_id))
    expiry_years = 1.0
    strike = base_spot
    notional = 1_000_000.0

    call = EuropeanFxOption("call", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)
    put = EuropeanFxOption("put", notional, strike, expiry_years, spot_id, vol_id, domestic_curve_id, foreign_curve_id)

    pv_call_base = pricer.price(call, base_market)
    pv_put_base = pricer.price(put, base_market)

    shocked_market = ParallelRateShock(
        name="domestic_rate_up_100bp",
        curve_id=domestic_curve_id,
        rate_shift=0.01,
    ).apply(base_market)

    pv_call_shocked = pricer.price(call, shocked_market)
    pv_put_shocked = pricer.price(put, shocked_market)

    assert pv_call_shocked > pv_call_base
    assert pv_put_shocked < pv_put_base