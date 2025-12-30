from __future__ import annotations

import pytest

from src.instruments.fx.linear.forward import FxForward
from src.marketdata.ids import MarketId
from src.marketdata.requests import MarketRequest, Universe
from src.marketdata.scenarios.shocks import SpotShock
from src.marketdata.synthetic.provider import SyntheticProvider
from src.pricers.linear.forward import LinearFxForwardPricer


@pytest.fixture(scope="module")
def fx_forward_market():
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")

    provider = SyntheticProvider(seed=123)
    market = provider.get_market(MarketRequest(asof="2025-12-29", universe=Universe([spot_id, rd_id, rf_id])))

    return market, spot_id, rd_id, rf_id


def test_forward_pv_formula_matches_definition(fx_forward_market) -> None:
    market, spot_id, rd_id, rf_id = fx_forward_market

    pricer = LinearFxForwardPricer()

    S = float(market.quote(spot_id))
    T = 1.0

    df_d = float(market.curve(rd_id).df(T))
    df_f = float(market.curve(rf_id).df(T))

    Nf = 1_000_000.0
    K = S  # arbitrary; we just want the formula check

    inst = FxForward(
        notional_foreign=Nf,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    pv = pricer.price(inst, market)
    expected = Nf * (S * df_f - K * df_d)

    assert pv == pytest.approx(expected, rel=1e-12, abs=1e-10)


def test_forward_is_near_zero_when_strike_is_fair_forward(fx_forward_market) -> None:
    """
    For K = F0 = S0 * df_f / df_d, PV should be ~ 0.
    """
    market, spot_id, rd_id, rf_id = fx_forward_market

    pricer = LinearFxForwardPricer()

    S = float(market.quote(spot_id))
    T = 1.0

    df_d = float(market.curve(rd_id).df(T))
    df_f = float(market.curve(rf_id).df(T))

    F0 = S * df_f / df_d

    inst = FxForward(
        notional_foreign=1_000_000.0,
        strike=F0,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    pv = pricer.price(inst, market)
    assert pv == pytest.approx(0.0, abs=1e-6)


def test_forward_pnl_under_spot_shock_matches_delta_times_dS(fx_forward_market) -> None:
    """
    Forward PV is linear in spot:
        dPV ≈ delta_spot * dS, where delta_spot = Nf * df_f(T)
    """
    market, spot_id, rd_id, rf_id = fx_forward_market

    pricer = LinearFxForwardPricer()

    T = 1.0
    Nf = 2_000_000.0

    S0 = float(market.quote(spot_id))
    df_d = float(market.curve(rd_id).df(T))
    df_f = float(market.curve(rf_id).df(T))
    F0 = S0 * df_f / df_d

    inst = FxForward(
        notional_foreign=Nf,
        strike=F0,
        expiry=T,
        spot_id=spot_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    base_pv = pricer.price(inst, market)

    bump = 0.01
    shocked_market = SpotShock(name="up_1pct", spot_id=spot_id, bump=bump, bump_mode="relative").apply(market)
    shocked_pv = pricer.price(inst, shocked_market)

    pnl = shocked_pv - base_pv

    dS = S0 * bump
    expected = (Nf * df_f) * dS

    assert pnl == pytest.approx(expected, rel=1e-10, abs=1e-6)