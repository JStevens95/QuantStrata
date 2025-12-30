# tests/unit/marketdata/scenarios/test_fx_vanilla_shocks.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import pytest

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.marketdata.ids import MarketId
from src.marketdata.scenarios.base import MarketView, ScenarioPack
from src.marketdata.scenarios.shocks import FlatVolShock, ParallelRateShock, SpotShock
from src.pricers.fx.european import FxEuropeanVanillaBsmPricer


# =============================================================================
# Minimal dummy market objects (only implement MarketView contract)
# =============================================================================

@dataclass(frozen=True, slots=True)
class _FlatDiscountCurve:
    """Flat continuously-compounded rate curve: df(t) = exp(-r t)."""
    r: float

    def df(self, t: float) -> float:
        t = float(t)
        if t < 0.0:
            raise ValueError("df(t) requires t >= 0.")
        return float(math.exp(-float(self.r) * t))


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    """Flat vol surface: sigma(expiry, strike) = const."""
    sigma: float

    def vol(self, expiry: float, strike: float) -> float:
        _ = float(strike)
        expiry = float(expiry)
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)


@dataclass(frozen=True, slots=True)
class _DummyMarket(MarketView):
    quotes: Dict[MarketId, float]
    curves: Dict[MarketId, object]
    vol_surfaces: Dict[MarketId, object]

    def quote(self, market_id: MarketId) -> float:
        return float(self.quotes[market_id])

    def curve(self, market_id: MarketId):
        return self.curves[market_id]

    def vol_surface(self, market_id: MarketId):
        return self.vol_surfaces[market_id]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def fx_market_and_ids() -> tuple[_DummyMarket, MarketId, MarketId, MarketId, MarketId]:
    """
    Returns (market, spot_id, vol_id, rd_id, rf_id) for a simple EURUSD setup.
    """
    spot_id = MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=())
    vol_id = MarketId(asset_class="FX", mkt_type="VOL", name="EURUSD.VOL", qualifiers=())
    rd_id = MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=())
    rf_id = MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=())

    S0 = 1.25
    rd = 0.03
    rf = 0.01
    sigma = 0.12

    market = _DummyMarket(
        quotes={spot_id: float(S0)},
        curves={
            rd_id: _FlatDiscountCurve(r=float(rd)),
            rf_id: _FlatDiscountCurve(r=float(rf)),
        },
        vol_surfaces={vol_id: _FlatVolSurface(sigma=float(sigma))},
    )
    return market, spot_id, vol_id, rd_id, rf_id


def _mk_call_put(
    *,
    notional_foreign: float,
    strike: float,
    expiry: float,
    spot_id: MarketId,
    vol_id: MarketId,
    domestic_curve_id: MarketId,
    foreign_curve_id: MarketId,
) -> tuple[EuropeanFxVanillaOption, EuropeanFxVanillaOption]:
    """
    IMPORTANT:
    Your EuropeanFxVanillaOption constructor does NOT accept keyword args
    like notional/notional_foreign, so we build it positionally.

    Expected signature (positional):
      EuropeanFxVanillaOption(option_type, notional, strike, expiry, spot_id, vol_id, rd_id, rf_id)
    """
    call = EuropeanFxVanillaOption(
        "call",
        float(notional_foreign),
        float(strike),
        float(expiry),
        spot_id,
        vol_id,
        domestic_curve_id,
        foreign_curve_id,
    )
    put = EuropeanFxVanillaOption(
        "put",
        float(notional_foreign),
        float(strike),
        float(expiry),
        spot_id,
        vol_id,
        domestic_curve_id,
        foreign_curve_id,
    )
    return call, put


def _assert_strictly_greater(a: float, b: float, *, rel_eps: float = 1e-12) -> None:
    assert float(a) > float(b) * (1.0 + float(rel_eps))


# =============================================================================
# Tests
# =============================================================================

def test_spot_up_increases_call_and_decreases_put(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = FxEuropeanVanillaBsmPricer()

    S0 = float(market.quote(spot_id))
    T = 1.0
    K = S0
    Nf = 1_000_000.0

    call, put = _mk_call_put(
        notional_foreign=Nf,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    pv_call_base = float(pricer.price(call, market))
    pv_put_base = float(pricer.price(put, market))

    shocked_market = SpotShock(
        name="SPOT_UP_1PCT",
        spot_id=spot_id,
        bump=+0.01,
        bump_mode="relative",
    ).apply(market)

    assert float(market.quote(spot_id)) == pytest.approx(S0)
    assert float(shocked_market.quote(spot_id)) == pytest.approx(S0 * 1.01)

    pv_call_up = float(pricer.price(call, shocked_market))
    pv_put_up = float(pricer.price(put, shocked_market))

    _assert_strictly_greater(pv_call_up, pv_call_base)
    assert pv_put_up < pv_put_base


def test_vol_up_increases_call_and_put(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = FxEuropeanVanillaBsmPricer()

    S0 = float(market.quote(spot_id))
    T = 1.0
    K = S0
    Nf = 1_000_000.0

    call, put = _mk_call_put(
        notional_foreign=Nf,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    pv_call_base = float(pricer.price(call, market))
    pv_put_base = float(pricer.price(put, market))

    shocked_market = FlatVolShock(
        name="VOL_UP_1PT",
        vol_id=vol_id,
        vol_bump=+0.01,
    ).apply(market)

    pv_call_up = float(pricer.price(call, shocked_market))
    pv_put_up = float(pricer.price(put, shocked_market))

    _assert_strictly_greater(pv_call_up, pv_call_base)
    _assert_strictly_greater(pv_put_up, pv_put_base)


def test_domestic_rate_up_increases_call_and_decreases_put(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = FxEuropeanVanillaBsmPricer()

    S0 = float(market.quote(spot_id))
    T = 1.0
    K = S0
    Nf = 1_000_000.0

    call, put = _mk_call_put(
        notional_foreign=Nf,
        strike=K,
        expiry=T,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    pv_call_base = float(pricer.price(call, market))
    pv_put_base = float(pricer.price(put, market))

    shocked_market = ParallelRateShock(
        name="RD_UP_10BP",
        curve_id=rd_id,
        rate_shift=+0.001,
    ).apply(market)

    pv_call_up = float(pricer.price(call, shocked_market))
    pv_put_up = float(pricer.price(put, shocked_market))

    _assert_strictly_greater(pv_call_up, pv_call_base)
    assert pv_put_up < pv_put_base


def test_scenario_pack_applies_multiple_shocks(fx_market_and_ids) -> None:
    market, spot_id, vol_id, rd_id, rf_id = fx_market_and_ids
    pricer = FxEuropeanVanillaBsmPricer()

    S0 = float(market.quote(spot_id))
    call, _ = _mk_call_put(
        notional_foreign=1_000_000.0,
        strike=S0,
        expiry=1.0,
        spot_id=spot_id,
        vol_id=vol_id,
        domestic_curve_id=rd_id,
        foreign_curve_id=rf_id,
    )

    pack = ScenarioPack(
        scenarios={
            "spot_up": SpotShock(name="spot_up", spot_id=spot_id, bump=+0.01, bump_mode="relative"),
            "vol_up": FlatVolShock(name="vol_up", vol_id=vol_id, vol_bump=+0.01),
            "rd_up": ParallelRateShock(name="rd_up", curve_id=rd_id, rate_shift=+0.001),
        }
    )

    shocked = pack.apply_all(market)
    assert set(shocked.keys()) == {"spot_up", "vol_up", "rd_up"}

    pv_base = float(pricer.price(call, market))
    pv_spot_up = float(pricer.price(call, shocked["spot_up"]))
    pv_vol_up = float(pricer.price(call, shocked["vol_up"]))
    pv_rd_up = float(pricer.price(call, shocked["rd_up"]))

    _assert_strictly_greater(pv_spot_up, pv_base)
    _assert_strictly_greater(pv_vol_up, pv_base)
    _assert_strictly_greater(pv_rd_up, pv_base)