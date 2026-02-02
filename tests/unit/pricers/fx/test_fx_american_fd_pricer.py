from __future__ import annotations

import math
import pytest
from typing import Dict
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.instruments.fx.options.vanilla import FxVanillaAmericanOption, FxVanillaEuropeanOption
from src.pricers.fx.european_bsm_fde import FxVanillaEuropeanOptionFdPricer
from src.pricers.fx.american_bsm_fde import FxVanillaAmericanOptionFdPricer


@dataclass(frozen=True, slots=True)
class _FlatCurve:
    rate: float

    def df(self, t: float) -> float:
        t = float(t)
        if t < 0.0:
            raise ValueError("t must be >= 0.")
        return float(math.exp(-float(self.rate) * t))


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    sigma: float

    def vol(self, *, expiry: float, strike: float) -> float:  # noqa: ARG002
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)


@dataclass(frozen=True, slots=True)
class _DummyMarket:
    spot: float
    rd: float
    rf: float
    sigma: float
    spot_id: MarketId
    vol_id: MarketId
    rd_id: MarketId
    rf_id: MarketId

    def quote(self, market_id: MarketId) -> float:
        if market_id != self.spot_id:
            raise KeyError(f"Unknown quote id: {market_id}")
        return float(self.spot)

    def curve(self, market_id: MarketId):
        if market_id == self.rd_id:
            return _FlatCurve(rate=self.rd)
        if market_id == self.rf_id:
            return _FlatCurve(rate=self.rf)
        raise KeyError(f"Unknown curve id: {market_id}")

    def vol_surface(self, market_id: MarketId):
        if market_id != self.vol_id:
            raise KeyError(f"Unknown vol id: {market_id}")
        return _FlatVolSurface(sigma=self.sigma)


@pytest.fixture(scope="module")
def ids() -> Dict[str, MarketId]:
    return {
        "spot": MarketId("FX", "SPOT", "EURUSD"),
        "vol": MarketId("FX", "VOL", "EURUSD.VOL"),
        "rd": MarketId("IR", "CURVE", "USD.OIS"),
        "rf": MarketId("IR", "CURVE", "EUR.OIS"),
    }


def _make_euro_trade(*, option_type: str, ids: Dict[str, MarketId], notional: float, strike: float, t: float) -> FxVanillaEuropeanOption:
    return FxVanillaEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(notional),
        strike=float(strike),
        expiry=float(t),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )


def _make_amer_trade(*, option_type: str, ids: Dict[str, MarketId], notional: float, strike: float, t: float) -> FxVanillaAmericanOption:
    return FxVanillaAmericanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(notional),
        strike=float(strike),
        expiry=float(t),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )


@pytest.mark.parametrize("option_type", ["put", "call"])
def test_american_price_is_at_least_european(
    ids: Dict[str, MarketId],
    option_type: str,
) -> None:
    """
    Fundamental no-arbitrage: American >= European (same model inputs).
    """
    market = _DummyMarket(
        spot=1.25,
        rd=0.03,
        rf=0.01,
        sigma=0.20,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    notional = 1_000_000.0
    strike = 1.25
    T = 1.0

    euro = _make_euro_trade(option_type=option_type, ids=ids, notional=notional, strike=strike, t=T)
    amer = _make_amer_trade(option_type=option_type, ids=ids, notional=notional, strike=strike, t=T)

    euro_fd = FxVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=240, n_std=6.0, theta=0.5, use_log_space=True)
    amer_fd = FxVanillaAmericanOptionFdPricer(n_space=401, n_time_steps=240, n_std=6.0, theta=0.5, use_log_space=True)

    pv_euro = float(euro_fd.price(euro, market))
    pv_amer = float(amer_fd.price(amer, market))

    assert math.isfinite(pv_euro)
    assert math.isfinite(pv_amer)

    # American >= European in theory, but PSOR/FD introduces tiny numerical noise.
    # Use a small PV tolerance that scales with notional (per-unit ~1e-8).
    pv_tol = 1e-8 * notional  # e.g. notional=1e6 -> 0.01 in PV units
    assert pv_amer + pv_tol >= pv_euro


def test_american_call_matches_european_call_when_rf_zero(
    ids: Dict[str, MarketId],
) -> None:
    """
    In GK mapping, foreign rate acts like a dividend yield q=r_f.
    When r_f = 0 (no yield), early exercise for a call should not be optimal,
    so American call ~= European call.
    """
    market = _DummyMarket(
        spot=1.25,
        rd=0.03,
        rf=0.0,
        sigma=0.20,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    notional = 1_000_000.0
    strike = 1.25
    T = 1.0

    euro = _make_euro_trade(option_type="call", ids=ids, notional=notional, strike=strike, t=T)
    amer = _make_amer_trade(option_type="call", ids=ids, notional=notional, strike=strike, t=T)

    euro_fd = FxVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=240, n_std=6.0, theta=0.5, use_log_space=True)
    amer_fd = FxVanillaAmericanOptionFdPricer(n_space=401, n_time_steps=240, n_std=6.0, theta=0.5, use_log_space=True)

    pv_euro = float(euro_fd.price(euro, market))
    pv_amer = float(amer_fd.price(amer, market))

    # Should be very close
    assert pv_amer == pytest.approx(pv_euro, rel=0.01, abs=1e-2)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_american_fd_greeks_basic_sanity(
    ids: Dict[str, MarketId],
    option_type: str,
) -> None:
    """
    Sanity checks on Greek signs/magnitudes (not a model-to-model equality test).
    """
    market = _DummyMarket(
        spot=1.25,
        rd=0.03,
        rf=0.01,
        sigma=0.20,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    trade = _make_amer_trade(option_type=option_type, ids=ids, notional=1_000_000.0, strike=1.25, t=1.0)

    amer_fd = FxVanillaAmericanOptionFdPricer(
        n_space=401,
        n_time_steps=240,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
    )

    g = amer_fd.greeks(trade, market)

    for k in ["delta", "gamma", "vega", "rho_domestic", "rho_foreign"]:
        assert k in g
        assert math.isfinite(float(g[k]))

    gamma = float(g["gamma"])
    vega = float(g["vega"])

    # Vanilla gamma/vega should be non-negative in normal regimes.
    assert gamma >= -1e-6
    assert vega >= -1e-6

    delta = float(g["delta"])
    notional = float(trade.notional)

    if option_type == "call":
        assert 0.0 <= delta <= notional * 1.05
    else:
        assert -notional * 1.05 <= delta <= 0.0


def test_american_fd_is_deterministic(
    ids: Dict[str, MarketId],
) -> None:
    market = _DummyMarket(
        spot=1.25,
        rd=0.03,
        rf=0.01,
        sigma=0.20,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    trade = _make_amer_trade(option_type="put", ids=ids, notional=1_000_000.0, strike=1.25, t=1.0)

    amer_fd = FxVanillaAmericanOptionFdPricer(n_space=301, n_time_steps=200, n_std=6.0, theta=0.5, use_log_space=True)

    pv1 = float(amer_fd.price(trade, market))
    pv2 = float(amer_fd.price(trade, market))

    assert pv1 == pytest.approx(pv2, rel=0.0, abs=0.0)