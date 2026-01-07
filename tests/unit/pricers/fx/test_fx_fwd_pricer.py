from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Dict, Callable

import pytest

from src.marketdata.core.ids import MarketId
from src.pricers.fx.forward import FxForwardPricer
from src.instruments.fx.linear.forward import FxForward


# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------

def _construct_dataclass(cls: type[Any], **kwargs: Any) -> Any:
    """
    Construct a class instance by filtering kwargs to its constructor signature.

    This keeps tests robust when instrument constructors evolve between Vn versions.
    """
    sig = inspect.signature(cls)
    params = sig.parameters

    filtered = {k: v for k, v in kwargs.items() if k in params}

    missing_required: list[str] = []
    for name, p in params.items():
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
            if p.default is inspect._empty and name not in filtered:
                missing_required.append(name)

    if missing_required:
        raise TypeError(f"Missing required constructor args for {cls.__name__}: {missing_required}")

    return cls(**filtered)


def _fd_first(f: Callable[[float], float], x0: float, eps: float) -> float:
    """Central finite difference for first derivative."""
    return (f(x0 + eps) - f(x0 - eps)) / (2.0 * eps)


@dataclass(frozen=True)
class _DummyCurve:
    """
    Continuously-compounded flat curve.

    df(T) = exp(-r*T)
    """
    rate: float

    def df(self, t: float) -> float:
        return float(math.exp(-float(self.rate) * float(t)))


@dataclass(frozen=True)
class _DummyMarket:
    """
    Minimal market stub for FxForwardPricer tests.

    The pricer needs:
      - quote(spot_id) -> float
      - curve(curve_id).df(T) -> float
    """
    spot_id: MarketId
    rd_id: MarketId
    rf_id: MarketId
    spot_value: float
    rd: float
    rf: float

    def quote(self, market_id: MarketId) -> float:
        if market_id != self.spot_id:
            raise KeyError(f"Unknown quote id: {market_id}")
        return float(self.spot_value)

    def curve(self, curve_id: MarketId) -> _DummyCurve:
        if curve_id == self.rd_id:
            return _DummyCurve(rate=float(self.rd))
        if curve_id == self.rf_id:
            return _DummyCurve(rate=float(self.rf))
        raise KeyError(f"Unknown curve id: {curve_id}")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture()
def ids() -> Dict[str, MarketId]:
    return {
        "spot": MarketId(asset_class="FX", mkt_type="SPOT", name="EURUSD", qualifiers=()),
        "rd": MarketId(asset_class="IR", mkt_type="CURVE", name="USD.OIS", qualifiers=()),
        "rf": MarketId(asset_class="IR", mkt_type="CURVE", name="EUR.OIS", qualifiers=()),
    }


@pytest.fixture()
def base_params() -> Dict[str, float]:
    return {
        "spot": 1.25,
        "strike": 1.30,
        "t": 1.0,
        "rd": 0.03,
        "rf": 0.01,
        "notional_foreign": 1_000_000.0,
    }


@pytest.fixture()
def pricer() -> FxForwardPricer:
    return FxForwardPricer()


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_fx_forward_price_matches_closed_form(ids: Dict[str, MarketId], base_params: Dict[str, float], pricer: FxForwardPricer) -> None:
    """
    Validate the pricing identity:

        PV = Nf * ( S * df_f(T) - K * df_d(T) )

    using flat continuously-compounded curves.
    """
    s = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])
    rd = float(base_params["rd"])
    rf = float(base_params["rf"])
    nf = float(base_params["notional_foreign"])

    market = _DummyMarket(
        spot_id=ids["spot"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot_value=s, rd=rd, rf=rf
    )

    trade = _construct_dataclass(
        FxForward,
        spot_id=ids["spot"],
        strike=k,
        expiry=t,
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
        notional_foreign=nf,   # preferred
        notional=nf,           # fallback if your instrument uses this
    )

    pv = pricer.price(trade, market)

    df_d = math.exp(-rd * t)
    df_f = math.exp(-rf * t)
    expected = nf * (s * df_f - k * df_d)

    assert pv == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fx_forward_greeks_match_finite_differences(ids: Dict[str, MarketId], base_params: Dict[str, float], pricer: FxForwardPricer) -> None:
    """
    Validate forward greeks against finite differences of pricer.price().

    We bump:
      - spot -> delta
      - rd   -> rho_domestic (through df_d)
      - rf   -> rho_foreign  (through df_f)

    Notes
    -----
    Forward PV is smooth, so FD tests should be very stable.
    """
    s0 = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])
    rd0 = float(base_params["rd"])
    rf0 = float(base_params["rf"])
    nf = float(base_params["notional_foreign"])

    trade = _construct_dataclass(
        FxForward,
        spot_id=ids["spot"],
        strike=k,
        expiry=t,
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
        notional_foreign=nf,
        notional=nf,
    )

    # Base market
    market0 = _DummyMarket(
        spot_id=ids["spot"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot_value=s0, rd=rd0, rf=rf0
    )

    g = pricer.greeks(trade, market0)

    # --- Delta: dPV/dS ---
    eps_s = 1e-6 * max(1.0, abs(s0))  # small absolute bump
    f_s = lambda s: pricer.price(trade, _DummyMarket(
        spot_id=ids["spot"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot_value=float(s), rd=rd0, rf=rf0
    ))
    delta_fd = _fd_first(f_s, s0, eps_s)
    assert g["delta"] == pytest.approx(delta_fd, rel=1e-7, abs=1e-7)

    # --- rho_domestic: dPV/dr_d ---
    eps_r = 1e-6  # 0.01bp
    f_rd = lambda rd: pricer.price(trade, _DummyMarket(
        spot_id=ids["spot"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot_value=s0, rd=float(rd), rf=rf0
    ))
    rho_d_fd = _fd_first(f_rd, rd0, eps_r)
    assert g["rho_domestic"] == pytest.approx(rho_d_fd, rel=1e-7, abs=1e-6)

    # --- rho_foreign: dPV/dr_f ---
    f_rf = lambda rf: pricer.price(trade, _DummyMarket(
        spot_id=ids["spot"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot_value=s0, rd=rd0, rf=float(rf)
    ))
    rho_f_fd = _fd_first(f_rf, rf0, eps_r)
    assert g["rho_foreign"] == pytest.approx(rho_f_fd, rel=1e-7, abs=1e-6)


def test_fx_forward_expiry_zero_price_and_zero_greeks(ids: Dict[str, MarketId], base_params: Dict[str, float], pricer: FxForwardPricer) -> None:
    """
    At expiry:
      - price should reduce to Nf*(S-K) (since df=1)
      - greeks are returned as zeros by design (stable convention across pricers)
    """
    s = float(base_params["spot"])
    k = float(base_params["strike"])
    nf = float(base_params["notional_foreign"])

    market = _DummyMarket(
        spot_id=ids["spot"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot_value=s, rd=float(base_params["rd"]), rf=float(base_params["rf"])
    )

    trade = _construct_dataclass(
        FxForward,
        spot_id=ids["spot"],
        strike=k,
        expiry=0.0,
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
        notional_foreign=nf,
        notional=nf,
    )

    pv = pricer.price(trade, market)
    assert pv == pytest.approx(nf * (s - k), rel=0.0, abs=0.0)

    greeks = pricer.greeks(trade, market)
    assert greeks == {"delta": 0.0, "rho_domestic": 0.0, "rho_foreign": 0.0}


@pytest.mark.parametrize("bad_strike", [0.0, -1.0])
def test_fx_forward_invalid_strike_rejected_by_instrument(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    pricer: FxForwardPricer,
    bad_strike: float,
) -> None:
    """
    Strike validation lives on the *instrument* (FxForward.__post_init__).

    This test asserts we fail fast at construction time, which is desirable:
    it prevents invalid trades from ever reaching pricers / risk.
    """
    # Market is irrelevant here (we never reach the pricer),
    # but we keep it to show the full intended test shape.
    _ = _DummyMarket(
        spot_id=ids["spot"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
        spot_value=float(base_params["spot"]),
        rd=float(base_params["rd"]),
        rf=float(base_params["rf"]),
    )

    with pytest.raises(ValueError, match=r"strike must be > 0"):
        _construct_dataclass(
            FxForward,
            spot_id=ids["spot"],
            strike=float(bad_strike),
            expiry=float(base_params["t"]),
            domestic_curve_id=ids["rd"],
            foreign_curve_id=ids["rf"],
            notional_foreign=float(base_params["notional_foreign"]),
            notional=float(base_params["notional_foreign"]),
        )