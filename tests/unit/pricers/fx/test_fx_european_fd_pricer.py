# tests/unit/pricers/fx/test_european_fde_pricer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.instruments.fx.options.digital import FxDigitalEuropeanOption

from src.pricers.fx.european_bsm import FxVanillaEuropeanOptionBsmPricer, FxDigitalEuropeanOptionBsmPricer
from src.pricers.fx.european_fde import FxVanillaEuropeanOptionFdPricer, FxDigitalEuropeanOptionFdPricer

from src.models.payoffs.vanilla import VanillaPayoff
from src.models.payoffs.digital import DigitalCashPayoff, DigitalAssetPayoff


# =============================================================================
# Minimal deterministic Market stub (enough for BSM + FD pricers)
# =============================================================================

@dataclass(frozen=True, slots=True)
class _FlatCurve:
    """Continuously-compounded flat curve: df(t) = exp(-r t)."""
    rate: float

    def df(self, t: float) -> float:
        t = float(t)
        if t < 0.0:
            raise ValueError("t must be >= 0.")
        return float(math.exp(-float(self.rate) * t))


@dataclass(frozen=True, slots=True)
class _FlatVolSurface:
    """Flat vol surface: vol(expiry, strike) = sigma."""
    sigma: float

    def vol(self, *, expiry: float, strike: float) -> float:  # noqa: ARG002 (strike unused)
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)


@dataclass(frozen=True, slots=True)
class _DummyMarket:
    """
    Minimal Market interface required by:
      - european_bsm.py adaptors (vanilla/digital)
      - european_fde.py pricers (vanilla/digital)
    """
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

    def curve(self, market_id: MarketId) -> _FlatCurve:
        if market_id == self.rd_id:
            return _FlatCurve(rate=self.rd)
        if market_id == self.rf_id:
            return _FlatCurve(rate=self.rf)
        raise KeyError(f"Unknown curve id: {market_id}")

    def vol_surface(self, market_id: MarketId) -> _FlatVolSurface:
        if market_id != self.vol_id:
            raise KeyError(f"Unknown vol id: {market_id}")
        return _FlatVolSurface(sigma=self.sigma)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def ids() -> Dict[str, MarketId]:
    return {
        "spot": MarketId("FX", "SPOT", "EURUSD"),
        "vol": MarketId("FX", "VOL", "EURUSD.VOL"),
        "rd": MarketId("IR", "CURVE", "USD.OIS"),
        "rf": MarketId("IR", "CURVE", "EUR.OIS"),
    }


@pytest.fixture(scope="module")
def base_params() -> Dict[str, float]:
    # Choose stable, not-too-extreme params for PDE accuracy and non-degenerate digitals.
    return {
        "spot": 1.25,
        "strike": 1.25,      # ATM
        "t": 1.0,
        "rd": 0.03,
        "rf": 0.01,
        "sigma": 0.20,
        "notional": 1_000_000.0,
        "cash_payout": 1234.5,
        "asset_units": 2.5,
    }


def _make_market(ids: Dict[str, MarketId], p: Dict[str, float], *, sigma: float | None = None, t: float | None = None) -> _DummyMarket:
    """
    Helper to build a market. `t` is not stored but included as a convenience
    for callers that want to emphasize “same trade different market”.
    """
    _ = t  # not used; trade carries expiry; kept for readability at call sites
    return _DummyMarket(
        spot=float(p["spot"]),
        rd=float(p["rd"]),
        rf=float(p["rf"]),
        sigma=float(p["sigma"] if sigma is None else sigma),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )


@pytest.fixture()
def market(ids: Dict[str, MarketId], base_params: Dict[str, float]) -> _DummyMarket:
    return _make_market(ids, base_params)


# =============================================================================
# Vanilla FD tests
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_fd_price_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    FD PV should be close to analytic BSM PV for European vanilla options.

    This also indirectly checks payoff-library integration because the FD pricer
    uses `VanillaPayoff` as its terminal condition.
    """
    trade = FxVanillaEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxVanillaEuropeanOptionBsmPricer()
    fd = FxVanillaEuropeanOptionFdPricer(
        n_space=401,
        n_time_steps=200,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
    )

    pv_bsm = float(bsm.price(trade, market))
    pv_fd = float(fd.price(trade, market))

    # PDE discretization error exists; choose a stable tolerance.
    assert pv_fd == pytest.approx(pv_bsm, rel=3e-3, abs=1e-8)


def test_fx_vanilla_fd_put_call_parity(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Put-call parity in FX (domestic PV):
        Call - Put = N_f * ( S*df_f - K*df_d )
    """
    s = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])
    rd = float(base_params["rd"])
    rf = float(base_params["rf"])
    n = float(base_params["notional"])

    df_d = math.exp(-rd * t)
    df_f = math.exp(-rf * t)

    call = FxVanillaEuropeanOption(
        option_type="call",
        notional=n,
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    put = FxVanillaEuropeanOption(
        option_type="put",
        notional=n,
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    fd = FxVanillaEuropeanOptionFdPricer(n_space=401, n_time_steps=200, theta=0.5, use_log_space=True)

    pv_call = float(fd.price(call, market))
    pv_put = float(fd.price(put, market))

    expected = n * (s * df_f - k * df_d)

    # Parity should hold quite well; discretization can still introduce small error.
    assert (pv_call - pv_put) == pytest.approx(expected, rel=2e-3, abs=1e-6)


def test_fx_vanilla_fd_t0_matches_payoff_library(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
) -> None:
    """
    For T=0, the FD pricer should return the immediate payoff at S0 (no discounting).
    This is a direct test that the payoff library is the source of truth.
    """
    p = dict(base_params)
    p["t"] = 0.0

    mkt = _make_market(ids, p)

    trade = FxVanillaEuropeanOption(
        option_type="call",
        notional=float(p["notional"]),
        strike=float(p["strike"]),
        expiry=0.0,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    payoff = VanillaPayoff(option_type="call", strike=float(p["strike"]))
    expected_per_unit = float(payoff.terminal(np.asarray([float(p["spot"])], dtype=np.float64))[0])
    expected = float(p["notional"]) * expected_per_unit

    fd = FxVanillaEuropeanOptionFdPricer()
    pv_fd = float(fd.price(trade, mkt))

    assert pv_fd == pytest.approx(expected, rel=0.0, abs=0.0)


def test_fx_vanilla_fd_zero_vol_matches_discounted_deterministic_forward(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
) -> None:
    """
    For sigma=0, the terminal spot is deterministic under GK:
        F0 = S0 * exp((rd-rf)*T)
    PV = exp(-rd*T) * payoff(F0) * notional
    """
    p = dict(base_params)
    mkt = _make_market(ids, p, sigma=0.0)

    S0 = float(p["spot"])
    K = float(p["strike"])
    T = float(p["t"])
    rd = float(p["rd"])
    rf = float(p["rf"])
    n = float(p["notional"])

    trade = FxVanillaEuropeanOption(
        option_type="put",
        notional=n,
        strike=K,
        expiry=T,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    F0 = S0 * math.exp((rd - rf) * T)
    disc = math.exp(-rd * T)

    payoff = VanillaPayoff(option_type="put", strike=K)
    expected_per_unit = float(payoff.terminal(np.asarray([F0], dtype=np.float64))[0])
    expected = n * disc * expected_per_unit

    fd = FxVanillaEuropeanOptionFdPricer()
    pv_fd = float(fd.price(trade, mkt))

    assert pv_fd == pytest.approx(expected, rel=0.0, abs=0.0)


# =============================================================================
# Digital FD tests (cash + asset)
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_digital_fd_cash_price_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Digital cash FD price should be reasonably close to analytic BSM digital cash.

    Note: digitals converge more slowly (discontinuity at K), so tolerance is looser.
    """
    trade = FxDigitalEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff="cash",
        payout_amount=float(base_params["cash_payout"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxDigitalEuropeanOptionBsmPricer()
    fd = FxDigitalEuropeanOptionFdPricer(
        n_space=801,
        n_time_steps=400,
        n_std=7.0,
        theta=0.5,
        use_log_space=True,
    )

    pv_bsm = float(bsm.price(trade, market))
    pv_fd = float(fd.price(trade, market))

    assert pv_fd == pytest.approx(pv_bsm, rel=0.03, abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_digital_fd_asset_price_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Asset-or-nothing digital FD price should be reasonably close to analytic BSM digital asset.
    """
    trade = FxDigitalEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff="asset",
        payout_amount=float(base_params["asset_units"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxDigitalEuropeanOptionBsmPricer()
    fd = FxDigitalEuropeanOptionFdPricer(
        n_space=801,
        n_time_steps=400,
        n_std=7.0,
        theta=0.5,
        use_log_space=True,
    )

    pv_bsm = float(bsm.price(trade, market))
    pv_fd = float(fd.price(trade, market))

    assert pv_fd == pytest.approx(pv_bsm, rel=0.03, abs=1e-8)


@pytest.mark.parametrize(
    "payoff_type,option_type",
    [
        ("cash", "call"),
        ("cash", "put"),
        ("asset", "call"),
        ("asset", "put"),
    ],
)
def test_fx_digital_fd_t0_matches_payoff_library(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    payoff_type: str,
    option_type: str,
) -> None:
    """
    For T=0, the FD digital pricer should return payoff(S0) (no discounting),
    which directly validates payoff-library integration.
    """
    p = dict(base_params)
    p["t"] = 0.0

    mkt = _make_market(ids, p)

    trade = FxDigitalEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff=payoff_type,       # type: ignore[arg-type]
        payout_amount=float(p["cash_payout"] if payoff_type == "cash" else p["asset_units"]),
        strike=float(p["strike"]),
        expiry=0.0,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    S0 = float(p["spot"])
    K = float(p["strike"])

    if payoff_type == "cash":
        payoff = DigitalCashPayoff(option_type=option_type, strike=K, cash=float(p["cash_payout"]))  # type: ignore[arg-type]
    else:
        payoff = DigitalAssetPayoff(option_type=option_type, strike=K, asset_units=float(p["asset_units"]))  # type: ignore[arg-type]

    expected = float(payoff.terminal(np.asarray([S0], dtype=np.float64))[0])

    fd = FxDigitalEuropeanOptionFdPricer()
    pv_fd = float(fd.price(trade, mkt))

    assert pv_fd == pytest.approx(expected, rel=0.0, abs=0.0)


@pytest.mark.parametrize(
    "payoff_type,option_type",
    [
        ("cash", "call"),
        ("cash", "put"),
        ("asset", "call"),
        ("asset", "put"),
    ],
)
def test_fx_digital_fd_zero_vol_matches_discounted_deterministic_forward(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    payoff_type: str,
    option_type: str,
) -> None:
    """
    For sigma=0, S(T) is deterministic: F0 = S0 * exp((rd-rf)*T)
    PV = exp(-rd*T) * payoff(F0)

    For asset digital, payoff(F0) already includes units*F0*1{...}.
    """
    p = dict(base_params)
    mkt = _make_market(ids, p, sigma=0.0)

    S0 = float(p["spot"])
    K = float(p["strike"])
    T = float(p["t"])
    rd = float(p["rd"])
    rf = float(p["rf"])

    payout = float(p["cash_payout"] if payoff_type == "cash" else p["asset_units"])

    trade = FxDigitalEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff=payoff_type,       # type: ignore[arg-type]
        payout_amount=payout,
        strike=K,
        expiry=T,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    F0 = S0 * math.exp((rd - rf) * T)
    disc = math.exp(-rd * T)

    if payoff_type == "cash":
        payoff = DigitalCashPayoff(option_type=option_type, strike=K, cash=payout)  # type: ignore[arg-type]
    else:
        payoff = DigitalAssetPayoff(option_type=option_type, strike=K, asset_units=payout)  # type: ignore[arg-type]

    expected = disc * float(payoff.terminal(np.asarray([F0], dtype=np.float64))[0])

    fd = FxDigitalEuropeanOptionFdPricer()
    pv_fd = float(fd.price(trade, mkt))

    assert pv_fd == pytest.approx(expected, rel=0.0, abs=1e-12)