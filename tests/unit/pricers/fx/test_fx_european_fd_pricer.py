from __future__ import annotations

import math
import pytest
from typing import Dict
from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.pricers.fx.european_fde import FxEuropeanVanillaFdPricer


# ============================================================================
# Minimal test market (flat curves + flat vol)
# ============================================================================

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

    def vol(self, *, expiry: float, strike: float) -> float:  # noqa: ARG002
        if expiry < 0.0:
            raise ValueError("expiry must be >= 0.")
        return float(self.sigma)


@dataclass(frozen=True, slots=True)
class _DummyMarket:
    """Enough of MarketView/Market for FX vanilla pricers."""
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


# ============================================================================
# Fixtures
# ============================================================================

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
    return {
        "spot": 1.25,
        "strike": 1.25,
        "t": 1.0,
        "rd": 0.03,
        "rf": 0.01,
        "sigma": 0.20,
        "notional": 1_000_000.0,
    }


@pytest.fixture()
def market(ids: Dict[str, MarketId], base_params: Dict[str, float]) -> _DummyMarket:
    return _DummyMarket(
        spot=float(base_params["spot"]),
        rd=float(base_params["rd"]),
        rf=float(base_params["rf"]),
        sigma=float(base_params["sigma"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )


def _make_trade(*, option_type: str, ids: Dict[str, MarketId], base_params: Dict[str, float]) -> EuropeanFxVanillaOption:
    return EuropeanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )


# ============================================================================
# Tests
# ============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_fd_price_is_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Price sanity: FD (constant vol) should match analytic BSM reasonably well.
    """
    trade = _make_trade(option_type=option_type, ids=ids, base_params=base_params)

    bsm = FxEuropeanVanillaBsmPricer()
    fd = FxEuropeanVanillaFdPricer(
        n_space=401,
        n_time_steps=240,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
        vol_abs_bump=1e-4,
        rate_abs_bump=1e-4,
    )

    pv_bsm = float(bsm.price(trade, market))
    pv_fd = float(fd.price(trade, market))

    assert math.isfinite(pv_fd)
    assert pv_fd == pytest.approx(pv_bsm, rel=0.01, abs=1e-2)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_fd_greeks_are_reasonable_vs_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Greek sanity: FD greeks should align with analytic BSM greeks within practical tolerances.

    Notes
    -----
    - Delta/Gamma should be stable if computed from the solved surface using
      grid derivatives + correct log-space chain rule.
    - Vega/Rhos require re-solves -> allow a wider tolerance.
    """
    trade = _make_trade(option_type=option_type, ids=ids, base_params=base_params)

    bsm = FxEuropeanVanillaBsmPricer()
    fd = FxEuropeanVanillaFdPricer(
        n_space=401,
        n_time_steps=240,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
        vol_abs_bump=1e-4,
        rate_abs_bump=1e-4,
    )

    g_bsm = bsm.greeks(trade, market)
    g_fd = fd.greeks(trade, market)

    for k in ["delta", "gamma", "vega", "rho_domestic", "rho_foreign"]:
        assert k in g_fd
        assert math.isfinite(float(g_fd[k]))

    # Delta: generally stable
    assert float(g_fd["delta"]) == pytest.approx(float(g_bsm["delta"]), rel=0.05, abs=1e-10)

    # Gamma should be positive for European vanilla under BSM/GK
    gamma_bsm = float(g_bsm["gamma"])
    gamma_fd = float(g_fd["gamma"])
    assert gamma_bsm > 0.0
    assert gamma_fd > 0.0

    # Gamma magnitude sanity guard (catches log-space chain rule / interpolation bugs)
    ratio = abs(gamma_fd) / max(1e-18, abs(gamma_bsm))
    assert 0.5 <= ratio <= 2.0, f"Gamma magnitude looks wrong (ratio={ratio:.3g})."

    # Gamma closeness (still allow wider tolerance than delta)
    assert gamma_fd == pytest.approx(gamma_bsm, rel=0.35, abs=1e-10)

    # Vega / rhos: bump-and-resolve -> allow moderate tolerance
    assert float(g_fd["vega"]) == pytest.approx(float(g_bsm["vega"]), rel=0.20, abs=1e-10)
    assert float(g_fd["rho_domestic"]) == pytest.approx(float(g_bsm["rho_domestic"]), rel=0.30, abs=1e-10)
    assert float(g_fd["rho_foreign"]) == pytest.approx(float(g_bsm["rho_foreign"]), rel=0.30, abs=1e-10)


def test_fx_vanilla_fd_is_deterministic_for_same_inputs(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Determinism: same inputs -> exactly the same PV (no RNG, no caching randomness).
    """
    trade = _make_trade(option_type="call", ids=ids, base_params=base_params)

    fd = FxEuropeanVanillaFdPricer(
        n_space=301,
        n_time_steps=200,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
        vol_abs_bump=1e-4,
        rate_abs_bump=1e-4,
    )

    pv_a = float(fd.price(trade, market))
    pv_b = float(fd.price(trade, market))

    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)