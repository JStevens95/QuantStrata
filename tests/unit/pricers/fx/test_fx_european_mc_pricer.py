# tests/unit/pricers/fx/test_fx_european_mc_pricer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import pytest

from src.marketdata.ids import MarketId
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.pricers.fx.european_bsm import FxEuropeanVanillaBsmPricer
from src.pricers.fx.european_mc import FxEuropeanVanillaMcPricer


# =============================================================================
# Minimal deterministic MarketView for pricer unit tests
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
    # Stable, non-degenerate parameters for monotone + parity-safe behavior
    return {
        "spot": 1.25,
        "strike": 1.25,   # ATM
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


# =============================================================================
# Tests
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_mc_price_is_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Monte Carlo should be close to analytic BSM for European FX vanilla.

    Notes
    -----
    - This is a stochastic test; we choose:
        * reasonably large path count
        * fixed seed
        * antithetic variates
      to make it stable and fast.
    """
    trade = EuropeanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxEuropeanVanillaBsmPricer()
    mc = FxEuropeanVanillaMcPricer(n_paths=120_000, seed=7, antithetic=True)

    pv_bsm = float(bsm.price(trade, market))
    pv_mc = float(mc.price(trade, market))

    # Relative tolerance chosen to avoid flaky CI while still being meaningful.
    assert pv_mc == pytest.approx(pv_bsm, rel=0.02, abs=1e-2)


def test_fx_vanilla_mc_scales_linearly_with_notional(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    PV must scale linearly with trade.notional.

    We force identical random draws by using the same seed and re-instantiating
    the pricer so the per-unit PV cancels exactly.
    """
    notional_1 = float(base_params["notional"])
    notional_2 = 2.0 * notional_1

    trade_1 = EuropeanFxVanillaOption(
        option_type="call",
        notional=notional_1,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    trade_2 = EuropeanFxVanillaOption(
        option_type="call",
        notional=notional_2,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_1 = float(FxEuropeanVanillaMcPricer(n_paths=80_000, seed=11, antithetic=True).price(trade_1, market))
    pv_2 = float(FxEuropeanVanillaMcPricer(n_paths=80_000, seed=11, antithetic=True).price(trade_2, market))

    assert pv_2 == pytest.approx(2.0 * pv_1, rel=1e-12, abs=1e-6)


def test_fx_vanilla_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Same seed => same PV (given identical code path and parameters).
    """
    trade = EuropeanFxVanillaOption(
        option_type="put",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_a = float(FxEuropeanVanillaMcPricer(n_paths=50_000, seed=999, antithetic=True).price(trade, market))
    pv_b = float(FxEuropeanVanillaMcPricer(n_paths=50_000, seed=999, antithetic=True).price(trade, market))

    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_vanilla_mc_changes_with_different_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Different seeds should (almost surely) produce different PVs.
    """
    trade = EuropeanFxVanillaOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_a = float(FxEuropeanVanillaMcPricer(n_paths=40_000, seed=1, antithetic=True).price(trade, market))
    pv_b = float(FxEuropeanVanillaMcPricer(n_paths=40_000, seed=2, antithetic=True).price(trade, market))

    assert pv_a != pv_b


@pytest.mark.parametrize(
    "scheme,n_steps,rel_tol",
    [
        ("exact", 1, 0.015),      # exact terminal distribution -> should be very close
        ("milstein", 64, 0.030),  # converges well; should be close to BSM/exact
        ("euler", 128, 0.060),    # weaker scheme; allow looser tolerance
    ],
)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_mc_schemes_are_reasonable_vs_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market,
    option_type: str,
    scheme: str,
    n_steps: int,
    rel_tol: float,
) -> None:
    """
    Sanity: different GBM discretization schemes produce a reasonable PV and
    converge toward the analytic BSM value as dt shrinks.

    Notes
    -----
    - We keep this test stable by:
      * fixed seed
      * antithetic variates
      * moderately large path count
    - Euler can be noisy / allow negative spots. We only require "close-ish".
    """
    trade = EuropeanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxEuropeanVanillaBsmPricer()
    pv_bsm = float(bsm.price(trade, market))

    mc = FxEuropeanVanillaMcPricer(
        n_paths=120_000,
        seed=123,
        antithetic=True,
        n_steps=int(n_steps),
        scheme=scheme,  # type: ignore[arg-type]
    )
    pv_mc = float(mc.price(trade, market))

    assert math.isfinite(pv_mc)
    # avoid division blow-up if PV is extremely small (deep OTM), though your tests are ATM-ish
    denom = max(1e-12, abs(pv_bsm))
    assert abs(pv_mc - pv_bsm) / denom <= rel_tol
