# tests/unit/pricers/fx/test_european_mc_pricer.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest

from src.marketdata.ids import MarketId
from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.digital import EuropeanFxDigitalOption

from src.pricers.fx.european_bsm import (
    FxEuropeanVanillaBsmPricer,
    FxEuropeanDigitalBsmPricer,
)
from src.pricers.fx.european_mc import (
    FxEuropeanVanillaMcPricer,
    FxEuropeanDigitalMcPricer,
)


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
    """Enough of Market/MarketView for FX vanilla + digital pricers."""
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
# Small numeric helpers
# =============================================================================

def _mean_stderr(x: np.ndarray) -> tuple[float, float]:
    """Return (mean, stderr) for 1D samples using ddof=1 when possible."""
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = int(v.size)
    if n <= 0:
        raise ValueError("empty samples")
    m = float(v.mean())
    if n == 1:
        return m, 0.0
    s2 = float(v.var(ddof=1))
    se = math.sqrt(max(0.0, s2 / n))
    return m, float(se)


def _assert_mc_close_to_analytic(
    *,
    pv_mc: float,
    stderr: float,
    pv_ref: float,
    rel_floor: float,
    abs_floor: float,
    n_sigma: float = 5.0,
) -> None:
    """
    Robust stochastic assertion:
      |mc - ref| <= max(n_sigma*stderr, rel_floor*|ref|, abs_floor)
    """
    tol = max(float(n_sigma) * float(stderr), float(rel_floor) * abs(float(pv_ref)), float(abs_floor))
    assert abs(float(pv_mc) - float(pv_ref)) <= tol


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
    # Stable, non-degenerate parameters
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
# Vanilla MC tests
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_mc_price_is_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
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

    assert pv_mc == pytest.approx(pv_bsm, rel=0.02, abs=1e-2)


def test_fx_vanilla_mc_scales_linearly_with_notional(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
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
        ("exact", 1, 0.015),
        ("milstein", 64, 0.030),
        ("euler", 128, 0.060),
    ],
)
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_mc_schemes_are_reasonable_vs_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
    scheme: str,
    n_steps: int,
    rel_tol: float,
) -> None:
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
    denom = max(1e-12, abs(pv_bsm))
    assert abs(pv_mc - pv_bsm) / denom <= rel_tol


# =============================================================================
# Digital MC tests (cash + asset)
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_digital_cash_mc_price_is_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    payout = 10_000.0  # domestic cash amount

    trade = EuropeanFxDigitalOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff="cash",
        payout_amount=float(payout),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxEuropeanDigitalBsmPricer()
    mc = FxEuropeanDigitalMcPricer(n_paths=250_000, seed=7, antithetic=True)

    pv_bsm = float(bsm.price(trade, market))
    sim = mc.run(trade, market, store_paths=False)
    pv_mc, se = _mean_stderr(sim.discounted_payoffs)

    # Digitals are high-variance; use stderr-aware tolerance.
    _assert_mc_close_to_analytic(pv_mc=pv_mc, stderr=se, pv_ref=pv_bsm, rel_floor=0.03, abs_floor=1e-8, n_sigma=6.0)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_digital_asset_mc_price_is_close_to_bsm(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    asset_units = 20_000.0  # foreign units

    trade = EuropeanFxDigitalOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff="asset",
        payout_amount=float(asset_units),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    bsm = FxEuropeanDigitalBsmPricer()
    mc = FxEuropeanDigitalMcPricer(n_paths=250_000, seed=7, antithetic=True)

    pv_bsm = float(bsm.price(trade, market))
    sim = mc.run(trade, market, store_paths=False)
    pv_mc, se = _mean_stderr(sim.discounted_payoffs)

    _assert_mc_close_to_analytic(pv_mc=pv_mc, stderr=se, pv_ref=pv_bsm, rel_floor=0.03, abs_floor=1e-8, n_sigma=6.0)


def test_fx_digital_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    trade = EuropeanFxDigitalOption(
        option_type="call",
        payoff="cash",
        payout_amount=5_000.0,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_a = float(FxEuropeanDigitalMcPricer(n_paths=120_000, seed=777, antithetic=True).price(trade, market))
    pv_b = float(FxEuropeanDigitalMcPricer(n_paths=120_000, seed=777, antithetic=True).price(trade, market))
    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_digital_mc_changes_with_different_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    trade = EuropeanFxDigitalOption(
        option_type="put",
        payoff="asset",
        payout_amount=10_000.0,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_a = float(FxEuropeanDigitalMcPricer(n_paths=90_000, seed=1, antithetic=True).price(trade, market))
    pv_b = float(FxEuropeanDigitalMcPricer(n_paths=90_000, seed=2, antithetic=True).price(trade, market))
    assert pv_a != pv_b


def test_fx_digital_cash_call_put_parity_mc(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Cash digital parity (continuous distribution):
      PV_call + PV_put = payout * df_d(T)
    """
    payout = 12_345.0
    t = float(base_params["t"])
    rd = float(base_params["rd"])
    df_d = math.exp(-rd * t)

    call = EuropeanFxDigitalOption(
        option_type="call",
        payoff="cash",
        payout_amount=payout,
        strike=float(base_params["strike"]),
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    put = EuropeanFxDigitalOption(
        option_type="put",
        payoff="cash",
        payout_amount=payout,
        strike=float(base_params["strike"]),
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    mc = FxEuropeanDigitalMcPricer(n_paths=300_000, seed=42, antithetic=True)

    sim_c = mc.run(call, market, store_paths=False)
    sim_p = mc.run(put, market, store_paths=False)

    pv_c, se_c = _mean_stderr(sim_c.discounted_payoffs)
    pv_p, se_p = _mean_stderr(sim_p.discounted_payoffs)

    lhs = pv_c + pv_p
    rhs = payout * df_d

    # combine stderrs conservatively (independent runs) even though seeds are same; keep it simple
    se_sum = math.sqrt(se_c * se_c + se_p * se_p)
    _assert_mc_close_to_analytic(pv_mc=lhs, stderr=se_sum, pv_ref=rhs, rel_floor=0.01, abs_floor=1e-8, n_sigma=6.0)


def test_fx_digital_asset_call_put_parity_mc(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Asset digital parity (continuous distribution):
      PV_call + PV_put = asset_units * S0 * df_f(T)
    because discounted E[S_T] under domestic measure equals S0*df_f.
    """
    asset_units = 7_500.0
    t = float(base_params["t"])
    s0 = float(base_params["spot"])
    rf = float(base_params["rf"])
    df_f = math.exp(-rf * t)

    call = EuropeanFxDigitalOption(
        option_type="call",
        payoff="asset",
        payout_amount=asset_units,
        strike=float(base_params["strike"]),
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    put = EuropeanFxDigitalOption(
        option_type="put",
        payoff="asset",
        payout_amount=asset_units,
        strike=float(base_params["strike"]),
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    mc = FxEuropeanDigitalMcPricer(n_paths=300_000, seed=99, antithetic=True)

    sim_c = mc.run(call, market, store_paths=False)
    sim_p = mc.run(put, market, store_paths=False)

    pv_c, se_c = _mean_stderr(sim_c.discounted_payoffs)
    pv_p, se_p = _mean_stderr(sim_p.discounted_payoffs)

    lhs = pv_c + pv_p
    rhs = asset_units * s0 * df_f

    se_sum = math.sqrt(se_c * se_c + se_p * se_p)
    _assert_mc_close_to_analytic(pv_mc=lhs, stderr=se_sum, pv_ref=rhs, rel_floor=0.01, abs_floor=1e-8, n_sigma=6.0)


@pytest.mark.parametrize(
    "payoff,option_type,spot,strike,expected_domestic",
    [
        ("cash", "call", 1.30, 1.20, 5.0),
        ("cash", "call", 1.10, 1.20, 0.0),
        ("cash", "put",  1.10, 1.20, 5.0),
        ("cash", "put",  1.30, 1.20, 0.0),
        ("asset", "call", 1.30, 1.20, 2.0 * 1.30),
        ("asset", "call", 1.10, 1.20, 0.0),
        ("asset", "put",  1.10, 1.20, 2.0 * 1.10),
        ("asset", "put",  1.30, 1.20, 0.0),
    ],
)
def test_fx_digital_mc_price_at_expiry_is_deterministic(
    ids: Dict[str, MarketId],
    payoff: str,
    option_type: str,
    spot: float,
    strike: float,
    expected_domestic: float,
) -> None:
    """
    At T=0, MC pricer should return the payoff evaluated at spot (df=1).
    """
    market0 = _DummyMarket(
        spot=float(spot),
        rd=0.05,
        rf=0.02,
        sigma=0.20,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    payout_amount = 5.0 if payoff == "cash" else 2.0

    trade = EuropeanFxDigitalOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff=payoff,            # type: ignore[arg-type]
        payout_amount=float(payout_amount),
        strike=float(strike),
        expiry=0.0,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    mc = FxEuropeanDigitalMcPricer(n_paths=10_000, seed=1, antithetic=True)
    pv = float(mc.price(trade, market0))

    assert pv == pytest.approx(float(expected_domestic), rel=0.0, abs=0.0)
