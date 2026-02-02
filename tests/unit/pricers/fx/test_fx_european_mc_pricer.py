from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pytest

from src.marketdata.core.ids import MarketId
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption
from src.instruments.fx.options.digital import FxDigitalEuropeanOption
from src.instruments.fx.options.barrier import FxBarrierEuropeanOption
from src.instruments.fx.options.asian import FxAsianEuropeanOption
from src.instruments.fx.options.lookback import FxLookbackEuropeanOption

from src.pricers.fx.european_bsm import (
    FxVanillaEuropeanOptionBsmPricer,
    FxDigitalEuropeanOptionBsmPricer,
    _rate_from_df
)
from src.pricers.fx.european_mc import (
    FxVanillaEuropeanOptionMcPricer,
    FxDigitalEuropeanOptionMcPricer,
    FxBarrierEuropeanOptionMcPricer,
    FxAsianEuropeanOptionMcPricer,
    FxLookbackEuropeanOptionMcPricer,
)

from src.models.payoffs.barrier import SingleBarrierPayoff


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
    mc = FxVanillaEuropeanOptionMcPricer(n_paths=120_000, seed=7, antithetic=True)

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

    trade_1 = FxVanillaEuropeanOption(
        option_type="call",
        notional=notional_1,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    trade_2 = FxVanillaEuropeanOption(
        option_type="call",
        notional=notional_2,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_1 = float(FxVanillaEuropeanOptionMcPricer(n_paths=80_000, seed=11, antithetic=True).price(trade_1, market))
    pv_2 = float(FxVanillaEuropeanOptionMcPricer(n_paths=80_000, seed=11, antithetic=True).price(trade_2, market))

    assert pv_2 == pytest.approx(2.0 * pv_1, rel=1e-12, abs=1e-6)


def test_fx_vanilla_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    trade = FxVanillaEuropeanOption(
        option_type="put",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_a = float(FxVanillaEuropeanOptionMcPricer(n_paths=50_000, seed=999, antithetic=True).price(trade, market))
    pv_b = float(FxVanillaEuropeanOptionMcPricer(n_paths=50_000, seed=999, antithetic=True).price(trade, market))

    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_vanilla_mc_changes_with_different_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    trade = FxVanillaEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_a = float(FxVanillaEuropeanOptionMcPricer(n_paths=40_000, seed=1, antithetic=True).price(trade, market))
    pv_b = float(FxVanillaEuropeanOptionMcPricer(n_paths=40_000, seed=2, antithetic=True).price(trade, market))

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
    pv_bsm = float(bsm.price(trade, market))

    mc = FxVanillaEuropeanOptionMcPricer(
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

    trade = FxDigitalEuropeanOption(
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

    bsm = FxDigitalEuropeanOptionBsmPricer()
    mc = FxDigitalEuropeanOptionMcPricer(n_paths=250_000, seed=7, antithetic=True)

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

    trade = FxDigitalEuropeanOption(
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

    bsm = FxDigitalEuropeanOptionBsmPricer()
    mc = FxDigitalEuropeanOptionMcPricer(n_paths=250_000, seed=7, antithetic=True)

    pv_bsm = float(bsm.price(trade, market))
    sim = mc.run(trade, market, store_paths=False)
    pv_mc, se = _mean_stderr(sim.discounted_payoffs)

    _assert_mc_close_to_analytic(pv_mc=pv_mc, stderr=se, pv_ref=pv_bsm, rel_floor=0.03, abs_floor=1e-8, n_sigma=6.0)


def test_fx_digital_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    trade = FxDigitalEuropeanOption(
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

    pv_a = float(FxDigitalEuropeanOptionMcPricer(n_paths=120_000, seed=777, antithetic=True).price(trade, market))
    pv_b = float(FxDigitalEuropeanOptionMcPricer(n_paths=120_000, seed=777, antithetic=True).price(trade, market))
    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_digital_mc_changes_with_different_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    trade = FxDigitalEuropeanOption(
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

    pv_a = float(FxDigitalEuropeanOptionMcPricer(n_paths=90_000, seed=1, antithetic=True).price(trade, market))
    pv_b = float(FxDigitalEuropeanOptionMcPricer(n_paths=90_000, seed=2, antithetic=True).price(trade, market))
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

    call = FxDigitalEuropeanOption(
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
    put = FxDigitalEuropeanOption(
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

    mc = FxDigitalEuropeanOptionMcPricer(n_paths=300_000, seed=42, antithetic=True)

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

    call = FxDigitalEuropeanOption(
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
    put = FxDigitalEuropeanOption(
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

    mc = FxDigitalEuropeanOptionMcPricer(n_paths=300_000, seed=99, antithetic=True)

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

    trade = FxDigitalEuropeanOption(
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

    mc = FxDigitalEuropeanOptionMcPricer(n_paths=10_000, seed=1, antithetic=True)
    pv = float(mc.price(trade, market0))

    assert pv == pytest.approx(float(expected_domestic), rel=0.0, abs=0.0)


# =============================================================================
# Barrier MC tests (single-barrier KO/KI with rebate, discrete monitoring)
# =============================================================================

def _make_barrier_trade(
    *,
    ids: Dict[str, MarketId],
    option_type: str,
    notional: float,
    strike: float,
    expiry: float,
    barrier_direction: str,
    barrier_style: str,
    barrier_level: float,
    rebate_amount: float,
) -> FxBarrierEuropeanOption:
    return FxBarrierEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(notional),
        strike=float(strike),
        expiry=float(expiry),
        barrier_direction=barrier_direction,  # type: ignore[arg-type]
        barrier_style=barrier_style,          # type: ignore[arg-type]
        barrier_level=float(barrier_level),
        rebate_amount=float(rebate_amount),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )


def _deterministic_gbm_paths_zero_vol(
    *,
    spot0: float,
    drift: float,
    maturity: float,
    n_steps: int,
) -> np.ndarray:
    """
    Build the deterministic GBM path when sigma=0 under:
      S(t) = S0 * exp(drift * t)
    Returns shape (1, n_steps+1).
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive.")
    t = np.linspace(0.0, float(maturity), int(n_steps) + 1, dtype=np.float64)
    path = float(spot0) * np.exp(float(drift) * t)
    return path.reshape(1, -1)


@pytest.mark.parametrize(
    "barrier_direction,barrier_level",
    [
        # Base params: S0=1.25, drift=rd-rf=0.02, deterministic S_T ~ 1.25*exp(0.02) ~ 1.275...
        ("up", 1.27),     # crossed by expiry (hit)
        ("up", 1.40),     # never reaches (no hit)
        ("down", 1.26),   # hit immediately because S0=1.25 <= 1.26
        ("down", 1.10),   # never goes that low (no hit)
    ],
)
@pytest.mark.parametrize("barrier_style", ["knock_out", "knock_in"])
@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_barrier_mc_zero_vol_matches_discounted_deterministic_path(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    barrier_direction: str,
    barrier_level: float,
    barrier_style: str,
    option_type: str,
) -> None:
    """
    With sigma=0 the path is deterministic, so the MC pricer should match a
    deterministic discrete-monitored payoff computed from that path.

    This regression test is intentionally strong (non-stochastic):
      - barrier hit logic (up/down, inclusive)
      - KO/KI logic
      - rebate usage
      - discounting + notional scaling
    """
    # Market with sigma=0 (deterministic path)
    mkt0 = _DummyMarket(
        spot=float(base_params["spot"]),
        rd=float(base_params["rd"]),
        rf=float(base_params["rf"]),
        sigma=0.0,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    s0 = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])

    notional = float(base_params["notional"])
    rebate = 0.1234  # per-unit-notional domestic rebate paid at expiry

    trade = _make_barrier_trade(
        ids=ids,
        option_type=option_type,
        notional=notional,
        strike=k,
        expiry=t,
        barrier_direction=barrier_direction,
        barrier_style=barrier_style,
        barrier_level=float(barrier_level),
        rebate_amount=rebate,
    )

    # Use multiple steps so discrete monitoring is meaningful.
    n_steps = 64

    mc = FxBarrierEuropeanOptionMcPricer(
        n_paths=50_000,
        seed=7,
        antithetic=True,
        n_steps=n_steps,
        scheme="exact",  # type: ignore[arg-type]
    )
    pv_mc = float(mc.price(trade, mkt0))

    # --- Expected PV computed using the SAME df->rate mapping as the pricer ---
    df_d = float(mkt0.curve(ids["rd"]).df(t))
    df_f = float(mkt0.curve(ids["rf"]).df(t))

    r_d = _rate_from_df(df=df_d, t=t)
    r_f = _rate_from_df(df=df_f, t=t)
    drift = float(r_d - r_f)

    paths = _deterministic_gbm_paths_zero_vol(spot0=s0, drift=drift, maturity=t, n_steps=n_steps)

    payoff = SingleBarrierPayoff(
        option_type=option_type,  # type: ignore[arg-type]
        strike=k,
        barrier_direction=barrier_direction,  # type: ignore[arg-type]
        barrier_style=barrier_style,          # type: ignore[arg-type]
        barrier_level=float(barrier_level),
        rebate_amount=rebate,
    )

    payoff_per_unit = float(payoff.terminal_from_paths(paths)[0])
    expected = float(notional) * float(df_d) * float(payoff_per_unit)

    # Avoid brittle exact-equality; numeric round-trips can differ at ~1e-9
    assert pv_mc == pytest.approx(expected, rel=0.0, abs=1e-8)


def test_fx_barrier_mc_scales_linearly_with_notional(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """Barrier PV must scale linearly with notional (foreign units)."""
    n1 = float(base_params["notional"])
    n2 = 3.0 * n1

    trade_1 = _make_barrier_trade(
        ids=ids,
        option_type="call",
        notional=n1,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        barrier_direction="up",
        barrier_style="knock_out",
        barrier_level=1.60,  # far barrier => stable regression behaviour
        rebate_amount=0.0,
    )
    trade_2 = _make_barrier_trade(
        ids=ids,
        option_type="call",
        notional=n2,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        barrier_direction="up",
        barrier_style="knock_out",
        barrier_level=1.60,
        rebate_amount=0.0,
    )

    mc = FxBarrierEuropeanOptionMcPricer(
        n_paths=120_000,
        seed=123,
        antithetic=True,
        n_steps=64,
        scheme="exact",  # type: ignore[arg-type]
    )

    pv_1 = float(mc.price(trade_1, market))
    pv_2 = float(mc.price(trade_2, market))

    assert pv_2 == pytest.approx(3.0 * pv_1, rel=1e-12, abs=1e-6)


def test_fx_barrier_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """Same seed => same PV."""
    trade = _make_barrier_trade(
        ids=ids,
        option_type="put",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        barrier_direction="down",
        barrier_style="knock_in",
        barrier_level=1.10,
        rebate_amount=0.05,
    )

    pv_a = float(
        FxBarrierEuropeanOptionMcPricer(
            n_paths=80_000,
            seed=999,
            antithetic=True,
            n_steps=64,
            scheme="exact",  # type: ignore[arg-type]
        ).price(trade, market)
    )
    pv_b = float(
        FxBarrierEuropeanOptionMcPricer(
            n_paths=80_000,
            seed=999,
            antithetic=True,
            n_steps=64,
            scheme="exact",  # type: ignore[arg-type]
        ).price(trade, market)
    )

    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_barrier_mc_changes_with_different_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """Different seeds => generally different PV (stochastic)."""
    trade = _make_barrier_trade(
        ids=ids,
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        barrier_direction="up",
        barrier_style="knock_out",
        barrier_level=1.35,
        rebate_amount=0.02,
    )

    pv_a = float(FxBarrierEuropeanOptionMcPricer(n_paths=60_000, seed=1, antithetic=True, n_steps=64, scheme="exact").price(trade, market))  # type: ignore[arg-type]
    pv_b = float(FxBarrierEuropeanOptionMcPricer(n_paths=60_000, seed=2, antithetic=True, n_steps=64, scheme="exact").price(trade, market))  # type: ignore[arg-type]

    assert pv_a != pv_b


def test_fx_barrier_mc_price_at_expiry_is_deterministic(
    ids: Dict[str, MarketId],
) -> None:
    """
    At T=0, barrier payoff is evaluated on the trivial path [S0].
    Because monitoring includes S0, the barrier may be hit immediately.
    """
    s0 = 1.25

    market0 = _DummyMarket(
        spot=float(s0),
        rd=0.03,
        rf=0.01,
        sigma=0.20,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    # Up KO with barrier below S0 => hit immediately => pays rebate (df=1 at T=0)
    trade = _make_barrier_trade(
        ids=ids,
        option_type="call",
        notional=1_000_000.0,
        strike=1.20,
        expiry=0.0,
        barrier_direction="up",
        barrier_style="knock_out",
        barrier_level=1.10,
        rebate_amount=0.25,  # domestic per unit notional
    )

    mc = FxBarrierEuropeanOptionMcPricer(n_paths=10_000, seed=1, antithetic=True, n_steps=1, scheme="exact")  # type: ignore[arg-type]
    pv = float(mc.price(trade, market0))

    expected = float(trade.notional) * float(trade.rebate_amount)
    assert pv == pytest.approx(expected, rel=0.0, abs=0.0)


# =============================================================================
# Asian MC tests (arithmetic + geometric averaging)
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("averaging_type", ["arithmetic", "geometric"])
def test_fx_asian_mc_price_is_positive(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
    averaging_type: str,
) -> None:
    """
    Test that Asian option prices are non-negative.

    This is a basic sanity check: option prices should never be negative.
    """
    trade = FxAsianEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type=averaging_type,  # type: ignore[arg-type]
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxAsianEuropeanOptionMcPricer(n_paths=50_000, seed=7, antithetic=True, n_steps=64)
    pv = float(pricer.price(trade, market))

    assert pv >= 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_asian_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Test that Asian pricer is reproducible with same seed.

    This ensures deterministic behavior, which is important for testing and debugging.
    """
    trade = FxAsianEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer_a = FxAsianEuropeanOptionMcPricer(n_paths=50_000, seed=999, antithetic=True, n_steps=64)
    pricer_b = FxAsianEuropeanOptionMcPricer(n_paths=50_000, seed=999, antithetic=True, n_steps=64)

    pv_a = float(pricer_a.price(trade, market))
    pv_b = float(pricer_b.price(trade, market))

    # With same seed, results should be identical
    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_asian_mc_scales_linearly_with_notional(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Test that Asian option price scales linearly with notional.

    This is a fundamental property: doubling notional should double PV.
    """
    notional_1 = float(base_params["notional"])
    notional_2 = 2.0 * notional_1

    trade_1 = FxAsianEuropeanOption(
        option_type="call",
        notional=notional_1,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    trade_2 = FxAsianEuropeanOption(
        option_type="call",
        notional=notional_2,
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxAsianEuropeanOptionMcPricer(n_paths=80_000, seed=11, antithetic=True, n_steps=64)
    pv_1 = float(pricer.price(trade_1, market))
    pv_2 = float(pricer.price(trade_2, market))

    assert pv_2 == pytest.approx(2.0 * pv_1, rel=1e-12, abs=1e-6)


# =============================================================================
# Asian: Comparison with vanilla options
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_asian_is_cheaper_than_vanilla(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Test that Asian options are cheaper than vanilla options.

    This is a fundamental property: averaging reduces volatility, making Asian
    options cheaper than their vanilla counterparts.
    """
    asian_trade = FxAsianEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    vanilla_trade = FxVanillaEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    asian_pricer = FxAsianEuropeanOptionMcPricer(n_paths=100_000, seed=7, antithetic=True, n_steps=64)
    vanilla_pricer = FxVanillaEuropeanOptionMcPricer(n_paths=100_000, seed=7, antithetic=True)

    pv_asian = float(asian_pricer.price(asian_trade, market))
    pv_vanilla = float(vanilla_pricer.price(vanilla_trade, market))

    # Asian should be cheaper (or equal in degenerate cases)
    assert pv_asian <= pv_vanilla * 1.01  # Allow small numerical error


def test_fx_asian_geometric_is_cheaper_than_arithmetic(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Test that geometric Asian is cheaper than arithmetic Asian.

    This follows from Jensen's inequality: geometric mean <= arithmetic mean.
    """
    arith_trade = FxAsianEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    geom_trade = FxAsianEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="geometric",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxAsianEuropeanOptionMcPricer(n_paths=100_000, seed=7, antithetic=True, n_steps=64)
    pv_arith = float(pricer.price(arith_trade, market))
    pv_geom = float(pricer.price(geom_trade, market))

    # Geometric should be cheaper (or equal in degenerate cases)
    assert pv_geom <= pv_arith * 1.01  # Allow small numerical error


# =============================================================================
# Asian: Edge cases
# =============================================================================

def test_fx_asian_mc_price_at_expiry_is_deterministic(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Test that Asian option at expiry gives deterministic payoff.

    At expiry (T=0), the path contains only S0, so average = S0 and payoff is deterministic.
    """
    trade = FxAsianEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=0.0,  # At expiry
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxAsianEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=1)
    pv = float(pricer.price(trade, market))

    # At expiry, average = S0 = 1.25, strike = 1.25, so payoff = max(1.25 - 1.25, 0) = 0
    # PV = notional * df_d(0) * 0 = 0
    expected_pv = 0.0
    assert pv == pytest.approx(expected_pv, abs=1e-10)


def test_fx_asian_mc_price_at_expiry_in_the_money(
    ids: Dict[str, MarketId],
    market: _DummyMarket,
) -> None:
    """Test Asian option at expiry when in-the-money."""
    trade = FxAsianEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        strike=1.20,  # Below spot (1.25), so in-the-money
        expiry=0.0,  # At expiry
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxAsianEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=1)
    pv = float(pricer.price(trade, market))

    # At expiry, average = S0 = 1.25, strike = 1.20, so payoff = max(1.25 - 1.20, 0) = 0.05
    # PV = notional * df_d(0) * 0.05 = 1_000_000 * 1.0 * 0.05 = 50_000
    expected_pv = 1_000_000.0 * 0.05
    assert pv == pytest.approx(expected_pv, abs=1e-6)


def test_fx_asian_mc_invalid_n_paths(
    ids: Dict[str, MarketId],
    market: _DummyMarket,
) -> None:
    """Test that invalid n_paths raises ValueError."""
    pricer = FxAsianEuropeanOptionMcPricer(n_paths=0, seed=7)
    trade = FxAsianEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        strike=1.25,
        expiry=1.0,
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    with pytest.raises(ValueError, match="n_paths must be positive"):
        pricer.price(trade, market)


def test_fx_asian_mc_invalid_n_steps(
    ids: Dict[str, MarketId],
    market: _DummyMarket,
) -> None:
    """Test that invalid n_steps raises ValueError."""
    pricer = FxAsianEuropeanOptionMcPricer(n_paths=10_000, seed=7, n_steps=0)
    trade = FxAsianEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        strike=1.25,
        expiry=1.0,
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    with pytest.raises(ValueError, match="n_steps must be positive"):
        pricer.price(trade, market)


# =============================================================================
# Asian: Simulation artifact tests
# =============================================================================

def test_fx_asian_mc_simulation_artifact_has_correct_structure(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """Test that simulation artifact has correct structure and fields."""
    trade = FxAsianEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        averaging_type="arithmetic",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxAsianEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=64)
    sim = pricer.run(trade, market, store_paths=True, paths_keep=100)

    # Verify all required fields exist
    assert hasattr(sim, "spot0")
    assert hasattr(sim, "strike")
    assert hasattr(sim, "maturity")
    assert hasattr(sim, "terminal_spots")
    assert hasattr(sim, "average_spots")
    assert hasattr(sim, "discounted_payoffs")
    assert hasattr(sim, "paths")

    # Verify shapes
    assert sim.terminal_spots.shape == (sim.n_paths_effective,)
    assert sim.average_spots.shape == (sim.n_paths_effective,)
    assert sim.discounted_payoffs.shape == (sim.n_paths_effective,)
    assert sim.paths is not None
    assert sim.paths.shape[0] == min(100, sim.n_paths_effective)
    assert sim.paths.shape[1] == sim.n_steps + 1

    # Verify that average_spots are computed correctly
    # (average should be mean of each path)
    for i in range(min(10, sim.n_paths_effective)):  # Check first 10 paths
        path = sim.paths[i, :]
        expected_avg = np.mean(path)
        assert sim.average_spots[i] == pytest.approx(expected_avg, abs=1e-10)


# =============================================================================
# Lookback MC tests (floating strike + fixed strike)
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("lookback_type", ["floating_strike", "fixed_strike"])
def test_fx_lookback_mc_price_is_positive(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
    lookback_type: str,
) -> None:
    """
    Test that lookback option prices are non-negative.

    This is a basic sanity check: option prices should never be negative.
    """
    # For fixed strike, use a strike price; for floating, use 0 (ignored)
    strike = float(base_params["strike"]) if lookback_type == "fixed_strike" else 0.0

    trade = FxLookbackEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        expiry=float(base_params["t"]),
        lookback_type=lookback_type,  # type: ignore[arg-type]
        strike=strike,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=50_000, seed=7, antithetic=True, n_steps=64)
    pv = float(pricer.price(trade, market))

    assert pv >= 0.0


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_lookback_mc_is_reproducible_for_same_seed(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Test that lookback pricer is reproducible with same seed.

    This ensures deterministic behavior for testing and debugging.
    """
    trade = FxLookbackEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        expiry=float(base_params["t"]),
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer_a = FxLookbackEuropeanOptionMcPricer(n_paths=50_000, seed=999, antithetic=True, n_steps=64)
    pricer_b = FxLookbackEuropeanOptionMcPricer(n_paths=50_000, seed=999, antithetic=True, n_steps=64)

    pv_a = float(pricer_a.price(trade, market))
    pv_b = float(pricer_b.price(trade, market))

    # With same seed, results should be identical
    assert pv_a == pytest.approx(pv_b, rel=0.0, abs=0.0)


def test_fx_lookback_mc_scales_linearly_with_notional(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Test that lookback option price scales linearly with notional.

    This is a fundamental property: doubling notional should double PV.
    """
    notional_1 = float(base_params["notional"])
    notional_2 = 2.0 * notional_1

    trade_1 = FxLookbackEuropeanOption(
        option_type="call",
        notional=notional_1,
        expiry=float(base_params["t"]),
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    trade_2 = FxLookbackEuropeanOption(
        option_type="call",
        notional=notional_2,
        expiry=float(base_params["t"]),
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=80_000, seed=11, antithetic=True, n_steps=64)
    pv_1 = float(pricer.price(trade_1, market))
    pv_2 = float(pricer.price(trade_2, market))

    assert pv_2 == pytest.approx(2.0 * pv_1, rel=1e-12, abs=1e-6)


# =============================================================================
# Lookback: Comparison with vanilla options
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_lookback_is_more_expensive_than_vanilla(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    option_type: str,
) -> None:
    """
    Test that lookback options are more expensive than vanilla options.

    This is a fundamental property: lookback captures optimal timing,
    so it must be worth at least as much as vanilla.
    """
    lookback_trade = FxLookbackEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        expiry=float(base_params["t"]),
        lookback_type="fixed_strike",
        strike=float(base_params["strike"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    vanilla_trade = FxVanillaEuropeanOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=float(base_params["notional"]),
        strike=float(base_params["strike"]),
        expiry=float(base_params["t"]),
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    lookback_pricer = FxLookbackEuropeanOptionMcPricer(n_paths=100_000, seed=7, antithetic=True, n_steps=64)
    vanilla_pricer = FxVanillaEuropeanOptionMcPricer(n_paths=100_000, seed=7, antithetic=True)

    pv_lookback = float(lookback_pricer.price(lookback_trade, market))
    pv_vanilla = float(vanilla_pricer.price(vanilla_trade, market))

    # Lookback should be more expensive (or equal in degenerate cases)
    assert pv_lookback >= pv_vanilla * 0.99  # Allow small numerical error


def test_fx_lookback_floating_equals_fixed_atm_at_zero_vol(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
) -> None:
    mkt0 = _DummyMarket(
        spot=float(base_params["spot"]),
        rd=float(base_params["rd"]),
        rf=float(base_params["rf"]),
        sigma=0.0,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
    )

    t = float(base_params["t"])
    s0 = float(base_params["spot"])

    floating = FxLookbackEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        expiry=t,
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    fixed = FxLookbackEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        expiry=t,
        lookback_type="fixed_strike",
        strike=s0,  # ATM
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=64, scheme="exact")  # if scheme exists
    pv_float = float(pricer.price(floating, mkt0))
    pv_fixed = float(pricer.price(fixed, mkt0))

    assert pv_float == pytest.approx(pv_fixed, rel=0.0, abs=1e-8)


# =============================================================================
# Lookback: Edge cases
# =============================================================================

def test_fx_lookback_mc_price_at_expiry_is_deterministic(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """
    Test that lookback option at expiry gives deterministic payoff.

    At expiry (T=0), the path contains only S0, so max = min = S0 = S_T.
    """
    trade = FxLookbackEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        expiry=0.0,  # At expiry
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=1)
    pv = float(pricer.price(trade, market))

    # At expiry, for floating strike call: S_T - min(S_t) = S0 - S0 = 0
    expected_pv = 0.0
    assert pv == pytest.approx(expected_pv, abs=1e-10)


def test_fx_lookback_mc_price_at_expiry_fixed_strike_in_the_money(
    ids: Dict[str, MarketId],
    market: _DummyMarket,
) -> None:
    """Test fixed strike lookback at expiry when in-the-money."""
    trade = FxLookbackEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        expiry=0.0,  # At expiry
        lookback_type="fixed_strike",
        strike=1.20,  # Below spot (1.25), so in-the-money
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=1)
    pv = float(pricer.price(trade, market))

    # At expiry, max(S_t) = S0 = 1.25, so payoff = max(1.25 - 1.20, 0) = 0.05
    # PV = notional * df_d(0) * 0.05 = 1_000_000 * 1.0 * 0.05 = 50_000
    expected_pv = 1_000_000.0 * 0.05
    assert pv == pytest.approx(expected_pv, abs=1e-6)


def test_fx_lookback_mc_invalid_n_paths(
    ids: Dict[str, MarketId],
    market: _DummyMarket,
) -> None:
    """Test that invalid n_paths raises ValueError."""
    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=0, seed=7)
    trade = FxLookbackEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        expiry=1.0,
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    with pytest.raises(ValueError, match="n_paths must be positive"):
        pricer.price(trade, market)


def test_fx_lookback_mc_invalid_n_steps(
    ids: Dict[str, MarketId],
    market: _DummyMarket,
) -> None:
    """Test that invalid n_steps raises ValueError."""
    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=10_000, seed=7, n_steps=0)
    trade = FxLookbackEuropeanOption(
        option_type="call",
        notional=1_000_000.0,
        expiry=1.0,
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    with pytest.raises(ValueError, match="n_steps must be positive"):
        pricer.price(trade, market)


# =============================================================================
# Lookback: Simulation artifact tests
# =============================================================================

def test_fx_lookback_mc_simulation_artifact_has_correct_structure(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
) -> None:
    """Test that simulation artifact has correct structure and fields."""
    trade = FxLookbackEuropeanOption(
        option_type="call",
        notional=float(base_params["notional"]),
        expiry=float(base_params["t"]),
        lookback_type="floating_strike",
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pricer = FxLookbackEuropeanOptionMcPricer(n_paths=10_000, seed=7, antithetic=True, n_steps=64)
    sim = pricer.run(trade, market, store_paths=True, paths_keep=100)

    # Verify all required fields exist
    assert hasattr(sim, "spot0")
    assert hasattr(sim, "strike")
    assert hasattr(sim, "maturity")
    assert hasattr(sim, "terminal_spots")
    assert hasattr(sim, "max_spots")
    assert hasattr(sim, "min_spots")
    assert hasattr(sim, "discounted_payoffs")
    assert hasattr(sim, "paths")

    # Verify shapes
    assert sim.terminal_spots.shape == (sim.n_paths_effective,)
    assert sim.max_spots.shape == (sim.n_paths_effective,)
    assert sim.min_spots.shape == (sim.n_paths_effective,)
    assert sim.discounted_payoffs.shape == (sim.n_paths_effective,)
    assert sim.paths is not None
    assert sim.paths.shape[0] == min(100, sim.n_paths_effective)
    assert sim.paths.shape[1] == sim.n_steps + 1

    # Verify that max_spots and min_spots are computed correctly
    for i in range(min(10, sim.n_paths_effective)):  # Check first 10 paths
        path = sim.paths[i, :]
        expected_max = np.max(path)
        expected_min = np.min(path)
        assert sim.max_spots[i] == pytest.approx(expected_max, abs=1e-10)
        assert sim.min_spots[i] == pytest.approx(expected_min, abs=1e-10)

