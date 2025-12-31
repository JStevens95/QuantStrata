from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import pytest

from src.marketdata.ids import MarketId
from src.models.analytic.black_scholes.european import BlackScholesMertonEuropean
from src.models.common.normal import std_norm_cdf

from src.instruments.fx.options.vanilla import EuropeanFxVanillaOption
from src.instruments.fx.options.digital import EuropeanFxDigitalOption

from src.pricers.fx.european_bsm import (
    FxAmericanVanillaPlaceholderPricer,
    FxEuropeanDigitalBsmPricer,
    FxEuropeanVanillaBsmPricer,
)


# =============================================================================
# Test-only minimal "Market" stub
# =============================================================================

@dataclass(frozen=True)
class _Curve:
    """Test curve with a single flat discount rate (continuous compounding)."""
    rate: float

    def df(self, t: float) -> float:
        # df(t) = exp(-r t)
        return float(math.exp(-float(self.rate) * float(t)))


@dataclass(frozen=True)
class _FlatVolSurface:
    """
    Test vol surface that returns a constant volatility.

    Important
    ---------
    Do NOT name the stored value `vol`, because that would shadow the `.vol(...)`
    method on the instance (leading to "'float' object is not callable").
    """
    sigma: float

    def vol(self, expiry: float, strike: float) -> float:  # noqa: ARG002 (strike unused by design)
        # Flat surface: ignore strike and return constant sigma for any expiry.
        return float(self.sigma)


class _DummyMarket:
    """
    Minimal Market stub implementing the interface required by pricers/fx/european_bsm.py.

    Required by the pricers:
      - quote(market_id) -> float
      - curve(curve_id).df(T) -> float
      - vol_surface(vol_id).vol(expiry, strike) -> float
    """

    def __init__(
        self,
        *,
        spot_id: MarketId,
        vol_id: MarketId,
        rd_id: MarketId,
        rf_id: MarketId,
        spot: float,
        rd: float,
        rf: float,
        sigma: float,
    ) -> None:
        self._spot_id = spot_id
        self._vol_id = vol_id
        self._rd_id = rd_id
        self._rf_id = rf_id

        self._spot = float(spot)
        self._curve_d = _Curve(rate=float(rd))
        self._curve_f = _Curve(rate=float(rf))
        self._vol = _FlatVolSurface(sigma=float(sigma))

    def quote(self, market_id: MarketId) -> float:
        if market_id != self._spot_id:
            raise KeyError(f"Unknown quote id: {market_id}")
        return float(self._spot)

    def curve(self, curve_id: MarketId) -> _Curve:
        if curve_id == self._rd_id:
            return self._curve_d
        if curve_id == self._rf_id:
            return self._curve_f
        raise KeyError(f"Unknown curve id: {curve_id}")

    def vol_surface(self, vol_id: MarketId) -> _FlatVolSurface:
        if vol_id != self._vol_id:
            raise KeyError(f"Unknown vol id: {vol_id}")
        return self._vol


# =============================================================================
# Small numeric helpers (finite differences)
# =============================================================================

def _fd_first(f: Callable[[float], float], x: float, eps: float) -> float:
    """Central finite difference first derivative."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def _fd_second(f: Callable[[float], float], x: float, eps: float) -> float:
    """Central finite difference second derivative."""
    return float((f(x + eps) - 2.0 * f(x) + f(x - eps)) / (eps * eps))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module")
def ids() -> Dict[str, MarketId]:
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    vol_id = MarketId("FX", "VOL", "EURUSD")
    rd_id = MarketId("IR", "CURVE", "USD.OIS")
    rf_id = MarketId("IR", "CURVE", "EUR.OIS")
    return {"spot": spot_id, "vol": vol_id, "rd": rd_id, "rf": rf_id}


@pytest.fixture(scope="module")
def base_params() -> Dict[str, float]:
    return {
        "spot": 100.0,
        "strike": 100.0,
        "t": 1.0,
        "rd": 0.03,
        "rf": 0.01,
        "sigma": 0.20,
        "notional": 1_000_000.0,
    }


@pytest.fixture()
def market(ids: Dict[str, MarketId], base_params: Dict[str, float]) -> _DummyMarket:
    return _DummyMarket(
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        rd_id=ids["rd"],
        rf_id=ids["rf"],
        spot=base_params["spot"],
        rd=base_params["rd"],
        rf=base_params["rf"],
        sigma=base_params["sigma"],
    )


@pytest.fixture(scope="module")
def vanilla_pricer() -> FxEuropeanVanillaBsmPricer:
    return FxEuropeanVanillaBsmPricer()


@pytest.fixture(scope="module")
def digital_pricer() -> FxEuropeanDigitalBsmPricer:
    return FxEuropeanDigitalBsmPricer()


# =============================================================================
# Vanilla pricer tests
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_price_matches_engine_scaled(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    vanilla_pricer: FxEuropeanVanillaBsmPricer,
    option_type: str,
) -> None:
    """
    The adapter should map FX market inputs -> (discount_rate=r_d, carry=r_d-r_f)
    and then scale PV by notional_foreign.
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

    pv = vanilla_pricer.price(trade, market)

    # Build the *expected* PV from the pure engine, using the same mapping.
    rd = float(base_params["rd"])
    rf = float(base_params["rf"])
    carry = rd - rf

    engine = BlackScholesMertonEuropean()
    pv_per_unit = engine.price(
        option_type=option_type,  # type: ignore[arg-type]
        spot=float(base_params["spot"]),
        strike=float(base_params["strike"]),
        time_to_expiry=float(base_params["t"]),
        discount_rate=rd,
        carry=carry,
        sigma=float(base_params["sigma"]),
    )
    expected = float(base_params["notional"]) * float(pv_per_unit)

    assert pv == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_fx_vanilla_put_call_parity(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    market: _DummyMarket,
    vanilla_pricer: FxEuropeanVanillaBsmPricer,
) -> None:
    """
    FX put-call parity under the domestic measure:

        Call - Put = N_f * ( S*df_f - K*df_d )

    where N_f is foreign notional.
    """
    t = float(base_params["t"])
    s = float(base_params["spot"])
    k = float(base_params["strike"])
    notional = float(base_params["notional"])
    rd = float(base_params["rd"])
    rf = float(base_params["rf"])

    df_d = math.exp(-rd * t)
    df_f = math.exp(-rf * t)

    call = EuropeanFxVanillaOption(
        option_type="call",
        notional=notional,
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )
    put = EuropeanFxVanillaOption(
        option_type="put",
        notional=notional,
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv_call = vanilla_pricer.price(call, market)
    pv_put = vanilla_pricer.price(put, market)

    expected = notional * (s * df_f - k * df_d)
    assert (pv_call - pv_put) == pytest.approx(expected, rel=1e-10, abs=1e-8)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_vanilla_greeks_match_finite_differences(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    vanilla_pricer: FxEuropeanVanillaBsmPricer,
    option_type: str,
) -> None:
    """
    Validate pricer greeks against central finite differences of pricer.price().

    This specifically tests the *FX chain-rule mapping*:
      rho_domestic = rho_r + rho_carry
      rho_foreign  = -rho_carry
    by bumping rd and rf through their discount factors.
    """
    s0 = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])
    rd0 = float(base_params["rd"])
    rf0 = float(base_params["rf"])
    sig0 = float(base_params["sigma"])
    notional = float(base_params["notional"])

    trade = EuropeanFxVanillaOption(
        option_type=option_type,  # type: ignore[arg-type]
        notional=notional,
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    # Base greeks
    market0 = _DummyMarket(
        spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot=s0, rd=rd0, rf=rf0, sigma=sig0,
    )
    g = vanilla_pricer.greeks(trade, market0)

    # --- Delta & Gamma w.r.t. spot ---
    eps_s = 1e-4 * s0  # 1bp of spot
    f_s = lambda s: vanilla_pricer.price(
        trade,
        _DummyMarket(
            spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
            spot=float(s), rd=rd0, rf=rf0, sigma=sig0,
        ),
    )

    delta_fd = _fd_first(f_s, s0, eps_s)
    gamma_fd = _fd_second(f_s, s0, eps_s)

    assert g["delta"] == pytest.approx(delta_fd, rel=5e-6, abs=5e-6)
    assert g["gamma"] == pytest.approx(gamma_fd, rel=5e-5, abs=5e-6)

    # --- Vega w.r.t. sigma (absolute vol) ---
    eps_v = 1e-5  # 0.1bp in absolute sigma terms
    f_sig = lambda sig: vanilla_pricer.price(
        trade,
        _DummyMarket(
            spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
            spot=s0, rd=rd0, rf=rf0, sigma=float(sig),
        ),
    )
    vega_fd = _fd_first(f_sig, sig0, eps_v)
    assert g["vega"] == pytest.approx(vega_fd, rel=5e-6, abs=5e-6)

    # --- Rho domestic: bump rd through df_d ---
    eps_r = 1e-6  # 0.01bp absolute rate
    f_rd = lambda rd: vanilla_pricer.price(
        trade,
        _DummyMarket(
            spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
            spot=s0, rd=float(rd), rf=rf0, sigma=sig0,
        ),
    )
    rho_d_fd = _fd_first(f_rd, rd0, eps_r)
    assert g["rho_domestic"] == pytest.approx(rho_d_fd, rel=5e-6, abs=5e-6)

    # --- Rho foreign: bump rf through df_f ---
    f_rf = lambda rf: vanilla_pricer.price(
        trade,
        _DummyMarket(
            spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
            spot=s0, rd=rd0, rf=float(rf), sigma=sig0,
        ),
    )
    rho_f_fd = _fd_first(f_rf, rf0, eps_r)
    assert g["rho_foreign"] == pytest.approx(rho_f_fd, rel=5e-6, abs=5e-6)


# =============================================================================
# Digital pricer tests (pricing only; greeks are intentionally "V1 light")
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_digital_cash_price_matches_closed_form(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    digital_pricer: FxEuropeanDigitalBsmPricer,
    option_type: str,
) -> None:
    """
    Cash-or-nothing digital under FX mapping:

      PV_call = payout * df_d * N(d2)
      PV_put  = payout * df_d * N(-d2)

    where:
      d1 = (ln(S/K) + (b + 0.5*sigma^2)T) / (sigma*sqrtT),
      d2 = d1 - sigma*sqrtT,
      b = r_d - r_f,
      df_d = exp(-r_d T)
    """
    s = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])
    rd = float(base_params["rd"])
    rf = float(base_params["rf"])
    sig = float(base_params["sigma"])

    payout = 1234.5

    market = _DummyMarket(
        spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot=s, rd=rd, rf=rf, sigma=sig,
    )

    trade = EuropeanFxDigitalOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff="cash",
        payout_amount=float(payout),
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv = digital_pricer.price(trade, market)

    b = rd - rf
    df_d = math.exp(-rd * t)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (b + 0.5 * sig * sig) * t) / (sig * sqrt_t)
    d2 = d1 - sig * sqrt_t

    if option_type == "call":
        expected = df_d * payout * std_norm_cdf(d2)
    else:
        expected = df_d * payout * std_norm_cdf(-d2)

    assert pv == pytest.approx(expected, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fx_digital_asset_price_matches_closed_form(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
    digital_pricer: FxEuropeanDigitalBsmPricer,
    option_type: str,
) -> None:
    """
    Asset-or-nothing (pays 'payout_amount' foreign units) under FX mapping:

      PV_call = payout * S * df_f * N(d1)
      PV_put  = payout * S * df_f * N(-d1)
    """
    s = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["t"])
    rd = float(base_params["rd"])
    rf = float(base_params["rf"])
    sig = float(base_params["sigma"])

    payout = 2.5  # foreign units

    market = _DummyMarket(
        spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot=s, rd=rd, rf=rf, sigma=sig,
    )

    trade = EuropeanFxDigitalOption(
        option_type=option_type,  # type: ignore[arg-type]
        payoff="asset",
        payout_amount=float(payout),
        strike=k,
        expiry=t,
        spot_id=ids["spot"],
        vol_id=ids["vol"],
        domestic_curve_id=ids["rd"],
        foreign_curve_id=ids["rf"],
    )

    pv = digital_pricer.price(trade, market)

    b = rd - rf
    df_f = math.exp(-rf * t)

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (b + 0.5 * sig * sig) * t) / (sig * sqrt_t)

    if option_type == "call":
        expected = payout * s * df_f * std_norm_cdf(d1)
    else:
        expected = payout * s * df_f * std_norm_cdf(-d1)

    assert pv == pytest.approx(expected, rel=1e-12, abs=1e-12)


# =============================================================================
# American placeholder tests
# =============================================================================

def test_fx_american_placeholder_raises_not_implemented(
    ids: Dict[str, MarketId],
    base_params: Dict[str, float],
) -> None:
    """
    Placeholder pricer must raise NotImplementedError for price/greeks.
    """
    pricer = FxAmericanVanillaPlaceholderPricer()

    # We only need a dummy market object for the call signature.
    market = _DummyMarket(
        spot_id=ids["spot"], vol_id=ids["vol"], rd_id=ids["rd"], rf_id=ids["rf"],
        spot=float(base_params["spot"]),
        rd=float(base_params["rd"]),
        rf=float(base_params["rf"]),
        sigma=float(base_params["sigma"]),
    )

    # If you haven't implemented AmericanFxVanillaOption yet, skip constructing it here.
    # This test only asserts the placeholder behavior.
    with pytest.raises(NotImplementedError):
        pricer.price(None, market)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError):
        pricer.greeks(None, market)  # type: ignore[arg-type]