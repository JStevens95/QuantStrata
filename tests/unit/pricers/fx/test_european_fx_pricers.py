from __future__ import annotations

import math
import pytest
from dataclasses import is_dataclass
from typing import Any, Dict, Optional, Sequence, Tuple


from src.pricers.fx.european_bsm import FxDigitalEuropeanOptionBsmPricer, FxVanillaEuropeanOptionBsmPricer
from src.pricers.fx.european_bsm_fde import FxDigitalEuropeanOptionFdPricer, FxVanillaEuropeanOptionFdPricer

from src.instruments.fx.options.digital import FxDigitalEuropeanOption
from src.instruments.fx.options.vanilla import FxVanillaEuropeanOption


# =============================================================================
# Test doubles: Curve / Vol / Market (duck-typed against your pricers)
# =============================================================================

class _FlatCurve:
    """Flat continuously-compounded rate curve. df(t) = exp(-r t)."""

    def __init__(self, r: float):
        self._r = float(r)

    def df(self, t: float) -> float:
        t = float(t)
        if t <= 0.0:
            return 1.0
        return float(math.exp(-self._r * t))


class _FlatVolSurface:
    """Flat implied vol surface. vol(expiry, strike) = sigma."""

    def __init__(self, sigma: float):
        self._sigma = float(sigma)

    def vol(self, *, expiry: float, strike: float) -> float:
        _ = float(expiry)
        _ = float(strike)
        return float(self._sigma)


class _TestMarket:
    """
    Minimal Market stub to satisfy:
      - market.quote(spot_id)
      - market.curve(curve_id).df(t)
      - market.vol_surface(vol_id).vol(expiry=T, strike=K)
    """

    def __init__(self, *, spot: float, r_d: float, r_f: float, sigma: float):
        self._spot = float(spot)
        self._curves = {
            "DOM": _FlatCurve(r_d),
            "FOR": _FlatCurve(r_f),
        }
        self._vols = {
            "VOL": _FlatVolSurface(sigma),
        }

    def quote(self, quote_id: str) -> float:
        # We ignore quote_id in tests (single-spot world).
        _ = str(quote_id)
        return float(self._spot)

    def curve(self, curve_id: str) -> _FlatCurve:
        return self._curves[str(curve_id)]

    def vol_surface(self, vol_id: str) -> _FlatVolSurface:
        return self._vols[str(vol_id)]


# =============================================================================
# Helpers: robust trade construction via dataclass introspection
# =============================================================================

def _set_if_present(dst: Dict[str, Any], fields: set[str], key: str, value: Any) -> None:
    if key in fields:
        dst[key] = value


def _set_first_alias(
    dst: Dict[str, Any],
    fields: set[str],
    aliases: Sequence[str],
    value: Any,
) -> None:
    for k in aliases:
        if k in fields:
            dst[k] = value
            return


def _construct_dataclass(cls: type, kwargs: Dict[str, Any]) -> Any:
    """
    Construct a dataclass instance, failing with a clear error message if required fields are missing.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass; update the test helper to match your instrument type.")

    fields = cls.__dataclass_fields__  # type: ignore[attr-defined]
    missing = [
        name
        for name, f in fields.items()
        if f.default is f.default_factory and name not in kwargs  # type: ignore[comparison-overlap]
        and f.default_factory is None  # type: ignore[attr-defined]
        and f.default is None
    ]
    # The above "missing" logic is conservative. We still try to instantiate and let Python tell us exact args.
    try:
        return cls(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed constructing {cls.__name__}.\n"
            f"Provided kwargs keys: {sorted(kwargs.keys())}\n"
            f"Dataclass fields: {sorted(fields.keys())}\n"
            f"Original error: {e}"
        ) from e


def make_vanilla_trade(
    *,
    option_type: str,
    strike: float,
    expiry: float,
    notional: float,
    spot_id: str = "SPOT",
    dom_curve_id: str = "DOM",
    for_curve_id: str = "FOR",
    vol_id: str = "VOL",
) -> FxVanillaEuropeanOption:
    """
    Build a FxVanillaEuropeanOption even if field names differ slightly across refactors.
    """
    cls = FxVanillaEuropeanOption
    fields = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    kw: Dict[str, Any] = {}

    _set_first_alias(kw, fields, ["option_type", "type"], option_type)
    _set_first_alias(kw, fields, ["strike", "K"], float(strike))
    _set_first_alias(kw, fields, ["expiry", "time_to_expiry", "T"], float(expiry))
    _set_first_alias(kw, fields, ["notional", "notional_foreign", "notional_fcy"], float(notional))

    _set_first_alias(kw, fields, ["spot_id", "spot_quote_id", "underlying_id"], str(spot_id))
    _set_first_alias(kw, fields, ["domestic_curve_id", "curve_domestic_id", "disc_curve_id"], str(dom_curve_id))
    _set_first_alias(kw, fields, ["foreign_curve_id", "curve_foreign_id", "carry_curve_id"], str(for_curve_id))
    _set_first_alias(kw, fields, ["vol_id", "vol_surface_id"], str(vol_id))

    return _construct_dataclass(cls, kw)


def make_digital_trade(
    *,
    option_type: str,
    payoff: str,  # "cash" | "asset"
    payout_amount: float,
    strike: float,
    expiry: float,
    spot_id: str = "SPOT",
    dom_curve_id: str = "DOM",
    for_curve_id: str = "FOR",
    vol_id: str = "VOL",
) -> FxDigitalEuropeanOption:
    """
    Build a EuropeanFxDigitalOption even if field names differ slightly across refactors.
    """
    cls = FxDigitalEuropeanOption
    fields = set(cls.__dataclass_fields__.keys())  # type: ignore[attr-defined]
    kw: Dict[str, Any] = {}

    _set_first_alias(kw, fields, ["option_type", "type"], option_type)
    _set_first_alias(kw, fields, ["payoff", "payoff_type"], payoff)
    _set_first_alias(kw, fields, ["payout_amount", "payout", "cash", "asset_units"], float(payout_amount))
    _set_first_alias(kw, fields, ["strike", "K"], float(strike))
    _set_first_alias(kw, fields, ["expiry", "time_to_expiry", "T"], float(expiry))

    _set_first_alias(kw, fields, ["spot_id", "spot_quote_id", "underlying_id"], str(spot_id))
    _set_first_alias(kw, fields, ["domestic_curve_id", "curve_domestic_id", "disc_curve_id"], str(dom_curve_id))
    _set_first_alias(kw, fields, ["foreign_curve_id", "curve_foreign_id", "carry_curve_id"], str(for_curve_id))
    _set_first_alias(kw, fields, ["vol_id", "vol_surface_id"], str(vol_id))

    return _construct_dataclass(cls, kw)


# =============================================================================
# Optional MC parity (skip cleanly if MC pricer not present)
# =============================================================================

def _try_import_mc() -> Optional[Tuple[Any, Any]]:
    try:
        from src.pricers.fx.european_bsm_mc import FxVanillaEuropeanOptionMcPricer, FxDigitalEuropeanOptionMcPricer  # type: ignore
        return FxVanillaEuropeanOptionMcPricer, FxDigitalEuropeanOptionMcPricer
    except Exception:
        return None


# =============================================================================
# Parity tests: PV
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K_mult", [0.9, 1.0, 1.1])
@pytest.mark.parametrize("T", [0.25, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [0.10, 0.20])
def test_vanilla_pv_bsm_vs_fd_parity(option_type: str, K_mult: float, T: float, sigma: float) -> None:
    S0 = 1.25
    K = S0 * K_mult
    r_d = 0.03
    r_f = 0.01
    notional = 1_000_000.0

    market = _TestMarket(spot=S0, r_d=r_d, r_f=r_f, sigma=sigma)
    trade = make_vanilla_trade(option_type=option_type, strike=K, expiry=T, notional=notional)

    bsm = FxVanillaEuropeanOptionBsmPricer()
    fd = FxVanillaEuropeanOptionFdPricer(
        n_space=501,
        n_time_steps=250,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
    )

    pv_bsm = bsm.price(trade, market)  # type: ignore[arg-type]
    pv_fd = fd.price(trade, market)    # type: ignore[arg-type]

    # Vanilla should match quite tightly, but short-dated / low-vol OTM cases are FD-stiff.
    sig_sqrt_t = float(sigma) * math.sqrt(float(T))
    rtol = 2e-3 if sig_sqrt_t >= 0.08 else 4e-3  # adapt for stiff corner cases
    atol = 1e-6 * notional
    assert math.isfinite(pv_bsm) and math.isfinite(pv_fd)
    assert pv_fd == pytest.approx(pv_bsm, rel=rtol, abs=atol)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("payoff", ["cash", "asset"])
@pytest.mark.parametrize("K_mult", [0.95, 1.0, 1.05])
@pytest.mark.parametrize("T", [0.25, 1.0])
@pytest.mark.parametrize("sigma", [0.15, 0.25])
def test_digital_pv_bsm_vs_fd_parity(
    option_type: str,
    payoff: str,
    K_mult: float,
    T: float,
    sigma: float,
) -> None:
    S0 = 1.10
    K = S0 * K_mult
    r_d = 0.02
    r_f = 0.005

    market = _TestMarket(spot=S0, r_d=r_d, r_f=r_f, sigma=sigma)

    # Cash payout is domestic; asset payout is "foreign units" (consistent with your digital instrument definition).
    payout_amount = 10_000.0 if payoff == "cash" else 5_000.0

    trade = make_digital_trade(
        option_type=option_type,
        payoff=payoff,
        payout_amount=payout_amount,
        strike=K,
        expiry=T,
    )

    bsm = FxDigitalEuropeanOptionBsmPricer()
    fd = FxDigitalEuropeanOptionFdPricer(
        n_space=901,
        n_time_steps=450,
        n_std=7.0,
        theta=0.5,
        use_log_space=True,
    )

    pv_bsm = bsm.price(trade, market)  # type: ignore[arg-type]
    pv_fd = fd.price(trade, market)    # type: ignore[arg-type]

    # Digitals converge slower (discontinuous payoff) => looser tolerance.
    atol = 1e-6 * max(1.0, abs(pv_bsm))
    rtol = 2.5e-2
    assert math.isfinite(pv_bsm) and math.isfinite(pv_fd)
    assert pv_fd == pytest.approx(pv_bsm, rel=rtol, abs=atol)


# =============================================================================
# Parity tests: Greeks (Vanilla only — digitals unstable unless smoothed)
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K_mult", [0.9, 1.0, 1.1])
@pytest.mark.parametrize("T", [0.5, 1.0])
def test_vanilla_greeks_bsm_vs_fd_parity(option_type: str, K_mult: float, T: float) -> None:
    S0 = 1.30
    K = S0 * K_mult
    r_d = 0.025
    r_f = 0.01
    sigma = 0.18
    notional = 2_000_000.0

    market = _TestMarket(spot=S0, r_d=r_d, r_f=r_f, sigma=sigma)
    trade = make_vanilla_trade(option_type=option_type, strike=K, expiry=T, notional=notional)

    bsm = FxVanillaEuropeanOptionBsmPricer()
    fd = FxVanillaEuropeanOptionFdPricer(
        n_space=601,
        n_time_steps=300,
        n_std=6.0,
        theta=0.5,
        use_log_space=True,
        vol_abs_bump=1e-3,
        rate_abs_bump=1e-4,
    )

    g_bsm = bsm.greeks(trade, market)  # type: ignore[arg-type]
    g_fd = fd.greeks(trade, market)    # type: ignore[arg-type]

    # Note: FD rhos/vega are bump-reprice on a PDE -> a bit noisier; use practical tolerances.
    checks = [
        ("delta", 5e-3, 1e-6 * notional),
        ("gamma", 1e-2, 1e-6 * notional),
        ("vega",  2e-2, 1e-6 * notional),
        ("rho_domestic", 2e-2, 1e-4 * notional),
        ("rho_foreign",  2e-2, 1e-4 * notional),
    ]

    for k, rtol, atol in checks:
        assert math.isfinite(float(g_bsm[k])) and math.isfinite(float(g_fd[k]))
        assert float(g_fd[k]) == pytest.approx(float(g_bsm[k]), rel=rtol, abs=atol)


# =============================================================================
# Optional parity vs MC (if present)
# =============================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("K_mult", [0.9, 1.0, 1.1])
@pytest.mark.parametrize("T", [0.5, 1.0])
def test_vanilla_pv_bsm_vs_mc_parity_if_available(option_type: str, K_mult: float, T: float) -> None:
    mc_mod = _try_import_mc()
    if mc_mod is None:
        pytest.skip("MC pricer not found at src.pricers.fx.european_mc (skipping).")

    FxEuropeanVanillaMcPricer, _FxEuropeanDigitalMcPricer = mc_mod

    S0 = 1.20
    K = S0 * K_mult
    r_d = 0.02
    r_f = 0.01
    sigma = 0.20
    notional = 1_000_000.0

    market = _TestMarket(spot=S0, r_d=r_d, r_f=r_f, sigma=sigma)
    trade = make_vanilla_trade(option_type=option_type, strike=K, expiry=T, notional=notional)

    bsm = FxVanillaEuropeanOptionBsmPricer()
    mc = FxEuropeanVanillaMcPricer(
        n_paths=250_000,
        seed=123,
        antithetic=True,
    )

    pv_bsm = bsm.price(trade, market)  # type: ignore[arg-type]
    pv_mc = mc.price(trade, market)    # type: ignore[arg-type]

    # MC is stochastic => use looser tolerances (or tighten by upping n_paths).
    assert math.isfinite(pv_bsm) and math.isfinite(pv_mc)
    assert pv_mc == pytest.approx(pv_bsm, rel=1.5e-2, abs=5e-4 * notional)