from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Callable


from src.models.analytic.black_scholes_merton.digital import BlackScholesMertonDigitalCash, BlackScholesMertonDigitalAsset



# ======================================================================================
# Numerical differentiation helpers
# ======================================================================================

def _central_diff_1(f: Callable[[float], float], x0: float, h: float) -> float:
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def _central_diff_2(f: Callable[[float], float], x0: float, h: float) -> float:
    return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h)


def _assert_close(a: float, b: float, *, rtol: float, atol: float, msg: str) -> None:
    if not math.isfinite(a) or not math.isfinite(b):
        raise AssertionError(f"{msg} | Non-finite values: a={a}, b={b}")
    if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{msg} | a={a:.16g}, b={b:.16g}, |a-b|={abs(a-b):.3e}")


# ======================================================================================
# Fixtures
# ======================================================================================

@dataclass(frozen=True, slots=True)
class _Params:
    spot: float
    strike: float
    t: float
    r: float
    b: float
    sigma: float
    cash: float
    asset_units: float


@pytest.fixture(scope="module")
def cash_engine() -> BlackScholesMertonDigitalCash:
    return BlackScholesMertonDigitalCash()


@pytest.fixture(scope="module")
def asset_engine() -> BlackScholesMertonDigitalAsset:
    return BlackScholesMertonDigitalAsset()


@pytest.fixture(scope="module")
def params() -> _Params:
    return _Params(
        spot=1.25,
        strike=1.20,
        t=1.0,
        r=0.03,
        b=0.02,
        sigma=0.20,
        cash=7.5,
        asset_units=3.0,
    )


# ======================================================================================
# Payoff compatibility (cash digital only — asset engine has explicit asset_units)
# ======================================================================================

@dataclass(frozen=True, slots=True)
class _CashPayoff:
    option_type: str
    strike: float
    cash: float


def test_digital_cash_payoff_object_is_respected(cash_engine: BlackScholesMertonDigitalCash, params: _Params) -> None:
    payoff = _CashPayoff(option_type="call", strike=params.strike, cash=params.cash)

    pv_from_payoff = cash_engine.price(
        option_type="put",  # should be overridden by payoff
        spot=params.spot,
        strike=999.0,       # should be overridden by payoff
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        cash=123.0,         # should be overridden by payoff
        payoff=payoff,
    )

    pv_direct = cash_engine.price(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        cash=params.cash,
    )

    _assert_close(pv_from_payoff, pv_direct, rtol=0.0, atol=0.0, msg="Cash payoff override failed")


# ======================================================================================
# Price identities
# ======================================================================================

def test_digital_cash_call_put_parity(cash_engine: BlackScholesMertonDigitalCash, params: _Params) -> None:
    """
    For continuous distributions (sigma>0, T>0):
      PV_call + PV_put = cash * df
    because P(S>K) + P(S<K) = 1 (and P(S=K)=0).
    """
    call = cash_engine.price(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        cash=params.cash,
    )
    put = cash_engine.price(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        cash=params.cash,
    )

    df = math.exp(-params.r * params.t)
    rhs = params.cash * df

    _assert_close(call + put, rhs, rtol=1e-12, atol=1e-12, msg="Cash digital parity failed")


def test_digital_asset_call_put_parity(asset_engine: BlackScholesMertonDigitalAsset, params: _Params) -> None:
    """
    For continuous distributions (sigma>0, T>0):
      PV_call + PV_put = asset_units * S * exp((b-r)T)
    because E[1_{S>K} + 1_{S<K}] = 1.
    """
    call = asset_engine.price(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        asset_units=params.asset_units,
    )
    put = asset_engine.price(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        asset_units=params.asset_units,
    )

    fwd_factor = math.exp((params.b - params.r) * params.t)
    rhs = params.asset_units * params.spot * fwd_factor

    _assert_close(call + put, rhs, rtol=1e-12, atol=1e-12, msg="Asset digital parity failed")


@pytest.mark.parametrize(
    "option_type,spot,strike,expected",
    [
        ("call", 1.30, 1.20, 5.0),
        ("call", 1.10, 1.20, 0.0),
        ("put",  1.10, 1.20, 5.0),
        ("put",  1.30, 1.20, 0.0),
    ],
)
def test_digital_cash_price_at_expiry(
    cash_engine: BlackScholesMertonDigitalCash,
    option_type: str,
    spot: float,
    strike: float,
    expected: float,
) -> None:
    pv = cash_engine.price(
        option_type=option_type,  # type: ignore[arg-type]
        spot=spot,
        strike=strike,
        time_to_expiry=0.0,
        discount_rate=0.05,
        carry=0.02,
        sigma=0.20,
        cash=5.0,
    )
    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Cash digital expiry payoff mismatch")


@pytest.mark.parametrize(
    "option_type,spot,strike,asset_units,expected",
    [
        ("call", 1.30, 1.20, 2.0, 2.0 * 1.30),
        ("call", 1.10, 1.20, 2.0, 0.0),
        ("put",  1.10, 1.20, 2.0, 2.0 * 1.10),
        ("put",  1.30, 1.20, 2.0, 0.0),
    ],
)
def test_digital_asset_price_at_expiry(
    asset_engine: BlackScholesMertonDigitalAsset,
    option_type: str,
    spot: float,
    strike: float,
    asset_units: float,
    expected: float,
) -> None:
    pv = asset_engine.price(
        option_type=option_type,  # type: ignore[arg-type]
        spot=spot,
        strike=strike,
        time_to_expiry=0.0,
        discount_rate=0.05,
        carry=0.02,
        sigma=0.20,
        asset_units=asset_units,
    )
    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Asset digital expiry payoff mismatch")


# ======================================================================================
# Finite-difference checks for greeks: cash digital
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_delta_matches_finite_difference(
    cash_engine: BlackScholesMertonDigitalCash,
    params: _Params,
    option_type: str,
) -> None:
    eps_rel = 2e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return cash_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            sigma=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_1(f, s0, h)
    ana = cash_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        cash=params.cash,
    )["delta"]

    _assert_close(ana, fd, rtol=2e-4, atol=5e-6, msg=f"Cash digital delta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_gamma_matches_finite_difference(
    cash_engine: BlackScholesMertonDigitalCash,
    params: _Params,
    option_type: str,
) -> None:
    eps_rel = 5e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return cash_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            sigma=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_2(f, s0, h)
    ana = cash_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        cash=params.cash,
    )["gamma"]

    _assert_close(ana, fd, rtol=5e-3, atol=5e-5, msg=f"Cash digital gamma FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_vega_matches_finite_difference(
    cash_engine: BlackScholesMertonDigitalCash,
    params: _Params,
    option_type: str,
) -> None:
    h = 2e-5
    sig0 = params.sigma

    def f(sig: float) -> float:
        return cash_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            sigma=sig,
            cash=params.cash,
        )

    fd = _central_diff_1(f, sig0, h)
    ana = cash_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=sig0,
        cash=params.cash,
    )["vega"]

    _assert_close(ana, fd, rtol=2e-4, atol=5e-6, msg=f"Cash digital vega FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_rho_discount_matches_finite_difference(
    cash_engine: BlackScholesMertonDigitalCash,
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    r0 = params.r

    def f(r: float) -> float:
        return cash_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=r,
            carry=params.b,  # carry held fixed by definition
            sigma=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_1(f, r0, h)
    ana = cash_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=r0,
        carry=params.b,
        sigma=params.sigma,
        cash=params.cash,
    )["rho_discount"]

    _assert_close(ana, fd, rtol=5e-5, atol=5e-6, msg=f"Cash digital rho_discount FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_rho_carry_matches_finite_difference(
    cash_engine: BlackScholesMertonDigitalCash,
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    b0 = params.b

    def f(b: float) -> float:
        return cash_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,  # discount held fixed by definition
            carry=b,
            sigma=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_1(f, b0, h)
    ana = cash_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=b0,
        sigma=params.sigma,
        cash=params.cash,
    )["rho_carry"]

    _assert_close(ana, fd, rtol=5e-5, atol=5e-6, msg=f"Cash digital rho_carry FD mismatch ({option_type})")


# ======================================================================================
# Finite-difference checks for greeks: asset digital
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_delta_matches_finite_difference(
    asset_engine: BlackScholesMertonDigitalAsset,
    params: _Params,
    option_type: str,
) -> None:
    eps_rel = 2e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return asset_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            sigma=params.sigma,
            asset_units=params.asset_units,
        )

    fd = _central_diff_1(f, s0, h)
    ana = asset_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=params.sigma,
        asset_units=params.asset_units,
    )["delta"]

    _assert_close(ana, fd, rtol=3e-4, atol=1e-5, msg=f"Asset digital delta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_vega_matches_finite_difference(
    asset_engine: BlackScholesMertonDigitalAsset,
    params: _Params,
    option_type: str,
) -> None:
    h = 2e-5
    sig0 = params.sigma

    def f(sig: float) -> float:
        return asset_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            sigma=sig,
            asset_units=params.asset_units,
        )

    fd = _central_diff_1(f, sig0, h)
    ana = asset_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        sigma=sig0,
        asset_units=params.asset_units,
    )["vega"]

    _assert_close(ana, fd, rtol=3e-4, atol=1e-5, msg=f"Asset digital vega FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_rho_discount_matches_finite_difference(
    asset_engine: BlackScholesMertonDigitalAsset,
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    r0 = params.r

    def f(r: float) -> float:
        return asset_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=r,
            carry=params.b,  # carry held fixed by definition
            sigma=params.sigma,
            asset_units=params.asset_units,
        )

    fd = _central_diff_1(f, r0, h)
    ana = asset_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=r0,
        carry=params.b,
        sigma=params.sigma,
        asset_units=params.asset_units,
    )["rho_discount"]

    _assert_close(ana, fd, rtol=1e-4, atol=1e-5, msg=f"Asset digital rho_discount FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_rho_carry_matches_finite_difference(
    asset_engine: BlackScholesMertonDigitalAsset,
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    b0 = params.b

    def f(b: float) -> float:
        return asset_engine.price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            time_to_expiry=params.t,
            discount_rate=params.r,  # discount held fixed by definition
            carry=b,
            sigma=params.sigma,
            asset_units=params.asset_units,
        )

    fd = _central_diff_1(f, b0, h)
    ana = asset_engine.greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        time_to_expiry=params.t,
        discount_rate=params.r,
        carry=b0,
        sigma=params.sigma,
        asset_units=params.asset_units,
    )["rho_carry"]

    _assert_close(ana, fd, rtol=1e-4, atol=1e-5, msg=f"Asset digital rho_carry FD mismatch ({option_type})")