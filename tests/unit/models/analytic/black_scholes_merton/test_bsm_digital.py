"""
Unit tests for Black-Scholes-Merton digital option formulas.

Tests pricing identities and Greeks against finite differences.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Callable

from src.models.analytic.black_scholes_merton import (
    digital_cash_price,
    digital_cash_greeks,
    digital_asset_price,
    digital_asset_greeks,
)


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
    )


# ======================================================================================
# Price identities
# ======================================================================================

def test_digital_cash_call_put_parity(params: _Params) -> None:
    """
    For continuous distributions (sigma>0, T>0):
      PV_call + PV_put = cash * df
    because P(S>K) + P(S<K) = 1 (and P(S=K)=0).
    """
    call = digital_cash_price(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
        cash=params.cash,
    )
    put = digital_cash_price(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
        cash=params.cash,
    )

    df = math.exp(-params.r * params.t)
    rhs = params.cash * df

    _assert_close(call + put, rhs, rtol=1e-12, atol=1e-12, msg="Cash digital parity failed")


def test_digital_asset_call_put_parity(params: _Params) -> None:
    """
    For continuous distributions (sigma>0, T>0):
      PV_call + PV_put = S * exp((b-r)T)
    because E[1_{S>K} + 1_{S<K}] = 1.
    """
    call = digital_asset_price(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )
    put = digital_asset_price(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )

    fwd_factor = math.exp((params.b - params.r) * params.t)
    rhs = params.spot * fwd_factor

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
    option_type: str,
    spot: float,
    strike: float,
    expected: float,
) -> None:
    pv = digital_cash_price(
        option_type=option_type,  # type: ignore[arg-type]
        spot=spot,
        strike=strike,
        expiry=0.0,
        discount_rate=0.05,
        carry=0.02,
        vol=0.20,
        cash=5.0,
    )
    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Cash digital expiry payoff mismatch")


@pytest.mark.parametrize(
    "option_type,spot,strike,expected",
    [
        ("call", 1.30, 1.20, 1.30),
        ("call", 1.10, 1.20, 0.0),
        ("put",  1.10, 1.20, 1.10),
        ("put",  1.30, 1.20, 0.0),
    ],
)
def test_digital_asset_price_at_expiry(
    option_type: str,
    spot: float,
    strike: float,
    expected: float,
) -> None:
    pv = digital_asset_price(
        option_type=option_type,  # type: ignore[arg-type]
        spot=spot,
        strike=strike,
        expiry=0.0,
        discount_rate=0.05,
        carry=0.02,
        vol=0.20,
    )
    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Asset digital expiry payoff mismatch")


# ======================================================================================
# Finite-difference checks for greeks: cash digital
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_delta_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    eps_rel = 2e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return digital_cash_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_1(f, s0, h)
    ana = digital_cash_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
        cash=params.cash,
    )["delta"]

    _assert_close(ana, fd, rtol=2e-4, atol=5e-6, msg=f"Cash digital delta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_gamma_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    eps_rel = 5e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return digital_cash_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_2(f, s0, h)
    ana = digital_cash_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
        cash=params.cash,
    )["gamma"]

    _assert_close(ana, fd, rtol=5e-3, atol=5e-5, msg=f"Cash digital gamma FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_vega_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 2e-5
    sig0 = params.sigma

    def f(sig: float) -> float:
        return digital_cash_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=sig,
            cash=params.cash,
        )

    fd = _central_diff_1(f, sig0, h)
    ana = digital_cash_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=sig0,
        cash=params.cash,
    )["vega"]

    _assert_close(ana, fd, rtol=2e-4, atol=5e-6, msg=f"Cash digital vega FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_theta_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-5
    t0 = params.t

    def f(t: float) -> float:
        return digital_cash_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
            cash=params.cash,
        )

    # Theta = -dPV/dt, so dPV/dT = -Theta
    fd = -_central_diff_1(f, t0, h)
    ana = digital_cash_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=t0,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
        cash=params.cash,
    )["theta"]

    _assert_close(ana, fd, rtol=5e-4, atol=5e-5, msg=f"Cash digital theta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_rho_discount_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    r0 = params.r

    def f(r: float) -> float:
        return digital_cash_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=r,
            carry=params.b,  # carry held fixed by definition
            vol=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_1(f, r0, h)
    ana = digital_cash_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=r0,
        carry=params.b,
        vol=params.sigma,
        cash=params.cash,
    )["rho_discount"]

    _assert_close(ana, fd, rtol=5e-5, atol=5e-6, msg=f"Cash digital rho_discount FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_cash_rho_carry_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    b0 = params.b

    def f(b: float) -> float:
        return digital_cash_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,  # discount held fixed by definition
            carry=b,
            vol=params.sigma,
            cash=params.cash,
        )

    fd = _central_diff_1(f, b0, h)
    ana = digital_cash_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=b0,
        vol=params.sigma,
        cash=params.cash,
    )["rho_carry"]

    _assert_close(ana, fd, rtol=5e-5, atol=5e-6, msg=f"Cash digital rho_carry FD mismatch ({option_type})")


# ======================================================================================
# Finite-difference checks for greeks: asset digital
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_delta_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    eps_rel = 2e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return digital_asset_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
        )

    fd = _central_diff_1(f, s0, h)
    ana = digital_asset_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["delta"]

    _assert_close(ana, fd, rtol=3e-4, atol=1e-5, msg=f"Asset digital delta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_vega_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 2e-5
    sig0 = params.sigma

    def f(sig: float) -> float:
        return digital_asset_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=sig,
        )

    fd = _central_diff_1(f, sig0, h)
    ana = digital_asset_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=sig0,
    )["vega"]

    _assert_close(ana, fd, rtol=3e-4, atol=1e-5, msg=f"Asset digital vega FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_theta_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-5
    t0 = params.t

    def f(t: float) -> float:
        return digital_asset_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
        )

    # Theta = -dPV/dt, so dPV/dT = -Theta
    fd = -_central_diff_1(f, t0, h)
    ana = digital_asset_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=t0,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["theta"]

    _assert_close(ana, fd, rtol=5e-4, atol=5e-5, msg=f"Asset digital theta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_rho_discount_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    r0 = params.r

    def f(r: float) -> float:
        return digital_asset_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=r,
            carry=params.b,  # carry held fixed by definition
            vol=params.sigma,
        )

    fd = _central_diff_1(f, r0, h)
    ana = digital_asset_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=r0,
        carry=params.b,
        vol=params.sigma,
    )["rho_discount"]

    _assert_close(ana, fd, rtol=1e-4, atol=1e-5, msg=f"Asset digital rho_discount FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_digital_asset_rho_carry_matches_finite_difference(
    params: _Params,
    option_type: str,
) -> None:
    h = 1e-6
    b0 = params.b

    def f(b: float) -> float:
        return digital_asset_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,  # discount held fixed by definition
            carry=b,
            vol=params.sigma,
        )

    fd = _central_diff_1(f, b0, h)
    ana = digital_asset_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=b0,
        vol=params.sigma,
    )["rho_carry"]

    _assert_close(ana, fd, rtol=1e-4, atol=1e-5, msg=f"Asset digital rho_carry FD mismatch ({option_type})")
