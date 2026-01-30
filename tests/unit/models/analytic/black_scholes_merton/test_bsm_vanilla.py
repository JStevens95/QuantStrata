"""
Unit tests for Black-Scholes-Merton vanilla option formulas.

Tests pricing identities and Greeks against finite differences.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Callable

from src.models.analytic.black_scholes_merton import (
    vanilla_price,
    vanilla_greeks,
)


# ======================================================================================
# Numerical differentiation helpers (central differences)
# ======================================================================================

def _central_diff_1(f: Callable[[float], float], x0: float, h: float) -> float:
    """Compute first derivative using central differences."""
    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def _central_diff_2(f: Callable[[float], float], x0: float, h: float) -> float:
    """Compute second derivative using central differences."""
    return (f(x0 + h) - 2.0 * f(x0) + f(x0 - h)) / (h * h)


def _assert_close(a: float, b: float, *, rtol: float, atol: float, msg: str) -> None:
    """Assert two floats are close with a good error message."""
    if not math.isfinite(a) or not math.isfinite(b):
        raise AssertionError(f"{msg} | Non-finite values: a={a}, b={b}")
    if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        raise AssertionError(f"{msg} | a={a:.16g}, b={b:.16g}, |a-b|={abs(a-b):.3e}")


# ======================================================================================
# Shared fixtures
# ======================================================================================

@dataclass(frozen=True, slots=True)
class _Params:
    spot: float
    strike: float
    t: float
    r: float
    b: float
    sigma: float


@pytest.fixture(scope="module")
def params() -> _Params:
    # Use a stable "typical" parameter set.
    return _Params(
        spot=1.25,
        strike=1.20,
        t=1.0,
        r=0.03,
        b=0.02,     # carry (e.g. r-q or r_d-r_f)
        sigma=0.20,
    )


# ======================================================================================
# Tests: price identities
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_price_is_intrinsic_at_expiry(params: _Params, option_type: str) -> None:
    """At T=0, vanilla price should equal intrinsic."""
    pv = vanilla_price(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=0.0,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )

    if option_type == "call":
        expected = max(params.spot - params.strike, 0.0)
    else:
        expected = max(params.strike - params.spot, 0.0)

    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Expiry intrinsic mismatch")


def test_vanilla_put_call_parity(params: _Params) -> None:
    """
    Put-call parity under generic carry:

      C - P = S*exp((b-r)T) - K*exp(-rT)

    This should hold for the model's parameterisation.
    """
    c = vanilla_price(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )
    p = vanilla_price(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )

    df = math.exp(-params.r * params.t)
    fwd_factor = math.exp((params.b - params.r) * params.t)
    rhs = params.spot * fwd_factor - params.strike * df

    _assert_close(c - p, rhs, rtol=1e-12, atol=1e-12, msg="Put-call parity failed")


# ======================================================================================
# Tests: greek sanity identities
# ======================================================================================

def test_vanilla_gamma_is_same_for_call_and_put(params: _Params) -> None:
    """Gamma for vanilla is identical for calls and puts under BSM."""
    g_call = vanilla_greeks(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["gamma"]
    g_put = vanilla_greeks(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["gamma"]

    _assert_close(g_call, g_put, rtol=1e-12, atol=1e-12, msg="Gamma call/put mismatch")


def test_vanilla_vega_is_same_for_call_and_put(params: _Params) -> None:
    """Vega for vanilla is identical for calls and puts under BSM."""
    v_call = vanilla_greeks(
        option_type="call",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["vega"]
    v_put = vanilla_greeks(
        option_type="put",
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["vega"]

    _assert_close(v_call, v_put, rtol=1e-12, atol=1e-12, msg="Vega call/put mismatch")


# ======================================================================================
# Tests: finite-difference validation for greeks (robust, model-agnostic)
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_delta_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Delta should match a central-difference bump on spot."""
    eps_rel = 1e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
        )

    fd = _central_diff_1(f, s0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["delta"]

    _assert_close(ana, fd, rtol=5e-7, atol=5e-9, msg=f"Delta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_gamma_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Gamma should match a second central-difference bump on spot."""
    eps_rel = 5e-5
    s0 = params.spot
    h = s0 * eps_rel

    def f(s: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=s,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=params.sigma,
        )

    fd = _central_diff_2(f, s0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=s0,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["gamma"]

    _assert_close(ana, fd, rtol=2e-5, atol=1e-7, msg=f"Gamma FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_vega_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Vega should match a central-difference bump on sigma (per +1.00 vol)."""
    h = 1e-5
    sig0 = params.sigma

    def f(sig: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,
            carry=params.b,
            vol=sig,
        )

    fd = _central_diff_1(f, sig0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=params.b,
        vol=sig0,
    )["vega"]

    _assert_close(ana, fd, rtol=5e-6, atol=5e-8, msg=f"Vega FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_theta_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Theta should match a central-difference bump on expiry."""
    h = 1e-5
    t0 = params.t

    def f(t: float) -> float:
        return vanilla_price(
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
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=t0,
        discount_rate=params.r,
        carry=params.b,
        vol=params.sigma,
    )["theta"]

    _assert_close(ana, fd, rtol=5e-5, atol=5e-6, msg=f"Theta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_rho_discount_matches_finite_difference(params: _Params, option_type: str) -> None:
    """rho_discount = dPV/d(discount_rate) holding carry fixed."""
    h = 1e-6
    r0 = params.r

    def f(r: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=r,
            carry=params.b,  # carry held fixed by definition
            vol=params.sigma,
        )

    fd = _central_diff_1(f, r0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=r0,
        carry=params.b,
        vol=params.sigma,
    )["rho_discount"]

    _assert_close(ana, fd, rtol=2e-6, atol=5e-7, msg=f"rho_discount FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_rho_carry_matches_finite_difference(params: _Params, option_type: str) -> None:
    """rho_carry = dPV/d(carry) holding discount_rate fixed."""
    h = 1e-6
    b0 = params.b

    def f(b: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            spot=params.spot,
            strike=params.strike,
            expiry=params.t,
            discount_rate=params.r,  # discount held fixed by definition
            carry=b,
            vol=params.sigma,
        )

    fd = _central_diff_1(f, b0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        spot=params.spot,
        strike=params.strike,
        expiry=params.t,
        discount_rate=params.r,
        carry=b0,
        vol=params.sigma,
    )["rho_carry"]

    _assert_close(ana, fd, rtol=2e-6, atol=5e-7, msg=f"rho_carry FD mismatch ({option_type})")
