"""
Unit tests for Black76 vanilla option formulas.

Tests pricing identities and Greeks against finite differences.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Callable

from src.models.analytic.black76 import (
    vanilla_price,
    vanilla_greeks,
    vanilla_delta,
    vanilla_gamma,
    vanilla_vega,
    vanilla_theta,
    vanilla_rho,
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
    """Test parameters for Black76 model."""
    forward: float      # Forward price F
    strike: float       # Strike price K
    t: float            # Time to expiry T
    r: float            # Discount rate
    sigma: float        # Volatility

    @property
    def df(self) -> float:
        """Discount factor."""
        return math.exp(-self.r * self.t)


@pytest.fixture(scope="module")
def params() -> _Params:
    """Stable typical parameter set for futures option."""
    return _Params(
        forward=75.0,       # Crude oil futures price
        strike=80.0,        # Strike price
        t=0.5,              # 6 months to expiry
        r=0.05,             # 5% discount rate
        sigma=0.30,         # 30% volatility
    )


@pytest.fixture(scope="module")
def params_atm() -> _Params:
    """At-the-money parameter set."""
    return _Params(
        forward=100.0,
        strike=100.0,
        t=1.0,
        r=0.03,
        sigma=0.20,
    )


# ======================================================================================
# Tests: price identities
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_price_is_intrinsic_at_expiry(params: _Params, option_type: str) -> None:
    """At T=0, Black76 price should equal discounted intrinsic."""
    pv = vanilla_price(
        option_type=option_type,  # type: ignore[arg-type]
        forward=params.forward,
        strike=params.strike,
        expiry=0.0,
        discount_factor=1.0,  # No discounting at T=0
        vol=params.sigma,
    )

    if option_type == "call":
        expected = max(params.forward - params.strike, 0.0)
    else:
        expected = max(params.strike - params.forward, 0.0)

    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Expiry intrinsic mismatch")


def test_vanilla_put_call_parity(params: _Params) -> None:
    """
    Put-call parity for Black76:

      C - P = DF × (F - K)
    """
    call_pv = vanilla_price(
        option_type="call",
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )
    put_pv = vanilla_price(
        option_type="put",
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )

    expected = params.df * (params.forward - params.strike)
    actual = call_pv - put_pv

    _assert_close(actual, expected, rtol=1e-12, atol=1e-14, msg="Put-call parity failed")


def test_vanilla_gamma_is_same_for_call_and_put(params: _Params) -> None:
    """Gamma should be identical for call and put at same parameters."""
    gamma_call = vanilla_gamma(
        option_type="call",
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )
    gamma_put = vanilla_gamma(
        option_type="put",
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )

    _assert_close(gamma_call, gamma_put, rtol=1e-14, atol=1e-16, msg="Gamma mismatch call/put")


def test_vanilla_vega_is_same_for_call_and_put(params: _Params) -> None:
    """Vega should be identical for call and put at same parameters."""
    vega_call = vanilla_vega(
        option_type="call",
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )
    vega_put = vanilla_vega(
        option_type="put",
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )

    _assert_close(vega_call, vega_put, rtol=1e-14, atol=1e-16, msg="Vega mismatch call/put")


# ======================================================================================
# Tests: Greeks via finite differences
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_delta_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Delta should match a central-difference bump on forward."""
    eps_rel = 1e-5
    f0 = params.forward
    h = f0 * eps_rel

    def f(fwd: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=fwd,
            strike=params.strike,
            expiry=params.t,
            discount_factor=params.df,
            vol=params.sigma,
        )

    fd = _central_diff_1(f, f0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=f0,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        discount_rate=params.r,
        vol=params.sigma,
    )["delta"]

    _assert_close(ana, fd, rtol=5e-7, atol=5e-9, msg=f"Delta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_gamma_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Gamma should match a second central-difference bump on forward."""
    eps_rel = 5e-5
    f0 = params.forward
    h = f0 * eps_rel

    def f(fwd: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=fwd,
            strike=params.strike,
            expiry=params.t,
            discount_factor=params.df,
            vol=params.sigma,
        )

    fd = _central_diff_2(f, f0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=f0,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        discount_rate=params.r,
        vol=params.sigma,
    )["gamma"]

    _assert_close(ana, fd, rtol=2e-5, atol=1e-7, msg=f"Gamma FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_vega_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Vega should match a central-difference bump on sigma."""
    h = 1e-5
    sig0 = params.sigma

    def f(sig: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=params.forward,
            strike=params.strike,
            expiry=params.t,
            discount_factor=params.df,
            vol=sig,
        )

    fd = _central_diff_1(f, sig0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        discount_rate=params.r,
        vol=sig0,
    )["vega"]

    _assert_close(ana, fd, rtol=5e-6, atol=5e-8, msg=f"Vega FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_theta_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Theta should match a central-difference bump on expiry."""
    h = 1e-5
    t0 = params.t

    def f(t: float) -> float:
        # Update discount factor consistently with expiry
        df_t = math.exp(-params.r * t)
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=params.forward,
            strike=params.strike,
            expiry=t,
            discount_factor=df_t,
            vol=params.sigma,
        )

    # Theta = -dPV/dt, so dPV/dT = -Theta
    fd = -_central_diff_1(f, t0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=params.forward,
        strike=params.strike,
        expiry=t0,
        discount_factor=params.df,
        discount_rate=params.r,
        vol=params.sigma,
    )["theta"]

    _assert_close(ana, fd, rtol=5e-5, atol=5e-6, msg=f"Theta FD mismatch ({option_type})")


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_rho_matches_finite_difference(params: _Params, option_type: str) -> None:
    """Rho = dPV/dr should match finite difference on discount rate."""
    h = 1e-6
    r0 = params.r

    def f(r: float) -> float:
        df_r = math.exp(-r * params.t)
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=params.forward,
            strike=params.strike,
            expiry=params.t,
            discount_factor=df_r,
            vol=params.sigma,
        )

    fd = _central_diff_1(f, r0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        discount_rate=r0,
        vol=params.sigma,
    )["rho"]

    _assert_close(ana, fd, rtol=2e-6, atol=5e-7, msg=f"Rho FD mismatch ({option_type})")


# ======================================================================================
# Tests: edge cases
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_price_zero_vol(params: _Params, option_type: str) -> None:
    """With zero volatility, price equals discounted intrinsic."""
    pv = vanilla_price(
        option_type=option_type,  # type: ignore[arg-type]
        forward=params.forward,
        strike=params.strike,
        expiry=params.t,
        discount_factor=params.df,
        vol=0.0,
    )

    if option_type == "call":
        expected = params.df * max(params.forward - params.strike, 0.0)
    else:
        expected = params.df * max(params.strike - params.forward, 0.0)

    _assert_close(pv, expected, rtol=0.0, atol=0.0, msg="Zero vol price mismatch")


def test_vanilla_price_deep_itm_call(params_atm: _Params) -> None:
    """Deep ITM call should be close to discounted intrinsic."""
    f = 150.0  # Deep ITM
    k = 100.0
    pv = vanilla_price(
        option_type="call",
        forward=f,
        strike=k,
        expiry=params_atm.t,
        discount_factor=params_atm.df,
        vol=params_atm.sigma,
    )

    # Should be at least intrinsic (no arbitrage)
    intrinsic = params_atm.df * (f - k)
    assert pv >= intrinsic - 1e-10, f"Deep ITM call below intrinsic: {pv} < {intrinsic}"


def test_vanilla_price_deep_otm_put(params_atm: _Params) -> None:
    """Deep OTM put should be close to zero."""
    f = 150.0  # Deep OTM for put
    k = 100.0
    pv = vanilla_price(
        option_type="put",
        forward=f,
        strike=k,
        expiry=params_atm.t,
        discount_factor=params_atm.df,
        vol=params_atm.sigma,
    )

    # Should be positive but small
    assert pv > 0, f"Put price should be positive: {pv}"
    assert pv < 1.0, f"Deep OTM put should be small: {pv}"


# ======================================================================================
# Tests: input validation
# ======================================================================================

def test_vanilla_price_invalid_forward() -> None:
    """Should raise for non-positive forward."""
    with pytest.raises(ValueError, match="forward must be > 0"):
        vanilla_price(
            option_type="call",
            forward=0.0,
            strike=100.0,
            expiry=1.0,
            discount_factor=0.95,
            vol=0.2,
        )


def test_vanilla_price_invalid_strike() -> None:
    """Should raise for non-positive strike."""
    with pytest.raises(ValueError, match="strike must be > 0"):
        vanilla_price(
            option_type="call",
            forward=100.0,
            strike=-10.0,
            expiry=1.0,
            discount_factor=0.95,
            vol=0.2,
        )


def test_vanilla_price_invalid_expiry() -> None:
    """Should raise for negative expiry."""
    with pytest.raises(ValueError, match="expiry must be >= 0"):
        vanilla_price(
            option_type="call",
            forward=100.0,
            strike=100.0,
            expiry=-1.0,
            discount_factor=0.95,
            vol=0.2,
        )
