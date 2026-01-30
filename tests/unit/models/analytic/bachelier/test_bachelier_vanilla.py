"""
Unit tests for Bachelier (Normal) vanilla option formulas.

Tests pricing identities and Greeks against finite differences.

Note: Bachelier model allows negative forward/strike values, which is
the key difference from BSM/Black76.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
import pytest
from dataclasses import dataclass
from typing import Callable

from src.models.analytic.bachelier import (
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
    """Test parameters for Bachelier model."""
    forward: float      # Forward price F (can be negative)
    strike: float       # Strike price K (can be negative)
    t: float            # Time to expiry T
    r: float            # Discount rate
    sigma: float        # Absolute volatility (in same units as forward)

    @property
    def df(self) -> float:
        """Discount factor."""
        return math.exp(-self.r * self.t)


@pytest.fixture(scope="module")
def params() -> _Params:
    """Stable typical parameter set (positive rates)."""
    return _Params(
        forward=0.05,       # 5% forward rate
        strike=0.04,        # 4% strike
        t=1.0,              # 1 year to expiry
        r=0.03,             # 3% discount rate
        sigma=0.0050,       # 50bp normal volatility
    )


@pytest.fixture(scope="module")
def params_negative() -> _Params:
    """Parameter set with negative forward/strike (negative rate environment)."""
    return _Params(
        forward=-0.005,     # -0.5% forward rate (negative)
        strike=-0.003,      # -0.3% strike (negative)
        t=1.0,              # 1 year to expiry
        r=0.01,             # 1% discount rate
        sigma=0.0040,       # 40bp normal volatility
    )


@pytest.fixture(scope="module")
def params_spread() -> _Params:
    """Parameter set for spread option (larger absolute values)."""
    return _Params(
        forward=2.50,       # Spread of 2.50
        strike=3.00,        # Strike at 3.00
        t=0.5,              # 6 months
        r=0.05,             # 5% discount rate
        sigma=1.20,         # Absolute vol of 1.20
    )


# ======================================================================================
# Tests: price identities
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_price_is_intrinsic_at_expiry(params: _Params, option_type: str) -> None:
    """At T=0, Bachelier price should equal discounted intrinsic."""
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

    _assert_close(pv, expected, rtol=0.0, atol=1e-16, msg="Expiry intrinsic mismatch")


def test_vanilla_put_call_parity(params: _Params) -> None:
    """
    Put-call parity for Bachelier:

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


def test_vanilla_put_call_parity_negative_rates(params_negative: _Params) -> None:
    """Put-call parity should hold even with negative forward and strike."""
    call_pv = vanilla_price(
        option_type="call",
        forward=params_negative.forward,
        strike=params_negative.strike,
        expiry=params_negative.t,
        discount_factor=params_negative.df,
        vol=params_negative.sigma,
    )
    put_pv = vanilla_price(
        option_type="put",
        forward=params_negative.forward,
        strike=params_negative.strike,
        expiry=params_negative.t,
        discount_factor=params_negative.df,
        vol=params_negative.sigma,
    )

    expected = params_negative.df * (params_negative.forward - params_negative.strike)
    actual = call_pv - put_pv

    _assert_close(actual, expected, rtol=1e-12, atol=1e-14, msg="Put-call parity failed (negative rates)")


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
    h = 1e-7  # Small absolute bump for small forward values
    f0 = params.forward

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
    h = 1e-5  # Larger bump for second derivative stability
    f0 = params.forward

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
    h = 1e-7
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

    _assert_close(ana, fd, rtol=5e-6, atol=5e-9, msg=f"Vega FD mismatch ({option_type})")


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

    _assert_close(ana, fd, rtol=5e-5, atol=5e-8, msg=f"Theta FD mismatch ({option_type})")


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

    _assert_close(ana, fd, rtol=2e-6, atol=5e-9, msg=f"Rho FD mismatch ({option_type})")


# ======================================================================================
# Tests: negative forward/strike (key Bachelier feature)
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_delta_matches_finite_difference_negative_rates(
    params_negative: _Params, option_type: str
) -> None:
    """Delta FD test with negative forward and strike."""
    h = 1e-7
    f0 = params_negative.forward

    def f(fwd: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=fwd,
            strike=params_negative.strike,
            expiry=params_negative.t,
            discount_factor=params_negative.df,
            vol=params_negative.sigma,
        )

    fd = _central_diff_1(f, f0, h)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=f0,
        strike=params_negative.strike,
        expiry=params_negative.t,
        discount_factor=params_negative.df,
        discount_rate=params_negative.r,
        vol=params_negative.sigma,
    )["delta"]

    _assert_close(ana, fd, rtol=5e-7, atol=5e-9, msg=f"Delta FD mismatch negative ({option_type})")


def test_vanilla_price_positive_for_negative_rates(params_negative: _Params) -> None:
    """Option prices should still be positive even with negative underlying."""
    call_pv = vanilla_price(
        option_type="call",
        forward=params_negative.forward,
        strike=params_negative.strike,
        expiry=params_negative.t,
        discount_factor=params_negative.df,
        vol=params_negative.sigma,
    )
    put_pv = vanilla_price(
        option_type="put",
        forward=params_negative.forward,
        strike=params_negative.strike,
        expiry=params_negative.t,
        discount_factor=params_negative.df,
        vol=params_negative.sigma,
    )

    # Both call and put should be positive (time value)
    assert call_pv > 0, f"Call price should be positive: {call_pv}"
    assert put_pv > 0, f"Put price should be positive: {put_pv}"


# ======================================================================================
# Tests: spread option scenario
# ======================================================================================

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_vanilla_greeks_spread_option(params_spread: _Params, option_type: str) -> None:
    """Test Greeks for spread option parameters (larger absolute values)."""
    h_fwd = 0.001  # Larger bump for larger values

    def f(fwd: float) -> float:
        return vanilla_price(
            option_type=option_type,  # type: ignore[arg-type]
            forward=fwd,
            strike=params_spread.strike,
            expiry=params_spread.t,
            discount_factor=params_spread.df,
            vol=params_spread.sigma,
        )

    fd = _central_diff_1(f, params_spread.forward, h_fwd)
    ana = vanilla_greeks(
        option_type=option_type,  # type: ignore[arg-type]
        forward=params_spread.forward,
        strike=params_spread.strike,
        expiry=params_spread.t,
        discount_factor=params_spread.df,
        discount_rate=params_spread.r,
        vol=params_spread.sigma,
    )["delta"]

    _assert_close(ana, fd, rtol=5e-6, atol=5e-6, msg=f"Delta FD mismatch spread ({option_type})")


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

    _assert_close(pv, expected, rtol=0.0, atol=1e-16, msg="Zero vol price mismatch")


def test_vanilla_price_atm_call_equals_put(params: _Params) -> None:
    """At-the-money (F=K), call and put should have equal price."""
    call_pv = vanilla_price(
        option_type="call",
        forward=params.forward,
        strike=params.forward,  # ATM: F = K
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )
    put_pv = vanilla_price(
        option_type="put",
        forward=params.forward,
        strike=params.forward,  # ATM: F = K
        expiry=params.t,
        discount_factor=params.df,
        vol=params.sigma,
    )

    _assert_close(call_pv, put_pv, rtol=1e-14, atol=1e-16, msg="ATM call != put")


# ======================================================================================
# Tests: input validation
# ======================================================================================

def test_vanilla_price_invalid_expiry() -> None:
    """Should raise for negative expiry."""
    with pytest.raises(ValueError, match="expiry must be >= 0"):
        vanilla_price(
            option_type="call",
            forward=0.05,
            strike=0.04,
            expiry=-1.0,
            discount_factor=0.95,
            vol=0.005,
        )


def test_vanilla_price_invalid_vol() -> None:
    """Should raise for negative volatility (when zero not allowed)."""
    # This test verifies the validation behavior
    # Note: vanilla_price allows zero vol (allow_zero_vol=True by default)
    with pytest.raises(ValueError, match="vol must be >= 0"):
        vanilla_price(
            option_type="call",
            forward=0.05,
            strike=0.04,
            expiry=1.0,
            discount_factor=0.95,
            vol=-0.001,  # Negative vol should fail
        )


def test_vanilla_price_allows_negative_forward() -> None:
    """Bachelier should allow negative forward (unlike BSM)."""
    # This should NOT raise
    pv = vanilla_price(
        option_type="call",
        forward=-0.01,  # Negative forward
        strike=-0.005,  # Negative strike
        expiry=1.0,
        discount_factor=0.98,
        vol=0.005,
    )
    assert pv > 0, "Price should be positive"
