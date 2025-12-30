from __future__ import annotations

import math
import pytest

from src.models.analytic.black_scholes.european import BlackScholesMertonEuropean


# -----------------------------------------------------------------------------
# Finite-difference helpers (central differences)
# -----------------------------------------------------------------------------

def _fd_first(f, x: float, eps: float) -> float:
    """Central first derivative."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


def _fd_second(f, x: float, eps: float) -> float:
    """Central second derivative."""
    return float((f(x + eps) - 2.0 * f(x) + f(x - eps)) / (eps * eps))


# -----------------------------------------------------------------------------
# Shared fixtures / constants
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine() -> BlackScholesMertonEuropean:
    return BlackScholesMertonEuropean()


@pytest.fixture(scope="module")
def base_params() -> dict[str, float]:
    # Reasonable, non-degenerate defaults (avoid extreme moneyness / tiny vol / tiny T).
    return {
        "spot": 100.0,
        "strike": 100.0,
        "time_to_expiry": 1.0,
        "discount_rate": 0.03,
        "carry": 0.01,
        "sigma": 0.2,
    }


# -----------------------------------------------------------------------------
# Core model identities
# -----------------------------------------------------------------------------

def test_put_call_parity(engine: BlackScholesMertonEuropean, base_params: dict[str, float]) -> None:
    """
    Put-call parity for cost-of-carry parameterisation.

    For this engine:
        C - P = S * exp((b - r)T) - K * exp(-rT)
    """
    s = base_params["spot"]
    k = base_params["strike"]
    t = base_params["time_to_expiry"]
    r = base_params["discount_rate"]
    b = base_params["carry"]
    sig = base_params["sigma"]

    call = engine.price(option_type="call", spot=s, strike=k, time_to_expiry=t, discount_rate=r, carry=b, sigma=sig)
    put = engine.price(option_type="put", spot=s, strike=k, time_to_expiry=t, discount_rate=r, carry=b, sigma=sig)

    lhs = float(call - put)
    rhs = float(s * math.exp((b - r) * t) - k * math.exp(-r * t))

    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-10)


# -----------------------------------------------------------------------------
# Greeks vs finite differences
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greeks_match_finite_differences(
    engine: BlackScholesMertonEuropean,
    base_params: dict[str, float],
    option_type: str,
) -> None:
    """
    Validate engine greeks against central finite differences of engine.price().

    Notes
    -----
    - We choose eps values that are small but not too small (avoid cancellation).
    - This is a *local* test and should remain stable across platforms.
    """
    s0 = float(base_params["spot"])
    k = float(base_params["strike"])
    t = float(base_params["time_to_expiry"])
    r0 = float(base_params["discount_rate"])
    b0 = float(base_params["carry"])
    sig0 = float(base_params["sigma"])

    greeks = engine.greeks(
        option_type=option_type, spot=s0, strike=k, time_to_expiry=t, discount_rate=r0, carry=b0, sigma=sig0
    )

    # --- Delta & Gamma w.r.t. spot ---
    eps_s = 1e-4 * s0  # 1bp of spot
    f_s = lambda s: engine.price(
        option_type=option_type, spot=float(s), strike=k, time_to_expiry=t, discount_rate=r0, carry=b0, sigma=sig0
    )

    delta_fd = _fd_first(f_s, s0, eps_s)
    gamma_fd = _fd_second(f_s, s0, eps_s)

    assert greeks["delta"] == pytest.approx(delta_fd, rel=5e-6, abs=5e-6)
    assert greeks["gamma"] == pytest.approx(gamma_fd, rel=5e-5, abs=5e-6)

    # --- Vega w.r.t. sigma (absolute vol) ---
    eps_v = 1e-5  # 0.1 bp vol in absolute sigma terms
    f_sig = lambda sig: engine.price(
        option_type=option_type, spot=s0, strike=k, time_to_expiry=t, discount_rate=r0, carry=b0, sigma=float(sig)
    )

    vega_fd = _fd_first(f_sig, sig0, eps_v)
    assert greeks["vega"] == pytest.approx(vega_fd, rel=5e-6, abs=5e-6)

    # --- rho_discount w.r.t discount_rate r (holding carry fixed) ---
    eps_r = 1e-6  # 0.01 bp rate in absolute terms
    f_r = lambda r: engine.price(
        option_type=option_type, spot=s0, strike=k, time_to_expiry=t, discount_rate=float(r), carry=b0, sigma=sig0
    )

    rho_disc_fd = _fd_first(f_r, r0, eps_r)
    assert greeks["rho_discount"] == pytest.approx(rho_disc_fd, rel=5e-6, abs=5e-6)

    # --- rho_carry w.r.t carry b (holding discount_rate fixed) ---
    eps_b = 1e-6
    f_b = lambda b: engine.price(
        option_type=option_type, spot=s0, strike=k, time_to_expiry=t, discount_rate=r0, carry=float(b), sigma=sig0
    )

    rho_carry_fd = _fd_first(f_b, b0, eps_b)
    assert greeks["rho_carry"] == pytest.approx(rho_carry_fd, rel=5e-6, abs=5e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_greek_sanity_signs(engine: BlackScholesMertonEuropean, base_params: dict[str, float], option_type: str) -> None:
    """
    Basic sanity checks: gamma and vega should be positive for standard European options.
    (Delta sign differs for call/put and depends on carry, so we don't assert it here.)
    """
    greeks = engine.greeks(option_type=option_type, **base_params)

    assert greeks["gamma"] > 0.0
    assert greeks["vega"] > 0.0


# -----------------------------------------------------------------------------
# Expiry edge-cases
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "option_type, spot, strike, expected",
    [
        ("call", 105.0, 100.0, 5.0),
        ("call", 95.0, 100.0, 0.0),
        ("put", 95.0, 100.0, 5.0),
        ("put", 105.0, 100.0, 0.0),
    ],
)
def test_expiry_zero_returns_intrinsic_and_zero_greeks(
    engine: BlackScholesMertonEuropean,
    option_type: str,
    spot: float,
    strike: float,
    expected: float,
) -> None:
    price = engine.price(
        option_type=option_type,
        spot=spot,
        strike=strike,
        time_to_expiry=0.0,
        discount_rate=0.03,
        carry=0.01,
        sigma=0.2,  # still required by signature, but not used at T=0
    )
    assert price == pytest.approx(expected)

    greeks = engine.greeks(
        option_type=option_type,
        spot=spot,
        strike=strike,
        time_to_expiry=0.0,
        discount_rate=0.03,
        carry=0.01,
        sigma=0.2,
    )
    assert greeks == {
        "delta": 0.0,
        "gamma": 0.0,
        "vega": 0.0,
        "rho_discount": 0.0,
        "rho_carry": 0.0,
    }


# -----------------------------------------------------------------------------
# Validation tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "spot, strike, t, sigma",
    [
        (0.0, 100.0, 1.0, 0.2),     # spot must be > 0
        (-1.0, 100.0, 1.0, 0.2),    # spot must be > 0
        (100.0, 0.0, 1.0, 0.2),     # strike must be > 0
        (100.0, -1.0, 1.0, 0.2),    # strike must be > 0
        (100.0, 100.0, -1.0, 0.2),  # expiry must be >= 0
        (100.0, 100.0, 1.0, 0.0),   # vol must be > 0
        (100.0, 100.0, 1.0, -0.1),  # vol must be > 0
    ],
)
def test_input_validation_raises(engine: BlackScholesMertonEuropean, spot: float, strike: float, t: float, sigma: float) -> None:
    with pytest.raises(ValueError):
        engine.price(
            option_type="call",
            spot=spot,
            strike=strike,
            time_to_expiry=t,
            discount_rate=0.03,
            carry=0.01,
            sigma=sigma,
        )

    with pytest.raises(ValueError):
        engine.greeks(
            option_type="call",
            spot=spot,
            strike=strike,
            time_to_expiry=t,
            discount_rate=0.03,
            carry=0.01,
            sigma=sigma,
        )