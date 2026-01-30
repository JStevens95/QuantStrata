# src/models/analytic/black76/base.py
"""
Black76 Model - Pure Functions.

Forward-based pricing model for options on futures, forwards, and forward rates.
Uses the forward price F directly (no cost-of-carry parameter).

Mathematical Framework
----------------------
The Black76 model assumes the forward price follows log-normal dynamics:

    dF = σ F dW

Closed-form solutions:
    Call: C = DF × [F N(d₁) - K N(d₂)]
    Put:  P = DF × [K N(-d₂) - F N(-d₁)]

Where:
    d₁ = [ln(F/K) + σ²T/2] / (σ√T)
    d₂ = d₁ - σ√T
    DF = exp(-rT) = discount factor

Use Cases
---------
- Options on commodity futures (oil, gold, etc.)
- Interest rate caps/floors (caplets/floorlets)
- Swaptions
- Options on FX forwards
- Options on equity index futures

Comparison with BSM
-------------------
- BSM: Uses spot S with cost-of-carry b → F = S × exp(bT)
- Black76: Uses forward F directly → simpler when forward is observable

Greek Conventions (Generic)
---------------------------
- delta: dPV/dF (forward sensitivity)
- gamma: d²PV/dF² (convexity)
- vega: dPV/dσ (per 1.0 absolute vol)
- theta: dPV/dT (per 1 year)
- rho: dPV/dr (discount rate sensitivity)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from typing import Literal, Dict

from src.models.common.normal import std_norm_cdf, std_norm_pdf

# Type alias for option type.
OptionType = Literal["call", "put"]

# Type alias for Greek names.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


# =============================================================================
# VALIDATION
# =============================================================================


def validate_inputs(
    *,
    forward: float,
    strike: float,
    expiry: float,
    vol: float,
    allow_zero_vol: bool = False,
) -> None:
    """
    Validate core Black76 inputs.

    Parameters
    ----------
    forward : float
        Forward price F (must be > 0).
    strike : float
        Strike price K (must be > 0).
    expiry : float
        Time to expiry T in years (must be >= 0).
    vol : float
        Volatility σ (must be > 0, or >= 0 if allow_zero_vol=True).
    allow_zero_vol : bool
        If True, allow vol=0 (for degenerate cases).

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    if float(forward) <= 0.0:
        raise ValueError(f"forward must be > 0; got {forward}.")
    if float(strike) <= 0.0:
        raise ValueError(f"strike must be > 0; got {strike}.")
    if float(expiry) < 0.0:
        raise ValueError(f"expiry must be >= 0; got {expiry}.")
    if allow_zero_vol:
        if float(vol) < 0.0:
            raise ValueError(f"vol must be >= 0; got {vol}.")
    else:
        if float(vol) <= 0.0:
            raise ValueError(f"vol must be > 0; got {vol}.")


# =============================================================================
# CORE HELPERS
# =============================================================================


def d1_d2(
    *,
    forward: float,
    strike: float,
    expiry: float,
    vol: float,
) -> tuple[float, float]:
    """
    Compute d₁ and d₂ for Black76.

    d₁ = [ln(F/K) + σ²T/2] / (σ√T)
    d₂ = d₁ - σ√T

    Parameters
    ----------
    forward : float
        Forward price F.
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T (must be > 0).
    vol : float
        Volatility σ (must be > 0).

    Returns
    -------
    tuple[float, float]
        (d₁, d₂)
    """
    t = float(expiry)
    f = float(forward)
    k = float(strike)
    sigma = float(vol)

    sqrt_t = math.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    ln_fk = math.log(f / k)

    d1 = (ln_fk + 0.5 * sigma * sigma * t) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    return float(d1), float(d2)


def intrinsic(*, option_type: OptionType, forward: float, strike: float) -> float:
    """
    Compute intrinsic value (undiscounted).

    Call: max(F - K, 0)
    Put:  max(K - F, 0)

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    forward : float
        Forward price F.
    strike : float
        Strike price K.

    Returns
    -------
    float
        Intrinsic value.
    """
    f = float(forward)
    k = float(strike)
    if option_type == "call":
        return max(f - k, 0.0)
    return max(k - f, 0.0)


# =============================================================================
# VANILLA OPTION FORMULAS
# =============================================================================


def vanilla_price(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    vol: float,
) -> float:
    """
    Black76 vanilla option price.

    Call: C = DF × [F N(d₁) - K N(d₂)]
    Put:  P = DF × [K N(-d₂) - F N(-d₁)]

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    forward : float
        Forward price F.
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T in years.
    discount_factor : float
        Discount factor DF = exp(-rT).
    vol : float
        Volatility σ.

    Returns
    -------
    float
        Option price (per unit notional).

    Examples
    --------
    Option on crude oil futures:
        >>> vanilla_price(option_type="call", forward=75.0, strike=80.0,
        ...               expiry=0.5, discount_factor=0.975, vol=0.30)

    Interest rate caplet:
        >>> vanilla_price(option_type="call", forward=0.05, strike=0.04,
        ...               expiry=1.0, discount_factor=0.95, vol=0.20)
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    f = float(forward)
    k = float(strike)
    t = float(expiry)
    df = float(discount_factor)
    sigma = float(vol)

    # Handle T=0: return discounted intrinsic value.
    if t == 0.0:
        return df * intrinsic(option_type=option_type, forward=f, strike=k)

    # Handle σ=0: forward is deterministic.
    if sigma == 0.0:
        return df * intrinsic(option_type=option_type, forward=f, strike=k)

    # Standard case.
    d1, d2 = d1_d2(forward=f, strike=k, expiry=t, vol=sigma)

    if option_type == "call":
        return df * (f * std_norm_cdf(d1) - k * std_norm_cdf(d2))
    return df * (k * std_norm_cdf(-d2) - f * std_norm_cdf(-d1))


def vanilla_delta(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    vol: float,
) -> float:
    """
    Black76 vanilla delta: dPV/dF.

    Call: Δ = DF × N(d₁)
    Put:  Δ = DF × [N(d₁) - 1] = -DF × N(-d₁)

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Delta (per unit notional).
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    t = float(expiry)
    df = float(discount_factor)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        # At expiry or with zero vol, delta is 0 or 1 depending on moneyness.
        # For smooth Greeks, return 0.
        return 0.0

    d1, _ = d1_d2(forward=forward, strike=strike, expiry=t, vol=sigma)

    if option_type == "call":
        return df * std_norm_cdf(d1)
    return df * (std_norm_cdf(d1) - 1.0)


def vanilla_gamma(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    vol: float,
) -> float:
    """
    Black76 vanilla gamma: d²PV/dF².

    Γ = DF × n(d₁) / (F σ √T)

    Same for call and put.

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Gamma (per unit notional).
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    f = float(forward)
    t = float(expiry)
    df = float(discount_factor)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(forward=f, strike=strike, expiry=t, vol=sigma)
    sqrt_t = math.sqrt(t)

    return df * std_norm_pdf(d1) / (f * sigma * sqrt_t)


def vanilla_vega(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    vol: float,
) -> float:
    """
    Black76 vanilla vega: dPV/dσ (per 1.0 absolute vol).

    ν = DF × F × n(d₁) × √T

    Same for call and put.

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Vega (per unit notional, per 1.0 vol).
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    f = float(forward)
    t = float(expiry)
    df = float(discount_factor)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(forward=f, strike=strike, expiry=t, vol=sigma)
    sqrt_t = math.sqrt(t)

    return df * f * std_norm_pdf(d1) * sqrt_t


def vanilla_theta(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    discount_rate: float,
    vol: float,
) -> float:
    """
    Black76 vanilla theta: -dPV/dt (time decay per year).

    Note: Requires discount_rate to compute the full theta.

    Parameters
    ----------
    forward, strike, expiry, discount_factor, vol : float
        Standard Black76 parameters.
    discount_rate : float
        Discount rate r (for computing dDF/dt contribution).

    Returns
    -------
    float
        Theta (per unit notional, per year).
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    f = float(forward)
    k = float(strike)
    t = float(expiry)
    df = float(discount_factor)
    r = float(discount_rate)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, d2 = d1_d2(forward=f, strike=k, expiry=t, vol=sigma)
    sqrt_t = math.sqrt(t)
    n_d1 = std_norm_pdf(d1)

    # Diffusion decay term.
    diffusion = -df * f * n_d1 * sigma / (2.0 * sqrt_t)

    # Rate effect (dDF/dt = r × DF).
    pv = vanilla_price(option_type=option_type, forward=f, strike=k, expiry=t,
                       discount_factor=df, vol=sigma)
    rate_effect = r * pv

    return diffusion + rate_effect


def vanilla_rho(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    vol: float,
) -> float:
    """
    Black76 vanilla rho: dPV/dr.

    ρ = -T × PV

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Rho (per unit notional, per 1.0 rate).
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    t = float(expiry)

    if t == 0.0:
        return 0.0

    pv = vanilla_price(option_type=option_type, forward=forward, strike=strike,
                       expiry=t, discount_factor=discount_factor, vol=vol)
    return -t * pv


def vanilla_greeks(
    *,
    option_type: OptionType,
    forward: float,
    strike: float,
    expiry: float,
    discount_factor: float,
    discount_rate: float,
    vol: float,
) -> Dict[GreekName, float]:
    """
    Compute all Black76 vanilla option Greeks in one call.

    Parameters
    ----------
    forward, strike, expiry, discount_factor, vol : float
        Standard Black76 parameters.
    discount_rate : float
        Discount rate r (needed for theta).

    Returns
    -------
    dict
        Keys: "delta", "gamma", "vega", "theta", "rho"
    """
    return {
        "delta": vanilla_delta(option_type=option_type, forward=forward, strike=strike,
                               expiry=expiry, discount_factor=discount_factor, vol=vol),
        "gamma": vanilla_gamma(option_type=option_type, forward=forward, strike=strike,
                               expiry=expiry, discount_factor=discount_factor, vol=vol),
        "vega": vanilla_vega(option_type=option_type, forward=forward, strike=strike,
                             expiry=expiry, discount_factor=discount_factor, vol=vol),
        "theta": vanilla_theta(option_type=option_type, forward=forward, strike=strike,
                               expiry=expiry, discount_factor=discount_factor,
                               discount_rate=discount_rate, vol=vol),
        "rho": vanilla_rho(option_type=option_type, forward=forward, strike=strike,
                           expiry=expiry, discount_factor=discount_factor, vol=vol),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "OptionType",
    "GreekName",
    # Validation
    "validate_inputs",
    # Core helpers
    "d1_d2",
    "intrinsic",
    # Vanilla
    "vanilla_price",
    "vanilla_delta",
    "vanilla_gamma",
    "vanilla_vega",
    "vanilla_theta",
    "vanilla_rho",
    "vanilla_greeks",
]
