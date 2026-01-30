# src/models/analytic/bachelier/base.py
"""
Bachelier (Normal) Model - Pure Functions.

Normal distribution model for options where the underlying can be negative
or where normal dynamics are more appropriate than log-normal.

Mathematical Framework
----------------------
The Bachelier model assumes the forward price follows normal (arithmetic) dynamics:

    dF = σ dW

Where σ is the absolute volatility (not percentage).

Closed-form solutions:
    Call: C = DF × [(F - K) N(d) + σ√T n(d)]
    Put:  P = DF × [(K - F) N(-d) + σ√T n(d)]

Where:
    d = (F - K) / (σ√T)
    DF = exp(-rT) = discount factor
    N(·) = standard normal CDF
    n(·) = standard normal PDF

Use Cases
---------
- Interest rate derivatives in negative rate environments
- Spread options (difference between two underlyings)
- Basis trades
- Options on swaps in certain markets
- Any case where the underlying can go negative

Key Differences from Black-Scholes/Black76
------------------------------------------
- Black-Scholes/Black76: Log-normal dynamics, percentage volatility
- Bachelier: Normal dynamics, absolute volatility (in same units as underlying)

Volatility Conventions
----------------------
- σ is quoted in the same units as the underlying (absolute volatility)
- For interest rates, often quoted in basis points (bp)
- Example: σ = 0.0050 for 50bp normal vol on a 5% rate

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
    Validate core Bachelier inputs.

    Note: forward and strike can be negative (unlike BSM/Black76).

    Parameters
    ----------
    forward : float
        Forward price F (can be negative).
    strike : float
        Strike price K (can be negative).
    expiry : float
        Time to expiry T in years (must be >= 0).
    vol : float
        Absolute volatility σ (must be > 0, or >= 0 if allow_zero_vol=True).
    allow_zero_vol : bool
        If True, allow vol=0 (for degenerate cases).

    Raises
    ------
    ValueError
        If any input is invalid.
    """
    # Note: forward and strike can be any real number (negative rates allowed).
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


def d_moneyness(
    *,
    forward: float,
    strike: float,
    expiry: float,
    vol: float,
) -> float:
    """
    Compute standardized moneyness d for Bachelier model.

    d = (F - K) / (σ√T)

    Parameters
    ----------
    forward : float
        Forward price F.
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T (must be > 0).
    vol : float
        Absolute volatility σ (must be > 0).

    Returns
    -------
    float
        Standardized moneyness d.
    """
    t = float(expiry)
    f = float(forward)
    k = float(strike)
    sigma = float(vol)

    sqrt_t = math.sqrt(t)
    return (f - k) / (sigma * sqrt_t)


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
    Bachelier (Normal) vanilla option price.

    Call: C = DF × [(F - K) N(d) + σ√T n(d)]
    Put:  P = DF × [(K - F) N(-d) + σ√T n(d)]

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    forward : float
        Forward price F (can be negative).
    strike : float
        Strike price K (can be negative).
    expiry : float
        Time to expiry T in years.
    discount_factor : float
        Discount factor DF = exp(-rT).
    vol : float
        Absolute volatility σ (same units as underlying).

    Returns
    -------
    float
        Option price (per unit notional).

    Examples
    --------
    Swaption with negative rates:
        >>> vanilla_price(option_type="call", forward=-0.005, strike=-0.003,
        ...               expiry=1.0, discount_factor=0.98, vol=0.005)  # 50bp normal vol

    Spread option:
        >>> vanilla_price(option_type="call", forward=2.5, strike=3.0,
        ...               expiry=0.5, discount_factor=0.975, vol=1.0)  # 1.0 absolute vol
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
    sqrt_t = math.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    d = (f - k) / sigma_sqrt_t

    if option_type == "call":
        # C = DF × [(F - K) N(d) + σ√T n(d)]
        return df * ((f - k) * std_norm_cdf(d) + sigma_sqrt_t * std_norm_pdf(d))
    # P = DF × [(K - F) N(-d) + σ√T n(d)]
    return df * ((k - f) * std_norm_cdf(-d) + sigma_sqrt_t * std_norm_pdf(d))


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
    Bachelier vanilla delta: dPV/dF.

    Call: Δ = DF × N(d)
    Put:  Δ = DF × [N(d) - 1] = -DF × N(-d)

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
        return 0.0

    sqrt_t = math.sqrt(t)
    d = (float(forward) - float(strike)) / (sigma * sqrt_t)

    if option_type == "call":
        return df * std_norm_cdf(d)
    return df * (std_norm_cdf(d) - 1.0)


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
    Bachelier vanilla gamma: d²PV/dF².

    Γ = DF × n(d) / (σ√T)

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

    t = float(expiry)
    df = float(discount_factor)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    sqrt_t = math.sqrt(t)
    d = (float(forward) - float(strike)) / (sigma * sqrt_t)

    return df * std_norm_pdf(d) / (sigma * sqrt_t)


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
    Bachelier vanilla vega: dPV/dσ (per 1.0 absolute vol).

    ν = DF × √T × n(d)

    Same for call and put.

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Vega (per unit notional, per 1.0 absolute vol).
    """
    validate_inputs(forward=forward, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    t = float(expiry)
    df = float(discount_factor)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    sqrt_t = math.sqrt(t)
    d = (float(forward) - float(strike)) / (sigma * sqrt_t)

    return df * sqrt_t * std_norm_pdf(d)


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
    Bachelier vanilla theta: -dPV/dt (time decay per year).

    Note: Requires discount_rate to compute the full theta.

    Parameters
    ----------
    forward, strike, expiry, discount_factor, vol : float
        Standard Bachelier parameters.
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

    sqrt_t = math.sqrt(t)
    d = (f - k) / (sigma * sqrt_t)
    n_d = std_norm_pdf(d)

    # Diffusion decay term: -DF × σ/(2√T) × n(d)
    diffusion = -df * sigma * n_d / (2.0 * sqrt_t)

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
    Bachelier vanilla rho: dPV/dr.

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
    Compute all Bachelier vanilla option Greeks in one call.

    Parameters
    ----------
    forward, strike, expiry, discount_factor, vol : float
        Standard Bachelier parameters.
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
    "d_moneyness",
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
