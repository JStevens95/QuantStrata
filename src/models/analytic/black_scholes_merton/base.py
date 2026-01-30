# src/models/analytic/black_scholes_merton/base.py
"""
Black-Scholes-Merton Model - Pure Functions.

Generalized BSM with cost-of-carry parameter that handles all asset classes:
- Equity (no dividends): b = r
- Equity (continuous dividends): b = r - q
- FX (Garman-Kohlhagen): b = r_d - r_f, discount_rate = r_d
- Commodities: b = r - convenience_yield + storage_cost

Note: For futures/forwards use Black76 model (separate module).

Mathematical Framework
----------------------
Under risk-neutral measure with cost-of-carry b:

    dS = b S dt + σ S dW

Closed-form solutions:
    Call: C = S exp((b-r)T) N(d₁) - K exp(-rT) N(d₂)
    Put:  P = K exp(-rT) N(-d₂) - S exp((b-r)T) N(-d₁)

Where:
    d₁ = [ln(S/K) + (b + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T

Greek Conventions (Generic)
---------------------------
- delta: dPV/dS (spot sensitivity)
- gamma: d²PV/dS² (convexity)
- vega: dPV/dσ (per 1.0 absolute vol)
- theta: dPV/dT (per 1 year, negative for long positions)
- rho_discount: dPV/dr (discount rate sensitivity, holding carry fixed)
- rho_carry: dPV/db (carry rate sensitivity, holding discount rate fixed)

Asset-Class Mapping (handled by pricers, not here):
- FX: rho_domestic = rho_discount + rho_carry, rho_foreign = -rho_carry
- Equity: rho = rho_discount + rho_carry

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from typing import Literal, Dict

from src.models.common.normal import std_norm_cdf, std_norm_pdf

# Type alias for option type.
OptionType = Literal["call", "put"]

# Type alias for Greek names.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho_discount", "rho_carry"]


# =============================================================================
# VALIDATION
# =============================================================================


def validate_inputs(
    *,
    spot: float,
    strike: float,
    expiry: float,
    vol: float,
    allow_zero_vol: bool = False,
) -> None:
    """
    Validate core BSM inputs.

    Parameters
    ----------
    spot : float
        Spot price S (must be > 0).
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
    if float(spot) <= 0.0:
        raise ValueError(f"spot must be > 0; got {spot}.")
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
    spot: float,
    strike: float,
    expiry: float,
    carry: float,
    vol: float,
) -> tuple[float, float]:
    """
    Compute d₁ and d₂ for generalized BSM.

    d₁ = [ln(S/K) + (b + σ²/2)T] / (σ√T)
    d₂ = d₁ - σ√T

    Parameters
    ----------
    spot : float
        Spot price S.
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T (must be > 0).
    carry : float
        Cost-of-carry rate b.
    vol : float
        Volatility σ (must be > 0).

    Returns
    -------
    tuple[float, float]
        (d₁, d₂)
    """
    t = float(expiry)
    s = float(spot)
    k = float(strike)
    b = float(carry)
    sigma = float(vol)

    sqrt_t = math.sqrt(t)
    sigma_sqrt_t = sigma * sqrt_t
    ln_sk = math.log(s / k)
    drift = (b + 0.5 * sigma * sigma) * t

    d1 = (ln_sk + drift) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t

    return float(d1), float(d2)


def forward_factor(
    *,
    carry: float,
    discount_rate: float,
    expiry: float,
) -> float:
    """
    Compute the forward factor exp((b-r)T).

    This factor converts spot to PV-adjusted forward in the BSM formula.

    Parameters
    ----------
    carry : float
        Cost-of-carry rate b.
    discount_rate : float
        Discount rate r.
    expiry : float
        Time to expiry T.

    Returns
    -------
    float
        exp((b-r)T)
    """
    return math.exp((float(carry) - float(discount_rate)) * float(expiry))


def discount_factor(*, rate: float, expiry: float) -> float:
    """
    Compute discount factor exp(-rT).

    Parameters
    ----------
    rate : float
        Discount rate r.
    expiry : float
        Time to expiry T.

    Returns
    -------
    float
        exp(-rT)
    """
    return math.exp(-float(rate) * float(expiry))


def intrinsic(*, option_type: OptionType, spot: float, strike: float) -> float:
    """
    Compute vanilla intrinsic value.

    Call: max(S - K, 0)
    Put:  max(K - S, 0)

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    spot : float
        Spot price S.
    strike : float
        Strike price K.

    Returns
    -------
    float
        Intrinsic value.
    """
    s = float(spot)
    k = float(strike)
    if option_type == "call":
        return max(s - k, 0.0)
    return max(k - s, 0.0)


# =============================================================================
# VANILLA OPTION FORMULAS
# =============================================================================


def vanilla_price(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Generalized BSM vanilla option price.

    Call: C = S exp((b-r)T) N(d₁) - K exp(-rT) N(d₂)
    Put:  P = K exp(-rT) N(-d₂) - S exp((b-r)T) N(-d₁)

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    spot : float
        Spot price S.
    strike : float
        Strike price K.
    expiry : float
        Time to expiry T in years.
    discount_rate : float
        Discount rate r.
    carry : float
        Cost-of-carry rate b.
    vol : float
        Volatility σ.

    Returns
    -------
    float
        Option price (per unit notional).

    Examples
    --------
    Equity with 2% dividend yield:
        >>> vanilla_price(option_type="call", spot=100, strike=100, expiry=1,
        ...               discount_rate=0.05, carry=0.03, vol=0.2)  # b = r - q = 0.05 - 0.02

    FX (EUR/USD, r_d=5%, r_f=3%):
        >>> vanilla_price(option_type="call", spot=1.10, strike=1.10, expiry=1,
        ...               discount_rate=0.05, carry=0.02, vol=0.1)  # b = r_d - r_f
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    k = float(strike)
    t = float(expiry)
    r = float(discount_rate)
    b = float(carry)
    sigma = float(vol)

    # Handle T=0: return intrinsic value.
    if t == 0.0:
        return intrinsic(option_type=option_type, spot=s, strike=k)

    # Handle σ=0: deterministic case.
    if sigma == 0.0:
        fwd = s * math.exp(b * t)
        df = math.exp(-r * t)
        if option_type == "call":
            return df * max(fwd - k, 0.0)
        return df * max(k - fwd, 0.0)

    # Standard case.
    d1, d2 = d1_d2(spot=s, strike=k, expiry=t, carry=b, vol=sigma)
    df = discount_factor(rate=r, expiry=t)
    ff = forward_factor(carry=b, discount_rate=r, expiry=t)

    if option_type == "call":
        return s * ff * std_norm_cdf(d1) - k * df * std_norm_cdf(d2)
    return k * df * std_norm_cdf(-d2) - s * ff * std_norm_cdf(-d1)


def vanilla_delta(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Vanilla option delta: dPV/dS.

    Call: Δ = exp((b-r)T) N(d₁)
    Put:  Δ = exp((b-r)T) [N(d₁) - 1]

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Delta (per unit notional).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    t = float(expiry)
    if t == 0.0:
        return 0.0

    sigma = float(vol)
    if sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=spot, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)

    if option_type == "call":
        return ff * std_norm_cdf(d1)
    return ff * (std_norm_cdf(d1) - 1.0)


def vanilla_gamma(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Vanilla option gamma: d²PV/dS².

    Γ = exp((b-r)T) n(d₁) / (S σ √T)

    Same for call and put.

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Gamma (per unit notional).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)

    return ff * std_norm_pdf(d1) / (s * sigma * sqrt_t)


def vanilla_vega(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Vanilla option vega: dPV/dσ (per 1.0 absolute vol).

    ν = S exp((b-r)T) n(d₁) √T

    Same for call and put.

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Vega (per unit notional, per 1.0 vol).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)

    return s * ff * std_norm_pdf(d1) * sqrt_t


def vanilla_theta(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Vanilla option theta: -dPV/dt (time decay per year).

    Formula with cost-of-carry:
        θ_call = -S exp((b-r)T) n(d₁) σ/(2√T) - (r-b) S exp((b-r)T) N(d₁) - r K exp(-rT) N(d₂)
        θ_put  = -S exp((b-r)T) n(d₁) σ/(2√T) + (r-b) S exp((b-r)T) N(-d₁) + r K exp(-rT) N(-d₂)

    Note: Returns negative value for long positions (value decays as time passes).

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Theta (per unit notional, per year).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    k = float(strike)
    t = float(expiry)
    r = float(discount_rate)
    b = float(carry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, d2 = d1_d2(spot=s, strike=k, expiry=t, carry=b, vol=sigma)
    df = discount_factor(rate=r, expiry=t)
    ff = forward_factor(carry=b, discount_rate=r, expiry=t)
    sqrt_t = math.sqrt(t)
    n_d1 = std_norm_pdf(d1)

    # Common diffusion decay term (always negative).
    diffusion = -s * ff * n_d1 * sigma / (2.0 * sqrt_t)

    if option_type == "call":
        # Carry adjustment + discounting effect.
        # θ_call = diffusion + (r-b)×S×ff×N(d₁) - r×K×df×N(d₂)
        return diffusion + (r - b) * s * ff * std_norm_cdf(d1) - r * k * df * std_norm_cdf(d2)
    # θ_put = diffusion - (r-b)×S×ff×N(-d₁) + r×K×df×N(-d₂)
    return diffusion - (r - b) * s * ff * std_norm_cdf(-d1) + r * k * df * std_norm_cdf(-d2)


def vanilla_rho_discount(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Vanilla option rho w.r.t. discount rate: dPV/dr (holding carry fixed).

    Derivation: C = S×exp((b-r)T)×N(d₁) - K×exp(-rT)×N(d₂)
    
    Since d₁,d₂ don't depend on r (when b is held fixed):
      ∂C/∂r = S×(-T)×ff×N(d₁) - K×(-T)×df×N(d₂)
            = T × [K×df×N(d₂) - S×ff×N(d₁)]

    For put: P = K×df×N(-d₂) - S×ff×N(-d₁)
      ∂P/∂r = T × [S×ff×N(-d₁) - K×df×N(-d₂)]

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Rho w.r.t. discount rate (per unit notional, per 1.0 rate).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    k = float(strike)
    t = float(expiry)
    r = float(discount_rate)
    b = float(carry)
    sigma = float(vol)

    if t == 0.0:
        return 0.0
    if sigma == 0.0:
        # Zero-vol limit: price is discounted intrinsic, dPV/dr = -T × PV
        pv = vanilla_price(option_type=option_type, spot=s, strike=k, expiry=t,
                           discount_rate=r, carry=b, vol=sigma)
        return -t * pv

    d1, d2 = d1_d2(spot=s, strike=k, expiry=t, carry=b, vol=sigma)
    df = discount_factor(rate=r, expiry=t)
    ff = forward_factor(carry=b, discount_rate=r, expiry=t)

    if option_type == "call":
        # ∂C/∂r = T × [K×df×N(d₂) - S×ff×N(d₁)]
        return t * (k * df * std_norm_cdf(d2) - s * ff * std_norm_cdf(d1))
    # ∂P/∂r = T × [S×ff×N(-d₁) - K×df×N(-d₂)]
    return t * (s * ff * std_norm_cdf(-d1) - k * df * std_norm_cdf(-d2))


def vanilla_rho_carry(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Vanilla option rho w.r.t. carry: dPV/db (holding discount rate fixed).

    ρ_b_call = T × S exp((b-r)T) N(d₁)
    ρ_b_put  = -T × S exp((b-r)T) N(-d₁)

    Parameters
    ----------
    (same as vanilla_price)

    Returns
    -------
    float
        Rho w.r.t. carry (per unit notional, per 1.0 rate).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)

    if option_type == "call":
        return t * s * ff * std_norm_cdf(d1)
    return -t * s * ff * std_norm_cdf(-d1)


def vanilla_greeks(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> Dict[GreekName, float]:
    """
    Compute all vanilla option Greeks in one call.

    Returns
    -------
    dict
        Keys: "delta", "gamma", "vega", "theta", "rho_discount", "rho_carry"
    """
    return {
        "delta": vanilla_delta(option_type=option_type, spot=spot, strike=strike,
                               expiry=expiry, discount_rate=discount_rate, carry=carry, vol=vol),
        "gamma": vanilla_gamma(option_type=option_type, spot=spot, strike=strike,
                               expiry=expiry, discount_rate=discount_rate, carry=carry, vol=vol),
        "vega": vanilla_vega(option_type=option_type, spot=spot, strike=strike,
                             expiry=expiry, discount_rate=discount_rate, carry=carry, vol=vol),
        "theta": vanilla_theta(option_type=option_type, spot=spot, strike=strike,
                               expiry=expiry, discount_rate=discount_rate, carry=carry, vol=vol),
        "rho_discount": vanilla_rho_discount(option_type=option_type, spot=spot, strike=strike,
                                             expiry=expiry, discount_rate=discount_rate, carry=carry, vol=vol),
        "rho_carry": vanilla_rho_carry(option_type=option_type, spot=spot, strike=strike,
                                       expiry=expiry, discount_rate=discount_rate, carry=carry, vol=vol),
    }


# =============================================================================
# DIGITAL CASH-OR-NOTHING FORMULAS
# =============================================================================


def digital_cash_price(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
    cash: float,
) -> float:
    """
    Cash-or-nothing digital option price.

    Pays fixed `cash` amount at expiry if ITM:
    - Call: pays cash if S_T > K
    - Put:  pays cash if S_T < K

    Call: C = cash × exp(-rT) × N(d₂)
    Put:  P = cash × exp(-rT) × N(-d₂)

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    spot, strike, expiry, discount_rate, carry, vol : float
        Standard BSM parameters.
    cash : float
        Cash payout if ITM.

    Returns
    -------
    float
        Option price.
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    k = float(strike)
    t = float(expiry)
    r = float(discount_rate)
    b = float(carry)
    sigma = float(vol)
    csh = float(cash)

    # Handle T=0: deterministic payoff.
    if t == 0.0:
        if option_type == "call":
            return csh if s > k else 0.0
        return csh if s < k else 0.0

    df = discount_factor(rate=r, expiry=t)

    # Handle σ=0: deterministic forward.
    if sigma == 0.0:
        fwd = s * math.exp(b * t)
        if option_type == "call":
            return csh * df if fwd > k else 0.0
        return csh * df if fwd < k else 0.0

    _, d2 = d1_d2(spot=s, strike=k, expiry=t, carry=b, vol=sigma)

    if option_type == "call":
        return csh * df * std_norm_cdf(d2)
    return csh * df * std_norm_cdf(-d2)


def digital_cash_delta(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
    cash: float,
) -> float:
    """
    Cash-or-nothing digital delta: dPV/dS.

    Δ_call = cash × exp(-rT) × n(d₂) / (S σ √T)
    Δ_put  = -cash × exp(-rT) × n(d₂) / (S σ √T)
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)
    csh = float(cash)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    _, d2 = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    df = discount_factor(rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)

    dd2_ds = 1.0 / (s * sigma * sqrt_t)
    sign = 1.0 if option_type == "call" else -1.0

    return sign * csh * df * std_norm_pdf(d2) * dd2_ds


def digital_cash_gamma(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
    cash: float,
) -> float:
    """
    Cash-or-nothing digital gamma: d²PV/dS².
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)
    csh = float(cash)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    _, d2 = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    df = discount_factor(rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)

    dd2_ds = 1.0 / (s * sigma * sqrt_t)
    sign = 1.0 if option_type == "call" else -1.0

    # d²/dS² = derivative of (sign * csh * df * n(d2) * dd2/dS)
    # = sign * csh * df * (-d2 * n(d2) * (dd2/dS)² - n(d2) * dd2/dS / S)
    return -sign * csh * df * std_norm_pdf(d2) * (d2 * dd2_ds * dd2_ds + dd2_ds / s)


def digital_cash_vega(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
    cash: float,
) -> float:
    """
    Cash-or-nothing digital vega: dPV/dσ.
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)
    csh = float(cash)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    _, d2 = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    df = discount_factor(rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)

    # dd2/dσ = -(d2/σ + √T)
    dd2_dsigma = -(d2 / sigma + sqrt_t)
    sign = 1.0 if option_type == "call" else -1.0

    return sign * csh * df * std_norm_pdf(d2) * dd2_dsigma


def digital_cash_theta(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
    cash: float,
) -> float:
    """
    Cash-or-nothing digital theta: -dPV/dT.

    θ = r × PV - cash × df × n(d₂) × dd₂/dT

    Where dd₂/dT = (b - σ²/2)/(σ√T) - d₂/(2T)
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    r = float(discount_rate)
    sigma = float(vol)
    csh = float(cash)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    pv = digital_cash_price(option_type=option_type, spot=s, strike=strike,
                            expiry=t, discount_rate=r, carry=carry,
                            vol=sigma, cash=csh)

    _, d2 = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    df = discount_factor(rate=r, expiry=t)
    sqrt_t = math.sqrt(t)

    # dd2/dT for fixed spot
    b = float(carry)
    dd2_dt = (b - 0.5 * sigma * sigma) / (sigma * sqrt_t) - d2 / (2.0 * t)

    sign = 1.0 if option_type == "call" else -1.0

    # θ = r × PV - sign × cash × df × n(d2) × dd2/dT
    return r * pv - sign * csh * df * std_norm_pdf(d2) * dd2_dt


def digital_cash_greeks(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
    cash: float,
) -> Dict[str, float]:
    """
    Compute all digital cash option Greeks.

    Returns
    -------
    dict
        Keys: "delta", "gamma", "vega", "theta", "rho_discount", "rho_carry"
    """
    t = float(expiry)
    pv = digital_cash_price(option_type=option_type, spot=spot, strike=strike,
                            expiry=expiry, discount_rate=discount_rate, carry=carry,
                            vol=vol, cash=cash)

    # rho_discount = -T × PV (since d2 doesn't depend on r when b is fixed)
    rho_discount = -t * pv if t > 0 else 0.0

    # rho_carry: need to compute dd2/db
    sigma = float(vol)
    if t > 0 and sigma > 0:
        _, d2 = d1_d2(spot=spot, strike=strike, expiry=t, carry=carry, vol=sigma)
        df = discount_factor(rate=discount_rate, expiry=t)
        sqrt_t = math.sqrt(t)
        dd2_db = sqrt_t / sigma
        sign = 1.0 if option_type == "call" else -1.0
        rho_carry = sign * float(cash) * df * std_norm_pdf(d2) * dd2_db
    else:
        rho_carry = 0.0

    return {
        "delta": digital_cash_delta(option_type=option_type, spot=spot, strike=strike,
                                    expiry=expiry, discount_rate=discount_rate, carry=carry,
                                    vol=vol, cash=cash),
        "gamma": digital_cash_gamma(option_type=option_type, spot=spot, strike=strike,
                                    expiry=expiry, discount_rate=discount_rate, carry=carry,
                                    vol=vol, cash=cash),
        "vega": digital_cash_vega(option_type=option_type, spot=spot, strike=strike,
                                  expiry=expiry, discount_rate=discount_rate, carry=carry,
                                  vol=vol, cash=cash),
        "theta": digital_cash_theta(option_type=option_type, spot=spot, strike=strike,
                                    expiry=expiry, discount_rate=discount_rate, carry=carry,
                                    vol=vol, cash=cash),
        "rho_discount": rho_discount,
        "rho_carry": rho_carry,
    }


# =============================================================================
# DIGITAL ASSET-OR-NOTHING FORMULAS
# =============================================================================


def digital_asset_price(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Asset-or-nothing digital option price.

    Pays the underlying asset at expiry if ITM:
    - Call: pays S_T if S_T > K
    - Put:  pays S_T if S_T < K

    Call: C = S × exp((b-r)T) × N(d₁)
    Put:  P = S × exp((b-r)T) × N(-d₁)

    Parameters
    ----------
    option_type : OptionType
        "call" or "put".
    spot, strike, expiry, discount_rate, carry, vol : float
        Standard BSM parameters.

    Returns
    -------
    float
        Option price (per unit payout).
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    k = float(strike)
    t = float(expiry)
    r = float(discount_rate)
    b = float(carry)
    sigma = float(vol)

    # Handle T=0.
    if t == 0.0:
        if option_type == "call":
            return s if s > k else 0.0
        return s if s < k else 0.0

    ff = forward_factor(carry=b, discount_rate=r, expiry=t)

    # Handle σ=0.
    if sigma == 0.0:
        fwd = s * math.exp(b * t)
        if option_type == "call":
            return s * ff if fwd > k else 0.0
        return s * ff if fwd < k else 0.0

    d1, _ = d1_d2(spot=s, strike=k, expiry=t, carry=b, vol=sigma)

    if option_type == "call":
        return s * ff * std_norm_cdf(d1)
    return s * ff * std_norm_cdf(-d1)


def digital_asset_delta(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Asset-or-nothing digital delta: dPV/dS.
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)
    vol_sqrt_t = sigma * sqrt_t

    dd1_ds = 1.0 / (s * vol_sqrt_t)
    sign = 1.0 if option_type == "call" else -1.0
    N_d1_signed = std_norm_cdf(sign * d1)

    # d/dS [S × ff × N(±d1)] = ff × [N(±d1) + S × n(d1) × sign × dd1/dS]
    return ff * (N_d1_signed + s * std_norm_pdf(d1) * sign * dd1_ds)


def digital_asset_gamma(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Asset-or-nothing digital gamma: d²PV/dS².

    PV = S × ff × N(±d₁)
    Delta = ff × [N(±d₁) + n(d₁) × sign / (σ√T)]
    Gamma = d(Delta)/dS
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)
    vol_sqrt_t = sigma * sqrt_t

    sign = 1.0 if option_type == "call" else -1.0
    dd1_ds = 1.0 / (s * vol_sqrt_t)

    # Gamma = ff × [sign × n(d1) × dd1/dS + d(n(d1) × sign / vol_sqrt_t)/dS]
    # d(n(d1))/dS = -d1 × n(d1) × dd1/dS
    # Gamma = ff × sign × dd1/dS × [n(d1) - d1 × n(d1) / vol_sqrt_t]
    #       = ff × sign × n(d1) × dd1/dS × [1 - d1 / vol_sqrt_t]
    return ff * sign * std_norm_pdf(d1) * dd1_ds * (2.0 - d1 / vol_sqrt_t)


def digital_asset_vega(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Asset-or-nothing digital vega: dPV/dσ.

    PV = S × ff × N(±d₁)
    Vega = S × ff × n(d₁) × sign × dd₁/dσ
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
    ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)
    sqrt_t = math.sqrt(t)

    # dd1/dσ = -d1/σ + √T (from differentiation of d1 formula)
    # Actually: dd1/dσ = √T - d1/σ
    dd1_dsigma = sqrt_t - d1 / sigma
    sign = 1.0 if option_type == "call" else -1.0

    return s * ff * std_norm_pdf(d1) * sign * dd1_dsigma


def digital_asset_theta(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> float:
    """
    Asset-or-nothing digital theta: -dPV/dT.

    PV = S × ff × N(±d₁)
    θ = (r-b) × PV - S × ff × n(d₁) × sign × dd₁/dT
    """
    validate_inputs(spot=spot, strike=strike, expiry=expiry, vol=vol, allow_zero_vol=True)

    s = float(spot)
    t = float(expiry)
    r = float(discount_rate)
    b = float(carry)
    sigma = float(vol)

    if t == 0.0 or sigma == 0.0:
        return 0.0

    pv = digital_asset_price(option_type=option_type, spot=s, strike=strike,
                             expiry=t, discount_rate=r, carry=b, vol=sigma)

    d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=b, vol=sigma)
    ff = forward_factor(carry=b, discount_rate=r, expiry=t)
    sqrt_t = math.sqrt(t)

    # dd1/dT = (b + σ²/2)/(σ√T) - d1/(2T)
    dd1_dt = (b + 0.5 * sigma * sigma) / (sigma * sqrt_t) - d1 / (2.0 * t)
    sign = 1.0 if option_type == "call" else -1.0

    # θ = (r-b) × PV - S × ff × n(d1) × sign × dd1/dT
    return (r - b) * pv - s * ff * std_norm_pdf(d1) * sign * dd1_dt


def digital_asset_greeks(
    *,
    option_type: OptionType,
    spot: float,
    strike: float,
    expiry: float,
    discount_rate: float,
    carry: float,
    vol: float,
) -> Dict[str, float]:
    """
    Compute all digital asset option Greeks.

    Returns
    -------
    dict
        Keys: "delta", "gamma", "vega", "theta", "rho_discount", "rho_carry"
    """
    s = float(spot)
    t = float(expiry)
    sigma = float(vol)

    pv = digital_asset_price(option_type=option_type, spot=s, strike=strike,
                             expiry=t, discount_rate=discount_rate, carry=carry, vol=sigma)

    # rho_discount = -T × PV
    rho_discount = -t * pv if t > 0 else 0.0

    # rho_carry
    if t > 0 and sigma > 0:
        d1, _ = d1_d2(spot=s, strike=strike, expiry=t, carry=carry, vol=sigma)
        ff = forward_factor(carry=carry, discount_rate=discount_rate, expiry=t)
        sqrt_t = math.sqrt(t)
        dd1_db = sqrt_t / sigma
        sign = 1.0 if option_type == "call" else -1.0
        rho_carry = t * pv + s * ff * std_norm_pdf(d1) * sign * dd1_db
    else:
        rho_carry = 0.0

    return {
        "delta": digital_asset_delta(option_type=option_type, spot=s, strike=strike,
                                     expiry=t, discount_rate=discount_rate, carry=carry, vol=sigma),
        "gamma": digital_asset_gamma(option_type=option_type, spot=s, strike=strike,
                                     expiry=t, discount_rate=discount_rate, carry=carry, vol=sigma),
        "vega": digital_asset_vega(option_type=option_type, spot=s, strike=strike,
                                   expiry=t, discount_rate=discount_rate, carry=carry, vol=sigma),
        "theta": digital_asset_theta(option_type=option_type, spot=s, strike=strike,
                                     expiry=t, discount_rate=discount_rate, carry=carry, vol=sigma),
        "rho_discount": rho_discount,
        "rho_carry": rho_carry,
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
    "forward_factor",
    "discount_factor",
    "intrinsic",
    # Vanilla
    "vanilla_price",
    "vanilla_delta",
    "vanilla_gamma",
    "vanilla_vega",
    "vanilla_theta",
    "vanilla_rho_discount",
    "vanilla_rho_carry",
    "vanilla_greeks",
    # Digital cash
    "digital_cash_price",
    "digital_cash_delta",
    "digital_cash_gamma",
    "digital_cash_vega",
    "digital_cash_theta",
    "digital_cash_greeks",
    # Digital asset
    "digital_asset_price",
    "digital_asset_delta",
    "digital_asset_gamma",
    "digital_asset_vega",
    "digital_asset_theta",
    "digital_asset_greeks",
]
