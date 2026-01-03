from __future__ import annotations  # Keep forward references available across the package.

import math  # Use the standard library math module for stable scalar math.
from dataclasses import dataclass  # Use dataclass for lightweight immutable containers.
from typing import Literal, Tuple  # Use Literal for option_type; Tuple for d1/d2 return type.

from src.models.common.normal import std_norm_cdf, std_norm_pdf  # Reuse your standard normal helpers.
from src.models.payoffs.types import OptionType


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------

def validate_bsm_inputs(*, spot: float, strike: float, expiry: float, vol: float) -> None:
    """
    Validate core BSM inputs.

    Notes
    -----
    - Rates (discount_rate/carry) are allowed to be negative.
    - expiry == 0 is allowed; callers handle expiry explicitly (intrinsic/terminal payoff).
    """
    if float(spot) <= 0.0:  # Spot must be strictly positive for lognormal dynamics.
        raise ValueError(f"spot must be > 0; got {spot}.")  # Raise a clear error.
    if float(strike) <= 0.0:  # Strike must be strictly positive for log(S/K).
        raise ValueError(f"strike must be > 0; got {strike}.")  # Raise a clear error.
    if float(expiry) < 0.0:  # Time to expiry cannot be negative.
        raise ValueError(f"expiry must be >= 0; got {expiry}.")  # Raise a clear error.
    if float(vol) <= 0.0:  # In this analytic package we require strictly positive vol (expiry==0 handled separately).
        raise ValueError(f"vol must be > 0; got {vol}.")  # Raise a clear error.


# --------------------------------------------------------------------------------------
# Core BSM helpers
# --------------------------------------------------------------------------------------

def d1_d2(*, spot: float, strike: float, expiry: float, carry: float, vol: float) -> Tuple[float, float]:
    """
    Compute (d1, d2) for Black–Scholes–Merton with carry.

    d1 = ( ln(S/K) + (carry + 0.5*σ²)*T ) / (σ*sqrt(T))
    d2 = d1 - σ*sqrt(T)

    Parameters
    ----------
    spot, strike, expiry, vol:
        Standard BSM inputs (spot>0, strike>0, expiry>0, vol>0 assumed).
    carry:
        Cost-of-carry rate used inside the drift term of d1/d2 (often b = r - q).

    Returns
    -------
    (d1, d2)
    """
    t = float(expiry)  # Convert expiry to float once.
    sqrt_t = math.sqrt(t)  # Compute sqrt(T) once (used repeatedly).
    sigma = float(vol)  # Convert vol to float once.
    sigma_sqrt_t = sigma * sqrt_t  # Compute σ*sqrt(T) once.
    ln_sk = math.log(float(spot) / float(strike))  # Compute ln(S/K) safely (inputs validated).
    drift = (float(carry) + 0.5 * sigma * sigma) * t  # Compute (b + 0.5σ²)T term.
    d1 = (ln_sk + drift) / sigma_sqrt_t  # Apply the definition of d1.
    d2 = d1 - sigma_sqrt_t  # Apply the definition of d2.
    return float(d1), float(d2)  # Return as plain floats.


@dataclass(frozen=True, slots=True)
class CarryDiscountTerms:
    """
    Convenience bundle for common discount/carry scalars.

    Definitions
    -----------
    df         = exp(-rT)
    fwd_factor = exp((carry - r)T)

    Interpretation
    --------------
    - In FX (Garman–Kohlhagen): r = r_d, carry = r_d - r_f, so fwd_factor = exp(-r_f T).
    - For equity with dividend yield q: carry = r - q, so fwd_factor = exp(-q T).
    """
    df: float  # Discount factor exp(-rT).
    fwd_factor: float  # Carry-vs-discount growth factor exp((b-r)T).
    t: float  # Time to expiry T (years).
    r: float  # Discount rate r (for df).
    b: float  # Carry rate b (for d1/d2 and fwd_factor).

    @staticmethod
    def from_rates(*, time_to_expiry: float, discount_rate: float, carry: float) -> "CarryDiscountTerms":
        t = float(time_to_expiry)  # Convert to float once.
        r = float(discount_rate)  # Convert to float once.
        b = float(carry)  # Convert to float once.
        df = math.exp(-r * t)  # Compute exp(-rT).
        fwd_factor = math.exp((b - r) * t)  # Compute exp((b-r)T).
        return CarryDiscountTerms(df=float(df), fwd_factor=float(fwd_factor), t=t, r=r, b=b)  # Return immutable bundle.


def intrinsic_vanilla(*, option_type: OptionType, spot: float, strike: float) -> float:
    """
    Vanilla intrinsic payoff at time 0 (or at expiry) for call/put.
    """
    s = float(spot)  # Convert to float once.
    k = float(strike)  # Convert to float once.
    if option_type == "call":  # Call intrinsic is max(S-K, 0).
        return float(max(s - k, 0.0))  # Return call intrinsic.
    return float(max(k - s, 0.0))  # Return put intrinsic.


__all__ = [
    "OptionType",  # Export the public option type.
    "validate_bsm_inputs",  # Export shared validation.
    "d1_d2",  # Export d1/d2 helper.
    "CarryDiscountTerms",  # Export terms bundle.
    "intrinsic_vanilla",  # Export intrinsic helper.
    "std_norm_cdf",  # Export normal cdf (so product modules only import from base).
    "std_norm_pdf",  # Export normal pdf (so product modules only import from base).
]