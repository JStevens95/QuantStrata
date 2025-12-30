from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, Tuple
from src.models.common.normal import std_norm_cdf, std_norm_pdf


# -----------------------------------------------------------------------------
# Public typing (kept minimal and explicit)
# -----------------------------------------------------------------------------

OptionType = Literal["call", "put"]

# Engine-level greeks:
# - "rho_discount" is dPV/d(discount_rate) holding carry fixed
# - "rho_carry"    is dPV/d(carry) holding discount_rate fixed
GreekName = Literal["delta", "gamma", "vega", "rho_discount", "rho_carry"]


# -----------------------------------------------------------------------------
# Defensive validation and core math helpers
# -----------------------------------------------------------------------------

def _validate_inputs(*, spot: float, strike: float, expiry: float, vol: float) -> None:
    """
    Validate BSM inputs.

    Notes
    -----
    - Rates (discount_rate/carry) may be negative in modern markets.
    - expiry == 0 is allowed (handled as intrinsic in public methods).
    """
    if float(spot) <= 0.0:
        raise ValueError(f"spot must be > 0; got {spot}.")
    if float(strike) <= 0.0:
        raise ValueError(f"strike must be > 0; got {strike}.")
    if float(expiry) < 0.0:
        raise ValueError(f"expiry must be >= 0; got {expiry}.")
    if float(vol) <= 0.0:
        raise ValueError(f"vol must be > 0; got {vol}.")


def _calculate_d1_d2(
    *,
    spot: float,
    strike: float,
    expiry: float,
    carry: float,
    vol: float,
) -> Tuple[float, float]:
    """
    Compute (d1, d2) for Black–Scholes–Merton with carry.

    Model form
    ----------
        d1 = ( ln(S/K) + (carry + 0.5*σ²)*T ) / (σ*sqrt(T))
        d2 = d1 - σ*sqrt(T)

    Parameters
    ----------
    spot, strike, expiry, vol:
        Standard BSM inputs.
    carry:
        The "cost-of-carry" rate used inside the drift term of d1/d2.

    Returns
    -------
    (d1, d2)
    """
    # sqrt(T) appears repeatedly, so compute once.
    sqrt_t = math.sqrt(expiry)

    # σ*sqrt(T) also appears repeatedly.
    vol_sqrt_t = float(vol) * sqrt_t

    # log(S/K) is safe after validation (spot>0, strike>0).
    log_moneyness = math.log(float(spot) / float(strike))

    # Apply the standard d1/d2 definitions.
    d1 = (log_moneyness + (float(carry) + 0.5 * float(vol) * float(vol)) * float(expiry)) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t

    return float(d1), float(d2)


# -----------------------------------------------------------------------------
# Public engine: European Black–Scholes–Merton with generic carry
# -----------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BlackScholesMertonEuropean:
    """
    Black–Scholes–Merton European engine with a generic carry parameter.

    Purpose
    -------
    This class is *pure maths*:
    - No Market dependencies
    - No instrument dependencies
    - No asset-class assumptions

    Inputs
    ------
    option_type:
        "call" or "put"
    spot:
        Spot level S
    strike:
        Strike K
    expiry:
        Time to expiry T (in years)
    discount_rate:
        Rate used for discounting: df = exp(-discount_rate * T)
    carry:
        Cost-of-carry parameter used in d1/d2

        Examples:
        - Equity with dividend yield q:
            carry = discount_rate - q
        - FX with domestic rate r_d and foreign rate r_f:
            discount_rate = r_d
            carry         = r_d - r_f

    vol:
        Volatility σ

    Greek conventions
    -----------------
    - vega is per +1.00 absolute vol (0.12 -> 0.13 corresponds to dσ = +0.01)
    - rho_discount is dPV/d(discount_rate) holding carry fixed
    - rho_carry is dPV/d(carry) holding discount_rate fixed
    """

    def price(
        self,
        *,
        option_type: OptionType,
        spot: float,
        strike: float,
        time_to_expiry: float,
        discount_rate: float,
        carry: float,
        sigma: float,
    ) -> float:
        """
        Compute the undiscounted-underlying / discounted-strike European option PV.

        Formula
        -------
        Let:
          df        = exp(-rT)
          fwd_factor= exp((carry - r)T)

        Call:
          PV = spot * fwd_factor * N(d1) - strike * df * N(d2)

        Put:
          PV = strike * df * N(-d2) - spot * fwd_factor * N(-d1)
        """
        _validate_inputs(spot=spot, strike=strike, expiry=time_to_expiry, vol=sigma)

        # Handle expiry==0 using intrinsic to avoid division-by-zero in d1/d2.
        if float(time_to_expiry) == 0.0:
            if option_type == "call":
                return float(max(float(spot) - float(strike), 0.0))
            return float(max(float(strike) - float(spot), 0.0))

        # Discount factor exp(-rT).
        df = math.exp(-float(discount_rate) * float(time_to_expiry))

        # "Forward factor" exp((carry - r)T).
        # For FX, this becomes exp(-r_f T) and matches S*df_f behaviour.
        fwd_factor = math.exp((float(carry) - float(discount_rate)) * float(time_to_expiry))

        # Compute d1/d2 using carry (not discount_rate).
        d1, d2 = _calculate_d1_d2(spot=spot, strike=strike, expiry=time_to_expiry, carry=carry, vol=sigma)

        if option_type == "call":
            pv = float(spot) * fwd_factor * std_norm_cdf(d1) - float(strike) * df * std_norm_cdf(d2)
        else:
            pv = float(strike) * df * std_norm_cdf(-d2) - float(spot) * fwd_factor * std_norm_cdf(-d1)

        return float(pv)

    def greeks(
        self,
        *,
        option_type: OptionType,
        spot: float,
        strike: float,
        time_to_expiry: float,
        discount_rate: float,
        carry: float,
        sigma: float,
    ) -> Dict[GreekName, float]:
        """
        Compute key BSM greeks for the engine's parameterisation.

        Returns
        -------
        dict
            Keys: delta, gamma, vega, rho_discount, rho_carry
        """
        _validate_inputs(spot=spot, strike=strike, expiry=time_to_expiry, vol=sigma)

        # At expiry the greeks are discontinuous/noisy; return zeros for stability.
        if float(time_to_expiry) == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho_discount": 0.0,
                "rho_carry": 0.0,
            }

        # Precompute scalars used in multiple expressions.
        t = float(time_to_expiry)
        s = float(spot)
        k = float(strike)
        sig = float(sigma)
        r = float(discount_rate)
        b = float(carry)

        # Discount factor and carry-vs-discount growth factor.
        df = math.exp(-r * t)
        fwd_factor = math.exp((b - r) * t)

        # Compute d1/d2 and their normal quantities.
        d1, d2 = _calculate_d1_d2(spot=s, strike=k, expiry=t, carry=b, vol=sig)

        n_d1 = std_norm_pdf(d1)
        N_d1 = std_norm_cdf(d1)

        sqrt_t = math.sqrt(t)

        # ---------------------------------------------------------------------
        # Delta: dPV/dS
        # Call: fwd_factor * N(d1)
        # Put : fwd_factor * (N(d1) - 1)
        # ---------------------------------------------------------------------
        if option_type == "call":
            delta = fwd_factor * N_d1
        else:
            delta = fwd_factor * (N_d1 - 1.0)

        # ---------------------------------------------------------------------
        # Gamma: d²PV/dS² (same for call and put)
        # gamma = fwd_factor * n(d1) / (S * σ * sqrt(T))
        # ---------------------------------------------------------------------
        gamma = fwd_factor * n_d1 / (s * sig * sqrt_t)

        # ---------------------------------------------------------------------
        # Vega: dPV/dσ (per 1.00 absolute vol)
        # vega = S * fwd_factor * n(d1) * sqrt(T)
        # ---------------------------------------------------------------------
        vega = s * fwd_factor * n_d1 * sqrt_t

        # ---------------------------------------------------------------------
        # rho_discount: dPV/dr holding carry fixed
        #
        # r appears in:
        # - df         = exp(-rT)
        # - fwd_factor = exp((b-r)T)
        #
        # Differentiate PV directly (compact and robust).
        # ---------------------------------------------------------------------
        if option_type == "call":
            # Call PV = S*fwd*N(d1) - K*df*N(d2)
            # d/dr PV = (-T)S*fwd*N(d1) + (+T)K*df*N(d2)
            rho_discount = (-t) * (s * fwd_factor * std_norm_cdf(d1)) + t * (k * df * std_norm_cdf(d2))
        else:
            # Put PV = K*df*N(-d2) - S*fwd*N(-d1)
            # d/dr PV = (-T)K*df*N(-d2) + (+T)S*fwd*N(-d1)
            rho_discount = t * (s * fwd_factor * std_norm_cdf(-d1)) - t * (k * df * std_norm_cdf(-d2))

        # ---------------------------------------------------------------------
        # rho_carry: dPV/db holding r fixed
        #
        # Only enters through fwd_factor = exp((b-r)T):
        # d/db fwd_factor = T * fwd_factor
        #
        # Put has a sign flip due to the N(-d1) term.
        # ---------------------------------------------------------------------
        if option_type == "call":
            rho_carry = t * (s * fwd_factor * std_norm_cdf(d1))
        else:
            rho_carry = -t * (s * fwd_factor * std_norm_cdf(-d1))

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho_discount": float(rho_discount),
            "rho_carry": float(rho_carry),
        }