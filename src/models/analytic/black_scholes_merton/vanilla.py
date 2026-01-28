from __future__ import annotations  # Enable forward references.

import math  # Use math for sqrt/exp.

from dataclasses import dataclass  # Use dataclass for a stateless engine object.
from typing import Dict, Literal  # Dict for greeks; Literal for greek keys.

from src.models.analytic.black_scholes_merton.base import (  # Import shared helpers.
    CarryDiscountTerms,
    OptionType,
    d1_d2,
    intrinsic_vanilla,
    std_norm_cdf,
    std_norm_pdf,
    validate_bsm_inputs,
)

GreekName = Literal["delta", "gamma", "vega", "theta", "rho_discount", "rho_carry"]  # Keep the greek set explicit.


@dataclass(frozen=True, slots=True)
class BlackScholesMertonVanilla:
    """
    Black–Scholes–Merton European vanilla (call/put) engine with generic carry.

    This is a pure-maths engine:
      - No Market dependencies
      - No instrument dependencies
      - No asset-class assumptions

    Signature is aligned to your existing engine style:
      price(*, option_type, spot, strike, time_to_expiry, discount_rate, carry, sigma)
      greeks(*, option_type, spot, strike, time_to_expiry, discount_rate, carry, sigma)
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
        validate_bsm_inputs(spot=spot, strike=strike, expiry=time_to_expiry, vol=sigma)  # Enforce core constraints.

        if float(time_to_expiry) == 0.0:  # Avoid division by zero in d1/d2.
            return intrinsic_vanilla(option_type=option_type, spot=spot, strike=strike)  # Return intrinsic at expiry.

        terms = CarryDiscountTerms.from_rates(  # Precompute df and carry factor scalars.
            time_to_expiry=float(time_to_expiry),  # Pass T.
            discount_rate=float(discount_rate),  # Pass r.
            carry=float(carry),  # Pass b.
        )

        d1, d2 = d1_d2(  # Compute d1 and d2 using carry b.
            spot=float(spot),  # Pass S.
            strike=float(strike),  # Pass K.
            expiry=terms.t,  # Use T from terms.
            carry=terms.b,  # Use b from terms.
            vol=float(sigma),  # Pass σ.
        )

        s = float(spot)  # Cache S as float.
        k = float(strike)  # Cache K as float.

        if option_type == "call":  # Branch on call/put formula.
            pv = s * terms.fwd_factor * std_norm_cdf(d1) - k * terms.df * std_norm_cdf(d2)  # Call PV.
        else:
            pv = k * terms.df * std_norm_cdf(-d2) - s * terms.fwd_factor * std_norm_cdf(-d1)  # Put PV.

        return float(pv)  # Return PV as a float.

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
        validate_bsm_inputs(spot=spot, strike=strike, expiry=time_to_expiry, vol=sigma)  # Enforce core constraints.

        if float(time_to_expiry) == 0.0:  # Greeks are not stable at expiry (discontinuous).
            return {  # Return safe zeros for portfolio stability.
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho_discount": 0.0,
                "rho_carry": 0.0,
            }

        s = float(spot)  # Cache S.
        k = float(strike)  # Cache K.
        t = float(time_to_expiry)  # Cache T.
        sig = float(sigma)  # Cache σ.

        terms = CarryDiscountTerms.from_rates(  # Precompute df and fwd_factor.
            time_to_expiry=t,  # Use cached T.
            discount_rate=float(discount_rate),  # Use r.
            carry=float(carry),  # Use b.
        )

        d1, d2 = d1_d2(  # Compute d1/d2.
            spot=s,  # Use cached S.
            strike=k,  # Use cached K.
            expiry=t,  # Use cached T.
            carry=terms.b,  # Use b.
            vol=sig,  # Use σ.
        )

        n_d1 = std_norm_pdf(d1)  # Compute n(d1).
        N_d1 = std_norm_cdf(d1)  # Compute N(d1).
        sqrt_t = math.sqrt(t)  # Compute sqrt(T).

        # Delta (dPV/dS) under this parameterisation (S * fwd_factor is the PV-carry underlying term).
        if option_type == "call":  # Call delta.
            delta = terms.fwd_factor * N_d1  # Call delta = fwd_factor * N(d1).
        else:  # Put delta.
            delta = terms.fwd_factor * (N_d1 - 1.0)  # Put delta = fwd_factor * (N(d1)-1).

        # Gamma (d²PV/dS²) is identical for call and put.
        gamma = terms.fwd_factor * n_d1 / (s * sig * sqrt_t)  # Gamma = fwd_factor*n(d1)/(Sσ√T).

        # Vega (dPV/dσ), per +1.00 absolute vol.
        vega = s * terms.fwd_factor * n_d1 * sqrt_t  # Vega = S*fwd_factor*n(d1)*√T.

        # rho_discount: dPV/dr holding carry fixed.
        # Here r enters df and fwd_factor, but d1/d2 use carry b (held fixed).
        if option_type == "call":  # Call rho_discount derivative.
            rho_discount = (-t) * (s * terms.fwd_factor * std_norm_cdf(d1)) + t * (k * terms.df * std_norm_cdf(d2))
        else:  # Put rho_discount derivative.
            rho_discount = t * (s * terms.fwd_factor * std_norm_cdf(-d1)) - t * (k * terms.df * std_norm_cdf(-d2))

        # rho_carry: dPV/db holding discount rate r fixed.
        # b only enters through fwd_factor (and also d1/d2). In this engine we keep the standard “carry greek”.
        # The clean convention consistent with your previous engine: treat carry as the drift parameter in d1/d2.
        if option_type == "call":  # Call rho_carry.
            rho_carry = t * (s * terms.fwd_factor * std_norm_cdf(d1))
        else:  # Put rho_carry.
            rho_carry = -t * (s * terms.fwd_factor * std_norm_cdf(-d1))

        # Theta (dPV/dT, time decay) - negative for long options (value decays as time passes).
        # Formula with cost-of-carry b:
        #   Theta_call = -S*fwd_factor*n(d1)*σ/(2√T) - r*K*df*N(d2) + (r-b)*S*fwd_factor*N(d1)
        #   Theta_put  = -S*fwd_factor*n(d1)*σ/(2√T) + r*K*df*N(-d2) - (r-b)*S*fwd_factor*N(-d1)
        # Where (r-b) is effectively the dividend yield q for equities.
        r = float(discount_rate)
        b = float(carry)
        q_eff = r - b  # Effective dividend yield (q for equity, r_f for FX)

        # Common term: diffusion decay (always negative, reduces option value)
        diffusion_decay = -s * terms.fwd_factor * n_d1 * sig / (2.0 * sqrt_t)

        if option_type == "call":
            # Rate effect (negative: discounting reduces strike PV)
            # Dividend effect (positive: dividends reduce spot drift, helping call holder via forward)
            theta = diffusion_decay - r * k * terms.df * std_norm_cdf(d2) + q_eff * s * terms.fwd_factor * N_d1
        else:
            # For puts, signs flip on the rate and dividend components
            theta = diffusion_decay + r * k * terms.df * std_norm_cdf(-d2) - q_eff * s * terms.fwd_factor * std_norm_cdf(-d1)

        return {  # Return greeks in a stable dict ordering.
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "theta": float(theta),
            "rho_discount": float(rho_discount),
            "rho_carry": float(rho_carry),
        }