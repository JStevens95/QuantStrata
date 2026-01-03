# src/models/analytic/black_scholes_merton/digital.py
from __future__ import annotations

import math  # We need exp/log/sqrt for closed-form BSM expressions.
from dataclasses import dataclass  # We use dataclass for a lightweight “engine” object.
from typing import Dict, Literal, Optional, Protocol, runtime_checkable  # We keep typing explicit and minimal.

from src.models.common.normal import std_norm_cdf, std_norm_pdf  # Standard normal CDF/PDF used by closed-form formulas.
from src.models.payoffs.types import OptionType
from src.models.analytic.black_scholes_merton.base import validate_bsm_inputs, CarryDiscountTerms, d1_d2


# -------------------------------------------------------------------------------------------------
# Public typing (kept consistent with your existing engine style)
# -------------------------------------------------------------------------------------------------

GreekName = Literal["delta", "gamma", "vega", "rho_discount", "rho_carry"]  # Same names as your BSM engine.


# -------------------------------------------------------------------------------------------------
# Optional: minimal “payoff compatibility” (so pricers can accept payoff objects cleanly)
# -------------------------------------------------------------------------------------------------

@runtime_checkable
class DigitalCashPayoffLike(Protocol):
    """Minimal protocol so we can accept your payoff-library digital-cash payoff without coupling tightly."""
    option_type: OptionType
    strike: float
    cash: float  # Cash amount paid at expiry if ITM (per unit notional / per contract, per your convention).


# -------------------------------------------------------------------------------------------------
# Internal helpers (self-contained, so digital.py is portable and easy to review)
# -------------------------------------------------------------------------------------------------

def _validate_inputs(*, spot: float, strike: float, expiry: float, sigma: float) -> None:
    """Validate core BSM inputs for digital pricing."""
    if float(spot) <= 0.0:  # Spot must be strictly positive for log(S/K).
        raise ValueError(f"spot must be > 0; got {spot}.")  # Fail fast with a clear message.
    if float(strike) <= 0.0:  # Strike must be strictly positive for log(S/K).
        raise ValueError(f"strike must be > 0; got {strike}.")  # Fail fast with a clear message.
    if float(expiry) < 0.0:  # Negative expiry is invalid.
        raise ValueError(f"expiry must be >= 0; got {expiry}.")  # Fail fast with a clear message.
    if float(sigma) < 0.0:  # Negative vol is invalid.
        raise ValueError(f"sigma must be >= 0; got {sigma}.")  # Fail fast with a clear message.


def _d2(*, spot: float, strike: float, expiry: float, carry: float, sigma: float) -> float:
    """
    Compute d2 for Black–Scholes–Merton with carry.

    d2 = [ ln(S/K) + (carry - 0.5*sigma^2)*T ] / (sigma*sqrt(T))
    """
    sqrt_t = math.sqrt(float(expiry))  # Compute sqrt(T) once (used multiple times).
    vol_sqrt_t = float(sigma) * sqrt_t  # Compute sigma*sqrt(T) once (the common denominator).
    log_moneyness = math.log(float(spot) / float(strike))  # Compute ln(S/K) safely after validation.
    numer = log_moneyness + (float(carry) - 0.5 * float(sigma) * float(sigma)) * float(expiry)  # Build numerator.
    return float(numer / vol_sqrt_t)  # Return d2 as a float for downstream normal calls.


# -------------------------------------------------------------------------------------------------
# Public engine: Cash-or-nothing digital under Black–Scholes–Merton with carry
# -------------------------------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BlackScholesMertonDigitalCash:
    """
    Closed-form cash-or-nothing digital option under Black–Scholes–Merton with carry.

    Model/parameterisation
    ----------------------
    - discount_rate: r  (used only for discounting via df = exp(-rT))
    - carry:        b  (used inside d2; e.g. FX: b = r_d - r_f)

    Payoff (cash-or-nothing)
    ------------------------
    - Call pays `cash` at expiry if S_T > K
    - Put  pays `cash` at expiry if S_T < K

    Price (closed form)
    -------------------
    PV_call = cash * exp(-rT) * N(d2)
    PV_put  = cash * exp(-rT) * N(-d2)

    Greek conventions
    -----------------
    - delta/gamma are w.r.t. spot S
    - vega is dPV/dsigma per +1.00 absolute vol
    - rho_discount is dPV/d(discount_rate) holding carry fixed
    - rho_carry    is dPV/d(carry) holding discount_rate fixed
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
        cash: float = 1.0,
        payoff: Optional[DigitalCashPayoffLike] = None,
    ) -> float:
        """
        Compute the digital cash-or-nothing PV in closed form.

        Notes
        -----
        - `cash` is the amount paid at expiry if ITM (per unit notional / per contract).
        - If `payoff` is provided, we read (option_type, strike, cash) from it (convenience only).
        """
        if payoff is not None:  # If the caller provides a payoff object, use it as the authoritative definition.
            option_type = payoff.option_type  # Pull call/put from the payoff.
            strike = float(payoff.strike)  # Pull strike from the payoff.
            cash = float(payoff.cash)  # Pull cash amount from the payoff.

        _validate_inputs(spot=spot, strike=strike, expiry=time_to_expiry, sigma=sigma)  # Validate core inputs.

        t = float(time_to_expiry)  # Convert expiry to float once.
        s = float(spot)  # Convert spot to float once.
        k = float(strike)  # Convert strike to float once.
        r = float(discount_rate)  # Convert discount rate to float once.
        b = float(carry)  # Convert carry to float once.
        sig = float(sigma)  # Convert sigma to float once.
        csh = float(cash)  # Convert cash to float once.

        if t == 0.0:  # At expiry, the payoff is deterministic (ignoring the boundary measure-zero issue).
            if option_type == "call":  # If it is a call digital...
                return float(csh if s > k else 0.0)  # Pay cash if strictly ITM, else zero.
            return float(csh if s < k else 0.0)  # Put pays cash if strictly ITM, else zero.

        df = math.exp(-r * t)  # Compute the discount factor exp(-rT).

        if sig == 0.0:  # With zero vol, S_T is deterministic under drift b (degenerate distribution).
            s_t = s * math.exp(b * t)  # Compute the deterministic terminal spot under carry drift.
            if option_type == "call":  # For a call digital...
                return float(csh * df if s_t > k else 0.0)  # Discounted cash if terminal is ITM.
            return float(csh * df if s_t < k else 0.0)  # Discounted cash if terminal is ITM.

        d2_val = _d2(spot=s, strike=k, expiry=t, carry=b, sigma=sig)  # Compute d2 in this engine's parameterisation.

        if option_type == "call":  # If call digital...
            pv = csh * df * std_norm_cdf(d2_val)  # PV = cash * df * N(d2).
        else:  # Otherwise put digital...
            pv = csh * df * std_norm_cdf(-d2_val)  # PV = cash * df * N(-d2).

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
        cash: float = 1.0,
        payoff: Optional[DigitalCashPayoffLike] = None,
    ) -> Dict[GreekName, float]:
        """
        Closed-form greeks for the cash-or-nothing digital, consistent with your engine conventions.
        """
        if payoff is not None:  # If the caller provides a payoff object, use it as the authoritative definition.
            option_type = payoff.option_type  # Pull call/put from the payoff.
            strike = float(payoff.strike)  # Pull strike from the payoff.
            cash = float(payoff.cash)  # Pull cash amount from the payoff.

        _validate_inputs(spot=spot, strike=strike, expiry=time_to_expiry, sigma=sigma)  # Validate core inputs.

        t = float(time_to_expiry)  # Convert expiry to float once.
        s = float(spot)  # Convert spot to float once.
        k = float(strike)  # Convert strike to float once.
        r = float(discount_rate)  # Convert discount rate to float once.
        b = float(carry)  # Convert carry to float once.
        sig = float(sigma)  # Convert sigma to float once.
        csh = float(cash)  # Convert cash to float once.

        if t == 0.0:  # At expiry the payoff is discontinuous, so analytical greeks are not stable.
            return {  # Return zeros for portfolio safety and to match your existing style.
                "delta": 0.0,  # Delta is not well-defined at expiry for digitals.
                "gamma": 0.0,  # Gamma is not well-defined at expiry for digitals.
                "vega": 0.0,  # Vega collapses at expiry.
                "rho_discount": 0.0,  # Rho collapses at expiry.
                "rho_carry": 0.0,  # Carry sensitivity collapses at expiry.
            }

        df = math.exp(-r * t)  # Compute discount factor exp(-rT).

        if sig == 0.0:  # Under zero vol, the distribution degenerates and greeks are effectively distributions (spikes).
            return {  # Return zeros to avoid misleading outputs in risk systems.
                "delta": 0.0,  # Deterministic indicator has zero sensitivity almost everywhere and undefined at boundary.
                "gamma": 0.0,  # Same reasoning for gamma.
                "vega": 0.0,  # No volatility sensitivity when sigma=0.
                "rho_discount": -t * float(self.price(  # Rho_discount is well-defined via PV = df * cash * indicator.
                    option_type=option_type,
                    spot=s,
                    strike=k,
                    time_to_expiry=t,
                    discount_rate=r,
                    carry=b,
                    sigma=sig,
                    cash=csh,
                )),
                "rho_carry": 0.0,  # Carry moves the deterministic terminal boundary; not stable as a “smooth” greek.
            }

        sqrt_t = math.sqrt(t)  # Compute sqrt(T) once.
        d2_val = _d2(spot=s, strike=k, expiry=t, carry=b, sigma=sig)  # Compute d2 in this engine's parameterisation.
        n_d2 = std_norm_pdf(d2_val)  # Compute phi(d2) once (used by most greeks).

        # Compute ∂d2/∂S = 1 / (S * sigma * sqrt(T)).
        dd2_dS = 1.0 / (s * sig * sqrt_t)  # This is the spot sensitivity of d2.

        # Compute ∂d2/∂b = sqrt(T) / sigma (since d2 numerator contains +bT).
        dd2_db = sqrt_t / sig  # This is the carry sensitivity of d2.

        # Compute ∂d2/∂sigma in a numerically stable form:
        # d2 = (A)/(sigma*sqrt(T)) - 0.5*sigma*sqrt(T), where A=ln(S/K)+bT
        # ∂d2/∂sigma = -(d2/sigma + sqrt(T)).
        dd2_dsigma = -(d2_val / sig + sqrt_t)  # Closed-form and stable.

        # Sign handling:
        # - Call uses N(d2) so derivative multiplier is +1.
        # - Put uses N(-d2) so derivative multiplier is -1 (because d/dx N(-x) = -phi(x)).
        sign = 1.0 if option_type == "call" else -1.0  # Capture call/put sign in one place.

        # Price term used by rho_discount (PV itself) is useful to compute once.
        pv = float(self.price(  # Reuse the closed-form price to guarantee consistent conventions.
            option_type=option_type,
            spot=s,
            strike=k,
            time_to_expiry=t,
            discount_rate=r,
            carry=b,
            sigma=sig,
            cash=csh,
        ))

        # Delta: dPV/dS = cash * df * phi(d2) * sign * ∂d2/∂S.
        delta = csh * df * n_d2 * sign * dd2_dS  # Closed-form delta.

        # Gamma: d²PV/dS².
        # For call: gamma_call = -cash*df*phi(d2) * ( d2*(∂d2/∂S)^2 + (∂d2/∂S)/S )
        # For put : gamma_put  = -gamma_call (because sign flips for the digital’s N(±d2)).
        g = dd2_dS  # Alias ∂d2/∂S to keep the expression readable.
        gamma_call = -csh * df * n_d2 * (d2_val * (g * g) + (g / s))  # Closed-form gamma for the call digital.
        gamma = gamma_call if option_type == "call" else -gamma_call  # Apply the put sign relation.

        # Vega: dPV/dsigma = cash * df * phi(d2) * sign * ∂d2/∂sigma.
        vega = csh * df * n_d2 * sign * dd2_dsigma  # Closed-form vega.

        # rho_discount: dPV/d(discount_rate) holding carry fixed.
        # PV = df * cash * N(±d2) and d2 does NOT depend on discount_rate when carry is held fixed.
        rho_discount = -t * pv  # Since d/ dr exp(-rT) = -T exp(-rT), rho_discount = -T * PV.

        # rho_carry: dPV/d(carry) holding discount_rate fixed.
        # PV = cash * df * N(±d2), so ∂PV/∂b = cash*df*phi(d2)*sign*∂d2/∂b.
        rho_carry = csh * df * n_d2 * sign * dd2_db  # Closed-form carry sensitivity.

        return {  # Return as floats with your existing key names.
            "delta": float(delta),  # Spot delta.
            "gamma": float(gamma),  # Spot gamma.
            "vega": float(vega),  # Vol sensitivity (per +1.00 absolute vol).
            "rho_discount": float(rho_discount),  # Sensitivity to discount_rate holding carry fixed.
            "rho_carry": float(rho_carry),  # Sensitivity to carry holding discount_rate fixed.
        }


@dataclass(frozen=True, slots=True)
class BlackScholesMertonDigitalAsset:
    """
    Asset-or-nothing digital option under Black–Scholes–Merton with generic carry.

    Payoff (per unit payout)
    ------------------------
    - Call: pays 1 unit of the underlying asset at expiry if S_T > K
    - Put : pays 1 unit of the underlying asset at expiry if S_T < K

    If you pass `asset_units`, the payoff becomes `asset_units` units of the underlying asset.

    Closed-form PV (generic carry)
    ------------------------------
    Let fwd_factor = exp((carry - discount_rate) * T).
    Then:

      PV_call = asset_units * S * fwd_factor * N(d1)
      PV_put  = asset_units * S * fwd_factor * N(-d1)

    Notes
    -----
    - This engine is *pure maths* (no Market, no instrument).
    - For FX, "asset units" means foreign currency units; PV is in domestic naturally via S*exp(-r_f T).
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
        asset_units: float = 1.0,
    ) -> float:
        validate_bsm_inputs(spot=spot, strike=strike, expiry=time_to_expiry, vol=max(float(sigma), 1e-300))  # Validate core inputs.

        s = float(spot)  # Cache S.
        k = float(strike)  # Cache K.
        t = float(time_to_expiry)  # Cache T.
        r = float(discount_rate)  # Cache r.
        b = float(carry)  # Cache b.
        sig = float(sigma)  # Cache sigma.
        q = float(asset_units)  # Cache payout in asset units.

        if t == 0.0:  # Handle expiry explicitly (avoid d1 division).
            if option_type == "call":  # Call pays asset if spot is above strike.
                return float(q * s if s > k else 0.0)  # Pay q*S if ITM.
            return float(q * s if s < k else 0.0)  # Put pays q*S if ITM.

        terms = CarryDiscountTerms.from_rates(  # Compute df and fwd_factor scalars.
            time_to_expiry=t,  # Pass T.
            discount_rate=r,  # Pass r.
            carry=b,  # Pass b.
        )

        if sig == 0.0:  # Handle degenerate distribution (deterministic terminal).
            s_t = s * math.exp(b * t)  # Deterministic terminal spot under drift b.
            pv = q * s * terms.fwd_factor if ((s_t > k) if option_type == "call" else (s_t < k)) else 0.0  # Discounted payoff.
            return float(pv)  # Return PV.

        d1, _d2 = d1_d2(  # Compute d1/d2; only d1 is needed here.
            spot=s,  # S.
            strike=k,  # K.
            expiry=t,  # T.
            carry=b,  # b in the drift term.
            vol=sig,  # sigma.
        )

        if option_type == "call":  # Call asset digital.
            pv = q * s * terms.fwd_factor * std_norm_cdf(d1)  # PV = q*S*fwd_factor*N(d1).
        else:  # Put asset digital.
            pv = q * s * terms.fwd_factor * std_norm_cdf(-d1)  # PV = q*S*fwd_factor*N(-d1).

        return float(pv)  # Return PV.

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
        asset_units: float = 1.0,
    ) -> Dict[GreekName, float]:
        validate_bsm_inputs(spot=spot, strike=strike, expiry=time_to_expiry, vol=max(float(sigma), 1e-300))  # Validate.

        s = float(spot)  # Cache S.
        k = float(strike)  # Cache K.
        t = float(time_to_expiry)  # Cache T.
        r = float(discount_rate)  # Cache r.
        b = float(carry)  # Cache b.
        sig = float(sigma)  # Cache sigma.
        q = float(asset_units)  # Cache payout.

        if t == 0.0:  # Greeks are discontinuous at expiry for digitals.
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho_discount": 0.0, "rho_carry": 0.0}  # Safe zeros.

        terms = CarryDiscountTerms.from_rates(  # Compute df and fwd_factor.
            time_to_expiry=t,  # T.
            discount_rate=r,  # r.
            carry=b,  # b.
        )

        if sig == 0.0:  # Degenerate distribution => greeks are not stable at the boundary.
            pv = self.price(  # Compute PV consistently.
                option_type=option_type,
                spot=s,
                strike=k,
                time_to_expiry=t,
                discount_rate=r,
                carry=b,
                sigma=sig,
                asset_units=q,
            )
            return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "rho_discount": float(-t * pv), "rho_carry": 0.0}  # Only df-effect is smooth.

        d1, _d2 = d1_d2(  # Compute d1.
            spot=s,  # S.
            strike=k,  # K.
            expiry=t,  # T.
            carry=b,  # b.
            vol=sig,  # sigma.
        )

        sqrt_t = math.sqrt(t)  # Compute sqrt(T).
        vol_sqrt_t = sig * sqrt_t  # Compute sigma*sqrt(T).
        phi_d1 = std_norm_pdf(d1)  # Compute phi(d1).

        sign = 1.0 if option_type == "call" else -1.0  # Call/put sign for N(±d1).

        Nd1_signed = std_norm_cdf(sign * d1)  # Compute N(±d1).

        # d1_S = 1 / (S * sigma * sqrt(T))
        dd1_dS = 1.0 / (s * vol_sqrt_t)  # Spot derivative of d1.

        # d1_sigma = sqrt(T) - d1/sigma
        dd1_dsigma = sqrt_t - (d1 / sig)  # Vol derivative of d1.

        # PV = q * S * fwd_factor * N(±d1)
        pv = float(q * s * terms.fwd_factor * Nd1_signed)  # Cache PV (used by rho_discount/rho_carry).

        # Delta = d/dS [ q*fwd*( S*N(±d1) ) ] = q*fwd*( N(±d1) + S*phi(d1)*sign*d1_S )
        delta = q * terms.fwd_factor * (Nd1_signed + (s * phi_d1 * sign * dd1_dS))  # Closed-form delta.

        # Gamma = d/dS [delta] = q*fwd*( N'(±d1) + (phi(d1)*sign*d1_S + S*d/dS[phi(d1)*sign*d1_S]) )
        # A compact closed form from differentiating delta = q*fwd*( N(±d1) + sign*phi(d1)/(sigma*sqrtT) ):
        gamma = q * terms.fwd_factor * (sign * phi_d1 / (s * vol_sqrt_t)) * (1.0 - (d1 / vol_sqrt_t))  # Closed-form gamma.

        # Vega = q*S*fwd * phi(d1) * sign * d1_sigma
        vega = q * s * terms.fwd_factor * phi_d1 * sign * dd1_dsigma  # Closed-form vega.

        # rho_discount: holding carry fixed => only fwd_factor depends on r (as exp((b-r)T))
        rho_discount = -t * pv  # d/dr exp((b-r)T) = -T exp((b-r)T).

        # rho_carry: holding r fixed => PV depends on b via fwd_factor and via d1
        # d/db PV = T*PV + q*S*fwd*phi(d1)*sign*d1_b, with d1_b = sqrt(T)/sigma
        dd1_db = sqrt_t / sig  # Carry derivative of d1.
        rho_carry = (t * pv) + (q * s * terms.fwd_factor * phi_d1 * sign * dd1_db)  # Closed-form carry rho.

        return {  # Return greeks.
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho_discount": float(rho_discount),
            "rho_carry": float(rho_carry),
        }