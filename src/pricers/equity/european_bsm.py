"""
Equity European BSM Pricers

Black-Scholes-Merton pricers for European equity options:
- Vanilla (call/put)
- Digital (cash-or-nothing, asset-or-nothing)

Author: QuantStrata Team
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
from scipy.stats import norm

from src.marketdata.core.market import Market
from src.instruments.equity.options.vanilla import EuropeanEquityVanillaOption
from src.instruments.equity.options.digital import EuropeanEquityDigitalOption
from src.models.analytic.black_scholes_merton.vanilla import BlackScholesMertonVanilla
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff
from src.models.payoffs.base import BasePayoff1D

# Greek names consistent with equity conventions (single rho, not domestic/foreign)
GreekName = Literal["delta", "gamma", "vega", "rho", "theta"]


def _rate_from_df(*, df: float, t: float) -> float:
    """
    Convert discount factor to continuously-compounded rate.

    df = exp(-r × T) => r = -ln(df) / T

    Parameters
    ----------
    df : float
        Discount factor
    t : float
        Time to maturity in years

    Returns
    -------
    float
        Continuously compounded rate
    """
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


def _terminal_value(payoff: BasePayoff1D, spot: float) -> float:
    """
    Evaluate terminal payoff at a single spot value.

    Parameters
    ----------
    payoff : BasePayoff1D
        Payoff object from payoff library
    spot : float
        Spot price

    Returns
    -------
    float
        Terminal payoff value
    """
    return float(payoff.terminal(np.asarray([float(spot)], dtype=np.float64))[0])


@dataclass(frozen=True, slots=True)
class EquityEuropeanVanillaBsmPricer:
    """
    Black-Scholes-Merton pricer for European equity vanilla options.

    Model
    -----
    Under risk-neutral measure with continuous dividend yield:

        dS = (r - q) S dt + σ S dW

    Where:
    - r = risk-free rate
    - q = continuous dividend yield
    - σ = volatility

    Cost-of-Carry
    -------------
    For equities: b = r - q

    This is different from FX where b = r_d - r_f (two separate curves).
    For equities, we have a single risk-free curve plus a dividend yield.

    Pricing
    -------
    PV = notional × BSM(S, K, T, r, b=r-q, σ)

    Greeks Mapping
    --------------
    The generic BSM engine returns rho_discount and rho_carry.
    For equity with b = r - q:

        d(PV)/dr = rho_discount + rho_carry  (total rho)

    We return a single "rho" which is the total sensitivity to the risk-free rate.
    """

    engine: BlackScholesMertonVanilla = BlackScholesMertonVanilla()

    def price(self, trade: EuropeanEquityVanillaOption, market: Market) -> float:
        """
        Calculate present value of European equity vanilla option.

        Parameters
        ----------
        trade : EuropeanEquityVanillaOption
            Option to price
        market : Market
            Market snapshot with spot, curve, and vol surface

        Returns
        -------
        float
            Present value in currency units
        """
        # Read market inputs
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # Validate inputs
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional == 0.0:
            return 0.0

        # Build terminal payoff from payoff library (single source of truth)
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # At expiry: PV = payoff(S) with no discounting
        if T == 0.0:
            return notional * _terminal_value(payoff, S)

        # Get discount factor and risk-free rate
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        # Cost-of-carry for equity: b = r - q
        b = float(r - q)

        # Get implied volatility
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Zero-vol shortcut: deterministic forward
        if sigma == 0.0:
            F = S * math.exp(b * T)  # Forward price
            disc = math.exp(-r * T)
            return float(notional * disc * _terminal_value(payoff, F))

        # Call BSM engine
        pv_per_unit = self.engine.price(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r,
            carry=b,
            sigma=sigma,
        )

        return float(notional) * float(pv_per_unit)

    def greeks(self, trade: EuropeanEquityVanillaOption, market: Market) -> Dict[GreekName, float]:
        """
        Calculate Greeks for European equity vanilla option.

        Parameters
        ----------
        trade : EuropeanEquityVanillaOption
            Option to analyze
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary with delta, gamma, vega, rho, theta
        """
        # Read market inputs
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # Validate
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S <= 0.0:
            raise ValueError("spot must be > 0.")

        # At expiry, greeks are unstable (discontinuity at strike)
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        # Get rate and vol
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)
        b = float(r - q)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Get greeks from engine
        g = self.engine.greeks(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            time_to_expiry=T,
            discount_rate=r,
            carry=b,
            sigma=sigma,
        )

        # Scale by notional
        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])
        theta = notional * float(g["theta"])

        # For equity: total rho = rho_discount + rho_carry
        # (both r effects combined since we only have one rate)
        rho = notional * (float(g["rho_discount"]) + float(g["rho_carry"]))

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho": float(rho),
            "theta": float(theta),
        }


# =============================================================================
# Digital Option Pricer
# =============================================================================


@dataclass(frozen=True, slots=True)
class EquityEuropeanDigitalBsmPricer:
    """
    BSM analytic pricer for European equity digital options.

    Payoff Types
    ------------
    **Cash-or-Nothing:**
    - Call: Pays `payout` if S_T > K, else 0
    - Put: Pays `payout` if S_T < K, else 0

    **Asset-or-Nothing:**
    - Call: Pays S_T if S_T > K, else 0
    - Put: Pays S_T if S_T < K, else 0

    Pricing Formulas
    ----------------
    Cash-or-Nothing Call:
        V = payout × exp(-rT) × N(d2)

    Cash-or-Nothing Put:
        V = payout × exp(-rT) × N(-d2)

    Asset-or-Nothing Call:
        V = S × exp(-qT) × N(d1)

    Asset-or-Nothing Put:
        V = S × exp(-qT) × N(-d1)

    Where:
        d1 = (ln(S/K) + (r-q+σ²/2)T) / (σ√T)
        d2 = d1 - σ√T
        q = dividend yield
        r = risk-free rate

    Greeks
    ------
    Digital options have discontinuous payoffs, leading to:
    - Very high gamma/vega near strike (delta spike)
    - Pin risk at expiry
    - Greeks may be unstable near ATM
    """

    def price(self, trade: EuropeanEquityDigitalOption, market: Market) -> float:
        """
        Calculate BSM price for equity digital option.

        Parameters
        ----------
        trade : EuropeanEquityDigitalOption
            The digital option to price
        market : Market
            Market snapshot with spot, curve, and vol

        Returns
        -------
        float
            Present value in currency units
        """
        # Read market data
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)
        payout = float(trade.payout)

        # Validate
        if S <= 0.0:
            raise ValueError("Spot must be > 0.")
        if T < 0.0:
            raise ValueError("Expiry must be >= 0.")

        # At expiry (T=0): return intrinsic value
        if T == 0.0:
            if trade.option_type == "call":
                itm = S > K
            else:
                itm = S < K

            if itm:
                if trade.digital_type == "cash":
                    return notional * payout
                else:  # asset
                    return notional * S
            else:
                return 0.0

        # Get rate from discount factor
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        # Get implied vol
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            raise ValueError("Implied vol must be > 0 for digital pricing.")

        # Compute d1 and d2
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Price based on digital type
        if trade.digital_type == "cash":
            # Cash-or-nothing
            if trade.option_type == "call":
                pv_per_unit = payout * math.exp(-r * T) * norm.cdf(d2)
            else:  # put
                pv_per_unit = payout * math.exp(-r * T) * norm.cdf(-d2)
        else:
            # Asset-or-nothing
            if trade.option_type == "call":
                pv_per_unit = S * math.exp(-q * T) * norm.cdf(d1)
            else:  # put
                pv_per_unit = S * math.exp(-q * T) * norm.cdf(-d1)

        return notional * pv_per_unit

    def greeks(
        self, trade: EuropeanEquityDigitalOption, market: Market
    ) -> Dict[GreekName, float]:
        """
        Calculate Greeks for equity digital option.

        Note: Digital Greeks are discontinuous near strike and can be large.

        Parameters
        ----------
        trade : EuropeanEquityDigitalOption
            The digital option
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary
        """
        # Read market data
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)
        payout = float(trade.payout)

        # At expiry: Greeks are discontinuous
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        # Get rate and vol
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))

        if sigma <= 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        # Compute d1 and d2
        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Standard normal PDF
        n_d2 = norm.pdf(d2)
        n_d1 = norm.pdf(d1)

        # Delta for cash-or-nothing (most common)
        if trade.digital_type == "cash":
            if trade.option_type == "call":
                # Delta = payout × exp(-rT) × n(d2) / (S × σ × √T)
                delta_per_unit = payout * math.exp(-r * T) * n_d2 / (S * sigma * sqrt_T)
            else:
                delta_per_unit = -payout * math.exp(-r * T) * n_d2 / (S * sigma * sqrt_T)
        else:
            # Asset-or-nothing delta
            N_d1 = norm.cdf(d1) if trade.option_type == "call" else norm.cdf(-d1)
            if trade.option_type == "call":
                delta_per_unit = math.exp(-q * T) * (N_d1 + n_d1 / (sigma * sqrt_T))
            else:
                delta_per_unit = math.exp(-q * T) * (-norm.cdf(-d1) + n_d1 / (sigma * sqrt_T))

        # Gamma (very high near strike for digitals)
        if trade.digital_type == "cash":
            if trade.option_type == "call":
                gamma_per_unit = (
                    -payout * math.exp(-r * T) * n_d2 * d1 / (S**2 * sigma**2 * T)
                )
            else:
                gamma_per_unit = (
                    payout * math.exp(-r * T) * n_d2 * d1 / (S**2 * sigma**2 * T)
                )
        else:
            # Asset digital gamma is more complex
            gamma_per_unit = 0.0  # Simplified

        # Vega
        if trade.digital_type == "cash":
            if trade.option_type == "call":
                vega_per_unit = -payout * math.exp(-r * T) * n_d2 * d1 / sigma
            else:
                vega_per_unit = payout * math.exp(-r * T) * n_d2 * d1 / sigma
        else:
            vega_per_unit = 0.0  # Simplified

        # Scale by notional
        return {
            "delta": float(notional * delta_per_unit),
            "gamma": float(notional * gamma_per_unit),
            "vega": float(notional * vega_per_unit),
            "rho": 0.0,  # Simplified
            "theta": 0.0,  # Simplified
        }
