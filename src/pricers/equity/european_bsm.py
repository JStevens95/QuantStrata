# src/pricers/equity/european_bsm.py
"""
Equity European BSM Pricers.

Adapter pricers that map equity instruments to the generic BSM model.

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.marketdata.core.market import Market
from src.instruments.equity.options.vanilla import EquityVanillaEuropeanOption
from src.instruments.equity.options.digital import EquityDigitalEuropeanOption

# Import pure functions from the generic BSM model.
from src.models.analytic.black_scholes_merton.base import (
    vanilla_price,
    vanilla_greeks,
    digital_cash_price,
    digital_cash_greeks,
    digital_asset_price,
    digital_asset_greeks,
    GreekName
)

# Payoff library: single source of truth for terminal condition + payout semantics.
from src.models.payoffs.factory import build_payoff_1d, require_terminal_payoff
from src.models.payoffs.base import BasePayoff1D


# Import common rate utility.
from src.core.math.rates import rate_from_df as _rate_from_df


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
class EquityVanillaEuropeanOptionBsmPricer:
    """
    Adapter pricer: EquityVanillaEuropeanOption -> BSM vanilla formulas.

    Equity Mapping
    --------------
    - discount_rate = r (risk-free rate)
    - carry = r - q (cost-of-carry with dividend yield q)
    - PV per unit notional, scaled by notional

    Greeks Mapping
    --------------
    The generic model returns rho_discount and rho_carry.
    For equity with b = r - q:
    - rho = rho_discount + rho_carry (total rate sensitivity)
    """

    def price(self, trade: EquityVanillaEuropeanOption, market: Market) -> float:
        """
        Calculate present value of European equity vanilla option.

        Parameters
        ----------
        trade : EquityVanillaEuropeanOption
            Option to price
        market : Market
            Market snapshot with spot, curve, and vol surface

        Returns
        -------
        float
            Present value in currency units
        """
        # Read market inputs.
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # Validate.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S <= 0.0:
            raise ValueError("spot must be > 0.")
        if notional == 0.0:
            return 0.0

        # Build terminal payoff (single source of truth).
        payoff = require_terminal_payoff(build_payoff_1d(trade))

        # Handle T=0: immediate payoff, no discounting.
        if T == 0.0:
            return notional * _terminal_value(payoff, S)

        # Get discount factor and risk-free rate.
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)

        # Equity cost-of-carry: b = r - q.
        b = float(r - q)

        # Get implied vol.
        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Handle σ=0: deterministic forward.
        if sigma == 0.0:
            F = S * math.exp(b * T)
            disc = math.exp(-r * T)
            return float(notional * disc * _terminal_value(payoff, F))

        # Call generic BSM formula.
        pv_per_unit = vanilla_price(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            expiry=T,
            discount_rate=r,
            carry=b,
            vol=sigma,
        )
        return float(notional) * float(pv_per_unit)

    def greeks(self, trade: EquityVanillaEuropeanOption, market: Market) -> Dict[GreekName, float]:
        """
        Calculate Greeks for European equity vanilla option.

        Parameters
        ----------
        trade : EquityVanillaEuropeanOption
            Option to analyze
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary with delta, gamma, vega, rho, theta
        """
        # Read market inputs.
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)

        # Validate.
        if T < 0.0:
            raise ValueError("expiry must be >= 0.")
        if S <= 0.0:
            raise ValueError("spot must be > 0.")

        # Handle T=0: return stable zeros.
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        # Get rate and vol.
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)
        b = float(r - q)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma < 0.0:
            raise ValueError("Implied vol must be non-negative.")

        # Call generic BSM Greeks.
        g = vanilla_greeks(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            expiry=T,
            discount_rate=r,
            carry=b,
            vol=sigma,
        )

        # Scale by notional.
        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])
        theta = notional * float(g["theta"])

        # Map generic rhos to equity rho (combined).
        rho = notional * (float(g["rho_discount"]) + float(g["rho_carry"]))

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho": float(rho),
            "theta": float(theta),
        }


@dataclass(frozen=True, slots=True)
class EquityDigitalEuropeanOptionBsmPricer:
    """
    Adapter pricer: EquityDigitalEuropeanOption -> BSM digital formulas.

    Routes to cash-or-nothing or asset-or-nothing based on digital_type.
    """

    def price(self, trade: EquityDigitalEuropeanOption, market: Market) -> float:
        """
        Calculate BSM price for equity digital option.

        Parameters
        ----------
        trade : EquityDigitalEuropeanOption
            The digital option to price
        market : Market
            Market snapshot with spot, curve, and vol

        Returns
        -------
        float
            Present value in currency units
        """
        # Read market data.
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)
        payout = float(trade.payout)

        # Validate.
        if S <= 0.0:
            raise ValueError("Spot must be > 0.")
        if T < 0.0:
            raise ValueError("Expiry must be >= 0.")

        # Handle T=0: immediate payoff.
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
            return 0.0

        # Get rate and vol.
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)
        b = float(r - q)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            raise ValueError("Implied vol must be > 0 for digital pricing.")

        # Route to appropriate formula.
        if trade.digital_type == "cash":
            pv_per_unit = digital_cash_price(
                option_type=trade.option_type,
                spot=S,
                strike=K,
                expiry=T,
                discount_rate=r,
                carry=b,
                vol=sigma,
                cash=payout,
            )
            return float(notional * pv_per_unit)

        # Asset-or-nothing.
        pv_per_unit = digital_asset_price(
            option_type=trade.option_type,
            spot=S,
            strike=K,
            expiry=T,
            discount_rate=r,
            carry=b,
            vol=sigma,
        )
        return float(notional * pv_per_unit)

    def greeks(
        self, trade: EquityDigitalEuropeanOption, market: Market
    ) -> Dict[GreekName, float]:
        """
        Calculate Greeks for equity digital option.

        Parameters
        ----------
        trade : EquityDigitalEuropeanOption
            The digital option
        market : Market
            Market snapshot

        Returns
        -------
        Dict[GreekName, float]
            Greeks dictionary
        """
        # Read market data.
        S = float(market.quote(trade.spot_id))
        K = float(trade.strike)
        T = float(trade.expiry)
        q = float(trade.dividend_yield)
        notional = float(trade.notional)
        payout = float(trade.payout)

        # Handle T=0: return stable zeros.
        if T == 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        # Get rate and vol.
        df = float(market.curve(trade.curve_id).df(T))
        r = _rate_from_df(df=df, t=T)
        b = float(r - q)

        sigma = float(market.vol_surface(trade.vol_id).vol(expiry=T, strike=K))
        if sigma <= 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "rho": 0.0,
                "theta": 0.0,
            }

        # Route to appropriate Greeks function.
        if trade.digital_type == "cash":
            g = digital_cash_greeks(
                option_type=trade.option_type,
                spot=S,
                strike=K,
                expiry=T,
                discount_rate=r,
                carry=b,
                vol=sigma,
                cash=payout,
            )
        else:
            g = digital_asset_greeks(
                option_type=trade.option_type,
                spot=S,
                strike=K,
                expiry=T,
                discount_rate=r,
                carry=b,
                vol=sigma,
            )

        # Scale by notional.
        delta = notional * float(g["delta"])
        gamma = notional * float(g["gamma"])
        vega = notional * float(g["vega"])
        theta = notional * float(g["theta"])

        # Map generic rhos to equity rho (combined).
        rho = notional * (float(g["rho_discount"]) + float(g["rho_carry"]))

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "vega": float(vega),
            "rho": float(rho),
            "theta": float(theta),
        }
