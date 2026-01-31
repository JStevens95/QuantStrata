# src/pricers/fx/european_bch.py
"""
FX European Option Bachelier Pricers.

Prices FX options using the Bachelier (normal) model. Appropriate for:
- Spread options (difference between two FX rates can be negative)
- Any case where normal dynamics are more suitable than lognormal

Mathematical Framework
----------------------
The underlying follows normal dynamics:
    dF = σ dW

Pricing:
    Call: PV = N × DF × [(F - K) N(d) + σ√T n(d)]
    Put:  PV = N × DF × [(K - F) N(-d) + σ√T n(d)]

Where:
- d = (F - K) / (σ√T)
- σ = absolute (normal) volatility

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.core.market import Market
from src.instruments.fx.options.spread import (
    EuropeanFxSpreadOption,
    EuropeanFxSpreadOptionSimple,
)
from src.instruments.core.types import OptionType

# Import pure functions from the Bachelier model.
from src.models.analytic.bachelier.base import (
    vanilla_price as bachelier_price,
    vanilla_delta as bachelier_delta,
    vanilla_gamma as bachelier_gamma,
    vanilla_vega as bachelier_vega,
    vanilla_theta as bachelier_theta,
    vanilla_rho as bachelier_rho,
)


# Greek name type.
GreekName = Literal["delta", "gamma", "vega", "theta", "rho"]


def _rate_from_df(*, df: float, t: float) -> float:
    """Convert discount factor to continuously-compounded rate."""
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


# =============================================================================
# FX SPREAD OPTION BACHELIER PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class FxEuropeanSpreadBchPricerSimple:
    """
    Bachelier pricer for FX spread option with direct parameters.
    """
    
    def price(self, trade: EuropeanFxSpreadOptionSimple) -> float:
        """
        Price an FX spread option using Bachelier model.
        
        Parameters
        ----------
        trade : EuropeanFxSpreadOptionSimple
            Spread option with direct parameters.
        
        Returns
        -------
        float
            Present value of the spread option.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T = float(trade.expiry)
        F = float(trade.forward_spread)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        # Handle expired option.
        if T <= 0.0:
            if trade.option_type == "call":
                intrinsic = max(F - K, 0.0)
            else:
                intrinsic = max(K - F, 0.0)
            return N * df * intrinsic
        
        # Bachelier price.
        unit_pv = bachelier_price(
            option_type=trade.option_type,
            forward=F,
            strike=K,
            expiry=T,
            discount_factor=df,
            vol=sigma,
        )
        
        return N * unit_pv
    
    def greeks(self, trade: EuropeanFxSpreadOptionSimple) -> Dict[GreekName, float]:
        """
        Compute Greeks for an FX spread option.
        
        Parameters
        ----------
        trade : EuropeanFxSpreadOptionSimple
            Spread option with direct parameters.
        
        Returns
        -------
        dict
            Greeks: delta, gamma, vega, theta, rho.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T = float(trade.expiry)
        F = float(trade.forward_spread)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        if T <= 0.0:
            return {
                "delta": 0.0,
                "gamma": 0.0,
                "vega": 0.0,
                "theta": 0.0,
                "rho": 0.0,
            }
        
        r = _rate_from_df(df=df, t=T)
        
        return {
            "delta": N * bachelier_delta(
                option_type=trade.option_type, forward=F, strike=K, expiry=T,
                discount_factor=df, vol=sigma,
            ),
            "gamma": N * bachelier_gamma(
                option_type=trade.option_type, forward=F, strike=K, expiry=T,
                discount_factor=df, vol=sigma,
            ),
            "vega": N * bachelier_vega(
                option_type=trade.option_type, forward=F, strike=K, expiry=T,
                discount_factor=df, vol=sigma,
            ),
            "theta": N * bachelier_theta(
                option_type=trade.option_type, forward=F, strike=K, expiry=T,
                discount_factor=df, discount_rate=r, vol=sigma,
            ),
            "rho": N * bachelier_rho(
                option_type=trade.option_type, forward=F, strike=K, expiry=T,
                discount_factor=df, vol=sigma,
            ),
        }


@dataclass(frozen=True, slots=True)
class FxEuropeanSpreadBchPricer:
    """
    Bachelier pricer for FX spread option with market data lookup.
    """
    
    def price(self, trade: EuropeanFxSpreadOption, market: Market) -> float:
        """
        Price an FX spread option using market data.
        
        Parameters
        ----------
        trade : EuropeanFxSpreadOption
            Spread option instrument.
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the spread option.
        """
        simple = self._to_simple(trade, market)
        return FxEuropeanSpreadBchPricerSimple().price(simple)
    
    def greeks(self, trade: EuropeanFxSpreadOption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for an FX spread option with market data."""
        simple = self._to_simple(trade, market)
        return FxEuropeanSpreadBchPricerSimple().greeks(simple)
    
    def _to_simple(
        self,
        trade: EuropeanFxSpreadOption,
        market: Market,
    ) -> EuropeanFxSpreadOptionSimple:
        """Convert market-based spread option to simple spread option."""
        curve = market.curve(trade.curve_id)
        
        # Get spot rates.
        spot1 = market.quote(trade.spot1_id)
        spot2 = market.quote(trade.spot2_id)
        
        # Get discount factor to expiry.
        df = float(curve.df(trade.expiry))
        
        # Forward spread (assuming no carry for simplicity).
        forward_spread = spot1 - spot2
        
        # Get spread volatility.
        vol_surface = market.vol_surface(trade.vol_id)
        sigma = float(vol_surface.vol(expiry=trade.expiry, strike=trade.strike))
        
        return EuropeanFxSpreadOptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            expiry=trade.expiry,
            forward_spread=forward_spread,
            vol=sigma,
            discount_factor=df,
            option_type=trade.option_type,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FxEuropeanSpreadBchPricer",
    "FxEuropeanSpreadBchPricerSimple",
]
