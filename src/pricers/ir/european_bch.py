# src/pricers/ir/european_bch.py
"""
Interest Rate Bachelier Pricers.

Pricers for swaptions using the Bachelier (normal) model.

Mathematical Framework
----------------------

Bachelier Model for Swaptions
-----------------------------
The swap rate follows normal dynamics:
    dS = σ dW

Where σ is the absolute (normal) volatility.

Swaption Pricing:
    Payer:   PV = A × N × [(F - K) N(d) + σ√T n(d)]
    Receiver: PV = A × N × [(K - F) N(-d) + σ√T n(d)]

Where:
- A = annuity (PV01) of the underlying swap
- N = notional
- F = forward swap rate
- K = strike rate
- σ = normal volatility
- d = (F - K) / (σ√T)
- T = time to option expiry

Greeks
------
- delta: dPV/dF (sensitivity to forward swap rate)
- gamma: d²PV/dF² (convexity)
- vega: dPV/dσ (per 1.0 absolute vol, or per 1bp)
- theta: dPV/dT (time decay per year)
- rho: dPV/dr (discount rate sensitivity)

Why Bachelier for Swaptions?
----------------------------
1. Handles negative rates naturally
2. Normal vol more stable than lognormal vol near zero rates
3. Industry standard for EUR, JPY, CHF swaptions
4. Additive vol interpretation (e.g., 50bp vol)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.core.market import Market
from src.instruments.ir.options.swaption import (
    Swaption,
    SwaptionSimple,
)
from src.instruments.ir.linear.swap import (
    InterestRateSwap,
    InterestRateSwapSimple,
    generate_swap_schedule,
    FixedLeg,
    FloatingLeg,
)
from src.instruments.ir.options.capfloor import compute_accrual_factor

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


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _forward_rate_from_dfs(
    *,
    df_start: float,
    df_end: float,
    accrual_factor: float,
) -> float:
    """Compute simple forward rate from discount factors."""
    if accrual_factor <= 0.0:
        raise ValueError("accrual_factor must be > 0")
    return (df_start / df_end - 1.0) / accrual_factor


def _rate_from_df(*, df: float, t: float) -> float:
    """Convert discount factor to continuously-compounded rate."""
    if t <= 0.0:
        return 0.0
    if df <= 0.0:
        raise ValueError(f"Discount factor must be > 0; got {df}.")
    return float(-math.log(df) / t)


# =============================================================================
# SWAPTION BACHELIER PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrEuropeanSwaptionBchPricerSimple:
    """
    Bachelier pricer for swaption with direct parameters.
    
    Uses the normal (Bachelier) model for pricing, appropriate for
    low/negative rate environments.
    """
    
    def price(self, trade: SwaptionSimple) -> float:
        """
        Price a swaption using Bachelier model.
        
        Parameters
        ----------
        trade : SwaptionSimple
            Swaption with direct parameters.
        
        Returns
        -------
        float
            Present value of the swaption.
        
        Notes
        -----
        PV = A × unit_bachelier_price
        
        Where unit_bachelier_price is the Bachelier call/put price
        scaled by notional.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T = float(trade.option_expiry)
        F = float(trade.forward_swap_rate)
        A = float(trade.annuity)
        sigma = float(trade.vol)
        df = float(trade.discount_factor)
        
        # Handle expired swaption.
        if T <= 0.0:
            if trade.swaption_type == "payer":
                intrinsic = max(F - K, 0.0)
            else:
                intrinsic = max(K - F, 0.0)
            return N * A * intrinsic
        
        # Map swaption type to option type.
        option_type = "call" if trade.swaption_type == "payer" else "put"
        
        # Bachelier price for unit notional.
        # Note: For swaptions, we use the annuity as the discount factor.
        # PV = A × [(F-K)N(d) + σ√T n(d)] for payer
        # This is equivalent to bachelier_price with df=1 and then multiply by A.
        unit_pv = bachelier_price(
            option_type=option_type,
            forward=F,
            strike=K,
            expiry=T,
            discount_factor=1.0,  # Annuity handles discounting
            vol=sigma,
        )
        
        # Scale by notional and annuity.
        return N * A * unit_pv
    
    def greeks(self, trade: SwaptionSimple) -> Dict[GreekName, float]:
        """
        Compute Greeks for a swaption.
        
        Parameters
        ----------
        trade : SwaptionSimple
            Swaption with direct parameters.
        
        Returns
        -------
        dict
            Greeks: delta, gamma, vega, theta, rho.
        """
        N = float(trade.notional)
        K = float(trade.strike)
        T = float(trade.option_expiry)
        F = float(trade.forward_swap_rate)
        A = float(trade.annuity)
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
        
        option_type = "call" if trade.swaption_type == "payer" else "put"
        
        # Compute discount rate for theta.
        r = _rate_from_df(df=df, t=T)
        
        # Scale factor.
        scale = N * A
        
        return {
            "delta": scale * bachelier_delta(
                option_type=option_type, forward=F, strike=K, expiry=T,
                discount_factor=1.0, vol=sigma,
            ),
            "gamma": scale * bachelier_gamma(
                option_type=option_type, forward=F, strike=K, expiry=T,
                discount_factor=1.0, vol=sigma,
            ),
            "vega": scale * bachelier_vega(
                option_type=option_type, forward=F, strike=K, expiry=T,
                discount_factor=1.0, vol=sigma,
            ),
            "theta": scale * bachelier_theta(
                option_type=option_type, forward=F, strike=K, expiry=T,
                discount_factor=1.0, discount_rate=r, vol=sigma,
            ),
            "rho": scale * bachelier_rho(
                option_type=option_type, forward=F, strike=K, expiry=T,
                discount_factor=1.0, vol=sigma,
            ),
        }
    
    def vega_bp(self, trade: SwaptionSimple) -> float:
        """
        Compute vega per 1 basis point of normal vol.
        
        More intuitive than vega per 1.0 absolute vol.
        """
        greeks = self.greeks(trade)
        return greeks["vega"] * 0.0001


@dataclass(frozen=True, slots=True)
class IrEuropeanSwaptionBchPricer:
    """
    Bachelier pricer for swaption with market data lookup.
    
    Automatically computes forward swap rate and annuity from the curve.
    """
    
    def price(self, trade: Swaption, market: Market) -> float:
        """
        Price a swaption using Bachelier model with market data.
        
        Parameters
        ----------
        trade : Swaption
            Swaption instrument.
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the swaption.
        """
        simple = self._to_simple(trade, market)
        return IrEuropeanSwaptionBchPricerSimple().price(simple)
    
    def greeks(self, trade: Swaption, market: Market) -> Dict[GreekName, float]:
        """Compute Greeks for a swaption with market data."""
        simple = self._to_simple(trade, market)
        return IrEuropeanSwaptionBchPricerSimple().greeks(simple)
    
    def vega_bp(self, trade: Swaption, market: Market) -> float:
        """Compute vega per 1bp of normal vol."""
        simple = self._to_simple(trade, market)
        return IrEuropeanSwaptionBchPricerSimple().vega_bp(simple)
    
    def forward_swap_rate(self, trade: Swaption, market: Market) -> float:
        """Compute the forward swap rate."""
        simple = self._to_simple(trade, market)
        return simple.forward_swap_rate
    
    def annuity(self, trade: Swaption, market: Market) -> float:
        """Compute the annuity (PV01) of the underlying swap."""
        simple = self._to_simple(trade, market)
        return simple.annuity
    
    def _to_simple(self, trade: Swaption, market: Market) -> SwaptionSimple:
        """Convert market-based swaption to simple swaption."""
        curve = market.curve(trade.curve_id)
        vol_surface = market.vol_surface(trade.vol_id)
        
        # Generate the underlying swap schedule.
        fixed_schedule = generate_swap_schedule(
            trade.swap_start,
            trade.swap_end,
            trade.fixed_frequency,
        )
        
        # Compute annuity = Σ[τ_i × DF_i]
        annuity = 0.0
        for t_start, t_end in fixed_schedule:
            tau = compute_accrual_factor(t_start, t_end, trade.fixed_day_count)
            df = float(curve.df(t_end))
            annuity += tau * df
        
        # Compute forward swap rate = [DF(T_start) - DF(T_end)] / Annuity
        # This is the simplified formula for a par swap starting at swap_start.
        df_start = float(curve.df(trade.swap_start))
        df_end = float(curve.df(trade.swap_end))
        
        forward_swap_rate = (df_start - df_end) / annuity
        
        # Get volatility (normal vol).
        sigma = float(vol_surface.vol(expiry=trade.option_expiry, strike=trade.strike))
        
        # Discount factor to option expiry.
        df_expiry = float(curve.df(trade.option_expiry))
        
        return SwaptionSimple(
            notional=trade.notional,
            strike=trade.strike,
            option_expiry=trade.option_expiry,
            swap_tenor=trade.swap_tenor,
            forward_swap_rate=forward_swap_rate,
            annuity=annuity,
            vol=sigma,
            discount_factor=df_expiry,
            swaption_type=trade.swaption_type,
            settlement=trade.settlement,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Swaption pricers
    "IrEuropeanSwaptionBchPricer",
    "IrEuropeanSwaptionBchPricerSimple",
]
