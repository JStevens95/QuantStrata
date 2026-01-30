# src/pricers/ir/linear.py
"""
Linear Interest Rate Pricers.

Pricers for Forward Rate Agreements (FRAs) and Interest Rate Swaps (IRS).

Mathematical Framework
----------------------

FRA Pricing:
    PV = N × τ × DF(T_end) × (F - K)  [for payer]
    PV = N × τ × DF(T_end) × (K - F)  [for receiver]

IRS Pricing:
    PV_fixed = N × K × Σ[τ_i × DF_i]
    PV_float = N × Σ[τ_i × DF_i × (F_i + spread)]
    PV = PV_fixed - PV_float  [for receiver]
    PV = PV_float - PV_fixed  [for payer]

Greeks
------
- DV01: Change in PV for 1bp parallel shift in rates
- Forward Delta: dPV/dF (sensitivity to forward rate)
- Gamma: d²PV/dF² (zero for linear instruments)

Author: QuantStrata Team
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Literal, List

from src.marketdata.core.market import Market
from src.instruments.ir.linear.fra import (
    ForwardRateAgreement,
    ForwardRateAgreementSimple,
)
from src.instruments.ir.linear.swap import (
    InterestRateSwap,
    InterestRateSwapSimple,
    FixedLeg,
    FloatingLeg,
    generate_swap_schedule,
)
from src.instruments.ir.options.capfloor import compute_accrual_factor


# Greek name type for linear instruments.
LinearGreekName = Literal["delta", "dv01", "pv01"]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _forward_rate_from_dfs(
    *,
    df_start: float,
    df_end: float,
    accrual_factor: float,
) -> float:
    """
    Compute simple forward rate from discount factors.
    
    F = (DF(T_start) / DF(T_end) - 1) / τ
    """
    if accrual_factor <= 0.0:
        raise ValueError("accrual_factor must be > 0")
    return (df_start / df_end - 1.0) / accrual_factor


# =============================================================================
# FRA PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class FRAPricerSimple:
    """
    Pricer for Forward Rate Agreement with direct parameters.
    """
    
    def price(self, trade: ForwardRateAgreementSimple) -> float:
        """
        Price a FRA.
        
        Parameters
        ----------
        trade : ForwardRateAgreementSimple
            FRA with direct parameters.
        
        Returns
        -------
        float
            Present value of the FRA.
        
        Notes
        -----
        PV = N × τ × DF × (F - K)  [payer]
        PV = N × τ × DF × (K - F)  [receiver]
        """
        N = float(trade.notional)
        K = float(trade.fixed_rate)
        F = float(trade.forward_rate)
        tau = float(trade.accrual_factor)
        df = float(trade.discount_factor)
        
        # Base PV (payer perspective: receive floating, pay fixed).
        base_pv = N * tau * df * (F - K)
        
        # Adjust for direction.
        if trade.direction == "payer":
            return base_pv
        return -base_pv  # Receiver is opposite sign
    
    def greeks(self, trade: ForwardRateAgreementSimple) -> Dict[LinearGreekName, float]:
        """
        Compute Greeks for a FRA.
        
        Parameters
        ----------
        trade : ForwardRateAgreementSimple
            FRA with direct parameters.
        
        Returns
        -------
        dict
            Greeks: delta (dPV/dF), dv01, pv01.
        """
        N = float(trade.notional)
        tau = float(trade.accrual_factor)
        df = float(trade.discount_factor)
        
        # Delta = dPV/dF = N × τ × DF
        # For payer, positive delta (benefit from rate increase)
        # For receiver, negative delta
        base_delta = N * tau * df
        delta = base_delta if trade.direction == "payer" else -base_delta
        
        # DV01 = change in PV for 1bp parallel shift
        # For a FRA, DV01 ≈ |delta| × 0.0001
        dv01 = abs(base_delta) * 0.0001
        
        # PV01 = N × τ × DF × 0.0001 (same as DV01 for single period)
        pv01 = dv01
        
        return {
            "delta": delta,
            "dv01": dv01,
            "pv01": pv01,
        }
    
    def par_rate(self, trade: ForwardRateAgreementSimple) -> float:
        """
        Return the par rate (forward rate).
        
        The par rate is the fixed rate K that makes PV = 0.
        """
        return float(trade.forward_rate)


@dataclass(frozen=True, slots=True)
class FRAPricer:
    """
    Pricer for Forward Rate Agreement with market data lookup.
    """
    
    def price(self, trade: ForwardRateAgreement, market: Market) -> float:
        """
        Price a FRA using market data.
        
        Parameters
        ----------
        trade : ForwardRateAgreement
            FRA instrument.
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the FRA.
        """
        simple = self._to_simple(trade, market)
        return FRAPricerSimple().price(simple)
    
    def greeks(self, trade: ForwardRateAgreement, market: Market) -> Dict[LinearGreekName, float]:
        """Compute Greeks for a FRA with market data."""
        simple = self._to_simple(trade, market)
        return FRAPricerSimple().greeks(simple)
    
    def par_rate(self, trade: ForwardRateAgreement, market: Market) -> float:
        """Compute the par rate for a FRA."""
        simple = self._to_simple(trade, market)
        return FRAPricerSimple().par_rate(simple)
    
    def _to_simple(
        self,
        trade: ForwardRateAgreement,
        market: Market,
    ) -> ForwardRateAgreementSimple:
        """Convert market-based FRA to simple FRA."""
        curve = market.curve(trade.curve_id)
        
        # Get discount factors.
        df_start = float(curve.df(trade.fixing_time))
        df_end = float(curve.df(trade.payment_time))
        
        # Compute accrual factor.
        tau = compute_accrual_factor(
            trade.fixing_time,
            trade.payment_time,
            trade.day_count,
        )
        
        # Compute forward rate.
        F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
        
        return ForwardRateAgreementSimple(
            notional=trade.notional,
            fixed_rate=trade.fixed_rate,
            fixing_time=trade.fixing_time,
            payment_time=trade.payment_time,
            accrual_factor=tau,
            forward_rate=F,
            discount_factor=df_end,
            direction=trade.direction,
        )


# =============================================================================
# IRS PRICERS
# =============================================================================


@dataclass(frozen=True, slots=True)
class IRSwapPricerSimple:
    """
    Pricer for Interest Rate Swap with direct parameters.
    """
    
    def price(self, trade: InterestRateSwapSimple) -> float:
        """
        Price an interest rate swap.
        
        Parameters
        ----------
        trade : InterestRateSwapSimple
            Swap with pre-computed leg cashflows.
        
        Returns
        -------
        float
            Present value of the swap.
        """
        # Fixed leg PV.
        fixed_pv = sum(
            leg.notional * leg.accrual_factor * leg.discount_factor * leg.fixed_rate
            for leg in trade.fixed_leg
        )
        
        # Floating leg PV.
        floating_pv = sum(
            leg.notional * leg.accrual_factor * leg.discount_factor * (leg.forward_rate + leg.spread)
            for leg in trade.floating_leg
        )
        
        # Net PV depends on direction.
        if trade.direction == "receiver":
            # Receive fixed, pay floating.
            return fixed_pv - floating_pv
        # Pay fixed, receive floating.
        return floating_pv - fixed_pv
    
    def greeks(self, trade: InterestRateSwapSimple) -> Dict[LinearGreekName, float]:
        """
        Compute Greeks for an interest rate swap.
        
        Parameters
        ----------
        trade : InterestRateSwapSimple
            Swap with pre-computed leg cashflows.
        
        Returns
        -------
        dict
            Greeks: delta, dv01, pv01.
        """
        # Annuity (PV01 factor).
        annuity = trade.annuity
        
        # DV01 ≈ N × A × 0.0001
        # For receiver, DV01 is negative (lose when rates rise)
        # For payer, DV01 is positive (gain when rates rise)
        base_dv01 = abs(float(trade.notional)) * annuity * 0.0001
        
        # Delta = dPV/dF (aggregate sensitivity to parallel shift in forward rates)
        # For receiver: negative (lose when rates rise)
        # For payer: positive (gain when rates rise)
        base_delta = float(trade.notional) * annuity
        
        if trade.direction == "payer":
            delta = base_delta
            dv01 = base_dv01
        else:
            delta = -base_delta
            dv01 = -base_dv01
        
        return {
            "delta": delta,
            "dv01": dv01,
            "pv01": base_dv01,  # Absolute value
        }
    
    def par_rate(self, trade: InterestRateSwapSimple) -> float:
        """
        Compute the par swap rate.
        
        The par rate is the fixed rate K that makes PV = 0.
        """
        return trade.par_rate
    
    def fixed_leg_pv(self, trade: InterestRateSwapSimple) -> float:
        """Compute PV of fixed leg only."""
        return sum(
            leg.notional * leg.accrual_factor * leg.discount_factor * leg.fixed_rate
            for leg in trade.fixed_leg
        )
    
    def floating_leg_pv(self, trade: InterestRateSwapSimple) -> float:
        """Compute PV of floating leg only."""
        return sum(
            leg.notional * leg.accrual_factor * leg.discount_factor * (leg.forward_rate + leg.spread)
            for leg in trade.floating_leg
        )


@dataclass(frozen=True, slots=True)
class IRSwapPricer:
    """
    Pricer for Interest Rate Swap with market data lookup.
    """
    
    def price(self, trade: InterestRateSwap, market: Market) -> float:
        """
        Price an interest rate swap using market data.
        
        Parameters
        ----------
        trade : InterestRateSwap
            Swap instrument.
        market : Market
            Market snapshot.
        
        Returns
        -------
        float
            Present value of the swap.
        """
        simple = self._to_simple(trade, market)
        return IRSwapPricerSimple().price(simple)
    
    def greeks(self, trade: InterestRateSwap, market: Market) -> Dict[LinearGreekName, float]:
        """Compute Greeks for a swap with market data."""
        simple = self._to_simple(trade, market)
        return IRSwapPricerSimple().greeks(simple)
    
    def par_rate(self, trade: InterestRateSwap, market: Market) -> float:
        """Compute the par swap rate."""
        simple = self._to_simple(trade, market)
        return IRSwapPricerSimple().par_rate(simple)
    
    def fixed_leg_pv(self, trade: InterestRateSwap, market: Market) -> float:
        """Compute PV of fixed leg only."""
        simple = self._to_simple(trade, market)
        return IRSwapPricerSimple().fixed_leg_pv(simple)
    
    def floating_leg_pv(self, trade: InterestRateSwap, market: Market) -> float:
        """Compute PV of floating leg only."""
        simple = self._to_simple(trade, market)
        return IRSwapPricerSimple().floating_leg_pv(simple)
    
    def _to_simple(
        self,
        trade: InterestRateSwap,
        market: Market,
    ) -> InterestRateSwapSimple:
        """Convert market-based swap to simple swap with pre-computed legs."""
        curve = market.curve(trade.curve_id)
        
        # Generate fixed leg schedule and cashflows.
        fixed_schedule = generate_swap_schedule(
            trade.start_time,
            trade.end_time,
            trade.fixed_frequency,
        )
        
        fixed_legs: List[FixedLeg] = []
        for t_start, t_end in fixed_schedule:
            tau = compute_accrual_factor(t_start, t_end, trade.fixed_day_count)
            df = float(curve.df(t_end))
            
            fixed_legs.append(FixedLeg(
                start_time=t_start,
                end_time=t_end,
                accrual_factor=tau,
                discount_factor=df,
                notional=trade.notional,
                fixed_rate=trade.fixed_rate,
            ))
        
        # Generate floating leg schedule and cashflows.
        floating_schedule = generate_swap_schedule(
            trade.start_time,
            trade.end_time,
            trade.floating_frequency,
        )
        
        floating_legs: List[FloatingLeg] = []
        for t_start, t_end in floating_schedule:
            tau = compute_accrual_factor(t_start, t_end, trade.floating_day_count)
            df_start = float(curve.df(max(t_start, 1e-9)))  # Avoid df(0) issues
            df_end = float(curve.df(t_end))
            
            # Compute forward rate for this period.
            if t_start < 1e-9:
                # First period uses spot rate.
                F = _forward_rate_from_dfs(df_start=1.0, df_end=df_end, accrual_factor=tau)
            else:
                F = _forward_rate_from_dfs(df_start=df_start, df_end=df_end, accrual_factor=tau)
            
            floating_legs.append(FloatingLeg(
                start_time=t_start,
                end_time=t_end,
                accrual_factor=tau,
                discount_factor=df_end,
                notional=trade.notional,
                forward_rate=F,
                spread=trade.spread,
            ))
        
        return InterestRateSwapSimple(
            notional=trade.notional,
            fixed_rate=trade.fixed_rate,
            fixed_leg=tuple(fixed_legs),
            floating_leg=tuple(floating_legs),
            direction=trade.direction,
        )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # FRA pricers
    "FRAPricer",
    "FRAPricerSimple",
    # IRS pricers
    "IRSwapPricer",
    "IRSwapPricerSimple",
]
