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

from dataclasses import dataclass
from typing import Dict, Literal, List

from src.marketdata.core.market import Market
from src.instruments.ir.linear.swap import (
    IrSwap,
    IrSwapSimple,
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

@dataclass(frozen=True, slots=True)
class IrSwapPricerSimple:
    """
    Pricer for Interest Rate Swap with direct parameters.
    """

    def price(self, trade: IrSwapSimple) -> float:
        """
        Price an interest rate swap.

        Parameters
        ----------
        trade : IrSwapSimple
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

    def greeks(self, trade: IrSwapSimple) -> Dict[LinearGreekName, float]:
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

    def par_rate(self, trade: IrSwapSimple) -> float:
        """
        Compute the par swap rate.

        The par rate is the fixed rate K that makes PV = 0.
        """
        return trade.par_rate

    def fixed_leg_pv(self, trade: IrSwapSimple) -> float:
        """Compute PV of fixed leg only."""
        return sum(
            leg.notional * leg.accrual_factor * leg.discount_factor * leg.fixed_rate
            for leg in trade.fixed_leg
        )

    def floating_leg_pv(self, trade: IrSwapSimple) -> float:
        """Compute PV of floating leg only."""
        return sum(
            leg.notional * leg.accrual_factor * leg.discount_factor * (leg.forward_rate + leg.spread)
            for leg in trade.floating_leg
        )


@dataclass(frozen=True, slots=True)
class IrSwapPricer:
    """
    Pricer for Interest Rate Swap with market data lookup.
    """

    def price(self, trade: IrSwap, market: Market) -> float:
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
        return IrSwapPricerSimple().price(simple)

    def greeks(self, trade: IrSwap, market: Market) -> Dict[LinearGreekName, float]:
        """Compute Greeks for a swap with market data."""
        simple = self._to_simple(trade, market)
        return IrSwapPricerSimple().greeks(simple)

    def par_rate(self, trade: IrSwap, market: Market) -> float:
        """Compute the par swap rate."""
        simple = self._to_simple(trade, market)
        return IrSwapPricerSimple().par_rate(simple)

    def fixed_leg_pv(self, trade: IrSwap, market: Market) -> float:
        """Compute PV of fixed leg only."""
        simple = self._to_simple(trade, market)
        return IrSwapPricerSimple().fixed_leg_pv(simple)

    def floating_leg_pv(self, trade: IrSwap, market: Market) -> float:
        """Compute PV of floating leg only."""
        simple = self._to_simple(trade, market)
        return IrSwapPricerSimple().floating_leg_pv(simple)

    def _to_simple(
            self,
            trade: IrSwap,
            market: Market,
    ) -> IrSwapSimple:
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

        return IrSwapSimple(
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
    # IRS pricers
    "IrSwapPricer",
    "IrSwapPricerSimple",
]
