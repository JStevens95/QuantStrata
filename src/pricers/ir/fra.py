# src/pricers/ir/fra.py
"""
Linear Interest Rate Pricers.

Pricers for Forward Rate Agreements (FRAs).

Mathematical Framework
----------------------

FRA Pricing:
    PV = N × τ × DF(T_end) × (F - K)  [for payer]
    PV = N × τ × DF(T_end) × (K - F)  [for receiver]

Greeks
------
- DV01: Change in PV for 1bp parallel shift in rates
- Forward Delta: dPV/dF (sensitivity to forward rate)
- Gamma: d²PV/dF² (zero for linear instruments)

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

from src.marketdata.core.market import Market
from src.instruments.ir.linear.fra import (
    IrForwardRateAgreement, IrForwardRateAgreementSimple,
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
class IrFraPricerSimple:
    """
    Pricer for Forward Rate Agreement with direct parameters.
    """

    def price(self, trade: IrForwardRateAgreementSimple) -> float:
        """
        Price a FRA.

        Parameters
        ----------
        trade : IrForwardRateAgreementSimple
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

    def greeks(self, trade: IrForwardRateAgreementSimple) -> Dict[LinearGreekName, float]:
        """
        Compute Greeks for a FRA.

        Parameters
        ----------
        trade : IrForwardRateAgreementSimple
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

    def par_rate(self, trade: IrForwardRateAgreementSimple) -> float:
        """
        Return the par rate (forward rate).

        The par rate is the fixed rate K that makes PV = 0.
        """
        return float(trade.forward_rate)


@dataclass(frozen=True, slots=True)
class IrFraPricer:
    """
    Pricer for Forward Rate Agreement with market data lookup.
    """

    def price(self, trade: IrForwardRateAgreement, market: Market) -> float:
        """
        Price a FRA using market data.

        Parameters
        ----------
        trade : IrForwardRateAgreement
            FRA instrument.
        market : Market
            Market snapshot.

        Returns
        -------
        float
            Present value of the FRA.
        """
        simple = self._to_simple(trade, market)
        return IrFraPricerSimple().price(simple)

    def greeks(self, trade: IrForwardRateAgreement, market: Market) -> Dict[LinearGreekName, float]:
        """Compute Greeks for a FRA with market data."""
        simple = self._to_simple(trade, market)
        return IrFraPricerSimple().greeks(simple)

    def par_rate(self, trade: IrForwardRateAgreement, market: Market) -> float:
        """Compute the par rate for a FRA."""
        simple = self._to_simple(trade, market)
        return IrFraPricerSimple().par_rate(simple)

    def _to_simple(
            self,
            trade: IrForwardRateAgreement,
            market: Market,
    ) -> IrForwardRateAgreementSimple:
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

        return IrForwardRateAgreementSimple(
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
# EXPORTS
# =============================================================================

__all__ = [
    # FRA pricers
    "IrFraPricer",
    "IrFraPricerSimple"
]