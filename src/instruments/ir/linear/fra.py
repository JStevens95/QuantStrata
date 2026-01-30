# src/instruments/ir/linear/fra.py
"""
Forward Rate Agreement (FRA) Instrument.

Mathematical Framework
----------------------
A Forward Rate Agreement is an OTC contract to exchange a fixed rate for a 
floating rate on a notional principal for a single future period.

Settlement at fixing date T_start (standard FRA settlement):
    PV = N × τ × (L - K) / (1 + L × τ)

Or settlement at payment date T_end:
    PV = N × τ × DF(T_end) × (L - K)

Where:
- N = notional principal
- τ = day count fraction for [T_start, T_end]
- L = LIBOR/SOFR rate fixing at T_start for period [T_start, T_end]
- K = agreed FRA rate (contract rate)
- DF(T_end) = discount factor to payment date

Sign Convention
---------------
- **Receiver FRA (receive fixed)**: Long receives K, pays L
  - PV = N × τ × DF × (K - F) where F is forward rate
  - Benefits when rates fall
  
- **Payer FRA (pay fixed)**: Long pays K, receives L
  - PV = N × τ × DF × (F - K)
  - Benefits when rates rise

Forward Rate
------------
The forward rate F for period [T_start, T_end] from discount factors:
    F = (DF(T_start) / DF(T_end) - 1) / τ

Par Rate
--------
The par FRA rate (K that makes PV = 0) equals the forward rate F.

Greeks
------
- DV01: Change in PV for 1bp parallel shift in rates
- Forward Delta: dPV/dF (sensitivity to forward rate)

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from src.marketdata.core.ids import MarketId
from src.instruments.core.types import FRADirection, DayCountConvention


# =============================================================================
# SIMPLE INSTRUMENT (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ForwardRateAgreementSimple:
    """
    Forward Rate Agreement with direct parameter input.
    
    A FRA is an agreement to exchange a fixed rate for a floating rate
    on a notional principal for a single future period.
    
    Parameters
    ----------
    notional : float
        Notional principal amount.
    fixed_rate : float
        Agreed FRA rate K (e.g., 0.05 for 5%).
    fixing_time : float
        Time to rate fixing T_start (years).
    payment_time : float
        Time to payment T_end (years).
    accrual_factor : float
        Day count fraction τ for the period.
    forward_rate : float
        Current forward rate F for the period.
    discount_factor : float
        Discount factor DF(T_end) to payment date.
    direction : FRADirection
        "payer" (pay fixed, receive floating) or "receiver" (receive fixed).
    
    Examples
    --------
    3x6 FRA (3 months to fixing, 6 months to payment):
        >>> fra = ForwardRateAgreementSimple(
        ...     notional=10_000_000,
        ...     fixed_rate=0.05,       # 5% contract rate
        ...     fixing_time=0.25,      # 3 months
        ...     payment_time=0.5,      # 6 months
        ...     accrual_factor=0.25,   # Quarterly period
        ...     forward_rate=0.052,    # Current 5.2% forward
        ...     discount_factor=0.975,
        ...     direction="payer",
        ... )
    """
    notional: float
    fixed_rate: float           # K = contract rate
    fixing_time: float          # T_start
    payment_time: float         # T_end (also payment date for FRA-in-arrears)
    accrual_factor: float       # τ = day count fraction
    forward_rate: float         # F = current forward rate
    discount_factor: float      # DF(T_end)
    direction: FRADirection = "payer"
    
    def __post_init__(self) -> None:
        """Validate FRA inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.fixed_rate) < 0.0:
            raise ValueError("fixed_rate must be >= 0.")
        if float(self.fixing_time) < 0.0:
            raise ValueError("fixing_time must be >= 0.")
        if float(self.payment_time) <= float(self.fixing_time):
            raise ValueError("payment_time must be > fixing_time.")
        if float(self.accrual_factor) <= 0.0:
            raise ValueError("accrual_factor must be > 0.")
        if float(self.forward_rate) < 0.0:
            # Allow zero or slightly negative rates for modern IR markets
            pass
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")
        if self.direction not in ("payer", "receiver"):
            raise ValueError(f"direction must be 'payer' or 'receiver'; got {self.direction}")
    
    @property
    def par_rate(self) -> float:
        """
        Par rate (rate that makes PV = 0).
        
        The par rate equals the forward rate.
        """
        return float(self.forward_rate)
    
    @property
    def is_in_the_money(self) -> bool:
        """
        Check if FRA is in-the-money.
        
        For payer: ITM when F > K (rates have risen)
        For receiver: ITM when F < K (rates have fallen)
        """
        if self.direction == "payer":
            return float(self.forward_rate) > float(self.fixed_rate)
        return float(self.forward_rate) < float(self.fixed_rate)


# =============================================================================
# MARKET DATA INSTRUMENT (Lookup from Market)
# =============================================================================


@dataclass(frozen=True, slots=True)
class ForwardRateAgreement:
    """
    Forward Rate Agreement with market data lookup.
    
    Parameters
    ----------
    notional : float
        Notional principal amount.
    fixed_rate : float
        Agreed FRA rate K.
    fixing_time : float
        Time to rate fixing T_start (years).
    payment_time : float
        Time to payment T_end (years).
    day_count : DayCountConvention
        Day count convention for accrual factor.
    direction : FRADirection
        "payer" or "receiver".
    curve_id : MarketId
        Market identifier for the discount/forward curve.
    
    Examples
    --------
    3x6 FRA on USD SOFR:
        >>> fra = ForwardRateAgreement(
        ...     notional=10_000_000,
        ...     fixed_rate=0.05,
        ...     fixing_time=0.25,
        ...     payment_time=0.5,
        ...     day_count="ACT/360",
        ...     direction="payer",
        ...     curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
        ... )
    """
    notional: float
    fixed_rate: float
    fixing_time: float
    payment_time: float
    day_count: DayCountConvention = "ACT/360"
    direction: FRADirection = "payer"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate FRA inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.fixed_rate) < 0.0:
            raise ValueError("fixed_rate must be >= 0.")
        if float(self.fixing_time) < 0.0:
            raise ValueError("fixing_time must be >= 0.")
        if float(self.payment_time) <= float(self.fixing_time):
            raise ValueError("payment_time must be > fixing_time.")
        if self.day_count not in ("ACT/360", "ACT/365", "30/360"):
            raise ValueError(f"Invalid day_count: {self.day_count}")
        if self.direction not in ("payer", "receiver"):
            raise ValueError(f"direction must be 'payer' or 'receiver'; got {self.direction}")
    
    @property
    def tenor_description(self) -> str:
        """
        Human-readable tenor description.
        
        E.g., "3x6" for a FRA starting in 3 months, ending in 6 months.
        """
        start_months = int(round(float(self.fixing_time) * 12))
        end_months = int(round(float(self.payment_time) * 12))
        return f"{start_months}x{end_months}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "FRADirection",
    "ForwardRateAgreement",
    "ForwardRateAgreementSimple",
]
