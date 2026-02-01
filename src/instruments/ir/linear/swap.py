# src/instruments/ir/linear/swap.py
"""
Interest Rate Swap (IRS) Instrument.

Mathematical Framework
----------------------
An Interest Rate Swap is an agreement to exchange fixed rate payments for 
floating rate payments on a notional principal over multiple periods.

Fixed Leg PV (receiver perspective):
    PV_fixed = N × K × Σ[τ_i × DF(T_i)]

Floating Leg PV (receiver perspective):
    PV_float = N × Σ[τ_i × DF(T_i) × F_i]

Where F_i is the forward rate for period i.

Total Swap PV (fixed receiver):
    PV = PV_fixed - PV_float = N × Σ[τ_i × DF(T_i) × (K - F_i)]

Sign Convention
---------------
- **Receiver Swap (receive fixed, pay floating)**:
  - PV = N × Σ[τ_i × DF_i × (K - F_i)]
  - Benefits when rates fall
  
- **Payer Swap (pay fixed, receive floating)**:
  - PV = N × Σ[τ_i × DF_i × (F_i - K)]
  - Benefits when rates rise

Key Quantities
--------------
- **Annuity (PV01)**: A = Σ[τ_i × DF(T_i)]
  - Present value of 1bp paid on each payment date
  
- **Par Swap Rate**: K_par = Σ[τ_i × DF_i × F_i] / A
  - Fixed rate that makes swap PV = 0
  
- **DV01**: Change in PV for 1bp parallel shift
  - DV01 ≈ N × A × 0.0001 for ATM swap

Swap Types
----------
- Vanilla IRS: Fixed vs floating (e.g., 5% fixed vs SOFR)
- Basis Swap: Floating vs floating (e.g., SOFR vs Fed Funds)
- OIS: Fixed vs overnight rate compounded

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, List

from src.marketdata.core.ids import MarketId
from src.instruments.core.types import SwapDirection, DayCountConvention


# =============================================================================
# SWAP LEG DEFINITIONS
# =============================================================================


@dataclass(frozen=True, slots=True)
class SwapLeg:
    """
    Base class for a single swap leg cashflow.
    
    Represents one payment period in a swap leg.
    """
    start_time: float       # Period start T_start
    end_time: float         # Period end T_end (payment date)
    accrual_factor: float   # τ = day count fraction
    discount_factor: float  # DF(T_end)
    notional: float         # N (can vary for amortizing swaps)


@dataclass(frozen=True, slots=True)
class FixedLeg(SwapLeg):
    """
    Fixed leg cashflow.
    
    Cashflow = N × τ × K
    PV = N × τ × DF × K
    """
    fixed_rate: float  # K = fixed rate


@dataclass(frozen=True, slots=True)
class FloatingLeg(SwapLeg):
    """
    Floating leg cashflow.
    
    Cashflow = N × τ × L (where L is the floating rate)
    PV = N × τ × DF × F (where F is the forward rate)
    """
    forward_rate: float   # F = forward rate for the period
    spread: float = 0.0   # Optional spread over floating rate


# =============================================================================
# SIMPLE INSTRUMENT (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrSwapSimple:
    """
    Interest Rate Swap with direct parameter input.
    
    A vanilla IRS exchanges fixed rate payments for floating rate payments.
    
    Parameters
    ----------
    notional : float
        Notional principal (constant for vanilla swap).
    fixed_rate : float
        Fixed rate K (e.g., 0.05 for 5%).
    fixed_leg : Tuple[FixedLeg, ...]
        Pre-computed fixed leg cashflows.
    floating_leg : Tuple[FloatingLeg, ...]
        Pre-computed floating leg cashflows.
    direction : SwapDirection
        "payer" (pay fixed) or "receiver" (receive fixed).
    
    Examples
    --------
    2-year annual swap:
        >>> fixed_legs = (
        ...     FixedLeg(start_time=0, end_time=1, accrual_factor=1.0,
        ...              discount_factor=0.95, notional=1e6, fixed_rate=0.05),
        ...     FixedLeg(start_time=1, end_time=2, accrual_factor=1.0,
        ...              discount_factor=0.90, notional=1e6, fixed_rate=0.05),
        ... )
        >>> floating_legs = (
        ...     FloatingLeg(start_time=0, end_time=1, accrual_factor=1.0,
        ...                 discount_factor=0.95, notional=1e6, forward_rate=0.048),
        ...     FloatingLeg(start_time=1, end_time=2, accrual_factor=1.0,
        ...                 discount_factor=0.90, notional=1e6, forward_rate=0.052),
        ... )
        >>> swap = IrSwapSimple(
        ...     notional=1_000_000,
        ...     fixed_rate=0.05,
        ...     fixed_leg=fixed_legs,
        ...     floating_leg=floating_legs,
        ...     direction="receiver",
        ... )
    """
    notional: float
    fixed_rate: float
    fixed_leg: Tuple[FixedLeg, ...] = field(default_factory=tuple)
    floating_leg: Tuple[FloatingLeg, ...] = field(default_factory=tuple)
    direction: SwapDirection = "receiver"
    
    def __post_init__(self) -> None:
        """Validate swap inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if len(self.fixed_leg) == 0:
            raise ValueError("fixed_leg must have at least one period.")
        if len(self.floating_leg) == 0:
            raise ValueError("floating_leg must have at least one period.")
        if self.direction not in ("payer", "receiver"):
            raise ValueError(f"direction must be 'payer' or 'receiver'; got {self.direction}")
    
    @property
    def annuity(self) -> float:
        """
        Calculate annuity (PV01 factor).
        
        A = Σ[τ_i × DF_i]
        
        This is the present value of receiving 1 unit at each payment date.
        """
        return sum(leg.accrual_factor * leg.discount_factor for leg in self.fixed_leg)
    
    @property
    def par_rate(self) -> float:
        """
        Calculate par swap rate.
        
        K_par = Σ[τ_i × DF_i × F_i] / A
        
        This is the fixed rate that makes the swap PV = 0.
        """
        annuity = self.annuity
        if annuity == 0.0:
            return 0.0
        
        floating_pv = sum(
            leg.accrual_factor * leg.discount_factor * (leg.forward_rate + leg.spread)
            for leg in self.floating_leg
        )
        
        return floating_pv / annuity
    
    @property
    def dv01(self) -> float:
        """
        Approximate DV01 (dollar value of 1 basis point).
        
        DV01 ≈ N × A × 0.0001
        
        This is the approximate change in PV for a 1bp parallel shift.
        """
        return abs(float(self.notional)) * self.annuity * 0.0001


# =============================================================================
# MARKET DATA INSTRUMENT (Lookup from Market)
# =============================================================================


@dataclass(frozen=True, slots=True)
class IrSwap:
    """
    Interest Rate Swap with market data lookup.
    
    Parameters
    ----------
    notional : float
        Notional principal amount.
    fixed_rate : float
        Fixed rate K.
    start_time : float
        Swap start date (years from today).
    end_time : float
        Swap maturity date (years from today).
    fixed_frequency : float
        Fixed leg payment frequency (e.g., 0.5 for semi-annual).
    floating_frequency : float
        Floating leg reset frequency (e.g., 0.25 for quarterly).
    fixed_day_count : DayCountConvention
        Day count for fixed leg.
    floating_day_count : DayCountConvention
        Day count for floating leg.
    direction : SwapDirection
        "payer" or "receiver".
    curve_id : MarketId
        Market identifier for discount/forward curve.
    spread : float
        Spread over floating rate (default 0).
    
    Examples
    --------
    5-year USD swap, semi-annual fixed vs quarterly floating:
        >>> swap = IrSwap(
        ...     notional=10_000_000,
        ...     fixed_rate=0.05,
        ...     start_time=0.0,
        ...     end_time=5.0,
        ...     fixed_frequency=0.5,      # Semi-annual
        ...     floating_frequency=0.25,   # Quarterly
        ...     fixed_day_count="30/360",
        ...     floating_day_count="ACT/360",
        ...     direction="receiver",
        ...     curve_id=MarketId("IR", "CURVE", "USD.SOFR"),
        ... )
    """
    notional: float
    fixed_rate: float
    start_time: float
    end_time: float
    fixed_frequency: float = 0.5       # Semi-annual default
    floating_frequency: float = 0.25   # Quarterly default
    fixed_day_count: DayCountConvention = "30/360"
    floating_day_count: DayCountConvention = "ACT/360"
    direction: SwapDirection = "receiver"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    spread: float = 0.0
    
    def __post_init__(self) -> None:
        """Validate swap inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.start_time) < 0.0:
            raise ValueError("start_time must be >= 0.")
        if float(self.end_time) <= float(self.start_time):
            raise ValueError("end_time must be > start_time.")
        if float(self.fixed_frequency) <= 0.0:
            raise ValueError("fixed_frequency must be > 0.")
        if float(self.floating_frequency) <= 0.0:
            raise ValueError("floating_frequency must be > 0.")
        if self.direction not in ("payer", "receiver"):
            raise ValueError(f"direction must be 'payer' or 'receiver'; got {self.direction}")
    
    @property
    def tenor(self) -> float:
        """Swap tenor in years."""
        return float(self.end_time) - float(self.start_time)
    
    @property
    def tenor_description(self) -> str:
        """Human-readable tenor description (e.g., '5Y')."""
        tenor = self.tenor
        if tenor >= 1.0:
            return f"{int(tenor)}Y"
        return f"{int(tenor * 12)}M"


# =============================================================================
# SCHEDULE GENERATION UTILITIES
# =============================================================================


def generate_swap_schedule(
    start_time: float,
    end_time: float,
    frequency: float,
) -> List[Tuple[float, float]]:
    """
    Generate payment schedule for a swap leg.
    
    Parameters
    ----------
    start_time : float
        Schedule start (years).
    end_time : float
        Schedule end (years).
    frequency : float
        Payment frequency (years).
    
    Returns
    -------
    List[Tuple[float, float]]
        List of (period_start, period_end) tuples.
    """
    schedule = []
    t = float(start_time)
    freq = float(frequency)
    end = float(end_time)
    
    while t < end - 1e-9:
        t_start = t
        t_end = min(t + freq, end)
        schedule.append((t_start, t_end))
        t = t_end
    
    return schedule


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "SwapDirection",
    # Leg types
    "SwapLeg",
    "FixedLeg",
    "FloatingLeg",
    # Instruments
    "IrSwap",
    "IrSwapSimple",
    # Utilities
    "generate_swap_schedule",
]
