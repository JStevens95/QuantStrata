# src/instruments/ir/options/capfloor.py
"""
Interest Rate Cap and Floor Instruments.

Mathematical Framework
----------------------
A caplet is a call option on a forward rate with payoff:

    Caplet Payoff = N × τ × max(L - K, 0)

Paid at T_end, where:
- N = notional principal
- τ = day count fraction for the period [T_start, T_end]
- L = LIBOR/SOFR rate fixing at T_start
- K = cap strike rate

A floorlet is a put option on a forward rate:

    Floorlet Payoff = N × τ × max(K - L, 0)

A cap is a portfolio of caplets, one for each reset period.
A floor is a portfolio of floorlets.

Black76 Pricing
---------------
Using Black76 model with forward rate F:

    Caplet PV = N × τ × DF(T_end) × [F × N(d₁) - K × N(d₂)]

Where:
- DF(T_end) = discount factor to payment date
- F = forward rate for [T_start, T_end]
- d₁ = [ln(F/K) + σ²T/2] / (σ√T)
- d₂ = d₁ - σ√T
- T = time to rate fixing (T_start)

Put-Call Parity (Cap vs Floor)
------------------------------
Cap - Floor = Swap (fixed payer)

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple, Literal

from src.marketdata.core.ids import MarketId
from src.instruments.core.types import DayCountConvention


# =============================================================================
# SIMPLE INSTRUMENTS (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class CapletSimple:
    """
    Single caplet with direct parameter input.
    
    A caplet is a call option on a forward interest rate. The payoff at 
    T_end is: N × τ × max(L - K, 0)
    
    This simple version takes all pricing parameters directly, useful
    for unit testing or when forward rate is directly observable.
    
    Parameters
    ----------
    notional : float
        Notional principal amount.
    strike : float
        Cap strike rate (e.g., 0.05 for 5%).
    fixing_time : float
        Time to rate fixing T_start (years).
    payment_time : float
        Time to payment T_end (years).
    accrual_factor : float
        Day count fraction τ for the period.
    forward_rate : float
        Forward rate F for the period.
    vol : float
        Black76 volatility (normal or log-normal depending on pricer).
    discount_factor : float
        Discount factor DF(T_end) to payment date.
    """
    notional: float
    strike: float
    fixing_time: float          # T_start - time to fixing
    payment_time: float         # T_end - time to payment
    accrual_factor: float       # τ = day count fraction
    forward_rate: float         # F = forward rate
    vol: float                  # σ = Black76 vol
    discount_factor: float      # DF(T_end)
    
    def __post_init__(self) -> None:
        """Validate caplet inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.fixing_time) < 0.0:
            raise ValueError("fixing_time must be >= 0.")
        if float(self.payment_time) <= float(self.fixing_time):
            raise ValueError("payment_time must be > fixing_time.")
        if float(self.accrual_factor) <= 0.0:
            raise ValueError("accrual_factor must be > 0.")
        if float(self.forward_rate) <= 0.0:
            raise ValueError("forward_rate must be > 0.")
        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")


@dataclass(frozen=True, slots=True)
class FloorletSimple:
    """
    Single floorlet with direct parameter input.
    
    A floorlet is a put option on a forward interest rate. The payoff at 
    T_end is: N × τ × max(K - L, 0)
    
    Parameters
    ----------
    (same as CapletSimple)
    """
    notional: float
    strike: float
    fixing_time: float
    payment_time: float
    accrual_factor: float
    forward_rate: float
    vol: float
    discount_factor: float
    
    def __post_init__(self) -> None:
        """Validate floorlet inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.fixing_time) < 0.0:
            raise ValueError("fixing_time must be >= 0.")
        if float(self.payment_time) <= float(self.fixing_time):
            raise ValueError("payment_time must be > fixing_time.")
        if float(self.accrual_factor) <= 0.0:
            raise ValueError("accrual_factor must be > 0.")
        if float(self.forward_rate) <= 0.0:
            raise ValueError("forward_rate must be > 0.")
        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")


@dataclass(frozen=True, slots=True)
class CapSimple:
    """
    Interest rate cap with direct parameter input.
    
    A cap is a portfolio of caplets. This simple version takes a list
    of pre-computed caplet parameters.
    
    Parameters
    ----------
    notional : float
        Notional principal amount (same for all caplets).
    strike : float
        Cap strike rate (same for all caplets).
    caplets : Tuple[CapletSimple, ...]
        Tuple of CapletSimple objects, one per reset period.
    """
    notional: float
    strike: float
    caplets: Tuple[CapletSimple, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        """Validate cap inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if len(self.caplets) == 0:
            raise ValueError("cap must have at least one caplet.")


@dataclass(frozen=True, slots=True)
class FloorSimple:
    """
    Interest rate floor with direct parameter input.
    
    A floor is a portfolio of floorlets.
    
    Parameters
    ----------
    notional : float
        Notional principal amount (same for all floorlets).
    strike : float
        Floor strike rate (same for all floorlets).
    floorlets : Tuple[FloorletSimple, ...]
        Tuple of FloorletSimple objects, one per reset period.
    """
    notional: float
    strike: float
    floorlets: Tuple[FloorletSimple, ...] = field(default_factory=tuple)
    
    def __post_init__(self) -> None:
        """Validate floor inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if len(self.floorlets) == 0:
            raise ValueError("floor must have at least one floorlet.")


# =============================================================================
# MARKET DATA INSTRUMENTS (Lookup from Market)
# =============================================================================


@dataclass(frozen=True, slots=True)
class Caplet:
    """
    Single caplet with market data lookup.
    
    Parameters
    ----------
    notional : float
        Notional principal amount.
    strike : float
        Cap strike rate.
    fixing_time : float
        Time to rate fixing T_start (years).
    payment_time : float
        Time to payment T_end (years).
    day_count : DayCountConvention
        Day count convention for accrual factor calculation.
    curve_id : MarketId
        Market identifier for the discount/forward curve.
    vol_id : MarketId
        Market identifier for the cap/floor volatility surface.
    """
    notional: float
    strike: float
    fixing_time: float
    payment_time: float
    day_count: DayCountConvention = "ACT/360"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("IR", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate caplet inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.fixing_time) < 0.0:
            raise ValueError("fixing_time must be >= 0.")
        if float(self.payment_time) <= float(self.fixing_time):
            raise ValueError("payment_time must be > fixing_time.")
        if self.day_count not in ("ACT/360", "ACT/365", "30/360"):
            raise ValueError(f"Invalid day_count: {self.day_count}")


@dataclass(frozen=True, slots=True)
class Floorlet:
    """
    Single floorlet with market data lookup.
    
    Parameters
    ----------
    (same as Caplet)
    """
    notional: float
    strike: float
    fixing_time: float
    payment_time: float
    day_count: DayCountConvention = "ACT/360"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("IR", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate floorlet inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.fixing_time) < 0.0:
            raise ValueError("fixing_time must be >= 0.")
        if float(self.payment_time) <= float(self.fixing_time):
            raise ValueError("payment_time must be > fixing_time.")
        if self.day_count not in ("ACT/360", "ACT/365", "30/360"):
            raise ValueError(f"Invalid day_count: {self.day_count}")


@dataclass(frozen=True, slots=True)
class Cap:
    """
    Interest rate cap with market data lookup.
    
    A cap protects against rising interest rates by paying the holder
    when the floating rate exceeds the strike.
    
    Parameters
    ----------
    notional : float
        Notional principal amount.
    strike : float
        Cap strike rate (e.g., 0.05 for 5%).
    start_time : float
        Time to first fixing (years).
    end_time : float
        Time to last payment (years).
    frequency : float
        Reset frequency (e.g., 0.25 for quarterly, 0.5 for semi-annual).
    day_count : DayCountConvention
        Day count convention.
    curve_id : MarketId
        Market identifier for the discount/forward curve.
    vol_id : MarketId
        Market identifier for the cap volatility surface.
    """
    notional: float
    strike: float
    start_time: float
    end_time: float
    frequency: float = 0.25  # Quarterly by default
    day_count: DayCountConvention = "ACT/360"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("IR", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate cap inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.start_time) < 0.0:
            raise ValueError("start_time must be >= 0.")
        if float(self.end_time) <= float(self.start_time):
            raise ValueError("end_time must be > start_time.")
        if float(self.frequency) <= 0.0:
            raise ValueError("frequency must be > 0.")
        if self.day_count not in ("ACT/360", "ACT/365", "30/360"):
            raise ValueError(f"Invalid day_count: {self.day_count}")


@dataclass(frozen=True, slots=True)
class Floor:
    """
    Interest rate floor with market data lookup.
    
    A floor protects against falling interest rates by paying the holder
    when the floating rate falls below the strike.
    
    Parameters
    ----------
    (same as Cap)
    """
    notional: float
    strike: float
    start_time: float
    end_time: float
    frequency: float = 0.25
    day_count: DayCountConvention = "ACT/360"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("IR", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate floor inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.strike) <= 0.0:
            raise ValueError("strike must be > 0.")
        if float(self.start_time) < 0.0:
            raise ValueError("start_time must be >= 0.")
        if float(self.end_time) <= float(self.start_time):
            raise ValueError("end_time must be > start_time.")
        if float(self.frequency) <= 0.0:
            raise ValueError("frequency must be > 0.")
        if self.day_count not in ("ACT/360", "ACT/365", "30/360"):
            raise ValueError(f"Invalid day_count: {self.day_count}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_accrual_factor(
    start_time: float,
    end_time: float,
    day_count: DayCountConvention = "ACT/360",
) -> float:
    """
    Compute accrual factor (day count fraction) for a period.
    
    Simplified version using year fractions directly.
    For production use, this should use actual dates and proper day counting.
    
    Parameters
    ----------
    start_time : float
        Start of period (years from today).
    end_time : float
        End of period (years from today).
    day_count : DayCountConvention
        Day count convention.
    
    Returns
    -------
    float
        Accrual factor τ.
    """
    period = float(end_time) - float(start_time)
    
    if day_count == "ACT/360":
        # Simplified: assume period in years, scale by 360/365
        return period * (365.0 / 360.0)
    elif day_count == "ACT/365":
        return period
    elif day_count == "30/360":
        # Simplified: same as ACT/365 for year fractions
        return period
    else:
        raise ValueError(f"Unknown day_count: {day_count}")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Types
    "DayCountConvention",
    # Simple instruments
    "CapletSimple",
    "FloorletSimple",
    "CapSimple",
    "FloorSimple",
    # Market data instruments
    "Caplet",
    "Floorlet",
    "Cap",
    "Floor",
    # Utilities
    "compute_accrual_factor",
]
