"""
Autocallable Option instruments.

Autocallables are structured products that may terminate early (autocall)
if the underlying exceeds a barrier on observation dates. They typically
pay periodic coupons if a coupon barrier is breached, with downside
protection via a put barrier at maturity.

Example:
    from src.instruments.equity.options.autocallable import EquityAutocallableOption
    
    autocall = EquityAutocallableOption(
        underlying_id="SPY",
        notional=1_000_000,
        start_date=date(2024, 1, 1),
        maturity_date=date(2025, 1, 1),
        observation_dates=[date(2024, 4, 1), date(2024, 7, 1), ...],
        autocall_barrier=1.0,      # 100% of initial spot
        coupon_barrier=0.8,        # 80% of initial spot
        put_barrier=0.7,           # 70% of initial spot
        coupon_rate=0.10,          # 10% p.a.
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from src.instruments.core.types import Currency


@dataclass
class EquityAutocallableOption:
    """
    Equity Autocallable Option (Phoenix / Athena structure).
    
    An autocallable pays:
    - Early redemption at 100% + coupon if spot >= autocall_barrier on any observation
    - Coupon payments if spot >= coupon_barrier on observation dates
    - At maturity: 100% if spot >= put_barrier, else (spot / initial - 1) * notional
    
    Attributes
    ----------
    underlying_id : str
        Identifier for the underlying asset.
    notional : float
        Notional amount.
    start_date : date
        Trade start date (determines initial fixing).
    maturity_date : date
        Final maturity date.
    observation_dates : list of date
        Dates for autocall and coupon observations.
    autocall_barrier : float
        Autocall trigger level as fraction of initial (e.g., 1.0 = 100%).
    coupon_barrier : float
        Coupon payment trigger as fraction of initial.
    put_barrier : float
        Put knock-in barrier as fraction of initial.
    coupon_rate : float
        Annual coupon rate (e.g., 0.10 for 10%).
    memory_coupon : bool
        If True, missed coupons accumulate and pay on next trigger.
    currency : Currency
        Settlement currency.
    
    Example
    -------
    >>> autocall = EquityAutocallableOption(
    ...     underlying_id="EUROSTOXX50",
    ...     notional=100_000,
    ...     start_date=date(2024, 1, 15),
    ...     maturity_date=date(2027, 1, 15),
    ...     observation_dates=[date(2024, 7, 15), date(2025, 1, 15), ...],
    ...     autocall_barrier=1.0,
    ...     coupon_barrier=0.8,
    ...     put_barrier=0.6,
    ...     coupon_rate=0.12,
    ...     memory_coupon=True,
    ... )
    """
    
    underlying_id: str
    notional: float
    start_date: date
    maturity_date: date
    observation_dates: List[date]
    autocall_barrier: float = 1.0
    coupon_barrier: float = 0.8
    put_barrier: float = 0.6
    coupon_rate: float = 0.10
    memory_coupon: bool = True
    currency: Currency = Currency.USD
    
    def __post_init__(self) -> None:
        """Validate autocallable parameters."""
        if self.notional <= 0:
            raise ValueError(f"Notional must be positive, got {self.notional}")
        
        if self.maturity_date <= self.start_date:
            raise ValueError("Maturity date must be after start date")
        
        if not self.observation_dates:
            raise ValueError("At least one observation date is required")
        
        if self.coupon_rate < 0:
            raise ValueError(f"Coupon rate cannot be negative, got {self.coupon_rate}")
        
        # Validate barrier ordering
        if not (0 < self.put_barrier <= self.coupon_barrier <= self.autocall_barrier):
            raise ValueError(
                f"Barrier ordering must be: 0 < put_barrier <= coupon_barrier <= autocall_barrier, "
                f"got put={self.put_barrier}, coupon={self.coupon_barrier}, autocall={self.autocall_barrier}"
            )
        
        # Sort and validate observation dates
        self.observation_dates = sorted(self.observation_dates)
        
        if self.observation_dates[0] < self.start_date:
            raise ValueError("First observation cannot be before start date")
        
        if self.observation_dates[-1] > self.maturity_date:
            raise ValueError("Last observation cannot be after maturity")
    
    @property
    def n_observations(self) -> int:
        """Number of observation dates."""
        return len(self.observation_dates)
    
    @property
    def time_to_maturity(self) -> float:
        """Time to maturity in years."""
        return (self.maturity_date - self.start_date).days / 365.0
    
    def get_coupon_payment(self, period_fraction: float) -> float:
        """
        Get coupon payment for a period.
        
        Parameters
        ----------
        period_fraction : float
            Fraction of year for the period.
        
        Returns
        -------
        float
            Coupon payment amount.
        """
        return self.notional * self.coupon_rate * period_fraction


@dataclass
class AutocallablePhoenix(EquityAutocallableOption):
    """
    Phoenix Autocallable with guaranteed coupons.
    
    A Phoenix pays coupons regardless of performance as long as
    the put barrier is not breached.
    """
    
    guaranteed_coupon: bool = True


@dataclass
class AutocallableAthena(EquityAutocallableOption):
    """
    Athena Autocallable with no coupon payments.
    
    Pure autocall structure without periodic coupons.
    Final payoff is notional + accumulated coupon if autocalled,
    or put payoff at maturity.
    """
    
    coupon_rate: float = 0.0  # No periodic coupons


__all__ = [
    "EquityAutocallableOption",
    "AutocallablePhoenix",
    "AutocallableAthena",
]
