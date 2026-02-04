"""
Range Accrual Note instruments.

Range accrual notes pay interest only for days when an underlying
rate or index is within a specified range.

Example:
    from src.instruments.ir.options.range_accrual import IrRangeAccrualNote
    
    note = IrRangeAccrualNote(
        notional=1_000_000,
        start_date=date(2024, 1, 1),
        maturity_date=date(2025, 1, 1),
        range_lower=0.03,       # 3% lower bound
        range_upper=0.05,       # 5% upper bound
        accrual_rate=0.06,      # 6% when in range
        reference_rate_id="USD3M",
        observation_frequency="daily",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import List, Optional

from src.instruments.core.types import Currency


class ObservationFrequency(Enum):
    """Frequency of range observations."""
    
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


@dataclass
class IrRangeAccrualNote:
    """
    Interest Rate Range Accrual Note.
    
    Pays accrual_rate for each day the reference rate is within
    [range_lower, range_upper]. The coupon is the sum of all
    accrued days divided by total days in the period.
    
    Payoff formula:
        Coupon = notional * accrual_rate * (days_in_range / total_days)
    
    Attributes
    ----------
    notional : float
        Notional amount.
    start_date : date
        Start date.
    maturity_date : date
        Maturity date.
    range_lower : float
        Lower bound of the accrual range.
    range_upper : float
        Upper bound of the accrual range.
    accrual_rate : float
        Annual rate paid when in range.
    reference_rate_id : str
        Identifier for the reference rate (e.g., "USD3M", "EURIBOR6M").
    observation_frequency : ObservationFrequency
        Frequency of observations.
    day_count_convention : str
        Day count convention (e.g., "ACT/360", "30/360").
    currency : Currency
        Payment currency.
    
    Example
    -------
    >>> note = IrRangeAccrualNote(
    ...     notional=1_000_000,
    ...     start_date=date(2024, 1, 1),
    ...     maturity_date=date(2024, 7, 1),
    ...     range_lower=0.04,
    ...     range_upper=0.06,
    ...     accrual_rate=0.08,
    ...     reference_rate_id="USD3M",
    ... )
    """
    
    notional: float
    start_date: date
    maturity_date: date
    range_lower: float
    range_upper: float
    accrual_rate: float
    reference_rate_id: str
    observation_frequency: ObservationFrequency = ObservationFrequency.DAILY
    day_count_convention: str = "ACT/360"
    currency: Currency = Currency.USD
    
    def __post_init__(self) -> None:
        """Validate range accrual parameters."""
        if self.notional <= 0:
            raise ValueError(f"Notional must be positive, got {self.notional}")
        
        if self.range_lower >= self.range_upper:
            raise ValueError(
                f"Range lower ({self.range_lower}) must be less than "
                f"upper ({self.range_upper})"
            )
        
        if self.accrual_rate < 0:
            raise ValueError(f"Accrual rate cannot be negative, got {self.accrual_rate}")
        
        if self.maturity_date <= self.start_date:
            raise ValueError("Maturity date must be after start date")
        
        # Convert string to enum if needed
        if isinstance(self.observation_frequency, str):
            self.observation_frequency = ObservationFrequency(self.observation_frequency.lower())
    
    @property
    def time_to_maturity(self) -> float:
        """Time to maturity in years."""
        return (self.maturity_date - self.start_date).days / 365.0
    
    @property
    def n_observation_days(self) -> int:
        """Approximate number of observation days."""
        total_days = (self.maturity_date - self.start_date).days
        
        if self.observation_frequency == ObservationFrequency.DAILY:
            return total_days
        elif self.observation_frequency == ObservationFrequency.WEEKLY:
            return total_days // 7
        elif self.observation_frequency == ObservationFrequency.MONTHLY:
            return (self.maturity_date.year - self.start_date.year) * 12 + \
                   (self.maturity_date.month - self.start_date.month)
        return total_days
    
    def max_coupon(self) -> float:
        """Maximum possible coupon (100% in range)."""
        T = self.time_to_maturity
        return self.notional * self.accrual_rate * T


@dataclass
class FxRangeAccrualNote:
    """
    FX Range Accrual Note.
    
    Pays accrual when FX rate is within a specified range.
    
    Attributes
    ----------
    notional : float
        Notional amount.
    start_date : date
        Start date.
    maturity_date : date
        Maturity date.
    range_lower : float
        Lower FX rate bound.
    range_upper : float
        Upper FX rate bound.
    accrual_rate : float
        Annual rate when in range.
    base_currency : Currency
        Base currency of FX pair.
    quote_currency : Currency
        Quote currency of FX pair.
    settlement_currency : Currency
        Payment currency.
    """
    
    notional: float
    start_date: date
    maturity_date: date
    range_lower: float
    range_upper: float
    accrual_rate: float
    base_currency: Currency
    quote_currency: Currency
    settlement_currency: Currency = Currency.USD
    observation_frequency: ObservationFrequency = ObservationFrequency.DAILY
    
    def __post_init__(self) -> None:
        """Validate FX range accrual parameters."""
        if self.notional <= 0:
            raise ValueError("Notional must be positive")
        
        if self.range_lower >= self.range_upper:
            raise ValueError("Range lower must be less than upper")
        
        if self.base_currency == self.quote_currency:
            raise ValueError("Base and quote currencies must differ")
    
    @property
    def pair_id(self) -> str:
        """FX pair identifier."""
        return f"{self.base_currency.value}{self.quote_currency.value}"


__all__ = [
    "IrRangeAccrualNote",
    "FxRangeAccrualNote",
    "ObservationFrequency",
]
