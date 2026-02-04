"""
Cliquet (Ratchet) Option instruments.

Cliquet options are path-dependent options that pay based on the sum of
capped and floored periodic returns. They are commonly used in equity-linked
structured notes and guaranteed return products.

Example:
    from src.instruments.equity.options.cliquet import EquityCliquetOption
    
    # 1-year cliquet with monthly resets
    cliquet = EquityCliquetOption(
        underlying_id="SPY",
        notional=1_000_000,
        start_date=date(2024, 1, 1),
        end_date=date(2025, 1, 1),
        reset_dates=[date(2024, i, 1) for i in range(1, 13)],
        local_cap=0.03,      # 3% per period max
        local_floor=-0.01,   # -1% per period min
        global_cap=0.20,     # 20% total max
        global_floor=0.0,    # 0% total min (principal protected)
        participation=1.0,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

from src.instruments.core.types import Currency


@dataclass
class EquityCliquetOption:
    """
    Equity Cliquet (Ratchet) Option.
    
    A cliquet option pays based on the sum of capped and floored periodic
    returns of the underlying asset. At each reset date, the return is
    calculated, capped/floored locally, and accumulated. At maturity,
    the total return is capped/floored globally and multiplied by the
    participation rate to determine the payoff.
    
    Payoff formula:
        Local return[i] = clamp(S[i]/S[i-1] - 1, local_floor, local_cap)
        Global return = clamp(sum(Local returns), global_floor, global_cap)
        Payoff = notional * participation * max(global_return, 0)
    
    Attributes
    ----------
    underlying_id : str
        Identifier for the underlying asset (e.g., "SPY", "EUROSTOXX50").
    notional : float
        Notional amount in the settlement currency.
    start_date : date
        Start date of the cliquet structure.
    end_date : date
        Maturity date.
    reset_dates : list of date
        Dates at which returns are measured and locked in.
        First reset uses start_date as reference.
    local_cap : float
        Maximum return per period (e.g., 0.03 for 3%).
    local_floor : float
        Minimum return per period (e.g., -0.01 for -1%).
    global_cap : float, optional
        Maximum total accumulated return. None means no cap.
    global_floor : float
        Minimum total accumulated return (e.g., 0 for principal protection).
    participation : float
        Participation rate applied to the final return (e.g., 1.0 for 100%).
    currency : Currency
        Settlement currency.
    
    Example
    -------
    >>> from datetime import date
    >>> cliquet = EquityCliquetOption(
    ...     underlying_id="SPY",
    ...     notional=1_000_000,
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2025, 1, 1),
    ...     reset_dates=[date(2024, 2, 1), date(2024, 3, 1), ...],
    ...     local_cap=0.03,
    ...     local_floor=-0.01,
    ...     global_floor=0.0,
    ...     participation=1.0,
    ... )
    """
    
    underlying_id: str
    notional: float
    start_date: date
    end_date: date
    reset_dates: List[date]
    local_cap: float
    local_floor: float
    global_cap: Optional[float] = None
    global_floor: float = 0.0
    participation: float = 1.0
    currency: Currency = Currency.USD
    
    def __post_init__(self) -> None:
        """Validate cliquet parameters."""
        if self.notional <= 0:
            raise ValueError(f"Notional must be positive, got {self.notional}")
        
        if self.local_floor > self.local_cap:
            raise ValueError(
                f"Local floor ({self.local_floor}) cannot exceed "
                f"local cap ({self.local_cap})"
            )
        
        if self.global_cap is not None and self.global_floor > self.global_cap:
            raise ValueError(
                f"Global floor ({self.global_floor}) cannot exceed "
                f"global cap ({self.global_cap})"
            )
        
        if self.participation <= 0:
            raise ValueError(f"Participation must be positive, got {self.participation}")
        
        if self.end_date <= self.start_date:
            raise ValueError("End date must be after start date")
        
        if not self.reset_dates:
            raise ValueError("At least one reset date is required")
        
        # Ensure reset dates are sorted
        self.reset_dates = sorted(self.reset_dates)
        
        # Validate reset dates are within bounds
        if self.reset_dates[0] < self.start_date:
            raise ValueError("First reset date cannot be before start date")
        
        if self.reset_dates[-1] > self.end_date:
            raise ValueError("Last reset date cannot be after end date")
    
    @property
    def n_periods(self) -> int:
        """Number of reset periods."""
        return len(self.reset_dates)
    
    @property
    def time_to_maturity(self) -> float:
        """Time to maturity in years (approximate)."""
        return (self.end_date - self.start_date).days / 365.0
    
    def get_observation_dates(self) -> List[date]:
        """Get all observation dates including start."""
        return [self.start_date] + self.reset_dates


@dataclass
class FxCliquetOption:
    """
    FX Cliquet Option.
    
    Similar to equity cliquet but on FX rate returns.
    
    Attributes
    ----------
    base_currency : Currency
        Base currency of the FX pair.
    quote_currency : Currency
        Quote currency of the FX pair.
    notional : float
        Notional in settlement currency.
    start_date : date
        Start date.
    end_date : date
        Maturity date.
    reset_dates : list of date
        Dates at which FX returns are locked.
    local_cap : float
        Maximum return per period.
    local_floor : float
        Minimum return per period.
    global_cap : float, optional
        Maximum total return.
    global_floor : float
        Minimum total return.
    participation : float
        Participation rate.
    settlement_currency : Currency
        Currency for settlement.
    """
    
    base_currency: Currency
    quote_currency: Currency
    notional: float
    start_date: date
    end_date: date
    reset_dates: List[date]
    local_cap: float
    local_floor: float
    global_cap: Optional[float] = None
    global_floor: float = 0.0
    participation: float = 1.0
    settlement_currency: Currency = Currency.USD
    
    def __post_init__(self) -> None:
        """Validate FX cliquet parameters."""
        if self.notional <= 0:
            raise ValueError(f"Notional must be positive, got {self.notional}")
        
        if self.local_floor > self.local_cap:
            raise ValueError("Local floor cannot exceed local cap")
        
        if self.base_currency == self.quote_currency:
            raise ValueError("Base and quote currencies must be different")
        
        self.reset_dates = sorted(self.reset_dates)
    
    @property
    def pair_id(self) -> str:
        """FX pair identifier."""
        return f"{self.base_currency.value}{self.quote_currency.value}"
    
    @property
    def n_periods(self) -> int:
        """Number of reset periods."""
        return len(self.reset_dates)


__all__ = [
    "EquityCliquetOption",
    "FxCliquetOption",
]
