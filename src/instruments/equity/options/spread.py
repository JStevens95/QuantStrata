# src/instruments/equity/options/spread.py
"""
Equity Spread Option Instrument.

Mathematical Framework
----------------------
A spread option pays off based on the difference between two equity prices or indices.

Payoff:
    Call: max(S1 - S2 - K, 0)
    Put:  max(K - (S1 - S2), 0)

Where:
- S1 = first equity/index price
- S2 = second equity/index price  
- K = spread strike

Bachelier (Normal) Model
------------------------
Spread options are typically priced using Bachelier because:
1. The spread can be negative
2. Normal dynamics better capture relative value moves
3. Avoids complex correlation modeling

Under Bachelier:
    dSpread = σ_spread dW

Use Cases
---------
- Pairs trading (long one stock, short another)
- Index arbitrage (S&P 500 vs NASDAQ)
- Sector rotation strategies
- Relative value between markets

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.marketdata.core.ids import MarketId
from src.instruments.core.types import OptionType


# =============================================================================
# SIMPLE INSTRUMENT (Direct Parameters)
# =============================================================================


@dataclass(frozen=True, slots=True)
class EuropeanEquitySpreadOptionSimple:
    """
    European equity spread option with direct parameter input.
    
    A spread option on the difference between two equity prices.
    
    Parameters
    ----------
    notional : float
        Notional amount (or number of spread units).
    strike : float
        Spread strike K (can be positive, zero, or negative).
    expiry : float
        Time to expiry in years.
    forward_spread : float
        Forward value of (F1 - F2).
    vol : float
        Normal (Bachelier) volatility of the spread.
    discount_factor : float
        Discount factor to expiry.
    option_type : OptionType
        "call" or "put".
    
    Examples
    --------
    S&P 500 - NASDAQ 100 spread option:
        >>> spread_opt = EuropeanEquitySpreadOptionSimple(
        ...     notional=100,             # 100 spread units
        ...     strike=500,               # Strike spread level
        ...     expiry=0.25,              # 3 months
        ...     forward_spread=550,       # F_SPX - F_NDX
        ...     vol=50.0,                 # 50 points absolute vol
        ...     discount_factor=0.99,
        ...     option_type="call",
        ... )
    """
    notional: float
    strike: float              # K (spread strike)
    expiry: float              # T
    forward_spread: float      # F1 - F2 (forward spread)
    vol: float                 # σ_spread (normal vol)
    discount_factor: float     # DF
    option_type: OptionType = "call"
    
    def __post_init__(self) -> None:
        """Validate inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
        if float(self.vol) < 0.0:
            raise ValueError("vol must be >= 0.")
        if float(self.discount_factor) <= 0.0:
            raise ValueError("discount_factor must be > 0.")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put'; got {self.option_type}")
    
    @property
    def is_in_the_money(self) -> bool:
        """Check if option is ITM."""
        if self.option_type == "call":
            return float(self.forward_spread) > float(self.strike)
        return float(self.forward_spread) < float(self.strike)


# =============================================================================
# MARKET DATA INSTRUMENT (Lookup from Market)
# =============================================================================


@dataclass(frozen=True, slots=True)
class EuropeanEquitySpreadOption:
    """
    European equity spread option with market data lookup.
    
    Parameters
    ----------
    notional : float
        Notional amount.
    strike : float
        Spread strike K.
    expiry : float
        Time to expiry in years.
    underlying1 : str
        First underlying (e.g., "SPX").
    underlying2 : str
        Second underlying (e.g., "NDX").
    option_type : OptionType
        "call" or "put".
    curve_id : MarketId
        Market identifier for discount curve.
    spot1_id : MarketId
        Market identifier for first spot price.
    spot2_id : MarketId
        Market identifier for second spot price.
    vol_id : MarketId
        Market identifier for spread volatility.
    
    Examples
    --------
    S&P 500 vs NASDAQ spread:
        >>> spread = EuropeanEquitySpreadOption(
        ...     notional=100,
        ...     strike=500,
        ...     expiry=0.25,
        ...     underlying1="SPX",
        ...     underlying2="NDX",
        ...     option_type="call",
        ...     curve_id=MarketId("IR", "CURVE", "USD"),
        ...     spot1_id=MarketId("EQ", "SPOT", "SPX"),
        ...     spot2_id=MarketId("EQ", "SPOT", "NDX"),
        ...     vol_id=MarketId("EQ", "VOL", "SPX-NDX"),
        ... )
    """
    notional: float
    strike: float
    expiry: float
    underlying1: str
    underlying2: str
    option_type: OptionType = "call"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    spot1_id: MarketId = field(default_factory=lambda: MarketId("EQ", "SPOT", "UNKNOWN"))
    spot2_id: MarketId = field(default_factory=lambda: MarketId("EQ", "SPOT", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("EQ", "VOL", "UNKNOWN"))
    
    def __post_init__(self) -> None:
        """Validate inputs."""
        if float(self.notional) == 0.0:
            raise ValueError("notional must be non-zero.")
        if float(self.expiry) < 0.0:
            raise ValueError("expiry must be >= 0.")
        if self.option_type not in ("call", "put"):
            raise ValueError(f"option_type must be 'call' or 'put'")
    
    @property
    def spread_description(self) -> str:
        """Description of the spread (e.g., 'SPX - NDX')."""
        return f"{self.underlying1} - {self.underlying2}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EuropeanEquitySpreadOption",
    "EuropeanEquitySpreadOptionSimple",
]
