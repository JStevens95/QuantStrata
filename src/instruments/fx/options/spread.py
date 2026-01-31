# src/instruments/fx/options/spread.py
"""
FX Spread Option Instrument.

Mathematical Framework
----------------------
A spread option pays off based on the difference between two FX rates.

Payoff:
    Call: max(S1 - S2 - K, 0)
    Put:  max(K - (S1 - S2), 0)

Where:
- S1 = first FX rate (e.g., EUR/USD)
- S2 = second FX rate (e.g., GBP/USD)
- K = spread strike

Bachelier (Normal) Model
------------------------
Spread options are typically priced using Bachelier because:
1. The spread can be negative
2. The spread follows approximately normal dynamics
3. Simpler than correlation-dependent models

Under Bachelier:
    dSpread = σ_spread dW

Where σ_spread depends on the volatilities and correlation of the two rates.

Use Cases
---------
- Cross-currency basis trading
- Relative value trades between currency pairs
- Hedging cross-rate exposures

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
class EuropeanFxSpreadOptionSimple:
    """
    European FX spread option with direct parameter input.
    
    A spread option on the difference between two FX rates.
    
    Parameters
    ----------
    notional : float
        Notional amount.
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
    EUR/USD - GBP/USD spread option:
        >>> spread_opt = EuropeanFxSpreadOptionSimple(
        ...     notional=1_000_000,
        ...     strike=0.10,              # Spread strike
        ...     expiry=0.5,               # 6 months
        ...     forward_spread=0.12,      # F_EUR - F_GBP
        ...     vol=0.05,                 # 5% absolute spread vol
        ...     discount_factor=0.975,
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
class EuropeanFxSpreadOption:
    """
    European FX spread option with market data lookup.
    
    Parameters
    ----------
    notional : float
        Notional amount.
    strike : float
        Spread strike K.
    expiry : float
        Time to expiry in years.
    pair1 : str
        First currency pair (e.g., "EURUSD").
    pair2 : str
        Second currency pair (e.g., "GBPUSD").
    option_type : OptionType
        "call" or "put".
    curve_id : MarketId
        Market identifier for discount curve.
    spot1_id : MarketId
        Market identifier for first spot rate.
    spot2_id : MarketId
        Market identifier for second spot rate.
    vol_id : MarketId
        Market identifier for spread volatility.
    
    Examples
    --------
    EUR/USD - GBP/USD spread call:
        >>> spread = EuropeanFxSpreadOption(
        ...     notional=1_000_000,
        ...     strike=0.10,
        ...     expiry=0.5,
        ...     pair1="EURUSD",
        ...     pair2="GBPUSD",
        ...     option_type="call",
        ...     curve_id=MarketId("IR", "CURVE", "USD"),
        ...     spot1_id=MarketId("FX", "SPOT", "EURUSD"),
        ...     spot2_id=MarketId("FX", "SPOT", "GBPUSD"),
        ...     vol_id=MarketId("FX", "VOL", "EURUSD-GBPUSD"),
        ... )
    """
    notional: float
    strike: float
    expiry: float
    pair1: str
    pair2: str
    option_type: OptionType = "call"
    curve_id: MarketId = field(default_factory=lambda: MarketId("IR", "CURVE", "UNKNOWN"))
    spot1_id: MarketId = field(default_factory=lambda: MarketId("FX", "SPOT", "UNKNOWN"))
    spot2_id: MarketId = field(default_factory=lambda: MarketId("FX", "SPOT", "UNKNOWN"))
    vol_id: MarketId = field(default_factory=lambda: MarketId("FX", "VOL", "UNKNOWN"))
    
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
        """Description of the spread (e.g., 'EURUSD - GBPUSD')."""
        return f"{self.pair1} - {self.pair2}"


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EuropeanFxSpreadOption",
    "EuropeanFxSpreadOptionSimple",
]
