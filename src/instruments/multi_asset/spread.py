"""
Spread Option Instruments.

A spread option is an option on the difference between two asset prices:

    Payoff_call = max(S1(T) - S2(T) - K, 0)
    Payoff_put  = max(K - (S1(T) - S2(T)), 0)

This module defines the instrument structure. Pricing logic lives in pricers/.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType


@dataclass(frozen=True, slots=True)
class MultiAssetSpreadEuropeanOption:
    """
    European spread option (call/put).

    Payoff_call = max(S1(T) - S2(T) - K, 0)
    Payoff_put  = max(K - (S1(T) - S2(T)), 0)

    Attributes
    ----------
    option_type : OptionType
        "call" or "put".
    underlying1 : MarketId
        Market ID for first underlying (long leg).
    underlying2 : MarketId
        Market ID for second underlying (short leg).
    strike : float
        Strike price (can be 0 for exchange option).
    expiry : float
        Time to expiry in years.
    notional : float
        Notional amount.
    """

    option_type: OptionType
    underlying1: MarketId
    underlying2: MarketId
    strike: float
    expiry: float
    notional: float = 1.0

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive.")


@dataclass(frozen=True, slots=True)
class MultiAssetExchangeEuropeanOption:
    """
    Exchange option (spread with K=0).

    Payoff = max(S1(T) - S2(T), 0)

    This is a special case with an exact closed-form solution (Margrabe's formula).

    Attributes
    ----------
    underlying1 : MarketId
        Market ID for first underlying (receive leg).
    underlying2 : MarketId
        Market ID for second underlying (deliver leg).
    expiry : float
        Time to expiry in years.
    notional : float
        Notional amount.
    """

    underlying1: MarketId
    underlying2: MarketId
    expiry: float
    notional: float = 1.0

    def __post_init__(self):
        if self.expiry <= 0:
            raise ValueError("expiry must be positive.")
