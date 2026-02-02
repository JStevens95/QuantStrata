"""
Basket Option Instruments.

A basket option is an option on a weighted portfolio of assets:

    Payoff_call = max(Σ w_i S_i(T) - K, 0)
    Payoff_put  = max(K - Σ w_i S_i(T), 0)

This module defines the instrument structure. Pricing logic lives in pricers/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType


@dataclass(frozen=True, slots=True)
class MultiAssetBasketEuropeanOption:
    """
    European basket option (call/put).

    Payoff_call = max(Σ w_i S_i(T) - K, 0)
    Payoff_put  = max(K - Σ w_i S_i(T), 0)

    Attributes
    ----------
    option_type : OptionType
        "call" or "put".
    underlyings : tuple[MarketId, ...]
        Market IDs for underlying assets.
    weights : tuple[float, ...]
        Portfolio weights (typically sum to 1).
    strike : float
        Strike price.
    expiry : float
        Time to expiry in years.
    notional : float
        Notional amount.
    correlation_id : MarketId, optional
        Market ID for correlation matrix (if from market data).
    """

    option_type: OptionType
    underlyings: tuple[MarketId, ...]
    weights: tuple[float, ...]
    strike: float
    expiry: float
    notional: float = 1.0
    correlation_id: MarketId | None = None

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if len(self.underlyings) < 2:
            raise ValueError("Basket must have at least 2 underlyings.")
        if len(self.weights) != len(self.underlyings):
            raise ValueError("weights must have same length as underlyings.")
        if self.strike < 0:
            raise ValueError("strike must be non-negative.")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive.")

    @property
    def n_assets(self) -> int:
        """Number of underlying assets."""
        return len(self.underlyings)

    @classmethod
    def from_lists(
        cls,
        option_type: OptionType,
        underlyings: Sequence[MarketId],
        weights: Sequence[float],
        strike: float,
        expiry: float,
        notional: float = 1.0,
        correlation_id: MarketId | None = None,
    ) -> "MultiAssetBasketEuropeanOption":
        """Create from lists (convenience constructor)."""
        return cls(
            option_type=option_type,
            underlyings=tuple(underlyings),
            weights=tuple(weights),
            strike=strike,
            expiry=expiry,
            notional=notional,
            correlation_id=correlation_id,
        )
