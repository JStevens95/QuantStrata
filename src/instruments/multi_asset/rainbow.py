"""
Rainbow Option Instruments (Best-of, Worst-of).

Rainbow options have payoffs that depend on the ordering of multiple assets:

- Best-of Call:  max(max(S1, S2, ..., Sn) - K, 0)
- Best-of Put:   max(K - max(S1, S2, ..., Sn), 0)
- Worst-of Call: max(min(S1, S2, ..., Sn) - K, 0)
- Worst-of Put:  max(K - min(S1, S2, ..., Sn), 0)

This module defines the instrument structure. Pricing logic lives in pricers/.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from src.marketdata.core.ids import MarketId
from src.models.payoffs.types import OptionType


@dataclass(frozen=True, slots=True)
class MultiAssetBestOfEuropeanOption:
    """
    European best-of option (call/put).

    Payoff_call = max(max(S1, S2, ..., Sn) - K, 0)
    Payoff_put  = max(K - max(S1, S2, ..., Sn), 0)

    The holder receives an option on the best performing asset.

    Attributes
    ----------
    option_type : OptionType
        "call" or "put".
    underlyings : tuple[MarketId, ...]
        Market IDs for underlying assets.
    strike : float
        Strike price.
    expiry : float
        Time to expiry in years.
    notional : float
        Notional amount.
    correlation_id : MarketId, optional
        Market ID for correlation matrix.
    """

    option_type: OptionType
    underlyings: tuple[MarketId, ...]
    strike: float
    expiry: float
    notional: float = 1.0
    correlation_id: MarketId | None = None

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if len(self.underlyings) < 2:
            raise ValueError("Rainbow option must have at least 2 underlyings.")
        if self.strike < 0:
            raise ValueError("strike must be non-negative.")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive.")

    @property
    def n_assets(self) -> int:
        """Number of underlying assets."""
        return len(self.underlyings)

    @classmethod
    def from_list(
        cls,
        option_type: OptionType,
        underlyings: Sequence[MarketId],
        strike: float,
        expiry: float,
        notional: float = 1.0,
        correlation_id: MarketId | None = None,
    ) -> "MultiAssetBestOfEuropeanOption":
        """Create from list (convenience constructor)."""
        return cls(
            option_type=option_type,
            underlyings=tuple(underlyings),
            strike=strike,
            expiry=expiry,
            notional=notional,
            correlation_id=correlation_id,
        )


@dataclass(frozen=True, slots=True)
class MultiAssetWorstOfEuropeanOption:
    """
    European worst-of option (call/put).

    Payoff_call = max(min(S1, S2, ..., Sn) - K, 0)
    Payoff_put  = max(K - min(S1, S2, ..., Sn), 0)

    The holder receives an option on the worst performing asset.
    Often cheaper than vanilla calls due to the worst-of feature.

    Attributes
    ----------
    option_type : OptionType
        "call" or "put".
    underlyings : tuple[MarketId, ...]
        Market IDs for underlying assets.
    strike : float
        Strike price.
    expiry : float
        Time to expiry in years.
    notional : float
        Notional amount.
    correlation_id : MarketId, optional
        Market ID for correlation matrix.
    """

    option_type: OptionType
    underlyings: tuple[MarketId, ...]
    strike: float
    expiry: float
    notional: float = 1.0
    correlation_id: MarketId | None = None

    def __post_init__(self):
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if len(self.underlyings) < 2:
            raise ValueError("Rainbow option must have at least 2 underlyings.")
        if self.strike < 0:
            raise ValueError("strike must be non-negative.")
        if self.expiry <= 0:
            raise ValueError("expiry must be positive.")

    @property
    def n_assets(self) -> int:
        """Number of underlying assets."""
        return len(self.underlyings)

    @classmethod
    def from_list(
        cls,
        option_type: OptionType,
        underlyings: Sequence[MarketId],
        strike: float,
        expiry: float,
        notional: float = 1.0,
        correlation_id: MarketId | None = None,
    ) -> "MultiAssetWorstOfEuropeanOption":
        """Create from list (convenience constructor)."""
        return cls(
            option_type=option_type,
            underlyings=tuple(underlyings),
            strike=strike,
            expiry=expiry,
            notional=notional,
            correlation_id=correlation_id,
        )
