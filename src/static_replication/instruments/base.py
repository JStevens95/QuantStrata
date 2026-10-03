"""
Base instrument abstractions.

All concrete instrument types (FX, IR) inherit from InstrumentSpec. This gives the trade generator, pricer and pnl
engine a uniform interface regardless of asset class.

Design:
    - to_pricer_params() --> flat dict of numeric inputs for vectorised pricing kernels.
    - to_trade_id) --> deterministic globally unique ID using make_trade_id()
    - to_elementary_trade() --> convert to ElementaryTrade for pipeline handoff.
"""
from __future__ import annotations

from typing import Any, Dict, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.static_replication.core.types import ElementaryTrade


@dataclass
class InstrumentSpec(ABC):
    """
    Abstract base for all instrument specifications.

    Subclasses define the concrete fields for their asset class while inheriting the common interface for pricing and
    trade ID generation.
    """

    # define parameters.
    asset_class: str
    payoff_type: str
    factor_id: str
    product_type: str
    product_subtype: str

    @abstractmethod
    def to_pricer_params(self) -> Dict[str, Any]:
        """
        Extract the parameter dict that the pricer needs.

        Returns a flat dictionary of numeric values for vectorised batch pricing.
        """
        ...

    @abstractmethod
    def to_trade_id(self) -> str:
        """
        Generate a deterministic, globally unique ID for this trade.

        Delegates to make_trade_ids() from core/registry.py
        """
        ...

    @abstractmethod
    def to_elementary_trade(self) -> ElementaryTrade:
        """
        Convert this instrument specification into a typed ElementaryTrade.

        This is a handoff object that crosses stage boundaries in the pipeline.
        """
        ...


@dataclass
class StrikeGrid:
    """
    Define a grid of strikes for option generation.

    Conventions:
        - absolute: raw strike values e.g. [1.05, 1.10, 1.15]
        - relative: fraction of spot e.g. [0.90, 1.00, 1.10]
        - delta: delta-based e.g. [0.10, 0.25, 0.50, 0,75, 0.90]
    """

    # define parameters.
    values: List[float] = field(default_factory=list)
    conventions: str = "absolute"
    reference_level: float = 1.0


@dataclass
class TenorGrid:
    """
    User-defined grid of tenors/expirires/maturities for option generation.

    Values are year fractions e.g. [0.083, 0.25, 0.5, 1.0]
    """

    # define parameters.
    values: List[float] = field(default_factory=list)
    convention: str = "year_fraction"