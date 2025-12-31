from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True, slots=True)
class Position:
    """
    A position is the atomic unit of a pricing portfolio.

    It wraps:
      - an instrument (any dataclass/object your library defines, e.g. EuropeanFxOption)
      - a quantity (can be negative for shorts)
      - optional metadata (book, tags, strategy, etc.)

    Notes
    -----
    - We keep Position generic: no assumptions about asset class or payoff.
    - Instrument typing is intentionally Any to allow mixed books.
      Routing happens in the PortfolioPricer via a pricer registry keyed by instrument type.

    Routing
    -------
    - `pricer_id` lets you select a non-default pricer for this position
      (e.g. "mc" instead of "default").
    - If pricer_id is None, the PortfolioPricer can still apply a global override
      (e.g. price the entire portfolio with "mc")
    """
    position_id: str
    instrument: Any
    quantity: float = 1.0
    pricer_id: Optional[str] = None
    metadata: Optional[Mapping[str, Any]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.position_id, str) or not self.position_id.strip():
            raise ValueError("position_id must be a non-empty string.")
        if not isinstance(self.quantity, (int, float)):
            raise TypeError("quantity must be a number.")
        if not math.isfinite(float(self.quantity)):
            raise ValueError("quantity must be finite (not NaN/inf).")
        if self.pricer_id is not None and (not isinstance(self.pricer_id, str) or not self.pricer_id.strip()):
            raise ValueError("pricer_id must be None or a non-empty string.")



@dataclass(frozen=True, slots=True)
class Portfolio:
    """
    A portfolio is simply a collection of positions.

    Design
    ------
    - We enforce unique position_id values to make reporting and debugging deterministic.
    - We do not embed pricing logic here; pricing lives in PortfolioPricer.
    """
    positions: Sequence[Position]

    def __post_init__(self) -> None:
        _validate_unique_position_ids(self.positions)

    def __iter__(self):
        return iter(self.positions)

    def __len__(self) -> int:
        return len(self.positions)


@dataclass(frozen=True, slots=True)
class PositionResult:
    """
    Pricing result for one position.
    """
    position_id: str
    instrument_type: str
    quantity: float
    pv: float
    greeks: Dict[str, float]


@dataclass(frozen=True, slots=True)
class PortfolioTotals:
    """
    Aggregated totals for the whole portfolio.
    """
    pv: float
    greeks: Dict[str, float]


@dataclass(frozen=True, slots=True)
class PortfolioResult:
    """
    Full portfolio pricing output.

    - per_position: list of PositionResult (one per input Position)
    - totals: aggregated PV and Greeks
    """
    per_position: List[PositionResult]
    totals: PortfolioTotals


def _validate_unique_position_ids(positions: Sequence[Position]) -> None:
    """
    Ensure position_id values are unique.

    This avoids silent overwrites and improves traceability in tests/reports.
    """
    seen: set[str] = set()
    duplicates: List[str] = []

    for p in positions:
        if p.position_id in seen:
            duplicates.append(p.position_id)
        seen.add(p.position_id)

    if duplicates:
        dup_str = ", ".join(sorted(set(duplicates)))
        raise ValueError(f"Duplicate position_id values found: {dup_str}")