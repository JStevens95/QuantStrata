from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from src.marketdata.core.ids import MarketId


def _parse_iso_date(value: str) -> date:
    """Parse and validate ISO date string 'YYYY-MM-DD' into a date object."""
    raw = str(value).strip()
    if not raw:
        raise ValueError("Date must be a non-empty ISO string 'YYYY-MM-DD'.")
    try:
        return date.fromisoformat(raw)
    except Exception as exc:
        raise ValueError(f"Invalid ISO date {value!r}. Expected 'YYYY-MM-DD'.") from exc


@dataclass(frozen=True, slots=True)
class Universe:
    """
    Immutable set of MarketIds required for a workflow.

    Stored as a tuple for stable hashing and reproducibility.
    """
    ids: Tuple[MarketId, ...]

    def __init__(self, ids: Iterable[MarketId]) -> None:
        # Materialize to tuple immediately (stable order preserved).
        ids_tuple = tuple(ids)
        if not ids_tuple:
            raise ValueError("Universe.ids must not be empty.")

        # Ensure uniqueness by canonical key.
        keys = [mid.key() for mid in ids_tuple]
        if len(set(keys)) != len(keys):
            raise ValueError("Universe.ids must not contain duplicate MarketId keys.")

        object.__setattr__(self, "ids", ids_tuple)


@dataclass(frozen=True, slots=True)
class MarketRequest:
    """Request for a single as-of Market snapshot."""
    asof: str
    universe: Universe
    scenario: Optional[int] = None

    def __post_init__(self) -> None:
        # Normalize asof to ISO string.
        d = _parse_iso_date(self.asof)
        object.__setattr__(self, "asof", d.isoformat())

        # Scenario must be non-negative if provided.
        if self.scenario is not None:
            if not isinstance(self.scenario, int):
                raise TypeError("MarketRequest.scenario must be an int when provided.")
            if self.scenario < 0:
                raise ValueError("MarketRequest.scenario must be >= 0 when provided.")


@dataclass(frozen=True, slots=True)
class TimeseriesRequest:
    """Request for a MarketDataset over a date range."""
    start: str
    end: str
    freq: str
    universe: Universe
    scenarios: int = 1

    def __post_init__(self) -> None:
        # Parse/normalize start/end.
        d0 = _parse_iso_date(self.start)
        d1 = _parse_iso_date(self.end)

        if d1 < d0:
            raise ValueError(f"TimeseriesRequest.end must be >= start. Got start={d0}, end={d1}.")

        object.__setattr__(self, "start", d0.isoformat())
        object.__setattr__(self, "end", d1.isoformat())

        # Validate frequency string.
        f = str(self.freq).strip()
        if not f:
            raise ValueError("TimeseriesRequest.freq must not be empty.")
        object.__setattr__(self, "freq", f)

        # Validate scenarios count.
        if int(self.scenarios) < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")