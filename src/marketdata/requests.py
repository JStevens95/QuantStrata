from __future__ import annotations

from datetime import date
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

from src.marketdata.ids import MarketId


def _parse_iso_date(value: str) -> str:
    """
    Validate and normalize ISO date strings.

    We store dates as ISO strings ('YYYY-MM-DD') for:
      - JSON friendliness
      - stable hashing/caching
      - minimal dependency footprint

    The validation ensures callers don't accidentally pass non-ISO formats.
    """
    try:
        d = date.fromisoformat(value)
    except Exception as exc:
        raise ValueError(f"Invalid ISO date '{value}'. Expected 'YYYY-MM-DD', got '{value}'.") from exc
    return d.isoformat()


@dataclass(frozen=True, slots=True)
class Universe:
    """
    Declares the set of MarketIds required for a workflow.

    This is the central mechanism for:
    - provider requests
    - caching keys
    - orchestrator step reproducibility

    Notes
    -----
    We store IDs as a tuple for immutability and stable hashing.
    """
    # initiate required variables.
    ids: Tuple[MarketId, ...]

    def __init__(self, ids: Iterable[MarketId]) -> None:
        # convert to a tuple immediately (stable order as provided).
        ids_tuple = tuple(ids)
        if not ids_tuple:
            raise ValueError("Universe.ids must not be empty.")

        # ensure uniqueness by canonical key to avoid subtle duplicates.
        keys = [mid.key() for mid in ids_tuple]
        if len(set(keys)) != len(keys):
            raise ValueError("Universe.ids must not contain duplicate MarketId keys.")

        # set ids in object
        object.__setattr__(self, "ids", ids_tuple)


@dataclass(frozen=True, slots=True)
class MarketRequest:
    """
    Request for a single as-of Market snapshot.

    Parameters
    ----------
    asof:
        ISO date string 'YYYY-MM-DD'
    universe:
        MarketIds to load
    scenario:
        Optional scenario index if the provider supports scenario-indexed snapshots.
        For simple providers, this can be ignored or require scenario=None/0.
    """
    # initiate required variables.
    asof: str
    universe: Universe
    scenario: Optional[int] = None

    def __post_init__(self) -> None:
        """Post initialization."""
        object.__setattr__(self, "asof", _parse_iso_date(self.asof))
        if self.scenario is not None and self.scenario < 0:
            raise ValueError("MarketRequest.scenario must be >= 0 when provided.")


@dataclass(frozen=True, slots=True)
class TimeseriesRequest:
    """
    Request for a MarketDataset over a date range (and optionally scenarios).

    Parameters
    ----------
    start, end:
        ISO date strings, inclusive bounds interpretation is provider-defined.
    freq:
        Frequency string (e.g. 'D', 'B', 'W', 'M').
        Providers may support more granular options later.
    universe:
        MarketIds to load across the panel
    scenarios:
        Number of scenarios requested (>=1). Synthetic providers can generate these.
    """
    # initiate required variables.
    start: str
    end: str
    freq: str
    universe: Universe
    scenarios: int = 1

    def __post_init__(self) -> None:
        """Post initialization."""
        object.__setattr__(self, "start", _parse_iso_date(self.start))
        object.__setattr__(self, "end", _parse_iso_date(self.end))

        if not self.freq or not self.freq.strip():
            raise ValueError("TimeseriesRequest.freq must not be empty.")

        if self.scenarios < 1:
            raise ValueError("TimeseriesRequest.scenarios must be >= 1.")
