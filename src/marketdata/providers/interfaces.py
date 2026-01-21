from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.market import Market
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest


@runtime_checkable
class MarketDataProvider(Protocol):
    """
    Provider interface for producing pricing-ready market data objects.

    Why this interface exists
    -------------------------
    - Keeps pricers/orchestrators decoupled from data sources.
    - Allows multiple implementations behind one stable API:
        * SyntheticProvider (deterministic generation)
        * StaticProvider (replay/frozen datasets)
        * ApiProvider(s) (Bloomberg/Refinitiv/internal feeds)
        * HybridProvider (primary + fallback + provenance)

    Design constraints
    ------------------
    - get_market(...) returns a pricing-ready Market snapshot.
    - get_timeseries(...) returns a MarketDataset suitable for:
        * snapshot(time_idx, scenario_idx)
        * plotting and backtests
        * scenario engines and repricing pipelines
    """

    name: str  # Human-friendly provider name for logging/provenance.

    def get_market(self, request: MarketRequest) -> Market:
        """
        Return a single as-of Market snapshot.

        Notes
        -----
        Implementations should be:
        - deterministic when configured with a seed or fixed store
        - dependency-safe (if a MarketId requires prerequisites, provider should handle it)
        """
        ...

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """
        Return a MarketDataset for a date range and scenario count.

        Notes
        -----
        - Dataset must be snapshot-able via MarketDataset.snapshot(...)
        - Providers should validate basic request invariants and raise ValueError
          for invalid inputs (e.g., scenarios < 1).
        """
        ...


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    """
    Optional metadata container for provider identification and provenance.

    This is not required by the protocol, but is useful for:
    - artifact stores (save/load)
    - logging
    - HybridProvider provenance chains later
    """
    name: str
    version: str = "v1"
    notes: str = ""