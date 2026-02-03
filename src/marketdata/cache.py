"""Market data caching: wrapper around MarketDataProvider that caches get_market by request."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Optional

from src.marketdata.core.market import Market
from src.marketdata.core.requests import MarketRequest


def _request_key(request: MarketRequest) -> tuple[str, tuple[str, ...], Optional[int]]:
    """Stable hashable key for a MarketRequest."""
    asof = getattr(request, "asof", None)
    universe = getattr(request, "universe", None)
    ids_tuple = tuple(m.key() for m in universe.ids) if universe else ()
    scenario = getattr(request, "scenario", None)
    return (str(asof), ids_tuple, scenario)


@dataclass
class CachingMarketDataProvider:
    """
    Wrapper that caches get_market(request) by a stable key from the request.

    Eviction: LRU by max_size; optional TTL (seconds). Thread-safe.

    Parameters
    ----------
    provider : MarketDataProvider
        Must implement get_market(request) -> Market (and optionally get_timeseries).
    max_size : int
        Max cached Market snapshots (LRU eviction).
    ttl_seconds : float or None
        If set, entries older than ttl_seconds are treated as miss.
    """

    provider: Any  # MarketDataProvider
    max_size: int = 512
    ttl_seconds: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("max_size must be >= 1.")
        self._cache: OrderedDict[tuple, tuple[Market, float]] = OrderedDict()
        self._lock = RLock()

    @property
    def name(self) -> str:
        return getattr(self.provider, "name", "CachingMarketDataProvider")

    def get_market(self, request: MarketRequest) -> Market:
        """Return cached Market if present and not expired; else call provider, cache, and return."""
        key = _request_key(request)
        now = time.monotonic()

        with self._lock:
            if key in self._cache:
                market, ts = self._cache[key]
                if self.ttl_seconds is None or (now - ts) <= self.ttl_seconds:
                    self._cache.move_to_end(key)
                    return market
                del self._cache[key]

        market = self.provider.get_market(request)

        with self._lock:
            while len(self._cache) >= self.max_size and self._cache:
                self._cache.popitem(last=False)
            self._cache[key] = (market, now)

        return market

    def get_timeseries(self, request: Any) -> Any:
        """Delegate to underlying provider (no caching of timeseries in this wrapper)."""
        return self.provider.get_timeseries(request)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
