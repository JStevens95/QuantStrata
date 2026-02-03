"""Pricer result caching: optional wrapper around PortfolioPricer."""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable, Optional

from src.portfolio.core import Portfolio, PortfolioResult


@dataclass
class CachingPortfolioPricer:
    """
    Wrapper that caches PortfolioResult by (portfolio_key, market_key).

    Eviction: LRU by max_size; optional TTL (seconds). Thread-safe for get/set.

    Parameters
    ----------
    portfolio_pricer : PortfolioPricer-like
        Must have price(portfolio, market, *, pricer_id) -> PortfolioResult.
    max_size : int
        Max cached entries (LRU eviction).
    ttl_seconds : float or None
        If set, entries older than ttl_seconds are treated as miss.
    portfolio_key_fn : callable or None
        (portfolio) -> hashable. Default: id(portfolio).
    market_key_fn : callable or None
        (market) -> hashable. Default: id(market).
    """

    portfolio_pricer: Any
    max_size: int = 1024
    ttl_seconds: Optional[float] = None
    portfolio_key_fn: Optional[Callable[[Portfolio], Any]] = None
    market_key_fn: Optional[Callable[[Any], Any]] = None

    def __post_init__(self) -> None:
        if self.max_size < 1:
            raise ValueError("max_size must be >= 1.")
        self._cache: OrderedDict[tuple[Any, Any], tuple[PortfolioResult, float]] = OrderedDict()
        self._lock = RLock()

    def _portfolio_key(self, portfolio: Portfolio) -> Any:
        if self.portfolio_key_fn is not None:
            return self.portfolio_key_fn(portfolio)
        return id(portfolio)

    def _market_key(self, market: Any) -> Any:
        if self.market_key_fn is not None:
            return self.market_key_fn(market)
        return id(market)

    def price(
        self,
        portfolio: Portfolio,
        market: Any,  # noqa: ANN401
        *,
        pricer_id: Optional[str] = None,
    ) -> PortfolioResult:
        """Return cached result if present and not expired; else compute, cache, and return."""
        pk = self._portfolio_key(portfolio)
        mk = self._market_key(market)
        key = (pk, mk)
        now = time.monotonic()

        with self._lock:
            if key in self._cache:
                result, ts = self._cache[key]
                if self.ttl_seconds is None or (now - ts) <= self.ttl_seconds:
                    self._cache.move_to_end(key)
                    return result
                del self._cache[key]

        result = self.portfolio_pricer.price(portfolio, market, pricer_id=pricer_id)

        with self._lock:
            while len(self._cache) >= self.max_size and self._cache:
                self._cache.popitem(last=False)
            self._cache[key] = (result, now)

        return result

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
