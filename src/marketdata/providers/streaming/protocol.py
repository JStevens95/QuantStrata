"""
Streaming market data protocol.

Produces a stream of (timestamp, Market) snapshots for consumption by
StreamingEngine. Aligns with patterns used by Alpaca (WebSocket subscribe,
quotes/bars) and IBKR (TWS callback stream); first implementation is
simulated (ReplayStreamProvider).
"""

from __future__ import annotations

from typing import AsyncIterator, Protocol, runtime_checkable

from src.marketdata.core.market import Market


@runtime_checkable
class StreamingMarketDataProtocol(Protocol):
    """
    Protocol for streaming market data: async iterator of (timestamp, Market).

    Implementations:
    - ReplayStreamProvider: replay from MarketDataset or list of (timestamp, Market).
    - Future: Alpaca WebSocket (subscribe symbols, map quotes/bars -> Market),
      IBKR TWS (callback stream -> Market).
    """

    async def stream(self) -> AsyncIterator[tuple[str, Market]]:
        """
        Yield (timestamp, market) snapshots in order.

        Timestamp is a string (e.g. ISO date or datetime); Market is
        pricing-ready (quotes, curves, vols).
        """
        ...
