"""
Streaming market data protocol and implementations.

StreamingMarketDataProtocol yields (timestamp, Market) via async iterator.
ReplayStreamProvider replays a MarketDataset. Future integrations: Alpaca
WebSocket (subscribe symbols, map quotes/bars -> Market), IBKR TWS (callback stream -> Market).
"""

from src.marketdata.providers.streaming.protocol import StreamingMarketDataProtocol
from src.marketdata.providers.streaming.replay import ReplayStreamProvider

__all__ = [
    "ReplayStreamProvider",
    "StreamingMarketDataProtocol",
]
