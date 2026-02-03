"""
Streaming and live trading infrastructure.

Provides:
- BrokerageAdapter protocol and PaperBrokerageAdapter (in-memory execution simulation)
- StreamingMarketDataProtocol and ReplayStreamProvider
- StreamingEngine consuming (timestamp, Market) and calling strategy(market, portfolio, context) -> orders

Designed to align with Alpaca (WebSocket/REST) and IBKR (TWS API) patterns;
real integrations implement the same protocols. Phase 5.5 uses simulated stream and paper adapter only.
"""

from src.streaming.brokerage.protocol import (
    BrokerageAdapter,
    OrderResult,
    Position,
)
from src.streaming.brokerage.paper import PaperBrokerageAdapter
from src.streaming.context import LiveContext
from src.streaming.engine import StreamingEngine, StreamingEngineConfig, StreamingRunResult

__all__ = [
    "BrokerageAdapter",
    "LiveContext",
    "OrderResult",
    "PaperBrokerageAdapter",
    "Position",
    "StreamingEngine",
    "StreamingEngineConfig",
    "StreamingRunResult",
]
