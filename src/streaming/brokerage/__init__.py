"""Brokerage adapter protocol and implementations (paper; live adapters in future)."""

from src.streaming.brokerage.protocol import (
    BrokerageAdapter,
    OrderResult,
    Position,
)
from src.streaming.brokerage.paper import PaperBrokerageAdapter

__all__ = [
    "BrokerageAdapter",
    "OrderResult",
    "PaperBrokerageAdapter",
    "Position",
]
