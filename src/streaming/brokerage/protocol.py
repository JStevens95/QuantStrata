"""
Brokerage adapter protocol for order execution and positions.

Implement this protocol for paper (simulated) or live brokers (e.g. Alpaca, IBKR).
Real integrations: Alpaca (REST + WebSocket), IBKR TWS API; implement BrokerageAdapter
and optionally wrap their native types into library Order/Position.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable


@runtime_checkable
class OrderLike(Protocol):
    """Minimal order interface; compatible with backtesting Order protocol."""

    @property
    def instrument_id(self) -> str:
        """Identifier for the instrument."""
        ...

    @property
    def quantity(self) -> float:
        """Signed quantity (positive = buy, negative = sell)."""
        ...


@dataclass(frozen=True, slots=True)
class Position:
    """
    A single position as returned by the brokerage.

    Attributes
    ----------
    instrument_id : str
        Instrument identifier.
    quantity : float
        Signed quantity (positive = long, negative = short).
    avg_price : float
        Average fill price.
    market_value : float
        Current market value (optional; 0 if not provided).
    unrealized_pnl : float
        Unrealized P&L (optional; 0 if not provided).
    """

    instrument_id: str
    quantity: float
    avg_price: float
    market_value: float = 0.0
    unrealized_pnl: float = 0.0


@dataclass(frozen=True, slots=True)
class OrderResult:
    """
    Result of submitting an order.

    Attributes
    ----------
    order_id : str
        Broker-assigned order id.
    status : str
        e.g. "filled", "pending", "cancelled", "rejected".
    """

    order_id: str
    status: str


@runtime_checkable
class BrokerageAdapter(Protocol):
    """
    Abstract interface for order execution and positions.

    Paper adapter simulates fills; live adapters (Alpaca, IBKR) call real APIs.
    """

    def submit_order(self, order: OrderLike) -> OrderResult:
        """
        Submit an order; returns order_id and status.
        """
        ...

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order by id. Returns True if cancel was accepted.
        """
        ...

    def get_positions(self) -> Sequence[Position]:
        """
        Return current positions.
        """
        ...
