"""
Paper brokerage adapter: in-memory execution simulation.

Fills orders at a price provided by the engine (e.g. from Market quote).
No external API; for testing and development. Live adapters (Alpaca, IBKR)
implement BrokerageAdapter with real API calls.
"""

from __future__ import annotations

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.streaming.brokerage.protocol import (
    BrokerageAdapter,
    OrderLike,
    OrderResult,
    Position,
)


@dataclass
class _PendingOrder:
    order_id: str
    instrument_id: str
    quantity: float


@dataclass
class PaperBrokerageAdapter:
    """
    In-memory brokerage adapter that simulates order execution.

    The engine should call apply_market(timestamp, market, get_price) when a new
    (timestamp, Market) arrives so pending orders are filled at that market's
    prices. get_price(instrument_id, market) should return the fill price
    (e.g. mid or last quote for that instrument).

    Parameters
    ----------
    initial_cash : float
        Starting cash balance.
    """

    initial_cash: float = 0.0

    def __post_init__(self) -> None:
        self._cash: float = float(self.initial_cash)
        self._positions: Dict[str, Position] = {}
        self._pending: OrderedDict[str, _PendingOrder] = OrderedDict()
        self._order_id_counter: int = 0

    def _next_order_id(self) -> str:
        self._order_id_counter += 1
        return f"paper-{uuid.uuid4().hex[:8]}-{self._order_id_counter}"

    def submit_order(self, order: OrderLike) -> OrderResult:
        order_id = self._next_order_id()
        self._pending[order_id] = _PendingOrder(
            order_id=order_id,
            instrument_id=order.instrument_id,
            quantity=order.quantity,
        )
        return OrderResult(order_id=order_id, status="pending")

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self._pending:
            del self._pending[order_id]
            return True
        return False

    def get_positions(self) -> Sequence[Position]:
        return list(self._positions.values())

    def get_cash(self) -> float:
        """Return current cash balance."""
        return self._cash

    def apply_market(
        self,
        market: Any,
        get_price: Callable[[str, Any], float],
    ) -> None:
        """
        Resolve pending orders to fills using current market prices.

        For each pending order, get_price(instrument_id, market) is called
        to obtain the fill price; the order is then filled and positions/cash
        are updated.

        Parameters
        ----------
        market : Market
            Current market snapshot (e.g. from stream).
        get_price : callable
            (instrument_id, market) -> fill price (float).
        """
        to_remove: List[str] = []
        for order_id, po in self._pending.items():
            try:
                price = get_price(po.instrument_id, market)
            except (KeyError, TypeError, ValueError):
                continue
            if not (price > 0 and abs(po.quantity) > 0):
                continue
            # Fill order: update cash and position
            cost = -price * po.quantity  # buy: negative cost (cash out)
            self._cash += cost
            existing = self._positions.get(po.instrument_id)
            if existing is None:
                self._positions[po.instrument_id] = Position(
                    instrument_id=po.instrument_id,
                    quantity=po.quantity,
                    avg_price=price,
                    market_value=po.quantity * price,
                    unrealized_pnl=0.0,
                )
            else:
                new_qty = existing.quantity + po.quantity
                if abs(new_qty) < 1e-12:
                    del self._positions[po.instrument_id]
                else:
                    new_avg = (
                        (existing.avg_price * existing.quantity + price * po.quantity)
                        / new_qty
                    )
                    self._positions[po.instrument_id] = Position(
                        instrument_id=po.instrument_id,
                        quantity=new_qty,
                        avg_price=new_avg,
                        market_value=new_qty * price,
                        unrealized_pnl=new_qty * (price - new_avg),
                    )
            to_remove.append(order_id)
        for oid in to_remove:
            del self._pending[oid]
