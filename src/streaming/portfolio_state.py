"""
Build portfolio state from brokerage adapter for strategy consumption.

Uses backtesting PortfolioState and Position so the same strategy
receives a consistent interface in both backtest and streaming.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from src.backtesting.core.engine import Position, PortfolioState
from src.streaming.brokerage.protocol import BrokerageAdapter


def portfolio_state_from_adapter(
    adapter: BrokerageAdapter,
    get_cash: Callable[[], float],
    market: Any,
    get_price: Callable[[str, Any], float],
) -> PortfolioState:
    """
    Build a backtest-compatible PortfolioState from adapter positions and cash.

    Uses get_price(instrument_id, market) to set market_value and unrealized_pnl
    for each position so the strategy sees up-to-date values.

    Parameters
    ----------
    adapter : BrokerageAdapter
        Brokerage adapter (e.g. PaperBrokerageAdapter).
    get_cash : callable
        () -> current cash balance (adapter.get_cash() if adapter supports it).
    market : Market
        Current market snapshot (for price lookup).
    get_price : callable
        (instrument_id, market) -> current price (float).

    Returns
    -------
    PortfolioState
        Backtest-compatible portfolio state.
    """
    cash = get_cash()
    positions: Dict[str, Position] = {}
    broker_positions = adapter.get_positions()
    for bp in broker_positions:
        try:
            price = get_price(bp.instrument_id, market)
        except (KeyError, TypeError, ValueError):
            price = bp.avg_price
        pos = Position(
            instrument_id=bp.instrument_id,
            quantity=bp.quantity,
            avg_price=bp.avg_price,
            market_value=bp.quantity * price,
            unrealized_pnl=bp.quantity * (price - bp.avg_price),
        )
        pos.update_market_value(price)
        positions[bp.instrument_id] = pos
    return PortfolioState(cash=cash, positions=positions)
