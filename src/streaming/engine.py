"""
StreamingEngine: event-driven engine consuming (timestamp, Market) and calling strategy.

Reuses strategy signature (market, portfolio, context) -> orders from backtesting.
Paper vs live is determined by which brokerage adapter is injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.backtesting.core.engine import BacktestContext, Order, StrategyFunc
from src.marketdata.core.market import Market
from src.marketdata.providers.streaming.protocol import StreamingMarketDataProtocol
from src.streaming.brokerage.protocol import BrokerageAdapter, OrderLike
from src.streaming.context import LiveContext
from src.streaming.portfolio_state import portfolio_state_from_adapter


def _parse_date_from_timestamp(ts: str) -> date:
    """Parse date from timestamp string (e.g. ISO date or datetime)."""
    if not ts or len(ts) < 10:
        return date.today()
    try:
        return date.fromisoformat(ts[:10])
    except (ValueError, TypeError):
        return date.today()


class _MarketSnapshotAdapter:
    """Wraps Market so it exposes .asof as date for backtest-compatible strategy."""

    def __init__(self, market: Market) -> None:
        self._market = market
        self._asof_date = _parse_date_from_timestamp(market.asof)

    @property
    def asof(self) -> date:
        return self._asof_date

    def quote(self, mkt_id: Any) -> float:
        return self._market.quote(mkt_id)

    def curve(self, mkt_id: Any) -> Any:
        return self._market.curve(mkt_id)

    def vol_surface(self, mkt_id: Any) -> Any:
        return self._market.vol_surface(mkt_id)


@dataclass
class StreamingEngineConfig:
    """Configuration for StreamingEngine (optional; extend later)."""

    pass


@dataclass
class StreamingRunResult:
    """
    Result of a streaming run (minimal for 5.5).

    Attributes
    ----------
    steps_processed : int
        Number of (timestamp, market) steps processed.
    orders_submitted : list
        Order results from submit_order (order_id, status).
    """

    steps_processed: int = 0
    orders_submitted: List[Any] = field(default_factory=list)


class StreamingEngine:
    """
    Event-driven engine that consumes a stream of (timestamp, Market) and runs a strategy.

    For each (timestamp, market): applies market to adapter (so paper adapter can fill
    pending orders), builds portfolio state from adapter, builds context, calls
    strategy(market, portfolio, context) -> orders, submits orders to adapter.

    Strategy signature is the same as backtesting: (market, portfolio, context) -> Sequence[Order].
    Market is wrapped so .asof is a date for compatibility. Portfolio and context are
    backtest-compatible (PortfolioState, LiveContext with current_date, step, total_steps).
    """

    def __init__(self, config: Optional[StreamingEngineConfig] = None) -> None:
        self._config = config or StreamingEngineConfig()

    async def run_async(
        self,
        stream_provider: StreamingMarketDataProtocol,
        brokerage_adapter: BrokerageAdapter,
        strategy: StrategyFunc,
        get_price: Callable[[str, Any], float],
        *,
        get_cash: Optional[Callable[[], float]] = None,
        user_data: Optional[Dict[str, Any]] = None,
    ) -> StreamingRunResult:
        """
        Run the strategy on the stream.

        Parameters
        ----------
        stream_provider : StreamingMarketDataProtocol
            Async stream of (timestamp, Market).
        brokerage_adapter : BrokerageAdapter
            Paper or live adapter; if PaperBrokerageAdapter, call apply_market before building portfolio.
        strategy : StrategyFunc
            (market, portfolio, context) -> Sequence[Order].
        get_price : callable
            (instrument_id, market) -> current price (float). Used for apply_market and portfolio state.
        get_cash : callable or None
            () -> current cash. If None, uses adapter.get_cash() when available.
        user_data : dict or None
            Initial user_data for context (persisted across steps).

        Returns
        -------
        StreamingRunResult
            Steps processed and orders submitted.
        """
        steps = 0
        orders_submitted: List[Any] = []
        ud = dict(user_data) if user_data else {}

        if get_cash is None:
            get_cash = getattr(brokerage_adapter, "get_cash", lambda: 0.0)

        apply_market = getattr(
            brokerage_adapter,
            "apply_market",
            None,
        )

        async for timestamp, market in stream_provider.stream():
            # Apply market to adapter (paper adapter fills pending orders)
            if apply_market is not None:
                apply_market(market, get_price)

            # Build portfolio state from adapter
            portfolio_state = portfolio_state_from_adapter(
                adapter=brokerage_adapter,
                get_cash=get_cash,
                market=market,
                get_price=get_price,
            )

            # Build context
            current_date = _parse_date_from_timestamp(timestamp)
            context = LiveContext(
                current_date=current_date,
                timestamp=timestamp,
                step=steps,
                total_steps=-1,
                user_data=ud,
            )
            backtest_context = BacktestContext(
                current_date=current_date,
                step=steps,
                total_steps=-1,
                user_data=ud,
            )

            # Wrap market so .asof is date for strategy
            snapshot = _MarketSnapshotAdapter(market)

            # Call strategy
            orders = strategy(snapshot, portfolio_state, backtest_context)
            if not isinstance(orders, (list, tuple)):
                orders = list(orders) if orders else []

            # Submit orders
            for order in orders:
                if order is None:
                    continue
                res = brokerage_adapter.submit_order(order)
                orders_submitted.append(res)

            steps += 1

        return StreamingRunResult(steps_processed=steps, orders_submitted=orders_submitted)
