# Streaming and Live Data – Technical Reference

**Module:** `src.streaming`, `src.marketdata.providers.streaming`

This document describes the streaming and live trading infrastructure: streaming market data protocol, event-driven engine, and brokerage adapter interface. The design is modeled on well-known free API brokerages (Alpaca WebSocket/REST, IBKR TWS API); Phase 5.5 delivers protocols plus simulated stream and paper adapter only.

---

## Overview

- **Streaming protocol:** Async iterator of (timestamp, Market) snapshots. Implementations: ReplayStreamProvider (replay from MarketDataset); future: Alpaca/IBKR adapters mapping their streams to Market.
- **StreamingEngine:** Consumes stream, applies market to adapter, builds portfolio state, calls strategy(market, portfolio, context) -> orders, submits orders to brokerage adapter. Same strategy signature as backtesting.
- **Brokerage adapter:** submit_order, cancel_order, get_positions. Paper adapter simulates fills; live adapters (future) call real APIs. Paper vs live is determined by which adapter is injected (and, for live, adapter config such as paper=True/False).

---

## Architecture

```
src/streaming/
├── brokerage/
│   ├── protocol.py   # BrokerageAdapter, Position, OrderResult, OrderLike
│   └── paper.py      # PaperBrokerageAdapter (in-memory, apply_market for fills)
├── context.py        # LiveContext (timestamp, step, current_date, user_data)
├── engine.py         # StreamingEngine, run_async(stream_provider, adapter, strategy, get_price)
└── portfolio_state.py  # portfolio_state_from_adapter (backtest-compatible PortfolioState)

src/marketdata/providers/streaming/
├── protocol.py       # StreamingMarketDataProtocol (async stream() -> (timestamp, Market))
└── replay.py         # ReplayStreamProvider (from MarketDataset or list of (ts, Market))
```

---

## Streaming Protocol

### StreamingMarketDataProtocol

```python
async def stream(self) -> AsyncIterator[tuple[str, Market]]:
    """Yield (timestamp, market) snapshots in order."""
```

- **ReplayStreamProvider:** Takes MarketDataset or list of (timestamp, Market); yields in order. No external API.
- **Future integrations:** Alpaca WebSocket (subscribe symbols, map quotes/bars -> Market); IBKR TWS (callback stream -> Market).

---

## Brokerage Adapter

### BrokerageAdapter (protocol)

- `submit_order(order: OrderLike) -> OrderResult` — order_id, status.
- `cancel_order(order_id: str) -> bool`
- `get_positions() -> Sequence[Position]`

### Position

- instrument_id, quantity, avg_price, market_value, unrealized_pnl

### PaperBrokerageAdapter

- initial_cash; in-memory positions and pending orders.
- `apply_market(market, get_price)` — resolve pending orders to fills using get_price(instrument_id, market).
- `get_cash()` — current cash (used by engine to build portfolio state).

Order type: any object with `.instrument_id` and `.quantity` (OrderLike / backtest Order protocol).

---

## StreamingEngine

- **run_async(stream_provider, brokerage_adapter, strategy, get_price, get_cash=None, user_data=None) -> StreamingRunResult**
- For each (timestamp, market): call adapter.apply_market(market, get_price) if present; build PortfolioState from adapter; build BacktestContext-compatible context; wrap market so .asof is date; call strategy(snapshot, portfolio_state, context); submit returned orders to adapter.
- **Strategy signature:** Same as backtesting: (market, portfolio, context) -> Sequence[Order]. Market is wrapped so .asof is date; portfolio is PortfolioState; context is BacktestContext (current_date, step, total_steps=-1, user_data).

---

## Paper vs Live

- **Paper:** Use PaperBrokerageAdapter (and/or simulated stream). No external API.
- **Live:** Use a live-configured adapter (future Alpaca/IBKR implementation) that implements BrokerageAdapter; adapter config (e.g. paper=True vs False, base URL) switches between broker’s paper and live environment. Engine and strategy are unchanged.

---

## References

- Alpaca: WebSocket streaming (quotes, bars, trades), REST orders/positions.
- IBKR TWS API: Callback-based streaming, placeOrder, reqPositions.
