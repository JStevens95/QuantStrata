# Streaming and Live Data – User Guide

This guide shows how to use QuantStrata's streaming and live trading infrastructure: simulated stream, paper brokerage adapter, and StreamingEngine with the same strategy signature as backtesting.

---

## Quick Start

### Replay stream + paper adapter + strategy

```python
import asyncio
from src.marketdata.providers.streaming import ReplayStreamProvider
from src.streaming import PaperBrokerageAdapter, StreamingEngine
from src.backtesting.core.engine import PortfolioState, Position

# Build a replay stream (e.g. from a MarketDataset or list of (timestamp, Market))
# For demo: list of snapshots (you would typically use ReplayStreamProvider(dataset=my_dataset))
from src.marketdata.core.market import Market
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.ids import MarketId
# ... build minimal market snapshots ...

snapshots = [(f"2024-01-{d:02d}", market_t) for d, market_t in enumerate([m1, m2, m3], start=1)]
stream_provider = ReplayStreamProvider(snapshots=snapshots)

# Paper adapter with initial cash
adapter = PaperBrokerageAdapter(initial_cash=100_000.0)

# Strategy: same signature as backtest (market, portfolio, context) -> orders
def my_strategy(market, portfolio, context):
    orders = []
    if context.step == 0:
        # Example: return an order (object with instrument_id, quantity)
        orders.append(SimpleOrder(instrument_id="FX.SPOT.EURUSD", quantity=10_000.0))
    return orders

class SimpleOrder:
    def __init__(self, instrument_id: str, quantity: float):
        self.instrument_id = instrument_id
        self.quantity = quantity

# get_price: (instrument_id, market) -> price (for fills and portfolio state)
def get_price(inst_id, market):
    # Map instrument_id to MarketId and return market.quote(mkt_id)
    return 1.10  # example

async def main():
    engine = StreamingEngine()
    result = await engine.run_async(
        stream_provider=stream_provider,
        brokerage_adapter=adapter,
        strategy=my_strategy,
        get_price=get_price,
    )
    print("Steps processed:", result.steps_processed)
    print("Orders submitted:", result.orders_submitted)

asyncio.run(main())
```

---

## Components

| Component | Role |
|-----------|------|
| **StreamingMarketDataProtocol** | Async stream of (timestamp, Market). Implement with ReplayStreamProvider or future Alpaca/IBKR adapter. |
| **ReplayStreamProvider** | Replay from MarketDataset or list of (timestamp, Market). No API keys. |
| **BrokerageAdapter** | submit_order, cancel_order, get_positions. Paper = simulated; live = future real API. |
| **PaperBrokerageAdapter** | In-memory fills; call apply_market(market, get_price) so engine can resolve pending orders. |
| **StreamingEngine** | Consumes stream, builds portfolio from adapter, calls strategy, submits orders. |

---

## Strategy Reuse

The same strategy function used in BacktestEngine works in StreamingEngine: (market, portfolio, context) -> Sequence[Order]. Market is wrapped so .asof is a date; portfolio is PortfolioState; context has current_date, step, total_steps, user_data. Orders are any object with .instrument_id and .quantity.

---

## Paper vs Live

- **Testing:** Use ReplayStreamProvider + PaperBrokerageAdapter (no external API).
- **Live later:** Replace with a live stream provider and a live brokerage adapter that implements the same protocols; switch paper/live via adapter config (e.g. Alpaca paper=True vs False).

---

*See also: [Streaming and Live Data (Reference)](../../reference/streaming_live_data.md) | [Tutorial](../tutorials/streaming/streaming_and_live_data.ipynb)*
