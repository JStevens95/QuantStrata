"""Unit tests for StreamingEngine."""

from __future__ import annotations

import asyncio
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.providers.streaming import ReplayStreamProvider
from src.streaming import PaperBrokerageAdapter, StreamingEngine


class SimpleOrder:
    def __init__(self, instrument_id: str, quantity: float):
        self.instrument_id = instrument_id
        self.quantity = quantity


@pytest.fixture
def stream_provider():
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    snapshots = [
        ("2024-01-01", Market(asof="2024-01-01", quotes={spot_id: Quote(value=1.08)}, curves={}, vols={})),
        ("2024-01-02", Market(asof="2024-01-02", quotes={spot_id: Quote(value=1.10)}, curves={}, vols={})),
    ]
    return ReplayStreamProvider(snapshots=snapshots)


@pytest.fixture
def spot_id():
    return MarketId("FX", "SPOT", "EURUSD")


def get_price(inst_id, market):
    mid = MarketId.parse(inst_id)
    return float(market.quote(mid))


def test_streaming_engine_processes_steps_and_submits_orders(stream_provider, spot_id):
    adapter = PaperBrokerageAdapter(initial_cash=100_000.0)

    def strategy(market, portfolio, context):
        if context.step == 0:
            return [SimpleOrder(spot_id.key(), 1_000.0)]
        return []

    engine = StreamingEngine()

    async def run():
        return await engine.run_async(
            stream_provider=stream_provider,
            brokerage_adapter=adapter,
            strategy=strategy,
            get_price=get_price,
        )

    result = asyncio.run(run())

    assert result.steps_processed == 2
    assert len(result.orders_submitted) == 1
    positions = adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].quantity == 1_000.0
    # Order submitted on step 0 is filled on step 1 when apply_market runs with next market
    assert positions[0].avg_price == pytest.approx(1.10)
