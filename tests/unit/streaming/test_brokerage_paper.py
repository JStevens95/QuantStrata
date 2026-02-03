"""Unit tests for PaperBrokerageAdapter."""

from __future__ import annotations

import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.streaming.brokerage import PaperBrokerageAdapter
from src.streaming.brokerage.protocol import OrderResult


class SimpleOrder:
    def __init__(self, instrument_id: str, quantity: float):
        self.instrument_id = instrument_id
        self.quantity = quantity


def test_paper_adapter_submit_returns_pending():
    adapter = PaperBrokerageAdapter(initial_cash=100_000.0)
    order = SimpleOrder("FX.SPOT.EURUSD", 10_000.0)
    res = adapter.submit_order(order)
    assert isinstance(res, OrderResult)
    assert res.status == "pending"
    assert res.order_id.startswith("paper-")


def test_paper_adapter_apply_market_fills_order():
    adapter = PaperBrokerageAdapter(initial_cash=100_000.0)
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    market = Market(asof="2024-01-01", quotes={spot_id: Quote(value=1.10)}, curves={}, vols={})

    order = SimpleOrder(spot_id.key(), 10_000.0)
    adapter.submit_order(order)

    def get_price(inst_id, mkt):
        return float(mkt.quote(spot_id))

    adapter.apply_market(market, get_price)

    positions = adapter.get_positions()
    assert len(positions) == 1
    assert positions[0].instrument_id == spot_id.key()
    assert positions[0].quantity == 10_000.0
    assert positions[0].avg_price == 1.10
    assert adapter.get_cash() == pytest.approx(100_000.0 - 10_000.0 * 1.10)


def test_paper_adapter_cancel_order():
    adapter = PaperBrokerageAdapter(initial_cash=100_000.0)
    order = SimpleOrder("FX.SPOT.EURUSD", 5_000.0)
    res = adapter.submit_order(order)
    order_id = res.order_id
    assert adapter.cancel_order(order_id) is True
    assert adapter.cancel_order(order_id) is False
    assert len(adapter.get_positions()) == 0
