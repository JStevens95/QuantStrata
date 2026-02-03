"""Unit tests for ReplayStreamProvider."""

from __future__ import annotations

import asyncio
import pytest

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.providers.streaming import ReplayStreamProvider


@pytest.fixture
def snapshots():
    """Three (timestamp, Market) snapshots."""
    spot_id = MarketId("FX", "SPOT", "EURUSD")
    return [
        ("2024-01-01", Market(asof="2024-01-01", quotes={spot_id: Quote(value=1.08)}, curves={}, vols={})),
        ("2024-01-02", Market(asof="2024-01-02", quotes={spot_id: Quote(value=1.10)}, curves={}, vols={})),
        ("2024-01-03", Market(asof="2024-01-03", quotes={spot_id: Quote(value=1.12)}, curves={}, vols={})),
    ]


def test_replay_stream_yields_snapshots_in_order(snapshots):
    async def run():
        provider = ReplayStreamProvider(snapshots=snapshots)
        collected = []
        async for ts, market in provider.stream():
            collected.append((ts, market))
        return collected

    collected = asyncio.run(run())
    assert len(collected) == 3
    assert collected[0][0] == "2024-01-01"
    assert collected[0][1].asof == "2024-01-01"
    assert collected[1][0] == "2024-01-02"
    assert collected[2][0] == "2024-01-03"


def test_replay_stream_markets_have_quotes(snapshots):
    spot_id = MarketId("FX", "SPOT", "EURUSD")

    async def run():
        provider = ReplayStreamProvider(snapshots=snapshots)
        out = []
        async for ts, market in provider.stream():
            out.append((ts, market.quote(spot_id)))
        return out

    out = asyncio.run(run())
    assert out[0][1] == pytest.approx(1.08)
    assert out[1][1] == pytest.approx(1.10)
    assert out[2][1] == pytest.approx(1.12)
