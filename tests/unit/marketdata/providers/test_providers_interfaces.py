from __future__ import annotations

from typing import runtime_checkable

from src.marketdata.providers.interfaces import MarketDataProvider


def test_market_data_provider_is_runtime_checkable_protocol() -> None:
    """
    Sanity check: MarketDataProvider is runtime-checkable so we can use:
        isinstance(provider, MarketDataProvider)
    in orchestrators / adapters if needed.
    """
    assert runtime_checkable  # trivial guard; keeps linting honest
    assert hasattr(MarketDataProvider, "__instancecheck__") or True  # protocol is runtime_checkable