"""
Historical Data Module for Backtesting.

This module provides data providers for backtesting:
- DictDataProvider / CsvDataProvider: Use marketdata HistoricalProvider under the hood
- BacktestDataAdapter: Use any MarketDataProvider (Static, Synthetic, Historical) with BacktestEngine
- SimpleMarketSnapshot: Standalone snapshot type (e.g. for tests)
"""

from src.backtesting.data.adapter import BacktestDataAdapter, MarketSnapshotAdapter
from src.backtesting.data.providers import (
    CsvDataProvider,
    DictDataProvider,
    HistoricalDataProvider,
    SimpleMarketSnapshot,
)

__all__ = [
    "BacktestDataAdapter",
    "MarketSnapshotAdapter",
    "HistoricalDataProvider",
    "CsvDataProvider",
    "DictDataProvider",
    "SimpleMarketSnapshot",
]
