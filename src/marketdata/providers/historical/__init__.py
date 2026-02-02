"""
Historical data provider for backtesting and time-series replay.

Loads simple price series from dict or CSV and implements MarketDataProvider,
so backtesting and other consumers can use the same provider interface as
StaticProvider and SyntheticProvider.
"""

from src.marketdata.providers.historical.provider import (
    HistoricalProvider,
    HistoricalProviderConfig,
)

__all__ = [
    "HistoricalProvider",
    "HistoricalProviderConfig",
]
