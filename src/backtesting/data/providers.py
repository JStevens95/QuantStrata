"""
Historical Data Providers for Backtesting.

This module provides data providers that implement the backtest DataProvider
protocol (get_dates, get_snapshot). Implementation is delegated to
marketdata/providers/historical (HistoricalProvider) so that loading logic
lives in one place and backtesting can use any MarketDataProvider via
BacktestDataAdapter.

- DictDataProvider: In-memory data (uses HistoricalProvider + BacktestDataAdapter)
- CsvDataProvider: Load from CSV (uses HistoricalProvider + BacktestDataAdapter)
- For using other marketdata providers (Static, Synthetic), use BacktestDataAdapter
  directly from src.backtesting.data.adapter.

Example
-------
>>> from src.backtesting.data import DictDataProvider, SimpleMarketSnapshot
>>> from datetime import date
>>>
>>> data = {
...     date(2024, 1, 1): {"AAPL": 150.0, "GOOGL": 140.0},
...     date(2024, 1, 2): {"AAPL": 151.0, "GOOGL": 142.0},
... }
>>> provider = DictDataProvider(data)
>>> snapshot = provider.get_snapshot(date(2024, 1, 1))
>>> print(snapshot.get_price("AAPL"))  # 150.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

# Integration: use marketdata HistoricalProvider + backtesting adapter
from src.backtesting.data.adapter import BacktestDataAdapter
from src.marketdata.providers.historical import (
    HistoricalProvider,
    HistoricalProviderConfig,
)


# =============================================================================
# Market Snapshot (standalone type for tests / backward compatibility)
# =============================================================================

@dataclass
class SimpleMarketSnapshot:
    """
    Simple market snapshot containing prices for instruments.

    Used when not going through marketdata (e.g. tests). When using
    DictDataProvider/CsvDataProvider backed by HistoricalProvider, get_snapshot
    returns MarketSnapshotAdapter instead; both implement get_price(id) and .asof.
    """

    asof: date
    prices: Dict[str, float] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)

    def get_price(self, instrument_id: str) -> float:
        """Get price for an instrument."""
        if instrument_id not in self.prices:
            raise KeyError(f"No price for {instrument_id} on {self.asof}")
        return self.prices[instrument_id]

    def get(self, key: str, default: Any = None) -> Any:
        """Get additional data."""
        return self.data.get(key, default)

    def __contains__(self, instrument_id: str) -> bool:
        """Check if instrument has price data."""
        return instrument_id in self.prices


# =============================================================================
# Base Provider (protocol)
# =============================================================================

class HistoricalDataProvider:
    """
    Protocol for historical data providers used by BacktestEngine.

    Implementations can be standalone (e.g. SimpleMarketSnapshot in tests)
    or delegate to marketdata via BacktestDataAdapter (DictDataProvider,
    CsvDataProvider).
    """

    def get_dates(self) -> Sequence[date]:
        """Return all available dates in chronological order."""
        ...

    def get_snapshot(self, dt: date):
        """Return market snapshot for a given date (get_price(id), .asof)."""
        ...

    @property
    def start_date(self) -> date:
        """First available date."""
        dates = self.get_dates()
        if not dates:
            raise ValueError("No dates available")
        return dates[0]

    @property
    def end_date(self) -> date:
        """Last available date."""
        dates = self.get_dates()
        if not dates:
            raise ValueError("No dates available")
        return dates[-1]

    @property
    def num_dates(self) -> int:
        """Number of available dates."""
        return len(self.get_dates())

    def get_instruments(self) -> List[str]:
        """Get list of all instruments in the dataset."""
        ...


# =============================================================================
# Dict Provider (delegates to marketdata HistoricalProvider + adapter)
# =============================================================================

class DictDataProvider(HistoricalDataProvider):
    """
    In-memory data provider from dictionary.

    Implemented via marketdata.providers.historical.HistoricalProvider and
    BacktestDataAdapter, so loading logic lives in marketdata.
    """

    def __init__(
        self,
        data: Mapping[date, Union[Dict[str, float], SimpleMarketSnapshot]],
        asset_class: str = "EQUITY",
    ) -> None:
        normalized: Dict[date, Dict[str, float]] = {}
        for dt, snapshot_or_dict in data.items():
            if hasattr(snapshot_or_dict, "prices"):
                normalized[dt] = dict(getattr(snapshot_or_dict, "prices"))
            else:
                normalized[dt] = dict(snapshot_or_dict)
        config = HistoricalProviderConfig(asset_class=asset_class)
        hp = HistoricalProvider(data=normalized, config=config)
        self._adapter = BacktestDataAdapter(provider=hp, universe=hp.universe)

    def get_dates(self) -> Sequence[date]:
        return self._adapter.get_dates()

    def get_snapshot(self, dt: date):
        return self._adapter.get_snapshot(dt)

    def get_instruments(self) -> List[str]:
        return sorted(mid.name for mid in self._adapter._universe.ids)


# =============================================================================
# CSV Provider (delegates to marketdata HistoricalProvider + adapter)
# =============================================================================

class CsvDataProvider(HistoricalDataProvider):
    """
    Load historical data from CSV file(s).

    Implemented via marketdata.providers.historical.HistoricalProvider and
    BacktestDataAdapter. Supports wide and long CSV formats.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        date_column: str = "date",
        date_format: str = "%Y-%m-%d",
        format: str = "wide",
        instrument_column: str = "instrument",
        price_column: str = "price",
        asset_class: str = "EQUITY",
    ) -> None:
        config = HistoricalProviderConfig(
            asset_class=asset_class,
            date_format=date_format,
        )
        hp = HistoricalProvider(
            data=Path(filepath),
            config=config,
            format=format,
            date_column=date_column,
            instrument_column=instrument_column,
            price_column=price_column,
        )
        self._adapter = BacktestDataAdapter(provider=hp, universe=hp.universe)

    def get_dates(self) -> Sequence[date]:
        return self._adapter.get_dates()

    def get_snapshot(self, dt: date):
        return self._adapter.get_snapshot(dt)

    def get_instruments(self) -> List[str]:
        return sorted(mid.name for mid in self._adapter._universe.ids)


# =============================================================================
# Factory
# =============================================================================

def create_data_provider(
    source: Union[str, Path, Dict],
    **kwargs: Any,
) -> HistoricalDataProvider:
    """
    Factory function to create appropriate data provider.

    Uses marketdata HistoricalProvider under the hood for dict and CSV sources.
    """
    if isinstance(source, dict):
        return DictDataProvider(source, **kwargs)
    path = Path(source)
    if path.suffix.lower() == ".csv":
        return CsvDataProvider(path, **kwargs)
    raise ValueError(f"Unknown data source type: {source}")
