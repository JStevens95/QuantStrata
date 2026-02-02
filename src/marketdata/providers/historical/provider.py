"""
Historical data provider for backtesting and time-series replay.

Loads simple price series from dict or CSV and implements MarketDataProvider.
Produces quote-only Market snapshots and MarketDatasets, so backtesting
and other consumers can use the same provider interface as StaticProvider.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Union

import numpy as np

from src.marketdata.core.dataset import MarketDataset
from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote
from src.marketdata.core.market import Market
from src.marketdata.core.panel import Panel
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest, Universe
from src.marketdata.providers.interfaces import MarketDataProvider


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True, slots=True)
class HistoricalProviderConfig:
    """
    Configuration for HistoricalProvider.

    Parameters
    ----------
    asset_class : str
        Asset class for MarketIds (e.g. "EQUITY", "FX").
    mkt_type : str
        Market type for MarketIds (e.g. "SPOT").
    date_format : str
        Format for parsing dates from CSV.
    """
    asset_class: str = "EQUITY"
    mkt_type: str = "SPOT"
    date_format: str = "%Y-%m-%d"


# =============================================================================
# HistoricalProvider
# =============================================================================

@dataclass
class HistoricalProvider(MarketDataProvider):
    """
    Provider that loads historical price series from dict or CSV.

    Implements MarketDataProvider so backtesting and other consumers can use
    the same interface as StaticProvider and SyntheticProvider. Produces
    quote-only Market snapshots (no curves/vols).

    Data can be supplied as:
    - dict: {date: {symbol: price}} or {date: {symbol: price}, ...}
    - CSV path: wide format (date,col1,col2,...) or long format (date,instrument,price)

    Parameters
    ----------
    data : dict or Path or str
        In-memory dict[date, dict[symbol, float]] or path to CSV file.
    config : HistoricalProviderConfig
        Optional config (asset class, date format).
    format : str
        For CSV: "wide" or "long".
    date_column : str
        Name of date column in CSV.
    instrument_column : str
        Name of instrument column (long format).
    price_column : str
        Name of price column (long format).
    """

    data: Union[Mapping[date, Mapping[str, float]], Path, str] = field()
    config: HistoricalProviderConfig = field(default_factory=HistoricalProviderConfig)
    format: str = "wide"
    date_column: str = "date"
    instrument_column: str = "instrument"
    price_column: str = "price"

    name: str = "HistoricalProvider"

    def __post_init__(self) -> None:
        self._dates: List[str]
        self._symbols: List[str]
        self._mid_to_series: Dict[MarketId, np.ndarray]
        self._load()

    def _load(self) -> None:
        """Load data from dict or CSV into internal arrays."""
        raw = self.data

        if isinstance(raw, (Path, str)):
            path = Path(raw)
            if not path.exists():
                raise FileNotFoundError(f"CSV file not found: {path}")
            raw = self._load_csv(path)

        # raw is dict[date, dict[symbol, float]]
        dates_sorted = sorted(raw.keys())
        if not dates_sorted:
            raise ValueError("HistoricalProvider: no dates in data.")

        # Collect all symbols
        all_symbols: set[str] = set()
        for d in dates_sorted:
            all_symbols.update(raw[d].keys())
        symbols_sorted = sorted(all_symbols)

        # Build arrays: date -> index, symbol -> MarketId, mid -> array[T]
        n_t = len(dates_sorted)
        ac = self.config.asset_class
        mt = self.config.mkt_type

        mid_to_series: Dict[MarketId, np.ndarray] = {}
        for sym in symbols_sorted:
            mid = MarketId(ac, mt, sym)
            series = np.full(n_t, np.nan, dtype=float)
            for i, d in enumerate(dates_sorted):
                if sym in raw[d]:
                    series[i] = float(raw[d][sym])
            mid_to_series[mid] = series

        object.__setattr__(self, "_dates", [d.isoformat() if isinstance(d, date) else str(d) for d in dates_sorted])
        object.__setattr__(self, "_symbols", symbols_sorted)
        object.__setattr__(self, "_mid_to_series", mid_to_series)

    def _load_csv(self, path: Path) -> Dict[date, Dict[str, float]]:
        """Load CSV into dict[date, dict[symbol, float]]."""
        out: Dict[date, Dict[str, float]] = {}
        fmt = self.config.date_format

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if self.format == "wide":
                for row in reader:
                    dt = datetime.strptime(row[self.date_column], fmt).date()
                    prices = {}
                    for col, val in row.items():
                        if col != self.date_column and val:
                            try:
                                prices[col] = float(val)
                            except ValueError:
                                pass
                    out[dt] = prices
            else:
                for row in reader:
                    dt = datetime.strptime(row[self.date_column], fmt).date()
                    sym = row[self.instrument_column]
                    price = float(row[self.price_column])
                    if dt not in out:
                        out[dt] = {}
                    out[dt][sym] = price
        return out

    @property
    def dates(self) -> List[str]:
        """Ordered list of as-of date strings."""
        return self._dates

    @property
    def universe(self) -> Universe:
        """Universe of all MarketIds in this provider."""
        return Universe(ids=list(self._mid_to_series.keys()))

    def get_market(self, request: MarketRequest) -> Market:
        """Return a single as-of Market snapshot (quotes only)."""
        asof_str = request.asof if isinstance(request.asof, str) else request.asof.isoformat()
        if asof_str not in self._dates:
            raise ValueError(f"HistoricalProvider: no data for date {asof_str}.")

        time_idx = self._dates.index(asof_str)
        quotes: Dict[MarketId, Quote] = {}
        for mid in request.universe.ids:
            if mid not in self._mid_to_series:
                continue
            arr = self._mid_to_series[mid]
            if time_idx < len(arr) and np.isfinite(arr[time_idx]):
                quotes[mid] = Quote(value=float(arr[time_idx]))

        return Market(asof=asof_str, quotes=quotes, curves={}, vols={}, meta=None)

    def get_timeseries(self, request: TimeseriesRequest) -> MarketDataset:
        """Return a MarketDataset for the requested date range (quote panels only)."""
        start_str = request.start if isinstance(request.start, str) else request.start
        end_str = request.end if isinstance(request.end, str) else request.end

        # Resolve time indices
        start_d = date.fromisoformat(start_str)
        end_d = date.fromisoformat(end_str)
        time_indices: List[int] = []
        for i, d_str in enumerate(self._dates):
            d = date.fromisoformat(d_str)
            if start_d <= d <= end_d:
                time_indices.append(i)

        if not time_indices:
            raise ValueError(
                f"HistoricalProvider: no data in range [{start_str}, {end_str}]."
            )

        # Build quote panels for requested universe
        panels: Dict[MarketId, Panel] = {}
        for mid in request.universe.ids:
            if mid not in self._mid_to_series:
                continue
            full_series = self._mid_to_series[mid]
            sliced = np.array([full_series[i] for i in time_indices], dtype=float)
            panels[mid] = Panel(data=sliced, axis_names=("time",))

        if not panels:
            raise ValueError(
                "HistoricalProvider: no overlap between request universe and stored symbols."
            )

        dates_slice = [self._dates[i] for i in time_indices]
        return MarketDataset(
            dates=dates_slice,
            n_scenarios=1,
            panels=panels,
            curve_params={},
            curve_factories={},
            vol_params={},
            vol_factories={},
            meta={"provider": self.name, "freq": request.freq},
        )
