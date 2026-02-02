"""
Adapter to use marketdata providers with the backtesting engine.

BacktestEngine expects a simple protocol: get_dates(), get_snapshot(date)
with get_price(instrument_id). MarketDataProvider returns Market snapshots
keyed by MarketId. This adapter wraps any MarketDataProvider so the engine
can use StaticProvider, SyntheticProvider, or HistoricalProvider without
duplicating data loading logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, List, Mapping, Optional, Sequence

from src.marketdata.core.ids import MarketId
from src.marketdata.core.market import Market
from src.marketdata.core.requests import MarketRequest, TimeseriesRequest, Universe
from src.marketdata.providers.interfaces import MarketDataProvider


# =============================================================================
# Market snapshot wrapper
# =============================================================================

@dataclass
class MarketSnapshotAdapter:
    """
    Wraps a Market snapshot to expose get_price(symbol) for backtesting.

    Strategies use string instrument ids (e.g. "AAPL"). Market uses MarketId.
    This adapter holds a mapping symbol -> MarketId and delegates get_price(symbol)
    to market.quote(mid).
    """

    asof: date
    market: Market
    symbol_to_mid: Mapping[str, MarketId]

    def get_price(self, instrument_id: str) -> float:
        """Get price for instrument (by symbol or MarketId key)."""
        mid = self.symbol_to_mid.get(instrument_id)
        if mid is None:
            # Try by name: maybe instrument_id is the MarketId name
            for m in self.symbol_to_mid.values():
                if m.name == instrument_id:
                    mid = m
                    break
        if mid is None:
            raise KeyError(f"No price for {instrument_id} on {self.asof}")
        return self.market.quote(mid)

    def __contains__(self, instrument_id: str) -> bool:
        """Check if instrument has price data."""
        if instrument_id in self.symbol_to_mid:
            return True
        return any(m.name == instrument_id for m in self.symbol_to_mid.values())


# =============================================================================
# BacktestDataAdapter
# =============================================================================

@dataclass
class BacktestDataAdapter:
    """
    Adapts a MarketDataProvider to the backtest DataProvider protocol.

    BacktestEngine expects:
    - get_dates() -> Sequence[date]
    - get_snapshot(dt: date) -> snapshot with get_price(instrument_id)

    This adapter wraps any MarketDataProvider: it uses get_timeseries() to get
    the date range, then get_market(asof=date) for each date and wraps each
    Market in MarketSnapshotAdapter with a symbol -> MarketId mapping.

    Parameters
    ----------
    provider : MarketDataProvider
        Any marketdata provider (HistoricalProvider, StaticProvider, etc.).
    universe : Universe, optional
        Universe of MarketIds to request. If None, uses provider.universe when
        the provider is HistoricalProvider; otherwise must be supplied.
    symbol_to_mid : dict, optional
        Mapping from strategy symbol (e.g. "AAPL") to MarketId. If None, symbols
        are taken as MarketId.name for each id in universe.
    """

    provider: MarketDataProvider
    universe: Optional[Universe] = None
    symbol_to_mid: Optional[Mapping[str, MarketId]] = None

    _dates: List[date] = field(init=False, default_factory=list)
    _universe: Optional[Universe] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Resolve universe: use provider.universe for HistoricalProvider if not given
        if self.universe is not None:
            object.__setattr__(self, "_universe", self.universe)
        elif hasattr(self.provider, "universe"):
            object.__setattr__(self, "_universe", self.provider.universe)
        else:
            raise ValueError(
                "BacktestDataAdapter requires universe= or a provider with .universe "
                "(e.g. HistoricalProvider)."
            )

    def get_dates(self) -> Sequence[date]:
        """Return all available dates in order."""
        if self._dates:
            return self._dates
        # Prefer provider.dates (HistoricalProvider) or provider.dataset.dates (StaticProvider)
        date_strings: List[str] = []
        if hasattr(self.provider, "dates") and getattr(self.provider, "dates", None):
            date_strings = list(self.provider.dates)
        elif hasattr(self.provider, "dataset") and getattr(self.provider.dataset, "dates", None):
            date_strings = list(self.provider.dataset.dates)
        if date_strings:
            dates = [date.fromisoformat(d) for d in date_strings]
            object.__setattr__(self, "_dates", dates)
            return dates
        # Fallback: request full range via get_timeseries
        req = TimeseriesRequest(
            start="2000-01-01",
            end="2030-12-31",
            freq="D",
            universe=self._universe,
            scenarios=1,
        )
        ds = self.provider.get_timeseries(req)
        dates = [date.fromisoformat(d) for d in ds.dates]
        object.__setattr__(self, "_dates", dates)
        return dates

    def get_snapshot(self, dt: date) -> MarketSnapshotAdapter:
        """Return a snapshot for the given date (wraps Market)."""
        if isinstance(dt, datetime):
            dt = dt.date()
        asof_str = dt.isoformat()
        req = MarketRequest(asof=asof_str, universe=self._universe)
        try:
            market = self.provider.get_market(req)
        except ValueError as e:
            if "no data" in str(e).lower() or "missing" in str(e).lower() or "not found" in str(e).lower():
                raise KeyError(f"No data for {asof_str}") from e
            raise
        # Build symbol -> MarketId: use provided map or name -> mid
        if self.symbol_to_mid is not None:
            symbol_to_mid = dict(self.symbol_to_mid)
        else:
            symbol_to_mid = {mid.name: mid for mid in market.quotes.keys()}
        return MarketSnapshotAdapter(
            asof=dt if isinstance(dt, date) else date.fromisoformat(str(dt)),
            market=market,
            symbol_to_mid=symbol_to_mid,
        )
