from __future__ import annotations

from typing import Mapping, Any
from dataclasses import dataclass

from src.marketdata.ids import MarketId
from src.marketdata.interfaces import Quote, Curve, VolSurface


@dataclass(frozen=True, slots=True)
class Market:
    """
    Immutable market snapshot used by pricing and risk.

    This object is the only thing pricers should depend on.
    It intentionally does NOT know where data came from (API/files/synthetic).

    Contents
    --------
    - quotes: MarketId -> Quote        (spots, fixings, scalar params)
    - curves: MarketId -> Curve        (discount/forecast curves)
    - vols:   MarketId -> VolSurface   (implied vol surfaces)
    """
    # initiate required variables.
    asof: str
    quotes: Mapping[MarketId, Quote]
    curves: Mapping[MarketId, Curve]
    vols: Mapping[MarketId, VolSurface]
    meta: Mapping[str, Any] | None = None

    def quote(self, mkt_id: MarketId) -> float:
        """Return a scalar quote value for the given MarketId."""
        try:
            return self.quotes[mkt_id].value
        except KeyError as exc:
            raise KeyError(f"Quote not found for MarketId {mkt_id.key()}") from exc

    def curve(self, mkt_id: MarketId) -> Curve:
        """Return a scalar curve value for the given MarketId."""
        try:
            return self.curves[mkt_id]
        except KeyError as exc:
            raise KeyError(f"Curve not found for MarketId {mkt_id.key()}") from exc

    def vol_surface(self, mkt_id: MarketId) -> VolSurface:
        """Return a vol surface object for the given MarketId."""
        try:
            return self.vols[mkt_id]
        except KeyError as exc:
            raise KeyError(f"VolSurface not found for MarketId {mkt_id.key()}") from exc

    def has(self, mkt_id: MarketId) -> bool:
        """Fast existence check across quotes/curves/vols."""
        return (mkt_id in self.quotes) or (mkt_id in self.curves) or (mkt_id in self.vols)
