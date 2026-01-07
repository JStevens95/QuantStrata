from __future__ import annotations

from typing import Any, Mapping
from dataclasses import dataclass

from src.marketdata.core.ids import MarketId
from src.marketdata.core.interfaces import Quote, Curve, VolSurface


@dataclass(frozen=True, slots=True)
class Market:
    """
    Immutable market snapshot consumed by pricing and risk.

    The key design rule:
    --------------------
    Pricers depend ONLY on Market (never on providers/files/apis).
    """
    asof: str
    quotes: Mapping[MarketId, Quote]
    curves: Mapping[MarketId, Curve]
    vols: Mapping[MarketId, VolSurface]
    meta: Mapping[str, Any] | None = None

    def quote(self, mkt_id: MarketId) -> float:
        """Return scalar quote value for mkt_id."""
        try:
            return float(self.quotes[mkt_id].value)
        except KeyError as exc:
            raise KeyError(f"Quote not found for MarketId {mkt_id.key()}") from exc

    def curve(self, mkt_id: MarketId) -> Curve:
        """Return curve object for mkt_id."""
        try:
            return self.curves[mkt_id]
        except KeyError as exc:
            raise KeyError(f"Curve not found for MarketId {mkt_id.key()}") from exc

    def vol_surface(self, mkt_id: MarketId) -> VolSurface:
        """Return vol surface object for mkt_id."""
        try:
            return self.vols[mkt_id]
        except KeyError as exc:
            raise KeyError(f"VolSurface not found for MarketId {mkt_id.key()}") from exc

    def has(self, mkt_id: MarketId) -> bool:
        """Fast existence check across quotes/curves/vols."""
        return (mkt_id in self.quotes) or (mkt_id in self.curves) or (mkt_id in self.vols)