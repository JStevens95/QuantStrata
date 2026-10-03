"""FX market snapshot — base market for one USD-facing pair."""
from __future__ import annotations

from dataclasses import dataclass

from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.fx.instruments import Spot, VolSurface
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot


@dataclass(frozen=True)
class FXSnapshot(MarketSnapshot):
    """Base FX market for one USD-facing pair (e.g. ``EURUSD``)."""
    spot: Spot = None  # type: ignore[assignment]
    domestic_curve: DiscountCurve = None  # numeraire / quote ccy (e.g. USD)
    foreign_curve: DiscountCurve = None   # asset / base ccy (e.g. EUR)
    vol_surface: VolSurface = None

    def forward(self, expiry: float) -> float:
        """Outright FX forward via covered interest parity: ``S * DF_f / DF_d``."""
        return self.spot.value * self.foreign_curve.df(expiry) / self.domestic_curve.df(expiry)
