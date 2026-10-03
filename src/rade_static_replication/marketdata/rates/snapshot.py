"""Rates market snapshot — base market for one currency curve (+ optional cube)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.rade_static_replication.marketdata.common.curves import DiscountCurve
from src.rade_static_replication.marketdata.rates.instruments import VolCube
from src.rade_static_replication.marketdata.snapshot import MarketSnapshot


@dataclass(frozen=True)
class RatesSnapshot(MarketSnapshot):
    """Base rates market for one currency curve (+ optional swaption cube)."""
    discount_curve: DiscountCurve = None  # type: ignore[assignment]
    vol_cube: Optional[VolCube] = None
