"""
STAR market-data client (WIRING STUB — your work API).

Implement each method to call the internal STAR market-data API and return the
payload type with the documented array shapes (see :mod:`clients.payloads` and the
mock client for a concrete reference implementation).
"""
from __future__ import annotations

from typing import Optional

from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)


class StarMarketDataClient:
    """Adapter onto the internal STAR market-data API."""

    def __init__(self, connection=None, **options) -> None:
        self.connection = connection
        self.options = options

    def fx_spot(self, pair: str, cob_date: str) -> float:
        raise NotImplementedError("TODO(wire): STAR FX spot")

    def discount_curve(self, currency: str, cob_date: str) -> CurvePayload:
        raise NotImplementedError("TODO(wire): STAR zero curve")

    def fx_vol_surface(self, pair: str, cob_date: str) -> SurfacePayload:
        raise NotImplementedError("TODO(wire): STAR FX vol surface")

    def fx_shocks(self, pair: str, cob_date: str) -> FXShockPayload:
        raise NotImplementedError("TODO(wire): STAR FX scenario states")

    def rates_vol_cube(self, factor_id: str, cob_date: str) -> Optional[CubePayload]:
        raise NotImplementedError("TODO(wire): STAR swaption cube (or None)")

    def rates_shocks(self, factor_id: str, cob_date: str) -> RatesShockPayload:
        raise NotImplementedError("TODO(wire): STAR rates scenario states")
