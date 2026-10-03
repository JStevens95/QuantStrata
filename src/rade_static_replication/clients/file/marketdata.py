"""
File-based market-data client (WIRING STUB).

Implement these to read market data + shocks from local files (e.g. STAR extracts).
Each method must return the payload type with the documented array shapes. Delete the
``NotImplementedError`` lines as you wire each one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)


class FileMarketDataClient:
    """Read market data + shocks from a directory layout you define."""

    def __init__(self, root) -> None:
        self.root = Path(root)

    def fx_spot(self, pair: str, cob_date: str) -> float:
        raise NotImplementedError("TODO(wire): read FX spot for `pair` as of `cob_date`")

    def discount_curve(self, currency: str, cob_date: str) -> CurvePayload:
        # return CurvePayload(tenors=np.array([...]), zero_rates=np.array([...]))
        raise NotImplementedError("TODO(wire): read zero curve for `currency`")

    def fx_vol_surface(self, pair: str, cob_date: str) -> SurfacePayload:
        raise NotImplementedError("TODO(wire): read FX vol surface for `pair`")

    def fx_shocks(self, pair: str, cob_date: str) -> FXShockPayload:
        raise NotImplementedError("TODO(wire): read FX scenario states for `pair`")

    def rates_vol_cube(self, factor_id: str, cob_date: str) -> Optional[CubePayload]:
        raise NotImplementedError("TODO(wire): read swaption cube for `factor_id` (or return None)")

    def rates_shocks(self, factor_id: str, cob_date: str) -> RatesShockPayload:
        raise NotImplementedError("TODO(wire): read rates scenario states for `factor_id`")
