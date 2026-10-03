"""
Client ports — the abstract interfaces the pipeline depends on.

The pipeline never touches Sage/STAR/CSV directly; it depends on these two
protocols. Any object with the right methods works (mock, file, or your API
adapters). This is the dependency-inversion seam: swap data source without touching
business logic.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)
from src.rade_static_replication.domain.contracts import RawPortfolio


@runtime_checkable
class PortfolioClient(Protocol):
    """Source of the target portfolio (trade attributes + scenario PnL)."""

    def load(self, cob_date: str) -> RawPortfolio:
        ...


@runtime_checkable
class MarketDataClient(Protocol):
    """Source of COB market data and scenario shocks, per risk factor."""

    def fx_spot(self, pair: str, cob_date: str) -> float: ...

    def discount_curve(self, currency: str, cob_date: str) -> CurvePayload: ...

    def fx_vol_surface(self, pair: str, cob_date: str) -> SurfacePayload: ...

    def fx_shocks(self, pair: str, cob_date: str) -> FXShockPayload: ...

    def rates_vol_cube(self, factor_id: str, cob_date: str) -> Optional[CubePayload]: ...

    def rates_shocks(self, factor_id: str, cob_date: str) -> RatesShockPayload: ...
