"""
Client layer (infrastructure / ports & adapters).

The pipeline depends only on the two *ports* (:class:`PortfolioClient`,
:class:`MarketDataClient`) and the payload dataclasses. Concrete adapters
(``mock``, ``file``, ``api``) are imported explicitly by the caller, keeping this
package import light.
"""
from src.rade_static_replication.clients.base import MarketDataClient, PortfolioClient
from src.rade_static_replication.clients.payloads import (
    CubePayload,
    CurvePayload,
    FXShockPayload,
    RatesShockPayload,
    SurfacePayload,
)

__all__ = [
    "PortfolioClient", "MarketDataClient",
    "CurvePayload", "SurfacePayload", "CubePayload", "FXShockPayload", "RatesShockPayload",
]
