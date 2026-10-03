"""Response schemas for trade-level endpoints."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

__all__ = [
    "TradeMetricRow",
    "TradeMetricsResponse",
    "TradeCatalogueRow",
    "TradeCatalogueResponse",
]


class TradeMetricRow(BaseModel):
    cluster_id: str
    trade_id: str
    mae: float
    rmse: float
    max_ae: float
    p95_ae: float
    mean_residual: float
    std_residual: float


class TradeMetricsResponse(BaseModel):
    split: str
    total: int
    offset: int
    limit: int
    rows: List[TradeMetricRow]


class TradeCatalogueRow(BaseModel):
    trade_id: str
    cluster_id: str
    desk: Optional[str] = None
    product: Optional[str] = None
    ccy: Optional[str] = None


class TradeCatalogueResponse(BaseModel):
    total: int
    rows: List[TradeCatalogueRow]
