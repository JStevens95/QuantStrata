"""Response schemas for graph / GNN endpoints."""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel

__all__ = [
    "VersionCompareResponse",
    "MetricDelta",
    "AdminReloadResponse",
    "HealthResponse",
]


class MetricDelta(BaseModel):
    metric: str
    current: float
    compare: float
    delta: float
    pct_change: Optional[float] = None


class VersionCompareResponse(BaseModel):
    current_version: str
    compare_version: str
    split: str
    deltas: List[MetricDelta]


class AdminReloadResponse(BaseModel):
    status: str
    version: str
    n_clusters: int
    n_trades: int


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str = "file"
    cached_splits: List[str]
