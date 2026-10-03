"""Response schemas for metadata endpoints."""
from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

__all__ = [
    "ManifestResponse",
    "ClusterAttributesResponse",
    "TradeClusterMapResponse",
    "GraphStatsEntry",
    "GraphStatsResponse",
    "VersionMetaResponse",
    "VersionListEntry",
    "VersionListResponse",
]


class ManifestResponse(BaseModel):
    trade_ids: List[str]
    cluster_ids: List[str]
    cluster_trade_indices: Dict[str, List[int]]
    splits_available: List[str]


class ClusterAttributesResponse(BaseModel):
    """``{cluster_id: {attr_name: value, ...}}``"""
    clusters: Dict[str, Dict[str, Any]]


class TradeClusterMapResponse(BaseModel):
    mapping: Dict[str, str] = Field(
        description="trade_id → cluster_id",
    )
    total_trades: int


class GraphStatsEntry(BaseModel):
    n_nodes: int = 0
    n_edges: int = 0
    density: float = 0.0
    mean_weight: float = 0.0


class GraphStatsResponse(BaseModel):
    clusters: Dict[str, GraphStatsEntry]


class VersionMetaResponse(BaseModel):
    version: str
    n_clusters: int
    n_trades: int
    splits_available: List[str]
    cluster_ids: List[str]


class VersionListEntry(BaseModel):
    version: str
    n_clusters: int = 0
    n_trades: int = 0


class VersionListResponse(BaseModel):
    versions: List[VersionListEntry]
    active_version: str
