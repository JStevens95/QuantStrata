"""Metadata endpoints — manifest, cluster attributes, trade map, graph stats."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.metadata import (
    ClusterAttributesResponse,
    GraphStatsEntry,
    GraphStatsResponse,
    ManifestResponse,
    TradeClusterMapResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/metadata")


@router.get("/manifest", response_model=ManifestResponse)
def get_manifest(cache: ArtifactCache = Depends(get_cache)):
    return ManifestResponse(**cache.manifest)


@router.get("/cluster-attributes", response_model=ClusterAttributesResponse)
def get_cluster_attributes(cache: ArtifactCache = Depends(get_cache)):
    return ClusterAttributesResponse(clusters=cache.cluster_attributes)


@router.get("/trade-cluster-map", response_model=TradeClusterMapResponse)
def get_trade_cluster_map(cache: ArtifactCache = Depends(get_cache)):
    return TradeClusterMapResponse(
        mapping=cache.trade_cluster_map,
        total_trades=len(cache.trade_cluster_map),
    )


@router.get("/graph-stats", response_model=GraphStatsResponse)
def get_graph_stats(cache: ArtifactCache = Depends(get_cache)):
    return GraphStatsResponse(
        clusters={
            cid: GraphStatsEntry(**stats)
            for cid, stats in cache.graph_stats.items()
        },
    )
