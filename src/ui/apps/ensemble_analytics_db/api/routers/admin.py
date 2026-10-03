"""Admin endpoints — health check, cache reload."""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.graph import (
    AdminReloadResponse,
    HealthResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/admin")


@router.get("/health", response_model=HealthResponse)
def health_check(cache: ArtifactCache = Depends(get_cache)):
    return HealthResponse(
        status="ok",
        version=cache.ensemble_version or cache.manifest.get("version", "unknown"),
        backend="cache",
        cached_splits=cache.cached_splits,
    )


@router.post("/reload", response_model=AdminReloadResponse)
def reload_cache(cache: ArtifactCache = Depends(get_cache)):
    """Force-reload all artifacts from disk (e.g. after a new eval run)."""
    cache.reload()
    return AdminReloadResponse(
        status="reloaded",
        version=cache.manifest.get("version", "unknown"),
        n_clusters=len(cache.cluster_attributes),
        n_trades=len(cache.trade_cluster_map),
    )
