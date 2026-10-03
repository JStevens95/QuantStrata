"""Metrics endpoints — ensemble KPIs and per-member metrics."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.metrics import (
    EnsembleMetricsResponse,
    MemberMetricsResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/metrics")


@router.get("/ensemble", response_model=EnsembleMetricsResponse)
def get_ensemble_metrics(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    data = cache.ensemble_metrics.get(split)
    if data is None:
        raise HTTPException(404, f"No ensemble metrics for split '{split}'")
    return EnsembleMetricsResponse(split=split, metrics=data)


@router.get("/members", response_model=MemberMetricsResponse)
def get_member_metrics(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    data = cache.member_metrics.get(split)
    if data is None:
        raise HTTPException(404, f"No member metrics for split '{split}'")
    return MemberMetricsResponse(split=split, members=data)
