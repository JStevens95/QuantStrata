"""``/prism/v1/metrics/*`` — ensemble and per-member aggregate metrics.

Reads:

- ``ensemble_metrics.parquet``    (B4) — one row per split
- ``per_member_metrics.parquet``  (B5) — one row per ``(cluster_id, split)``
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.metrics import (
    EnsembleMetricsResponse,
    EnsembleSplitMetrics,
    PerMemberMetric,
    PerMemberMetricsResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1/metrics", tags=["metrics"])


@router.get("/ensemble", response_model=EnsembleMetricsResponse)
def get_ensemble_metrics(
    reader: ArtifactReader = Depends(get_reader),
) -> EnsembleMetricsResponse:
    """Return portfolio-level aggregate metrics (one row per split)."""
    df = reader.ensemble_metrics()
    metrics = [
        EnsembleSplitMetrics(**row) for row in df.to_dict(orient="records")
    ]
    return EnsembleMetricsResponse(metrics=metrics)


@router.get("/per-member", response_model=PerMemberMetricsResponse)
def get_per_member_metrics(
    split: Optional[str] = Query(
        None, description="Filter to one split (train/val/test). Optional."
    ),
    cluster_id: Optional[str] = Query(
        None, description="Filter to one cluster. Optional."
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> PerMemberMetricsResponse:
    """Return per-cluster aggregate metrics, optionally filtered."""
    df = reader.per_member_metrics(split=split, cluster_id=cluster_id)
    metrics = [
        PerMemberMetric(**row) for row in df.to_dict(orient="records")
    ]
    return PerMemberMetricsResponse(metrics=metrics)
