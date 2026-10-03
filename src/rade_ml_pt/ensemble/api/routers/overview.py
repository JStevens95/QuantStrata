"""``/prism/v1/overview`` — headline numbers for the PRISM Overview tab.

This endpoint is the template for every subsequent router:

1. Router declared at module scope with a ``/prism/v1`` prefix.
2. Path operation takes ``reader: ArtifactReader = Depends(get_reader)``.
3. Body is pure orchestration: pull the needed DataFrames/dicts off the
   reader, convert to the pydantic response model, return.
4. No business logic — any serialisation quirks that are shared by
   multiple endpoints move into :mod:`ArtifactReader`.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.metrics import EnsembleSplitMetrics
from src.rade_ml_pt.ensemble.api.models.overview import OverviewResponse
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["overview"])


@router.get("/overview", response_model=OverviewResponse)
def get_overview(
    reader: ArtifactReader = Depends(get_reader),
) -> OverviewResponse:
    """Return headline numbers for the active ensemble version."""
    manifest = reader.manifest()
    metrics_df = reader.ensemble_metrics()

    metrics = [
        EnsembleSplitMetrics(**row)
        for row in metrics_df.to_dict(orient="records")
    ]

    return OverviewResponse(
        version=str(manifest["version"]),
        evaluated_at=str(manifest["evaluated_at"]),
        n_clusters=len(manifest.get("cluster_ids", [])),
        n_trades=len(manifest.get("trade_ids", [])),
        splits_available=list(manifest.get("splits_available", [])),
        ensemble_metrics=metrics,
    )
