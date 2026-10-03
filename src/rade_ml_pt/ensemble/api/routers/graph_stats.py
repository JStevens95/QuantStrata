"""``/prism/v1/graph-stats`` — per-cluster graph topology stats.

Reads ``graph_stats.parquet`` (B8 in the PRISM eval contract).
Ensemble-scoped: one row per cluster, no split dimension — used by the
Cluster Deep Dive "Graph Statistics" card and by the Trade Graph tab's
right-hand metadata panel.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.graph_stats import (
    GraphStatsResponse,
    GraphStatsRow,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["graph-stats"])


@router.get("/graph-stats", response_model=GraphStatsResponse)
def get_graph_stats(
    cluster_id: Optional[str] = Query(
        None, description="Filter to one cluster. Optional."
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> GraphStatsResponse:
    try:
        df = reader.graph_stats(cluster_id=cluster_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="graph_stats.parquet not found for the active version.",
        )

    if cluster_id is not None and df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"cluster_id '{cluster_id}' not found in graph_stats.parquet.",
        )

    rows = [GraphStatsRow(**row) for row in df.to_dict(orient="records")]
    return GraphStatsResponse(n_clusters=len(rows), rows=rows)
