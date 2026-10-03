"""Cluster endpoints — summary timeseries and per-cluster predictions."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.timeseries import (
    ClusterPredictionsResponse,
    ClusterTimeseriesEntry,
    ClusterTimeseriesResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/clusters")


@router.get("/timeseries", response_model=ClusterTimeseriesResponse)
def get_cluster_timeseries(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    """Return per-cluster summed predictions and targets (1-D per cluster)."""
    data = cache.get_cluster_summary(split)
    if not data:
        raise HTTPException(404, f"No cluster summary for split '{split}'")
    first = next(iter(data.values()))
    n_scenarios = len(first.get("predictions", []))
    return ClusterTimeseriesResponse(
        split=split,
        n_scenarios=n_scenarios,
        clusters={
            cid: ClusterTimeseriesEntry(**entry)
            for cid, entry in data.items()
        },
    )


@router.get(
    "/{cluster_id}/predictions",
    response_model=ClusterPredictionsResponse,
)
def get_cluster_predictions(
    cluster_id: str,
    split: str = "test",
    mode: str = Query(
        "summary",
        description="'summary' for summed 1-D, 'full' for 2-D matrix, "
                    "'trades' for specific trade columns",
    ),
    trade_indices: Optional[str] = Query(
        None,
        description="Comma-separated column indices (only with mode='trades')",
    ),
    cache: ArtifactCache = Depends(get_cache),
):
    """Load per-trade predictions for a single cluster.

    Three modes control response size:

    * ``summary`` — cluster-level sums only (~20KB)
    * ``full`` — complete ``[n_scenarios x n_trades]`` matrix
    * ``trades`` — selected trade columns via ``trade_indices``
    """
    raw = cache.get_cluster_predictions(cluster_id, split)
    if raw is None:
        raise HTTPException(
            404, f"No predictions for cluster '{cluster_id}', split '{split}'"
        )

    preds, tgts = raw["predictions"], raw["targets"]
    if preds.ndim == 1:
        import numpy as np
        preds = preds.reshape(-1, 1)
        tgts = tgts.reshape(-1, 1)

    n_scenarios, n_trades = preds.shape

    if mode == "summary":
        return ClusterPredictionsResponse(
            cluster_id=cluster_id,
            split=split,
            mode=mode,
            n_scenarios=n_scenarios,
            n_trades=n_trades,
            cluster_pred=preds.sum(axis=1).tolist(),
            cluster_target=tgts.sum(axis=1).tolist(),
        )

    if mode == "trades" and trade_indices:
        idx = [int(i) for i in trade_indices.split(",")]
        idx = [i for i in idx if 0 <= i < n_trades]
        return ClusterPredictionsResponse(
            cluster_id=cluster_id,
            split=split,
            mode=mode,
            n_scenarios=n_scenarios,
            n_trades=len(idx),
            predictions=preds[:, idx].tolist(),
            targets=tgts[:, idx].tolist(),
            trade_indices=idx,
        )

    return ClusterPredictionsResponse(
        cluster_id=cluster_id,
        split=split,
        mode="full",
        n_scenarios=n_scenarios,
        n_trades=n_trades,
        predictions=preds.tolist(),
        targets=tgts.tolist(),
    )
