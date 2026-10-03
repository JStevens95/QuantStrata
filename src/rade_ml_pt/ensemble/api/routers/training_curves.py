"""``/prism/v1/clusters/{cluster_id}/training-curves`` — per-cluster
training curves.

Reads ``members/{cluster_id}/training_curves.parquet``, staged by the
eval pipeline from the training registry (trainer-side contract
§11.15.1).  Used by the Evaluation → Cluster Deep-Dive UI tab to render
a multi-series training curve plot with a metric-picker chip row.

The endpoint is cluster-scoped because each ensemble member carries its
own training history — the UI lets the user pivot to whichever cluster
they're inspecting without refetching other clusters' curves.
"""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.training_curves import (
    TrainingCurvesResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["training-curves"])


@router.get(
    "/clusters/{cluster_id}/training-curves",
    response_model=TrainingCurvesResponse,
)
def get_training_curves(
    cluster_id: str,
    reader: ArtifactReader = Depends(get_reader),
) -> TrainingCurvesResponse:
    """Return per-epoch training curves for one cluster member.

    Responses are columnar — one array per metric — so the UI can plot
    any subset of metrics without having to re-zip rows.  ``train_loss``
    is always present; other series are whatever the trainer emitted.
    """
    try:
        df = reader.training_curves(cluster_id=cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if df is None or df.empty or "epoch" not in df.columns or "train_loss" not in df.columns:
        raise HTTPException(
            status_code=500,
            detail=(
                f"training_curves.parquet for cluster '{cluster_id}' "
                "is present but malformed (missing 'epoch' / 'train_loss'). "
                "This is a pipeline contract break — file a bug."
            ),
        )

    df_sorted = df.sort_values("epoch").reset_index(drop=True)
    n_epochs = int(len(df_sorted))

    series: dict[str, list[float]] = {
        col: _to_clean_floats(df_sorted[col])
        for col in df_sorted.columns
        if col != "epoch"
    }
    metrics = sorted(k for k in series if k != "train_loss")

    return TrainingCurvesResponse(
        cluster_id=cluster_id,
        n_epochs=n_epochs,
        epoch=[int(v) for v in df_sorted["epoch"].tolist()],
        series=series,
        metrics=metrics,
    )


def _to_clean_floats(series) -> list[float]:
    """Coerce to ``list[float]`` and replace ``NaN`` / ``inf`` with ``0.0``.

    Pydantic / JSON can't round-trip non-finite floats, so anything
    pathological is clamped before serialisation.  Training runs that
    diverge (``nan`` loss) still produce a plottable payload; the UI
    flags the divergence separately via the summary chip strip.
    """
    out: list[float] = []
    for v in series.tolist():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        if not math.isfinite(fv):
            fv = 0.0
        out.append(fv)
    return out
