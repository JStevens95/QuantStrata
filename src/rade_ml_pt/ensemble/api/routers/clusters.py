"""``/prism/v1/clusters`` — per-cluster metadata (attributes + trade counts).

Reads ``cluster_attributes.parquet`` (B7 in the PRISM eval contract).
The parquet has dynamic columns, so this router picks out the two
fixed ones (``cluster_id``, ``n_trades``) and puts the rest into an
``attributes`` bag on each row.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.clusters import (
    ClusterInfo,
    ClustersResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["clusters"])


_FIXED_COLS = {"cluster_id", "n_trades"}


def _none_if_na(value: object) -> Optional[str]:
    """Coerce pandas missing-value markers to ``None`` for JSON output."""
    if value is None or pd.isna(value):
        return None
    return str(value)


@router.get("/clusters", response_model=ClustersResponse)
def get_clusters(
    cluster_id: Optional[str] = Query(
        None, description="Filter to one cluster. Optional."
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> ClustersResponse:
    """Return per-cluster metadata, optionally filtered to one cluster."""
    df = reader.cluster_attributes()

    if cluster_id is not None:
        df = df[df["cluster_id"] == cluster_id]
        if df.empty:
            raise HTTPException(
                status_code=404,
                detail=f"cluster_id '{cluster_id}' not found.",
            )

    attr_names = sorted(c for c in df.columns if c not in _FIXED_COLS)

    clusters = [
        ClusterInfo(
            cluster_id=str(row["cluster_id"]),
            n_trades=int(row["n_trades"]),
            attributes={k: _none_if_na(row[k]) for k in attr_names},
        )
        for _, row in df.iterrows()
    ]

    return ClustersResponse(
        attribute_names=attr_names,
        clusters=clusters,
    )
