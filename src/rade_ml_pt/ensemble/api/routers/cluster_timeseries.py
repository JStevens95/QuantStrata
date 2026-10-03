"""``/prism/v1/cluster-timeseries`` — per-cluster portfolio series.

Reads ``cluster_timeseries_{split}.parquet`` (B2 in the PRISM eval
contract, schema v3).  Each cluster becomes one :class:`ClusterTimeseries`
entry with parallel arrays for predictions/targets/error columns.
``?space=scaled|original`` selects which PnL space's columns to
surface — see
:func:`src.rade_ml_pt.ensemble.api.services.reader._apply_space`.

Warning
-------
Omitting ``cluster_id`` returns every cluster's full series — payload
can easily reach tens of MB for large ensembles.  The typical UI
pattern is per-cluster; multi-cluster calls should be reserved for
small-multiples views on modest ensemble sizes.
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.cluster_timeseries import (
    ClusterTimeseries,
    ClusterTimeseriesResponse,
)
from src.rade_ml_pt.ensemble.api.routers.portfolio import _nullable_floats
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["cluster-timeseries"])


@router.get("/cluster-timeseries", response_model=ClusterTimeseriesResponse)
def get_cluster_timeseries(
    split: str = Query(..., description="train / val / test"),
    cluster_id: Optional[str] = Query(
        None,
        description=(
            "Filter to one cluster.  Omit to return every cluster (caution: "
            "response size scales with n_clusters × n_scenarios)."
        ),
    ),
    space: Literal["scaled", "original"] = Query(
        "scaled",
        description=(
            "PnL space for the measure columns on every nested cluster.  "
            "``scaled`` (default) is the model's z-space output; "
            "``original`` is inverse-scaled PnL in notional units.  "
            "Original-space values are ``null`` for clusters without "
            "scaler coverage."
        ),
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> ClusterTimeseriesResponse:
    """Return per-cluster portfolio series for one split."""
    try:
        df = reader.cluster_timeseries(split, cluster_id=cluster_id, space=space)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"cluster_timeseries_{split}.parquet not found. "
                f"Splits available: {reader.available_splits()}."
            ),
        )

    if cluster_id is not None and df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"cluster_id '{cluster_id}' not found in split '{split}'.",
        )

    clusters = []
    for cid, g in df.groupby("cluster_id", sort=True):
        clusters.append(
            ClusterTimeseries(
                cluster_id=str(cid),
                n_scenarios=len(g),
                scenario_idx=[int(x) for x in g["scenario_idx"]],
                scenario_label=[str(x) for x in g["scenario_label"]],
                predictions=_nullable_floats(g["predictions"]),
                targets=_nullable_floats(g["targets"]),
                error=_nullable_floats(g["error"]),
                abs_error=_nullable_floats(g["abs_error"]),
                squared_error=_nullable_floats(g["squared_error"]),
            )
        )

    return ClusterTimeseriesResponse(split=split, space=space, clusters=clusters)
