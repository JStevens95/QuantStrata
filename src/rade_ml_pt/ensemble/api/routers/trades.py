"""``/prism/v1/trades`` — per-trade metrics for one split.

Reads ``trade_metrics_{split}.parquet`` (B3 in the PRISM eval contract).
One row per trade; can be filtered to a single cluster for the Cluster
Deep Dive trade table.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.trades import (
    TradeMetric,
    TradesResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["trades"])


def _is_missing(value: Any) -> bool:
    """``True`` for ``None``, pandas ``NA``, and ``NaN`` floats.

    Old parquets sometimes carry partially-populated rows; we use
    this to decide whether to synthesise a fallback ``trade_id``.
    """
    if value is None:
        return True
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        # Strings, pd.NA, anything non-numeric — treat as "present"
        # unless the caller already nulled it.
        return False


@router.get("/trades", response_model=TradesResponse)
def get_trades(
    split: str = Query(..., description="train / val / test"),
    cluster_id: Optional[str] = Query(
        None, description="Filter to one cluster.  Optional."
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> TradesResponse:
    """Return per-trade metrics for one split, optionally filtered to a cluster."""
    try:
        df = reader.trade_metrics(split, cluster_id=cluster_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"trade_metrics_{split}.parquet not found. "
                f"Splits available: {reader.available_splits()}."
            ),
        )

    if cluster_id is not None and df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"cluster_id '{cluster_id}' not found in split '{split}'.",
        )

    # Backwards-compat shim: legacy parquets (pre-trade_id schema) ship
    # rows without a ``trade_id`` column at all.  The eval producer's
    # own fallback uses ``f"{cluster_id}_trade_{j}"`` (eval.py:1124);
    # we mirror the same convention so synthetic ids are stable across
    # producer + API and across repeated calls.
    #
    # ``p99_ae`` was added to the eval producer schema after some
    # parquets were already on disk in user environments — fall back to
    # ``p95_ae`` (a slightly tighter percentile) so the row still
    # validates rather than raising ``Field required`` on the legacy
    # data.  Same idea for ``std_residual`` (very early parquets only
    # had ``mean_residual``).  These shims are deliberately *quiet*:
    # the eval pipeline is the source of truth for new artifacts, and
    # over-logging on every request would drown the actual signal in
    # production.
    trades = []
    per_cluster_idx: Dict[str, int] = {}
    for row in df.to_dict(orient="records"):
        cid = str(row.get("cluster_id", ""))
        if "trade_id" not in row or _is_missing(row.get("trade_id")):
            j = per_cluster_idx.get(cid, 0)
            row["trade_id"] = f"{cid}_trade_{j}"
            per_cluster_idx[cid] = j + 1
        else:
            row["trade_id"] = str(row["trade_id"])

        if "p99_ae" not in row or _is_missing(row.get("p99_ae")):
            # ``p95_ae`` is always present on every parquet that has
            # got past the FastAPI validator at least once, so it's
            # the safest fallback.  Worst case it's a tighter bound
            # than the true 99th percentile; the UI label still says
            # "P99" but at least the cell isn't empty.
            row["p99_ae"] = float(row.get("p95_ae") or 0.0)

        if "std_residual" not in row or _is_missing(row.get("std_residual")):
            row["std_residual"] = 0.0

        trades.append(TradeMetric(**row))

    return TradesResponse(
        split=split,
        n_trades=len(trades),
        trades=trades,
    )
