"""``/prism/v1/clusters/{cluster_id}/elementary-pnl`` — per-scenario PnL
of one or more elementary trades inside a cluster.

Reads ``members/{cluster_id}/elementary_pnl.parquet``, staged by the
eval pipeline from the training registry alongside
``training_curves.parquet`` (trainer-side data contract).  Used by the
Cluster Deep-Dive "Elementary PnL Explorer" row: the user picks one or
more elementary trades from the explorer table, the UI fetches their
per-scenario PnL series and plots them as a multi-line chart.

Elementary trades are *model inputs* (the atomic legs / hedge
instruments composing target structures), so this endpoint
deliberately returns the raw PnL only — no predictions, no targets.

Filtering
---------
``trade_ids`` is a required, repeatable query parameter:

.. code-block:: text

    GET /prism/v1/clusters/cluster_0/elementary-pnl
        ?trade_ids=t_eur_usd_atmv
        &trade_ids=t_eur_usd_25rr
        &trade_ids=t_eur_usd_25fly

The filter is applied server-side so payloads stay tight (5 000
scenarios × 50 trades × 4 bytes ≈ 1 MB raw, ≈ 4 MB JSON-serialised).
Trades that don't exist in the cluster's parquet are silently
omitted; the response's ``n_trades`` plus
``set(payload.pnl) ⊆ set(trade_ids)`` lets the UI flag dropped ids
without a second call.
"""
from __future__ import annotations

import math
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.elementary_pnl import (
    ElementaryPnlResponse,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["elementary-pnl"])


# Cap the per-request column count to keep the JSON payload bounded
# even if a misbehaving caller asks for the entire cluster.  The UI's
# multi-select on the Explorer table caps at 12 trades by default;
# 50 leaves comfortable headroom for power-user URLs without making
# the worst case unbounded.
_MAX_TRADES_PER_REQUEST = 50


@router.get(
    "/clusters/{cluster_id}/elementary-pnl",
    response_model=ElementaryPnlResponse,
)
def get_elementary_pnl(
    cluster_id: str,
    trade_ids: List[str] = Query(
        ...,
        description=(
            "Elementary trade ids to project from the cluster's "
            "elementary_pnl.parquet.  At least one required."
        ),
        min_length=1,
        max_length=_MAX_TRADES_PER_REQUEST,
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> ElementaryPnlResponse:
    """Return per-scenario PnL for the requested elementary trades."""
    try:
        df = reader.elementary_pnl(cluster_id=cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if df is None or df.empty:
        raise HTTPException(
            status_code=500,
            detail=(
                f"elementary_pnl.parquet for cluster '{cluster_id}' "
                "is present but empty.  This is a pipeline contract "
                "break — file a bug."
            ),
        )

    # Project to the requested columns; drop unknown trade ids
    # silently — the UI uses ``len(pnl) < len(trade_ids)`` as the
    # tripwire for "some of your selected trades aren't in this
    # cluster" rather than failing the whole request.
    available = [tid for tid in trade_ids if tid in df.columns]
    if not available:
        raise HTTPException(
            status_code=404,
            detail=(
                f"None of the requested trade_ids exist in cluster "
                f"'{cluster_id}'.  Requested: {trade_ids}.  "
                f"Available (first 5): {list(df.columns[:5])}…"
            ),
        )

    n_scenarios = int(len(df))
    pnl: dict[str, list[float]] = {
        tid: _to_clean_floats(df[tid]) for tid in available
    }

    return ElementaryPnlResponse(
        cluster_id=cluster_id,
        n_scenarios=n_scenarios,
        n_trades=len(pnl),
        # 0-indexed integer axis — the parquet's RangeIndex is the
        # canonical scenario index across the rest of the API
        # (cluster_timeseries, predictions, …) so we don't try to
        # introspect a label column here.
        scenario_idx=list(range(n_scenarios)),
        pnl=pnl,
    )


def _to_clean_floats(series) -> list[float]:
    """Coerce a pandas Series to ``list[float]`` with no non-finite values.

    Same shape as the helper in ``training_curves.py`` — JSON can't
    round-trip ``NaN`` / ``inf`` so we clamp those to ``0.0``.  In
    practice the elementary PnL parquet is float32 and finite, but
    legacy runs occasionally carry an isolated ``NaN`` from a
    pricer hiccup; we'd rather plot a zero than 500 the whole
    cluster.
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
