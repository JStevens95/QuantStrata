"""``/prism/v1/quality/*`` — per-feature data-quality artifacts (Phase 5e).

Two sibling endpoints, both filtered the same way (``split`` required,
``cluster_id`` optional):

- ``/prism/v1/quality/completeness``    — F2 schema (missingness & shape).
- ``/prism/v1/quality/feature-summary`` — F4 schema (numeric stats).

Both sources share the same ``cluster_id`` / ``feature_name`` key pair,
so clients can compose them in memory if they need a combined view.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.quality import (
    CompletenessResponse,
    CompletenessRow,
    FeatureSummaryResponse,
    FeatureSummaryRow,
)
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1/quality", tags=["quality"])


# ── Helpers ───────────────────────────────────────────────────────────

def _none_if_na(value: object) -> Optional[float]:
    """Convert pandas missing values (NaN, pd.NA, None) to ``None``."""
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _nan_safe_record(row: Dict[str, Any], float_cols: List[str]) -> Dict[str, Any]:
    """Replace pandas missing floats with ``None`` for pydantic coercion."""
    return {
        k: (_none_if_na(v) if k in float_cols else v)
        for k, v in row.items()
    }


_FEATURE_SUMMARY_FLOAT_COLS = [
    "mean", "std", "p01", "p50", "p99", "min", "max",
]


# ── Endpoints ─────────────────────────────────────────────────────────

@router.get("/completeness", response_model=CompletenessResponse)
def get_completeness(
    split: str = Query(..., description="train / val / test"),
    cluster_id: Optional[str] = Query(
        None, description="Filter to one cluster. Optional."
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> CompletenessResponse:
    try:
        df = reader.completeness(split, cluster_id=cluster_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"quality/completeness_{split}.parquet not found. "
                f"Splits available: {reader.available_splits()}."
            ),
        )

    if cluster_id is not None and df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"cluster_id '{cluster_id}' not found in split '{split}'.",
        )

    rows = [CompletenessRow(**row) for row in df.to_dict(orient="records")]
    return CompletenessResponse(
        split=split,
        n_clusters=df["cluster_id"].nunique() if not df.empty else 0,
        n_features_total=len(rows),
        rows=rows,
    )


@router.get("/feature-summary", response_model=FeatureSummaryResponse)
def get_feature_summary(
    split: str = Query(..., description="train / val / test"),
    cluster_id: Optional[str] = Query(
        None, description="Filter to one cluster. Optional."
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> FeatureSummaryResponse:
    try:
        df = reader.feature_summary(split, cluster_id=cluster_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"quality/feature_summary_{split}.parquet not found. "
                f"Splits available: {reader.available_splits()}."
            ),
        )

    if cluster_id is not None and df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"cluster_id '{cluster_id}' not found in split '{split}'.",
        )

    rows = [
        FeatureSummaryRow(**_nan_safe_record(row, _FEATURE_SUMMARY_FLOAT_COLS))
        for row in df.to_dict(orient="records")
    ]
    return FeatureSummaryResponse(
        split=split,
        n_clusters=df["cluster_id"].nunique() if not df.empty else 0,
        n_features_total=len(rows),
        rows=rows,
    )
