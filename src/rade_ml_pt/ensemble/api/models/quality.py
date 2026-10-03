"""Pydantic response schemas for the ``/prism/v1/quality/*`` endpoints.

Backs Phase 5e in the PRISM eval contract — per-split, per-cluster,
per-feature data-quality parquets concatenated across the ensemble.

Two flavours:

- **completeness** — missing / inf / zero / distinct counts.
- **feature-summary** — numeric summary statistics (mean/std/percentiles).

Both carry the same ``cluster_id`` / ``feature_name`` key pair, so a
client can join the two into a single "row per feature" view in memory.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Completeness ──────────────────────────────────────────────────────

class CompletenessRow(BaseModel):
    """Per-feature completeness record (F2 schema)."""

    cluster_id: str
    feature_name: str
    dtype: str = Field(..., description="Original pandas dtype, for auditability.")
    n_rows: int
    n_null: int = Field(..., description="``pd.isna`` count — subsumes NaN and pd.NA.")
    null_rate: float
    n_distinct: int = Field(..., description="Distinct count over non-null values.")
    n_zero: int = Field(..., description="Exact zeros, excluding nulls and inf.")
    n_inf: int = Field(..., description="Positive + negative infinity count.")
    n_nan: int = Field(..., description="IEEE-754 NaN count specifically.")


class CompletenessResponse(BaseModel):
    split: str
    n_clusters: int
    n_features_total: int = Field(
        ..., description="Sum of feature rows across all clusters in the response."
    )
    rows: List[CompletenessRow]


# ── Feature summary ───────────────────────────────────────────────────

class FeatureSummaryRow(BaseModel):
    """Per-feature numeric summary (F4 schema).

    Stats are computed on **finite values only**.  If a column has zero
    finite values all stat fields are ``None``.
    """

    cluster_id: str
    feature_name: str
    count: int = Field(..., description="Number of finite values in the column.")
    mean: Optional[float]
    std: Optional[float] = Field(..., description="Population std (ddof=0).")
    p01: Optional[float]
    p50: Optional[float]
    p99: Optional[float]
    min: Optional[float] = Field(..., alias="min")
    max: Optional[float] = Field(..., alias="max")

    model_config = {"populate_by_name": True}


class FeatureSummaryResponse(BaseModel):
    split: str
    n_clusters: int
    n_features_total: int
    rows: List[FeatureSummaryRow]
