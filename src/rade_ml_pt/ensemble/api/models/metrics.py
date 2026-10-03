"""Pydantic response schemas for the metrics endpoints.

Shared across ``/prism/v1/metrics/*`` and the headline section of
``/prism/v1/overview``.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class EnsembleSplitMetrics(BaseModel):
    """Portfolio-level aggregate metrics for one split.

    Also embedded in :class:`OverviewResponse` — keep the fields in
    lockstep with what the evaluation pipeline writes to
    ``ensemble_metrics.parquet`` (B4 schema v1).
    """

    split: str = Field(..., description="train / val / test")
    mae: float
    mse: float
    rmse: float
    max_ae: float
    p95_ae: float
    p99_ae: float


class PerMemberMetric(BaseModel):
    """Per-cluster aggregate metrics for one split.

    Mirrors ``per_member_metrics.parquet`` (B5 schema v1) — one row per
    ``(cluster_id, split)`` pair.
    """

    cluster_id: str
    split: str
    mae: float
    mse: float
    rmse: float
    max_ae: float
    p95_ae: float
    p99_ae: float
    n_targets: int
    n_scenarios: int


class EnsembleMetricsResponse(BaseModel):
    """Response wrapper for ``GET /prism/v1/metrics/ensemble``."""

    metrics: List[EnsembleSplitMetrics]


class PerMemberMetricsResponse(BaseModel):
    """Response wrapper for ``GET /prism/v1/metrics/per-member``."""

    metrics: List[PerMemberMetric]
