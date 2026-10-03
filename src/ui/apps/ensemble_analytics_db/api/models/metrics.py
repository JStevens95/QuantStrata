"""Response schemas for metrics endpoints."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel

__all__ = [
    "EnsembleMetricsResponse",
    "MemberMetricsResponse",
]


class EnsembleMetricsResponse(BaseModel):
    """Portfolio-level KPIs for a single split."""
    split: str
    metrics: Dict[str, float]


class MemberMetricsResponse(BaseModel):
    """Per-cluster metrics for a single split.

    Shape: ``{cluster_id: {metric_name: value}}``
    """
    split: str
    members: Dict[str, Dict[str, Any]]
