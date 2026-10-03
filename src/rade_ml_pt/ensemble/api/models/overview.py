"""Pydantic response schemas for the ``/prism/v1/overview`` endpoint."""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from src.rade_ml_pt.ensemble.api.models.metrics import EnsembleSplitMetrics


class OverviewResponse(BaseModel):
    """Headline summary of one ensemble evaluation run.

    Powers the PRISM Overview tab (version banner, count cards,
    per-split metric tiles).
    """

    version: str = Field(..., description="Concrete ensemble version directory name.")
    evaluated_at: str = Field(
        ...,
        description="ISO-8601 UTC timestamp recorded when eval ran.",
    )
    n_clusters: int
    n_trades: int
    splits_available: List[str] = Field(
        ...,
        description="Splits for which predictions were evaluated.",
    )
    ensemble_metrics: List[EnsembleSplitMetrics] = Field(
        ...,
        description="One entry per split in ``splits_available``.",
    )
