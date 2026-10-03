"""Pydantic response schemas for the ``/prism/v1/cluster-timeseries`` endpoint.

Mirrors ``cluster_timeseries_{split}.parquet`` (B2 schema v3): one
series per cluster, all sharing the same ``scenario_idx`` axis for the
split.  Measure-column types are nullable (``List[Optional[float]]``)
so original-space requests on runs with partial / no scaler coverage
can serialise NaN positions as JSON ``null`` — see
:class:`~src.rade_ml_pt.ensemble.api.models.portfolio.PortfolioTimeseriesResponse`
for the same rationale on the portfolio side.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ClusterTimeseries(BaseModel):
    """Per-cluster predictions vs targets series for one split.

    Columnar (parallel arrays) — see
    :class:`~src.rade_ml_pt.ensemble.api.models.portfolio.PortfolioTimeseriesResponse`
    for rationale.  All arrays share the same length ``n_scenarios``.
    """

    cluster_id: str
    n_scenarios: int

    scenario_idx: List[int]
    scenario_label: List[str]
    predictions: List[Optional[float]] = Field(..., description="Cluster predicted PnL per scenario.")
    targets: List[Optional[float]] = Field(..., description="Cluster actual (ground-truth) PnL per scenario.")
    error: List[Optional[float]] = Field(..., description="predictions − targets")
    abs_error: List[Optional[float]]
    squared_error: List[Optional[float]]


class ClusterTimeseriesResponse(BaseModel):
    """Collection response for ``GET /prism/v1/cluster-timeseries``.

    ``clusters`` has 1 element when ``cluster_id`` is supplied, ``N``
    elements otherwise (one per cluster in the ensemble).  ``space``
    identifies the PnL units of every nested cluster's measure columns
    (Phase 3.2).
    """

    split: str
    space: Literal["scaled", "original"] = Field(
        "scaled",
        description=(
            "PnL units of the measure columns on every nested cluster."
        ),
    )
    clusters: List[ClusterTimeseries]
