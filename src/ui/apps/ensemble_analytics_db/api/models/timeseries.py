"""Response schemas for timeseries / array endpoints."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

__all__ = [
    "PortfolioTimeseriesResponse",
    "PortfolioStatRow",
    "PortfolioStatsResponse",
    "WorstScenarioRow",
    "WorstScenariosResponse",
    "ClusterTimeseriesEntry",
    "ClusterTimeseriesResponse",
    "ClusterPredictionsResponse",
    "GroupTimeseriesEntry",
    "GroupTimeseriesResponse",
    "GroupSummaryEntry",
    "GroupSummariesResponse",
    "CorrelationMatrix",
    "GroupCorrelationsResponse",
]


class PortfolioTimeseriesResponse(BaseModel):
    split: str
    n_scenarios: int
    predictions: List[float]
    targets: List[float]


class PortfolioStatRow(BaseModel):
    metric: str
    predicted: Optional[float] = None
    target: Optional[float] = None
    diff: Optional[float] = None
    abs_error: Optional[float] = None


class PortfolioStatsResponse(BaseModel):
    split: str
    rows: List[PortfolioStatRow]


class WorstScenarioRow(BaseModel):
    rank: int
    scenario: int
    target: float
    prediction: float
    abs_error: float


class WorstScenariosResponse(BaseModel):
    split: str
    rows: List[WorstScenarioRow]


class ClusterTimeseriesEntry(BaseModel):
    predictions: List[float]
    targets: List[float]


class ClusterTimeseriesResponse(BaseModel):
    split: str
    n_scenarios: int
    clusters: Dict[str, ClusterTimeseriesEntry]


class ClusterPredictionsResponse(BaseModel):
    """Per-trade 2-D arrays for a single cluster.

    When ``mode="summary"`` only the summed 1-D arrays and metadata
    are returned.  ``mode="full"`` includes the complete matrix.
    """
    cluster_id: str
    split: str
    mode: str
    n_scenarios: int
    n_trades: int
    cluster_pred: Optional[List[float]] = None
    cluster_target: Optional[List[float]] = None
    predictions: Optional[List[List[float]]] = None
    targets: Optional[List[List[float]]] = None
    trade_indices: Optional[List[int]] = None


class GroupTimeseriesEntry(BaseModel):
    predictions: List[float]
    targets: List[float]
    n_clusters: int
    n_trades: int


class GroupTimeseriesResponse(BaseModel):
    """Pre-aggregated timeseries grouped by a cluster attribute.

    Shape: ``{attr_key: {group_value: {predictions, targets, ...}}}``
    """
    split: str
    groups: Dict[str, Dict[str, GroupTimeseriesEntry]]


class GroupSummaryEntry(BaseModel):
    mae: float
    rmse: float
    n_clusters: int
    n_trades: int


class GroupSummariesResponse(BaseModel):
    """Scalar metrics per group per attribute dimension."""
    split: str
    groups: Dict[str, Dict[str, GroupSummaryEntry]]


class CorrelationMatrix(BaseModel):
    columns: List[str]
    values: List[List[float]]


class GroupCorrelationsResponse(BaseModel):
    split: str
    correlations: Dict[str, CorrelationMatrix]
