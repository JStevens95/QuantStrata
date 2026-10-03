"""Group endpoints — pre-aggregated timeseries and correlations."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.timeseries import (
    CorrelationMatrix,
    GroupCorrelationsResponse,
    GroupSummariesResponse,
    GroupSummaryEntry,
    GroupTimeseriesEntry,
    GroupTimeseriesResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/groups")


@router.get("/timeseries", response_model=GroupTimeseriesResponse)
def get_group_timeseries(
    split: str = "test",
    attr_key: Optional[str] = Query(
        None,
        description="Filter to a single attribute (e.g. 'desk', 'ccy'). "
                    "Omit to get all attribute dimensions.",
    ),
    cache: ArtifactCache = Depends(get_cache),
):
    """Pre-aggregated timeseries grouped by cluster attributes.

    Returns cluster predictions summed by each group value
    (e.g. all FLOW_RATES clusters → one timeseries).
    """
    all_groups = cache.get_group_aggregations(split)

    if attr_key:
        all_groups = {k: v for k, v in all_groups.items() if k == attr_key}

    result = {}
    for ak, group_dict in all_groups.items():
        result[ak] = {
            gv: GroupTimeseriesEntry(**gdata)
            for gv, gdata in group_dict.items()
        }

    return GroupTimeseriesResponse(split=split, groups=result)


@router.get("/summaries", response_model=GroupSummariesResponse)
def get_group_summaries(
    split: str = "test",
    attr_key: Optional[str] = Query(
        None,
        description="Filter to a single attribute (e.g. 'desk', 'ccy'). "
                    "Omit to get all attribute dimensions.",
    ),
    cache: ArtifactCache = Depends(get_cache),
):
    """Pre-computed scalar metrics (MAE, RMSE, n_trades) per group."""
    all_summaries = cache.get_group_summaries(split)
    if attr_key:
        all_summaries = {k: v for k, v in all_summaries.items() if k == attr_key}
    result = {
        ak: {gv: GroupSummaryEntry(**gdata) for gv, gdata in group_dict.items()}
        for ak, group_dict in all_summaries.items()
    }
    return GroupSummariesResponse(split=split, groups=result)


@router.get("/correlations", response_model=GroupCorrelationsResponse)
def get_group_correlations(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    """Cross-group residual correlation matrices per attribute key."""
    raw = cache.get_group_correlations(split)
    return GroupCorrelationsResponse(
        split=split,
        correlations={
            k: CorrelationMatrix(**v) for k, v in raw.items()
        },
    )
