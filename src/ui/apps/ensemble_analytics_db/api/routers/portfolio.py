"""Portfolio endpoints — timeseries, percentiles, worst scenarios."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.timeseries import (
    PortfolioStatRow,
    PortfolioStatsResponse,
    PortfolioTimeseriesResponse,
    WorstScenarioRow,
    WorstScenariosResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/portfolio")


@router.get("/timeseries", response_model=PortfolioTimeseriesResponse)
def get_portfolio_timeseries(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    data = cache.get_portfolio(split)
    if data is None:
        raise HTTPException(404, f"No portfolio summary for split '{split}'")
    return PortfolioTimeseriesResponse(
        split=split,
        n_scenarios=data["n_scenarios"],
        predictions=data["predictions"],
        targets=data["targets"],
    )


@router.get("/stats", response_model=PortfolioStatsResponse)
def get_portfolio_stats(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    """Percentile distribution + scalar error metrics (MAE, RMSE, etc.)."""
    rows = cache.get_portfolio_percentiles(split)
    if not rows:
        raise HTTPException(404, f"No portfolio data for split '{split}'")
    return PortfolioStatsResponse(
        split=split,
        rows=[PortfolioStatRow(**r) for r in rows],
    )


@router.get("/worst-scenarios", response_model=WorstScenariosResponse)
def get_worst_scenarios(
    split: str = "test",
    cache: ArtifactCache = Depends(get_cache),
):
    rows = cache.get_worst_scenarios(split)
    if not rows:
        raise HTTPException(404, f"No portfolio data for split '{split}'")
    return WorstScenariosResponse(
        split=split,
        rows=[WorstScenarioRow(**r) for r in rows],
    )
