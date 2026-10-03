"""Trade endpoints — per-trade metrics with pagination and filtering."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query

from src.ui.apps.ensemble_analytics_db.api.dependencies import get_cache
from src.ui.apps.ensemble_analytics_db.api.models.trades import (
    TradeCatalogueResponse,
    TradeCatalogueRow,
    TradeMetricRow,
    TradeMetricsResponse,
)
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

router = APIRouter(prefix="/trades")


@router.get("/metrics", response_model=TradeMetricsResponse)
def get_trade_metrics(
    split: str = "test",
    cluster_id: Optional[str] = Query(None, description="Filter by cluster"),
    sort_by: str = Query("mae", description="Sort column"),
    descending: bool = Query(True),
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=5000),
    cache: ArtifactCache = Depends(get_cache),
):
    """Paginated, filterable trade-level metrics."""
    rows = cache.get_trade_metrics(split)

    if cluster_id:
        rows = [r for r in rows if r.get("cluster_id") == cluster_id]

    if sort_by and rows and sort_by in rows[0]:
        rows = sorted(rows, key=lambda r: r.get(sort_by, 0), reverse=descending)

    total = len(rows)
    page = rows[offset : offset + limit]

    return TradeMetricsResponse(
        split=split,
        total=total,
        offset=offset,
        limit=limit,
        rows=[TradeMetricRow(**r) for r in page],
    )


@router.get("/catalogue", response_model=TradeCatalogueResponse)
def get_trade_catalogue(
    cluster_id: Optional[str] = None,
    desk: Optional[str] = None,
    ccy: Optional[str] = None,
    product: Optional[str] = None,
    offset: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=5000),
    cache: ArtifactCache = Depends(get_cache),
):
    """Trade catalogue — the join of trade_cluster_map + cluster_attributes."""
    rows = cache.trade_catalogue

    if cluster_id:
        rows = [r for r in rows if r.get("cluster_id") == cluster_id]
    if desk:
        rows = [r for r in rows if r.get("desk") == desk]
    if ccy:
        rows = [r for r in rows if r.get("ccy") == ccy]
    if product:
        rows = [r for r in rows if r.get("product") == product]

    total = len(rows)
    page = rows[offset : offset + limit]
    return TradeCatalogueResponse(
        total=total,
        rows=[TradeCatalogueRow(**{k: r.get(k) for k in TradeCatalogueRow.model_fields}) for r in page],
    )
