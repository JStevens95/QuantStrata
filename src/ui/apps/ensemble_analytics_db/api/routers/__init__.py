"""API router registration."""
from fastapi import APIRouter

from src.ui.apps.ensemble_analytics_db.api.routers import (
    admin,
    clusters,
    governance,
    groups,
    metadata,
    metrics,
    portfolio,
    registry,
    trades,
)

_PREFIX = "/api/v1"

_SUB_ROUTERS = [
    (metadata.router, "metadata"),
    (metrics.router, "metrics"),
    (portfolio.router, "portfolio"),
    (clusters.router, "clusters"),
    (trades.router, "trades"),
    (groups.router, "groups"),
    (governance.router, "governance"),
    (registry.router, "registry"),
    (admin.router, "admin"),
]


def build_root_router() -> APIRouter:
    root = APIRouter(prefix=_PREFIX)
    for sub, tag in _SUB_ROUTERS:
        root.include_router(sub, tags=[tag])
    return root
