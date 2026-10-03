"""
FastAPI dependency injection.

Provides a single ``get_cache()`` dependency that all routers use to
access the :class:`ArtifactCache`.
"""
from __future__ import annotations

from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

_cache: ArtifactCache | None = None


def set_cache(cache: ArtifactCache) -> None:
    """Called once during the lifespan startup event."""
    global _cache
    _cache = cache


def get_cache() -> ArtifactCache:
    """FastAPI ``Depends`` — returns the singleton cache."""
    if _cache is None:
        raise RuntimeError("ArtifactCache not initialised. Server not ready.")
    return _cache
