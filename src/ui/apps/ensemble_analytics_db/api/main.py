"""
PRISM API — FastAPI application factory.

Serves ensemble evaluation artifacts from RAM for low-latency
dashboard consumption.

Usage — env vars
----------------
.. code-block:: bash

    export PRISM_ARTIFACTS_DIR=/path/to/artifacts
    export PRISM_REGISTRY_DIR=/path/to/registry
    export PRISM_ENSEMBLE_VERSION=latest          # optional, default "latest"

    uvicorn src.ui.apps.ensemble_analytics_db.api.main:app --port 8000

Usage — programmatic (from a script)
-------------------------------------
.. code-block:: python

    from src.ui.apps.ensemble_analytics_db.api.main import create_app
    from src.ui.apps.ensemble_analytics_db.api.config import Settings, set_settings

    set_settings(Settings(
        artifacts_dir="/path/to/artifacts",
        registry_dir="/path/to/registry",
        ensemble_version="latest",
    ))
    app = create_app()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

OpenAPI docs are available at ``http://localhost:8000/docs``.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.ui.apps.ensemble_analytics_db.api.config import (
    Settings,
    get_settings,
    set_settings,
)
from src.ui.apps.ensemble_analytics_db.api.dependencies import set_cache
from src.ui.apps.ensemble_analytics_db.api.routers import build_root_router
from src.ui.apps.ensemble_analytics_db.api.services.artifact_loader import (
    ArtifactCache,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: resolve version, build eval_dir, load all artifacts into RAM."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    logger.info(
        "PRISM API starting — version=%s, artifacts=%s, registry=%s",
        settings.ensemble_version,
        settings.artifacts_dir,
        settings.registry_dir,
    )

    cache = ArtifactCache(
        artifacts_dir=settings.artifacts_path,
        registry_dir=settings.registry_path,
        version=settings.resolved_version,
    )
    cache.load_all()
    set_cache(cache)

    logger.info(
        "PRISM API ready — version=%s, clusters=%d, trades=%d — %s",
        cache.ensemble_version,
        len(cache.cluster_attributes),
        len(cache.trade_cluster_map),
        f"http://{settings.host}:{settings.port}/docs",
    )

    yield

    logger.info("PRISM API shutting down.")


def create_app(
    artifacts_dir: Optional[str] = None,
    registry_dir: Optional[str] = None,
    ensemble_version: str = "latest",
) -> FastAPI:
    """Build the FastAPI application.

    Parameters
    ----------
    artifacts_dir, registry_dir, ensemble_version
        If provided, these override environment variables.  Convenient
        for script-based launch without touching env vars.
    """
    if artifacts_dir is not None and registry_dir is not None:
        set_settings(Settings(
            artifacts_dir=artifacts_dir,
            registry_dir=registry_dir,
            ensemble_version=ensemble_version,
        ))

    settings = get_settings()

    app = FastAPI(
        title="PRISM — Quantitative Model Intelligence",
        description=(
            "REST API serving ensemble evaluation artifacts for the "
            "PRISM dashboard.  Consumed by Dash (ApiBackend) and Retool."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_root_router())

    return app


def get_app() -> FastAPI:
    """Return a module-level app for ``uvicorn api.main:get_app --factory``.

    For programmatic usage, call ``set_settings(...)`` then
    ``create_app()`` directly (see script 10).
    """
    return create_app()
