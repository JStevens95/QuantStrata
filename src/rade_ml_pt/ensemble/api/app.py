"""PRISM API — FastAPI application factory.

Serves ensemble evaluation artifacts (parquet + JSON) for low-latency
dashboard consumption.

CLI launch
----------
.. code-block:: bash

    export PRISM_ARTIFACTS_DIR=/path/to/artifacts
    export PRISM_REGISTRY_DIR=/path/to/registry
    export PRISM_ENSEMBLE_VERSION=latest          # optional, default "latest"

    uvicorn src.rade_ml_pt.ensemble.api.app:get_app --factory --port 8000

Programmatic launch
-------------------
.. code-block:: python

    from src.rade_ml_pt.ensemble.api.app import create_app
    from src.rade_ml_pt.ensemble.api.config import Settings, set_settings

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

from src.rade_ml_pt.ensemble.api.config import (
    Settings,
    get_settings,
    set_settings,
)
from src.rade_ml_pt.ensemble.api.dependencies import (
    set_inference_state_manager,
    set_monitoring_result_reader,
    set_monitoring_state_manager,
    set_reader,
    set_result_reader,
)
from src.rade_ml_pt.ensemble.api.models.meta import HealthResponse, VersionsResponse
from src.rade_ml_pt.ensemble.api.routers.cluster_timeseries import (
    router as cluster_timeseries_router,
)
from src.rade_ml_pt.ensemble.api.routers.clusters import router as clusters_router
from src.rade_ml_pt.ensemble.api.routers.elementary_pnl import (
    router as elementary_pnl_router,
)
from src.rade_ml_pt.ensemble.api.routers.governance import (
    router as governance_router,
)
from src.rade_ml_pt.ensemble.api.routers.inference import (
    router as inference_router,
)
from src.rade_ml_pt.ensemble.api.routers.graph_stats import (
    router as graph_stats_router,
)
from src.rade_ml_pt.ensemble.api.routers.group_correlations import (
    router as group_correlations_router,
)
from src.rade_ml_pt.ensemble.api.routers.metrics import router as metrics_router
from src.rade_ml_pt.ensemble.api.routers.monitoring import (
    router as monitoring_router,
)
from src.rade_ml_pt.ensemble.api.routers.overview import router as overview_router
from src.rade_ml_pt.ensemble.api.routers.portfolio import router as portfolio_router
from src.rade_ml_pt.ensemble.api.routers.predictions import (
    router as predictions_router,
)
from src.rade_ml_pt.ensemble.api.routers.quality import router as quality_router
from src.rade_ml_pt.ensemble.api.routers.trade_graph import (
    router as trade_graph_router,
)
from src.rade_ml_pt.ensemble.api.routers.trades import router as trades_router
from src.rade_ml_pt.ensemble.api.routers.training_curves import (
    router as training_curves_router,
)
from src.rade_ml_pt.ensemble.api.services.inference_state import (
    InferenceStateManager,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import (
    MonitoringResultReader,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_state import (
    MonitoringStateManager,
)
from src.rade_ml_pt.ensemble.api.services.paths import ArtifactPaths
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader
from src.rade_ml_pt.ensemble.api.services.result_reader import (
    InferenceResultReader,
)
from src.rade_ml_pt.ensemble.api.services.version import list_versions

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: resolve version, build paths, register the reader."""
    settings = get_settings()
    logging.basicConfig(level=settings.log_level.upper())

    version = settings.resolved_version
    logger.info(
        "PRISM API starting — version=%s (requested=%s), artifacts=%s, registry=%s",
        version,
        settings.ensemble_version,
        settings.artifacts_dir,
        settings.registry_dir,
    )

    paths = ArtifactPaths(
        artifacts_dir=settings.artifacts_path,
        version=version,
    )
    reader = ArtifactReader(paths)
    set_reader(reader)

    # Inference state manager — holds the live EnsembleInferencePipeline
    # for the /inference router.  Empty until POST /inference/load.
    set_inference_state_manager(InferenceStateManager())

    # Inference result reader — serves the on-disk artifacts written
    # by EnsembleInferencePipeline.post_infer (Stage 6) to the
    # historical-run endpoints under /inference/runs/... (Stage 10).
    # Pure read-side; no warm-loading required.
    set_result_reader(InferenceResultReader(settings.artifacts_dir))

    # Monitoring state manager — holds the live EnsembleMonitoringPipeline
    # for the /monitoring router (M.4).  Empty until POST /monitoring/load.
    set_monitoring_state_manager(MonitoringStateManager())

    # Monitoring result reader — serves the on-disk artifacts written
    # by EnsembleMonitoringPipeline.compute_drift / promote_to_predictions
    # to the historical-run endpoints under /monitoring/runs/... (M.4).
    # Pure read-side; mirrors InferenceResultReader's lifecycle.
    set_monitoring_result_reader(MonitoringResultReader(settings.artifacts_dir))

    splits = reader.available_splits()
    logger.info(
        "PRISM API ready — version=%s, splits=%s — http://%s:%d/docs",
        version, splits, settings.host, settings.port,
    )

    yield

    logger.info("PRISM API shutting down.")


def create_app(
    artifacts_dir: Optional[str] = None,
    registry_dir: Optional[str] = None,
    ensemble_version: str = "latest",
) -> FastAPI:
    """Build the FastAPI application.

    If *artifacts_dir* and *registry_dir* are passed they override any
    env-var-driven settings — convenient for script-based launches.
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
            "PRISM dashboard.  Consumed by Dash and (optionally) a "
            "TypeScript/React/Tailwind front-end."
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

    # ── Discovery / liveness endpoints ────────────────────────────
    # Inlined here because they don't fit any artifact-family router
    # and aren't worth their own module.

    @app.get("/health", tags=["meta"], response_model=HealthResponse)
    def health() -> HealthResponse:
        s = get_settings()
        return HealthResponse(
            status="ok",
            version=s.resolved_version,
            artifacts_dir=s.artifacts_dir,
        )

    @app.get("/versions", tags=["meta"], response_model=VersionsResponse)
    def versions() -> VersionsResponse:
        s = get_settings()
        return VersionsResponse(
            active=s.resolved_version,
            available=list_versions(s.registry_path),
        )

    # ── Artifact routers ──────────────────────────────────────────
    app.include_router(overview_router)
    app.include_router(portfolio_router)
    app.include_router(metrics_router)
    app.include_router(clusters_router)
    app.include_router(cluster_timeseries_router)
    app.include_router(trades_router)
    app.include_router(group_correlations_router)
    app.include_router(graph_stats_router)
    app.include_router(trade_graph_router)
    app.include_router(training_curves_router)
    app.include_router(elementary_pnl_router)
    app.include_router(quality_router)
    app.include_router(predictions_router)
    app.include_router(governance_router)
    app.include_router(inference_router)
    app.include_router(monitoring_router)

    return app


def get_app() -> FastAPI:
    """Factory entry-point for ``uvicorn ... --factory`` CLI launch."""
    return create_app()
