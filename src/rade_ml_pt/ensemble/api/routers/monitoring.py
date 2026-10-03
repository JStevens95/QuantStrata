"""``/prism/v1/monitoring`` — staged drift-monitoring workflow over HTTP.

Mirrors the inference router 1:1 — see
:mod:`ensemble.api.routers.inference` for the parallel design rationale
(state machine, non-blocking ``/run``, historical-runs data plane).
This router adds one extra control endpoint on top of the four
inference stages: ``POST /promote``, which triggers M.3's
:meth:`EnsembleMonitoringPipeline.promote_to_predictions` against a
completed drift run.

Endpoint map
------------
Control plane (active run)::

    POST /prism/v1/monitoring/load        → cold-load ensemble.
    POST /prism/v1/monitoring/scenarios   → load + parse a scenario folder.
    POST /prism/v1/monitoring/validate    → run validation on loaded scenarios.
    POST /prism/v1/monitoring/run         → compute drift on a worker thread.
    POST /prism/v1/monitoring/promote     → forward-pass inside the same run.
    GET  /prism/v1/monitoring/status      → cheap status probe.
    GET  /prism/v1/monitoring/events      → cursor-paginated activity log tail.
    GET  /prism/v1/monitoring/manifest    → active run's manifest.json.

Data plane (historical runs)::

    GET  /prism/v1/monitoring/runs                                 → list every run on disk.
    GET  /prism/v1/monitoring/runs/{run_id}/manifest               → read any run's manifest.
    GET  /prism/v1/monitoring/runs/{run_id}/drift_summary          → portfolio drift KPIs.
    GET  /prism/v1/monitoring/runs/{run_id}/clusters               → per-cluster severity index.
    GET  /prism/v1/monitoring/runs/{run_id}/clusters/{cid}/drift   → per-cluster drift table.

State model
-----------
Single :class:`MonitoringRunState` on the process-wide
:class:`MonitoringStateManager` (Option A — single user).  Same
migration story to Option B as the inference router.

Data plane (historical runs) goes straight through
:class:`MonitoringResultReader` to disk — stateless, safe for any
number of concurrent UI sessions.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.config import Settings, get_settings
from src.rade_ml_pt.ensemble.api.models.monitoring import (
    ClusterDriftResponse,
    ClusterDriftRow,
    ClusterRoutingDecisionModel,
    ClusterSeverityResponse,
    ClusterSeverityRow,
    DriftSummaryResponse,
    EventsResponse,
    LoadResponse,
    LoadScenariosRequest,
    LoadScenariosResponse,
    ManifestResponse,
    MonitoringEventModel,
    PromoteResponse,
    RunRequest,
    RunResponse,
    RunSummary,
    RunsListResponse,
    StatusResponse,
    ValidateResponse,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_reader import (
    MonitoringResultReader,
    get_monitoring_result_reader,
)
from src.rade_ml_pt.ensemble.api.services.monitoring_state import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_LOADED,
    STATUS_LOADING,
    STATUS_PROMOTED,
    STATUS_PROMOTING,
    STATUS_RUNNING,
    STATUS_SCENARIOS,
    STATUS_VALIDATED,
    MonitoringStateManager,
    build_monitoring_ensemble_config,
    get_monitoring_state_manager,
    per_run_monitoring_artifacts_dir,
)
from src.rade_ml_pt.monitoring.run_paths import (
    MANIFEST_FILENAME,
    MONITORING_SUBDIRNAME,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prism/v1/monitoring", tags=["monitoring"])


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _opt_float(x) -> float | None:
    """Coerce a parquet cell to ``Optional[float]`` for the wire schema.

    Parquet's NA marker varies (``pd.NA``, ``np.nan``, plain Python
    ``None``); the Pydantic models declare these fields as
    ``Optional[float]`` so the router normalises all NA flavours to
    a real ``None`` here.  This keeps the wire shape JSON-strict
    (no NaN-encoded floats slipping through) regardless of how the
    writer side normalised the parquet.
    """
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _opt_int(x) -> int | None:
    """Coerce a parquet cell to ``Optional[int]``."""
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════
# Control plane (active run)
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# POST /load
# ──────────────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
def load_ensemble(
    settings: Settings                  = Depends(get_settings),
    manager:  MonitoringStateManager    = Depends(get_monitoring_state_manager),
) -> LoadResponse:
    """Cold-load the ensemble model + per-cluster contexts.

    Wraps :meth:`EnsembleMonitoringPipeline.load` (which forwards to
    :meth:`EnsembleInferencePipeline.load` under the hood).  Any prior
    active monitoring run is silently replaced — appropriate for the
    single-user dashboard today.
    """
    ensemble_config = build_monitoring_ensemble_config(
        registry_dir  = settings.registry_dir,
        artifacts_dir = settings.artifacts_dir,
    )

    state = manager.create_run(
        ensemble_config  = ensemble_config,
        ensemble_version = settings.resolved_version,
    )

    state.transition(STATUS_LOADING)
    try:
        state.pipeline.load()
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Monitoring ensemble load failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Ensemble load failed: {exc}")

    # Reach into the composed inference pipeline to count clusters —
    # mirrors the inference router's ``len(state.pipeline._ensemble.members)``
    # lookup.  We can't call a public accessor because none exists on
    # the inference pipeline either (would couple us to infer.py).
    inference_pipeline = state.pipeline._inference_pipeline  # noqa: SLF001
    n_clusters = (
        len(inference_pipeline._ensemble.members)            # noqa: SLF001
        if inference_pipeline._ensemble is not None else 0   # noqa: SLF001
    )
    state.transition(STATUS_LOADED)

    return LoadResponse(
        run_id           = state.run_id,
        ensemble_version = state.ensemble_version,
        n_clusters       = n_clusters,
        status           = state.status,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /scenarios
# ──────────────────────────────────────────────────────────────────────

@router.post("/scenarios", response_model=LoadScenariosResponse)
def load_scenarios(
    body:    LoadScenariosRequest,
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> LoadScenariosResponse:
    """Parse a folder of shock CSVs into the monitoring pipeline state.

    Wraps :meth:`EnsembleMonitoringPipeline.load_scenarios`.  Same
    server-side-path contract inference uses; file-upload deferred.
    """
    state = manager.get_run()
    if state.status not in (
        STATUS_LOADED, STATUS_SCENARIOS, STATUS_VALIDATED,
        STATUS_COMPLETE, STATUS_PROMOTED, STATUS_FAILED,
    ):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot load scenarios from status '{state.status}'. Call /load first.",
        )

    try:
        report = state.pipeline.load_scenarios(new_scenario_dir=body.new_scenario_dir)
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Monitoring scenario load failed for run %s", state.run_id)
        raise HTTPException(status_code=400, detail=f"Scenario load failed: {exc}")

    state.transition(STATUS_SCENARIOS)
    return LoadScenariosResponse(**report.to_dict())


# ──────────────────────────────────────────────────────────────────────
# POST /validate
# ──────────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ValidateResponse)
def validate_scenarios(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> ValidateResponse:
    """Compute per-cluster routing decisions + run-level validation errors.

    Wraps :meth:`EnsembleMonitoringPipeline.validate_scenarios`.
    Does NOT raise on user-input errors (those land in ``report.errors``);
    only system failures bubble up as 500s.  A non-empty ``errors``
    list keeps the run in ``scenarios_loaded`` so the user can fix
    and re-validate.
    """
    state = manager.get_run()
    if state.status not in (STATUS_SCENARIOS, STATUS_VALIDATED):
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot validate from status '{state.status}'. "
                "Call /scenarios first."
            ),
        )

    try:
        report = state.pipeline.validate_scenarios()
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Monitoring validation failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")

    if report.is_valid:
        state.transition(STATUS_VALIDATED)

    decisions = [
        ClusterRoutingDecisionModel(**d.to_dict()) for d in report.cluster_decisions
    ]
    return ValidateResponse(
        ensemble_version       = report.ensemble_version,
        n_scenarios            = report.n_scenarios,
        scenario_labels        = list(report.scenario_labels),
        cluster_decisions      = decisions,
        errors                 = list(report.errors),
        warnings               = list(report.warnings),
        is_valid               = report.is_valid,
        affected_cluster_ids   = report.affected_cluster_ids,
        unaffected_cluster_ids = report.unaffected_cluster_ids,
        affected_count         = report.affected_count,
        unaffected_count       = report.unaffected_count,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /run
# ──────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=RunResponse)
def run_monitoring(
    body:     RunRequest                = RunRequest(),  # noqa: B008 — FastAPI default
    settings: Settings                  = Depends(get_settings),
    manager:  MonitoringStateManager    = Depends(get_monitoring_state_manager),
) -> RunResponse:
    """Dispatch the drift-compute stage onto a background thread.

    Wraps :meth:`EnsembleMonitoringPipeline.compute_drift`.  Non-
    blocking — returns immediately with ``status='running'``; the UI
    polls ``GET /status`` until terminal.  Mirror of
    ``POST /inference/run`` in shape.

    Gated on:
      * ``status == 'validated'`` — validation must have produced an
        error-free report.
      * ``not state.is_alive`` — no prior background thread still
        executing.

    Side effects
    ------------
    Pins ``config.artifacts_dir`` on the composed inference pipeline
    so the monitoring pipeline writes its run under
    ``<artifacts_dir>/monitoring_runs/<run_id>/``.  The router doesn't
    know the final run_id at dispatch (it's minted by
    :func:`monitoring_run_id` inside ``compute_drift``); the on-disk
    convention puts every run under the same parent, so the path
    returned here is the parent ``monitoring_runs/<provisional_id>``
    directory.  The state's ``manifest_path`` is populated by the
    worker after compute_drift completes.
    """
    state = manager.get_run()
    if state.status != STATUS_VALIDATED:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot run from status '{state.status}'. "
                "Validation must complete cleanly (status='validated')."
            ),
        )
    if state.is_alive:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run {state.run_id} is already executing in the "
                "background.  Wait for completion before re-dispatching."
            ),
        )

    # Pin the per-run artifacts root.  The monitoring pipeline mints
    # its own run_id inside compute_drift; we point ``artifacts_dir``
    # at the BASE (where ``monitoring_runs/<run_id>/`` will land) and
    # surface the resolved final dir on the worker-callback below
    # once the pipeline returns its MonitoringResult.
    base_artifacts_dir = settings.artifacts_dir
    state.pipeline.config.artifacts_dir = base_artifacts_dir
    state.artifacts_dir                  = base_artifacts_dir
    state.transition(STATUS_RUNNING)

    def _execute() -> None:
        try:
            result = state.pipeline.compute_drift()
        except Exception as exc:
            state.transition(STATUS_FAILED, error=str(exc))
            logger.exception(
                "Monitoring run failed for run %s", state.run_id,
            )
            return

        # MonitoringResult exposes the canonical run_id (the pipeline
        # mints this internally via :func:`monitoring_run_id`).
        # Persist it on the state so subsequent /promote and /status
        # calls refer to the same id the manifest carries.
        state.run_id        = result.run_id
        state.artifacts_dir = str(result.artifacts_dir)
        state.manifest_path = (
            str(result.manifest_path) if result.manifest_path.exists() else None
        )
        state.transition(STATUS_COMPLETE)

    manager.start_run_in_background(_execute)

    return RunResponse(
        run_id        = state.run_id,
        status        = state.status,                              # "running"
        artifacts_dir = per_run_monitoring_artifacts_dir(
            base_artifacts_dir, state.run_id,
        ),
    )


# ──────────────────────────────────────────────────────────────────────
# POST /promote
# ──────────────────────────────────────────────────────────────────────

@router.post("/promote", response_model=PromoteResponse)
def promote_to_predictions(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> PromoteResponse:
    """Dispatch promote-to-predictions onto a background thread.

    Wraps :meth:`EnsembleMonitoringPipeline.promote_to_predictions`.
    Non-blocking — same dispatch model as ``/run``.  The UI polls
    ``GET /status`` until terminal (``status=='promoted'``) and then
    reads the updated manifest via ``GET /manifest``.

    Gated on:
      * ``status == 'complete'`` — the drift run must have finished.
      * ``not state.is_alive`` — no prior background thread still
        executing.

    The pipeline itself enforces single-promote-per-instance
    (raises if a prior promote succeeded); we surface that as a 409
    when the second dispatch reaches the worker.
    """
    state = manager.get_run()
    if state.status != STATUS_COMPLETE:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Cannot promote from status '{state.status}'. "
                "Drift run must complete first (status='complete')."
            ),
        )
    if state.is_alive:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Run {state.run_id} is already executing in the "
                "background.  Wait for completion before re-dispatching."
            ),
        )

    state.transition(STATUS_PROMOTING)

    def _execute() -> None:
        try:
            promote_result = state.pipeline.promote_to_predictions()
        except Exception as exc:
            state.transition(STATUS_FAILED, error=str(exc))
            logger.exception(
                "Monitoring promote failed for run %s", state.run_id,
            )
            return

        state.predictions_dir = str(promote_result.predictions_dir)
        # promote_to_predictions rewrites the same monitoring manifest,
        # so manifest_path stays the same — but defensively refresh it
        # in case the pipeline ever changes that contract.
        state.manifest_path = (
            str(promote_result.manifest_path)
            if promote_result.manifest_path.exists() else state.manifest_path
        )
        state.transition(STATUS_PROMOTED)

    manager.start_run_in_background(_execute)

    return PromoteResponse(
        run_id = state.run_id,
        status = state.status,                                 # "promoting"
    )


# ──────────────────────────────────────────────────────────────────────
# GET /status
# ──────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
def get_status(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> StatusResponse:
    """Cheap status probe — drives the next-button gate in the UI."""
    if not manager.has_active_run:
        return StatusResponse(has_active_run=False)

    state = manager.get_run()
    return StatusResponse(
        has_active_run   = True,
        run_id           = state.run_id,
        ensemble_version = state.ensemble_version,
        status           = state.status,
        last_error       = state.last_error,
        n_events         = len(state.activity_log),
        created_at       = state.created_at,
        artifacts_dir    = state.artifacts_dir,
        manifest_path    = state.manifest_path,
        predictions_dir  = state.predictions_dir,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /events
# ──────────────────────────────────────────────────────────────────────

@router.get("/events", response_model=EventsResponse)
def get_events(
    cursor:  int                       = Query(0, ge=0, description="Number of events the caller has already seen."),
    manager: MonitoringStateManager    = Depends(get_monitoring_state_manager),
) -> EventsResponse:
    """Cursor-paginated tail of the activity log.

    Same cursor protocol as ``GET /inference/events``.  Falls back to
    an empty response when no run has been created so the UI can mount
    the polling loop before /load.
    """
    if not manager.has_active_run:
        return EventsResponse(events=[], next_cursor=0)

    events, next_cursor = manager.events_since(run_id=None, cursor=cursor)
    return EventsResponse(
        events      = [MonitoringEventModel(**e) for e in events],
        next_cursor = next_cursor,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/manifest", response_model=ManifestResponse)
def get_manifest(
    manager: MonitoringStateManager = Depends(get_monitoring_state_manager),
) -> ManifestResponse:
    """Return the active run's monitoring ``manifest.json``.

    Only available after the drift run completes (``status='complete'``
    or ``'promoted'``).  The manifest is the dashboard's entry point —
    lists per-cluster paths + the portfolio drift_summary +
    (if promoted) the predictions block.
    """
    state = manager.get_run()
    if state.status not in (STATUS_COMPLETE, STATUS_PROMOTED) or state.manifest_path is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Manifest not available — run status is '{state.status}'.  "
                "Wait for status='complete' (drift) or 'promoted' (with predictions)."
            ),
        )

    path = Path(state.manifest_path)
    if not path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Manifest path recorded but file is missing: {path}",
        )

    with open(path, "r") as f:
        manifest = json.load(f)

    return ManifestResponse(run_id=state.run_id, manifest=manifest)


# ══════════════════════════════════════════════════════════════════════
# Data plane (historical runs)
# ══════════════════════════════════════════════════════════════════════
#
# All endpoints below are stateless: they go straight through
# :class:`MonitoringResultReader` to disk and don't touch
# :class:`MonitoringStateManager` at all.
#
# Error model
# -----------
# * ``FileNotFoundError`` from the reader → 404.  Covers the
#   common case of polling for results before the worker thread
#   finished writing them.
# * Any other reader exception → 500 (data corruption).
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# GET /runs — discovery
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=RunsListResponse)
def list_runs(
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> RunsListResponse:
    """Return every monitoring run on disk, most recent first."""
    run_ids = reader.list_run_ids()
    runs    = [RunSummary(**reader.run_summary(rid)) for rid in run_ids]
    return RunsListResponse(runs=runs, count=len(runs))


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/manifest", response_model=ManifestResponse)
def get_run_manifest(
    run_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> ManifestResponse:
    """Read ``manifest.json`` for any historical monitoring run."""
    try:
        manifest = reader.load_manifest(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ManifestResponse(run_id=run_id, manifest=manifest)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/drift_summary
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/drift_summary", response_model=DriftSummaryResponse,
)
def get_run_drift_summary(
    run_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> DriftSummaryResponse:
    """Return the portfolio-level drift KPIs for ``run_id``.

    Sourced from ``drift_summary.json``.  Cheap (small JSON), so
    safe to poll from the UI's main dashboard view.
    """
    try:
        summary = reader.load_drift_summary(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return DriftSummaryResponse(run_id=run_id, summary=summary)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/clusters", response_model=ClusterSeverityResponse,
)
def get_run_clusters(
    run_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> ClusterSeverityResponse:
    """Return the per-cluster severity index for ``run_id``.

    Derived from the manifest's portfolio drift_summary — cheap
    enough to call on every UI navigation event without paying for
    a per-cluster parquet read.
    """
    try:
        rows_raw = reader.load_cluster_severity_index(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        ClusterSeverityRow(
            cluster_id = str(r.get("cluster_id", "")),
            severity   = str(r.get("severity", "no_data")),
            max_psi    = _opt_float(r.get("max_psi")),
            mean_psi   = _opt_float(r.get("mean_psi")),
            n_features = _opt_int(r.get("n_features")),
        )
        for r in rows_raw
    ]
    return ClusterSeverityResponse(run_id=run_id, rows=rows, count=len(rows))


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters/{cluster_id}/drift
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/clusters/{cluster_id}/drift",
    response_model=ClusterDriftResponse,
)
def get_run_cluster_drift(
    run_id:     str,
    cluster_id: str,
    reader: MonitoringResultReader = Depends(get_monitoring_result_reader),
) -> ClusterDriftResponse:
    """Return the long-format drift table for one cluster.

    Sourced from
    ``<run>/monitoring/clusters/<cid>/drift_table.parquet``.  Output
    of :func:`monitoring.drift.build_drift_table`.
    """
    try:
        df = reader.load_cluster_drift_table(run_id, cluster_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        ClusterDriftRow(
            cluster_id    = str(r.get("cluster_id", cluster_id)),
            feature_name  = str(r.get("feature_name", "")),
            psi           = _opt_float(r.get("psi")),
            js_divergence = _opt_float(r.get("js_divergence")),
            mean_shift    = _opt_float(r.get("mean_shift")),
            std_ratio     = _opt_float(r.get("std_ratio")),
            severity      = str(r.get("severity", "no_data")),
        )
        for r in df.to_dict(orient="records")
    ]
    return ClusterDriftResponse(
        run_id     = run_id,
        cluster_id = cluster_id,
        rows       = rows,
        n_features = len(rows),
    )


# Re-export the constants the active-run /manifest path uses so the
# router and the on-disk writer can never drift on the manifest
# filename / sub-directory name.  Lint-friendly: explicit reference
# so unused-import linting doesn't strip them.
_ = (MANIFEST_FILENAME, MONITORING_SUBDIRNAME)
