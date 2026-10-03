"""``/prism/v1/inference`` — staged inference workflow over HTTP.

Exposes the four stages of :class:`EnsembleInferencePipeline` as REST
endpoints, plus two polling endpoints for the UI activity log and
status probe, plus a manifest fetcher for completed runs, plus the
historical run reader (Stage 10) for cross-session dashboarding.

Endpoint map
------------
Control plane (active run)::

    POST /prism/v1/inference/load          → load the ensemble (cold-load).
    POST /prism/v1/inference/scenarios     → load + parse a scenario folder.
    POST /prism/v1/inference/validate      → run validation on loaded scenarios.
    POST /prism/v1/inference/run           → dispatch inference onto a worker thread.
    GET  /prism/v1/inference/status        → cheap status probe (gate the next button).
    GET  /prism/v1/inference/events        → cursor-based activity log poll.
    GET  /prism/v1/inference/manifest      → read the active run's manifest.json.

Data plane (historical runs, Stage 10)::

    GET  /prism/v1/inference/runs                                            → list every run on disk.
    GET  /prism/v1/inference/runs/{run_id}/manifest                          → read any run's manifest.
    GET  /prism/v1/inference/runs/{run_id}/portfolio                         → portfolio-level summary parquet.
    GET  /prism/v1/inference/runs/{run_id}/clusters                          → cluster-level summary parquet.
    GET  /prism/v1/inference/runs/{run_id}/clusters/{cid}/trades?space=...   → per-cluster trade-level wide parquet.
    GET  /prism/v1/inference/runs/{run_id}/validation                        → ValidationReport snapshot from manifest.
    GET  /prism/v1/inference/runs/{run_id}/scenarios                         → LoadedScenariosReport snapshot from manifest.

State model
-----------
Today (Option A) a single :class:`InferenceRunState` is held on the
process-wide :class:`InferenceStateManager`.  Each ``POST /load`` call
replaces any prior run.  See
:mod:`services.inference_state` for the migration path to multi-user.

The data-plane endpoints are stateless — they go straight through
:class:`InferenceResultReader` to disk, so they're safe to call from
any number of concurrent UI sessions.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.config import Settings, get_settings
from src.rade_ml_pt.ensemble.api.dependencies import (
    get_inference_state_manager,
    get_result_reader,
)
from src.rade_ml_pt.ensemble.api.models.inference import (
    ClusterRoutingDecisionModel,
    ClusterSummaryResponse,
    ClusterSummaryRow,
    ClusterTradesResponse,
    EventModel,
    EventsResponse,
    LoadResponse,
    LoadScenariosRequest,
    LoadScenariosResponse,
    ManifestResponse,
    PortfolioResponse,
    PortfolioRow,
    RunRequest,
    RunResponse,
    RunSummary,
    RunsListResponse,
    ScenariosSnapshotResponse,
    StatusResponse,
    ValidateResponse,
    ValidationSnapshotResponse,
)
from src.rade_ml_pt.ensemble.api.services.inference_state import (
    STATUS_COMPLETE,
    STATUS_FAILED,
    STATUS_LOADED,
    STATUS_LOADING,
    STATUS_RUNNING,
    STATUS_SCENARIOS,
    STATUS_VALIDATED,
    InferenceStateManager,
    build_ensemble_config,
    per_run_artifacts_dir,
)
from src.rade_ml_pt.ensemble.api.services.result_reader import (
    VALID_SPACES,
    InferenceResultReader,
)
# Single source of truth for the on-disk layout — same constants the
# writer (post_infer) and reader (InferenceResultReader) import.
# Keeping the router on these constants means renames in the writer
# propagate transparently and ``state.manifest_path`` never silently
# drifts off the real file location.
from src.rade_ml_pt.pipelines.ensemble.infer import (
    INFERENCE_DIRNAME,
    MANIFEST_FILENAME,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/prism/v1/inference", tags=["inference"])


# ──────────────────────────────────────────────────────────────────────
# POST /load
# ──────────────────────────────────────────────────────────────────────

@router.post("/load", response_model=LoadResponse)
def load_ensemble(
    settings: Settings                  = Depends(get_settings),
    manager:  InferenceStateManager     = Depends(get_inference_state_manager),
) -> LoadResponse:
    """Cold-load the ensemble model + per-cluster inference contexts.

    Wraps :meth:`EnsembleInferencePipeline.load`.  Constructs a fresh
    run state on the manager and returns the assigned ``run_id`` plus
    a count of clusters that the UI can render immediately.

    The active run (if any) is silently replaced — appropriate for
    the single-user dashboard today; Option B will key on per-request
    ``run_id`` and reject the implicit replacement.
    """
    # Build a minimal EnsembleConfig from settings.  artifacts_dir is
    # filled with a placeholder; the real per-run path is set when
    # /run is called (post_infer needs it before writing).
    ensemble_config = build_ensemble_config(
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
        logger.exception("Ensemble load failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Ensemble load failed: {exc}")

    n_clusters = (
        len(state.pipeline._ensemble.members)  # noqa: SLF001 — internal-but-stable
        if state.pipeline._ensemble is not None else 0
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
    body:     LoadScenariosRequest,
    manager:  InferenceStateManager = Depends(get_inference_state_manager),
) -> LoadScenariosResponse:
    """Parse a folder of shock CSVs into the pipeline state.

    Wraps :meth:`EnsembleInferencePipeline.load_scenarios`.  The path
    must be readable by the API host process.  File-upload (multipart)
    is deferred to a later stage.

    Side-effect: invalidates any prior validate result (the pipeline
    resets ``_validation_report`` to ``None`` internally).
    """
    state = manager.get_run()
    if state.status not in (STATUS_LOADED, STATUS_SCENARIOS, STATUS_VALIDATED, STATUS_COMPLETE, STATUS_FAILED):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot load scenarios from status '{state.status}'. Call /load first.",
        )

    try:
        report = state.pipeline.load_scenarios(new_scenario_dir=body.new_scenario_dir)
    except Exception as exc:
        state.transition(STATUS_FAILED, error=str(exc))
        logger.exception("Scenario load failed for run %s", state.run_id)
        raise HTTPException(status_code=400, detail=f"Scenario load failed: {exc}")

    state.transition(STATUS_SCENARIOS)
    return LoadScenariosResponse(**report.to_dict())


# ──────────────────────────────────────────────────────────────────────
# POST /validate
# ──────────────────────────────────────────────────────────────────────

@router.post("/validate", response_model=ValidateResponse)
def validate_scenarios(
    manager: InferenceStateManager = Depends(get_inference_state_manager),
) -> ValidateResponse:
    """Compute the per-cluster routing decisions + run-level errors.

    Wraps :meth:`EnsembleInferencePipeline.validate_scenarios`.  Does
    NOT raise for user-input issues (those land in
    ``report.errors``); only system failures bubble up as 500s.

    A non-empty ``errors`` list keeps the run in the
    ``scenarios_loaded`` state so the user can fix and re-validate
    without re-loading scenarios.
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
        logger.exception("Validation failed for run %s", state.run_id)
        raise HTTPException(status_code=500, detail=f"Validation failed: {exc}")

    # Only advance to "validated" when the report is clean — keeps the
    # state machine truthful (an invalid report shouldn't unlock /run).
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
        cheap_path_used        = report.cheap_path_used,
    )


# ──────────────────────────────────────────────────────────────────────
# POST /run
# ──────────────────────────────────────────────────────────────────────

@router.post("/run", response_model=RunResponse)
def run_inference(
    body:     RunRequest                = RunRequest(),
    settings: Settings                  = Depends(get_settings),
    manager:  InferenceStateManager     = Depends(get_inference_state_manager),
) -> RunResponse:
    """Dispatch the inference run onto a background thread.

    Wraps :meth:`EnsembleInferencePipeline.run_inference`.  Stage 9:
    the call is **non-blocking** — the pipeline executes on a worker
    thread and the HTTP response returns immediately with
    ``status='running'``.  The UI then polls
    :func:`get_status` (terminal-state probe) and :func:`get_events`
    (live activity-log tail) until ``status`` becomes ``'complete'``
    or ``'failed'``.

    Gated on:
      * ``status == 'validated'`` — validation must have produced an
        error-free report.
      * ``not state.is_alive`` — no prior run thread still executing
        (defends against double-dispatch from a refreshed browser).

    Side effects
    ------------
    Pins ``config.artifacts_dir`` on the pipeline to a per-run path
    so ``post_infer`` writes its manifest + parquets into a stable
    predictable place the API can serve.  The resolved path is
    returned immediately for the UI to deep-link to even while the
    run is still in flight.

    Terminal metrics (``n_scenarios``, ``n_clusters``,
    ``n_predictions``, ``manifest_path``) are unknown at dispatch —
    they are populated on the state object by the worker thread once
    the run completes, and surfaced through ``GET /status`` and
    ``GET /manifest``.
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

    # Resolve and pin the per-run artifacts directory BEFORE dispatch
    # so the worker closure captures a stable path — the request
    # scope ends before the worker thread does.
    artifacts_dir = body.artifacts_dir or per_run_artifacts_dir(
        settings.artifacts_dir, state.run_id,
    )
    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    state.pipeline.config.artifacts_dir = artifacts_dir
    state.artifacts_dir                  = artifacts_dir
    state.transition(STATUS_RUNNING)

    # Worker closure — captured by the background thread.  Any
    # exception lands on ``state.last_error`` via the FAILED
    # transition; the HTTP response has already been returned so the
    # client must poll /status to discover the outcome.
    def _execute() -> None:
        try:
            state.pipeline.run_inference()
        except Exception as exc:
            state.transition(STATUS_FAILED, error=str(exc))
            logger.exception(
                "Inference run failed for run %s", state.run_id,
            )
            return

        # post_infer writes <artifacts_dir>/<INFERENCE_DIRNAME>/<MANIFEST_FILENAME> —
        # pin the path on the state so /manifest can read it back
        # without re-deriving the layout.  Use the same constants as
        # the writer (pipelines.ensemble.infer) and reader
        # (services.result_reader) so the three sides cannot drift.
        manifest_path = Path(artifacts_dir) / INFERENCE_DIRNAME / MANIFEST_FILENAME
        state.manifest_path = (
            str(manifest_path) if manifest_path.exists() else None
        )
        state.transition(STATUS_COMPLETE)

    manager.start_run_in_background(_execute)

    return RunResponse(
        run_id        = state.run_id,
        status        = state.status,        # "running"
        artifacts_dir = artifacts_dir,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /status
# ──────────────────────────────────────────────────────────────────────

@router.get("/status", response_model=StatusResponse)
def get_status(
    manager: InferenceStateManager = Depends(get_inference_state_manager),
) -> StatusResponse:
    """Cheap status probe — drives the next-button gate in the UI.

    Designed to be polled at 1-2 Hz between stages without
    appreciable load on the server.  Returns a no-op-shaped response
    when no run has been created yet.
    """
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
    )


# ──────────────────────────────────────────────────────────────────────
# GET /events
# ──────────────────────────────────────────────────────────────────────

@router.get("/events", response_model=EventsResponse)
def get_events(
    cursor:  int                       = Query(0, ge=0, description="Number of events the caller has already seen."),
    manager: InferenceStateManager     = Depends(get_inference_state_manager),
) -> EventsResponse:
    """Cursor-paginated tail of the activity log.

    Used by the UI to render the live activity feed.  Returns only
    events strictly after ``cursor``; the response's ``next_cursor``
    is the total event count so far — pass it back on the next poll
    to chain efficiently.

    Falls back to an empty response when no run has been created so
    the UI can mount the polling loop before /load is called.
    """
    if not manager.has_active_run:
        return EventsResponse(events=[], next_cursor=0)

    events, next_cursor = manager.events_since(run_id=None, cursor=cursor)
    return EventsResponse(
        events      = [EventModel(**e) for e in events],
        next_cursor = next_cursor,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/manifest", response_model=ManifestResponse)
def get_manifest(
    manager: InferenceStateManager = Depends(get_inference_state_manager),
) -> ManifestResponse:
    """Return the per-run ``manifest.json`` written by ``post_infer``.

    Only available after the run completes successfully.  The
    manifest is the dashboard's entry point — it lists the artifact
    paths for every per-cluster parquet plus the run-level
    portfolio / cluster summaries.
    """
    state = manager.get_run()
    if state.status != STATUS_COMPLETE or state.manifest_path is None:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Manifest not available — run status is "
                f"'{state.status}'.  Wait for status='complete'."
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
# Stage 10 — historical run reader
#
# All endpoints below are stateless: they go straight through
# :class:`InferenceResultReader` to disk and don't touch
# :class:`InferenceStateManager` at all.  Safe to call from any
# number of concurrent UI sessions.
#
# Error model
# -----------
# * ``FileNotFoundError`` from the reader → 404.  Covers the
#   common case of polling for results before the worker thread
#   finished writing them (manifest exists but parquets don't yet,
#   or run_id is invalid).
# * Any other reader exception → 500 (signals data corruption).
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# GET /runs — discovery
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs", response_model=RunsListResponse)
def list_runs(
    reader: InferenceResultReader = Depends(get_result_reader),
) -> RunsListResponse:
    """Return every inference run on disk, most recent first.

    Reads only ``manifest.json`` per run (small JSON), not parquets,
    so the response is fast even with hundreds of runs.  Returns an
    empty list when the ``inference_runs/`` directory doesn't exist
    yet (fresh deployment).
    """
    run_ids = reader.list_run_ids()
    runs    = [RunSummary(**reader.run_summary(rid)) for rid in run_ids]
    return RunsListResponse(runs=runs, count=len(runs))


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/manifest
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/manifest", response_model=ManifestResponse)
def get_run_manifest(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ManifestResponse:
    """Read ``manifest.json`` for any historical run.

    Counterpart to ``GET /manifest`` (active run); use this one
    when navigating the run-history table.
    """
    try:
        manifest = reader.load_manifest(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ManifestResponse(run_id=run_id, manifest=manifest)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/portfolio
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/portfolio", response_model=PortfolioResponse)
def get_run_portfolio(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> PortfolioResponse:
    """Return the per-scenario portfolio summary for ``run_id``.

    Sourced from ``portfolio_predictions.parquet``.  Long-format:
    one row per scenario with ``sum_pnl_*`` (additive across
    clusters) and ``n_clusters`` (sanity check for missing
    contributions).
    """
    try:
        df = reader.load_portfolio(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        PortfolioRow(
            scenario_label   = str(r["scenario_label"]),
            sum_pnl_scaled   = float(r["sum_pnl_scaled"]),
            sum_pnl_original = float(r["sum_pnl_original"]),
            n_clusters       = int(r["n_clusters"]),
        )
        for r in df.to_dict(orient="records")
    ]
    return PortfolioResponse(run_id=run_id, n_scenarios=len(rows), rows=rows)


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

@router.get("/runs/{run_id}/clusters", response_model=ClusterSummaryResponse)
def get_run_clusters(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ClusterSummaryResponse:
    """Return the per-cluster × per-scenario summary for ``run_id``.

    Sourced from ``cluster_predictions.parquet``.  Long-format —
    one row per cluster × scenario, schema matches
    :meth:`HybridGnnRnnInferencePipeline.transform_predictions`.
    """
    try:
        df = reader.load_clusters_summary(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    rows = [
        ClusterSummaryRow(
            scenario_label    = str(r["scenario_label"]),
            cluster_id        = str(r["cluster_id"]),
            sum_pnl_scaled    = float(r["sum_pnl_scaled"]),
            sum_pnl_original  = float(r["sum_pnl_original"]),
            mean_pnl_original = float(r["mean_pnl_original"]),
            std_pnl_original  = float(r["std_pnl_original"]),
            min_pnl_original  = float(r["min_pnl_original"]),
            max_pnl_original  = float(r["max_pnl_original"]),
        )
        for r in df.to_dict(orient="records")
    ]
    return ClusterSummaryResponse(
        run_id      = run_id,
        n_clusters  = int(df["cluster_id"].nunique()) if len(df) else 0,
        n_scenarios = int(df["scenario_label"].nunique()) if len(df) else 0,
        rows        = rows,
    )


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/clusters/{cluster_id}/trades
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/clusters/{cluster_id}/trades",
    response_model=ClusterTradesResponse,
)
def get_run_cluster_trades(
    run_id:     str,
    cluster_id: str,
    space:      str = Query(
        "original",
        description=(
            "Output space — 'original' (inverse-scaled + notional-restored) "
            "or 'scaled' (model output space)."
        ),
        pattern=f"^({'|'.join(VALID_SPACES)})$",
    ),
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ClusterTradesResponse:
    """Return the per-cluster trade-level wide parquet.

    Wide-format on the wire (matrix + axis labels) rather than
    long-format because typical sizes are ~30k cells and a row-per-
    cell response would be 5-10× heavier.  The dashboard's AG Grid
    consumes the matrix directly; histograms / aggregates pivot
    cheaply from this shape.
    """
    try:
        df = reader.load_cluster_trades(run_id, cluster_id, space)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        # space validated by the regex above, but defence-in-depth.
        raise HTTPException(status_code=400, detail=str(e)) from e

    return ClusterTradesResponse(
        run_id          = run_id,
        cluster_id      = cluster_id,
        space           = space,
        n_scenarios     = int(len(df)),
        n_trades        = int(len(df.columns)),
        scenario_labels = [str(i) for i in df.index],
        trade_ids       = [str(c) for c in df.columns],
        values          = df.astype(float).values.tolist(),
    )


# ──────────────────────────────────────────────────────────────────────
# GET /runs/{run_id}/validation
# GET /runs/{run_id}/scenarios
# ──────────────────────────────────────────────────────────────────────

@router.get(
    "/runs/{run_id}/validation", response_model=ValidationSnapshotResponse,
)
def get_run_validation(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ValidationSnapshotResponse:
    """Return the ``ValidationReport`` snapshot embedded in the run's manifest.

    Cheaper than the active run's ``GET /validate`` because no
    on-disk parquet is involved — ``post_infer`` captures the
    report into the manifest at write time.
    """
    try:
        snapshot = reader.load_validation_snapshot(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ValidationSnapshotResponse(run_id=run_id, snapshot=snapshot)


@router.get(
    "/runs/{run_id}/scenarios", response_model=ScenariosSnapshotResponse,
)
def get_run_scenarios(
    run_id: str,
    reader: InferenceResultReader = Depends(get_result_reader),
) -> ScenariosSnapshotResponse:
    """Return the ``LoadedScenariosReport`` snapshot embedded in the manifest."""
    try:
        snapshot = reader.load_scenarios_snapshot(run_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return ScenariosSnapshotResponse(run_id=run_id, snapshot=snapshot)
