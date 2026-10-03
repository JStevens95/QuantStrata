"""Pydantic schemas for the ``/inference`` router.

Type-safe wire contracts for the three-button workflow:

    POST /inference/load          → LoadResponse
    POST /inference/scenarios     ← LoadScenariosRequest    → LoadScenariosResponse
    POST /inference/validate                                → ValidateResponse
    POST /inference/run           ← RunRequest              → RunResponse
    GET  /inference/status                                  → StatusResponse
    GET  /inference/events?cursor=N                         → EventsResponse
    GET  /inference/manifest                                → ManifestResponse

Most response models mirror the dataclass ``to_dict()`` shapes the
pipeline already exposes (``LoadedScenariosReport.to_dict``,
``ValidationReport.to_dict``).  Wrapping them in Pydantic gives the
typed client and OpenAPI docs full schema coverage without duplicating
the underlying state.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Activity-log event (mirror of ``infer_events.ActivityEntry``)
# ──────────────────────────────────────────────────────────────────────

class EventModel(BaseModel):
    """One activity-log event emitted by the pipeline.

    Wire shape identical to :data:`infer_events.ActivityEntry` (a plain
    dict).  Re-declared as a Pydantic model purely for OpenAPI schema
    coverage — the pipeline itself stays Pydantic-free.
    """

    id:     str            = Field(..., description="Hex UUID, set at event-construction time.")
    stage:  str            = Field(..., description="One of 'ingest', 'validate', 'inference'.")
    phase:  str            = Field(..., description="Human-readable phase label (e.g. 'Loading new-scenario shocks').")
    status: str            = Field(..., description="One of 'ok', 'running', 'fail', 'pending'.")
    ts:     str            = Field(..., description="UTC ISO-8601 timestamp.")
    target: Optional[str]  = Field(None, description="Optional secondary label (filename, cluster id, etc.).")
    detail: Optional[str]  = Field(None, description="Optional detail / error text.")


# ──────────────────────────────────────────────────────────────────────
# POST /inference/load
# ──────────────────────────────────────────────────────────────────────

class LoadResponse(BaseModel):
    """Result of the cold-load stage.

    Returned after :meth:`EnsembleInferencePipeline.load` completes.
    ``n_clusters`` lets the UI render the routing table skeleton before
    scenarios are loaded.
    """

    run_id:           str  = Field(..., description="Identifier for the newly-created run.")
    ensemble_version: str  = Field(..., description="Resolved ensemble version.")
    n_clusters:       int  = Field(..., description="Number of clusters in the loaded ensemble.")
    status:           str  = Field(..., description="Run lifecycle state (one of 'loaded' / 'failed').")


# ──────────────────────────────────────────────────────────────────────
# POST /inference/scenarios
# ──────────────────────────────────────────────────────────────────────

class LoadScenariosRequest(BaseModel):
    """Request body for ``POST /inference/scenarios``.

    For v1 the API accepts a server-side path; file-upload (multipart)
    is deferred to a later stage to keep the contract simple.  The
    path must be readable by the API process.
    """

    new_scenario_dir: str = Field(
        ...,
        description=(
            "Path on the API host to a folder containing one CSV "
            "per shocked risk factor.  Filename minus '.csv' is the "
            "risk-factor key."
        ),
    )


class LoadScenariosResponse(BaseModel):
    """Mirror of :meth:`LoadedScenariosReport.to_dict`."""

    new_scenario_dir:  str
    risk_factor_names: List[str]
    n_risk_factors:    int
    n_scenarios:       int
    scenario_labels:   List[str]


# ──────────────────────────────────────────────────────────────────────
# POST /inference/validate
# ──────────────────────────────────────────────────────────────────────

class ClusterRoutingDecisionModel(BaseModel):
    """Per-cluster decision in :class:`ValidationReport`."""

    cluster_id:                str
    is_affected:               bool
    intersecting_risk_factors: List[str]
    n_elementary_trades:       int
    n_target_trades:           int
    missing_scenario_labels:   List[str]


class ValidateResponse(BaseModel):
    """Mirror of :meth:`ValidationReport.to_dict`, including derived fields.

    Derived properties (``is_valid``, counts, etc.) are returned so
    the UI can render the validate-stage report without re-deriving
    them client-side.
    """

    ensemble_version:       str
    n_scenarios:            int
    scenario_labels:        List[str]
    cluster_decisions:      List[ClusterRoutingDecisionModel]
    errors:                 List[str]
    warnings:               List[str]

    is_valid:               bool
    affected_cluster_ids:   List[str]
    unaffected_cluster_ids: List[str]
    affected_count:         int
    unaffected_count:       int
    cheap_path_used:        bool


# ──────────────────────────────────────────────────────────────────────
# POST /inference/run
# ──────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Optional knobs for the run stage.

    Both fields are optional — sensible defaults are taken from the
    API settings + the run state when omitted.
    """

    artifacts_dir: Optional[str] = Field(
        None,
        description=(
            "Override for the per-run artifacts directory.  Defaults "
            "to <settings.artifacts_dir>/inference_runs/<run_id>."
        ),
    )
    batch_size:    Optional[int] = Field(
        None,
        description=(
            "Per-cluster chunk size for the chunked forward pass "
            "(see HybridGnnRnnInferencePipeline.predict_member_chunked). "
            "Defaults to the pipeline's built-in default."
        ),
    )


class RunResponse(BaseModel):
    """Acknowledgement of a background ``POST /inference/run``.

    Returned immediately after the pipeline is dispatched onto a
    worker thread (Stage 9).  ``status`` is therefore ``'running'``
    on success; the run's terminal outcome (``'complete'`` /
    ``'failed'``) and its metrics are surfaced through
    ``GET /status`` and ``GET /manifest`` once the worker finishes.

    The metric fields below remain on the schema so the client type
    stays stable even when Stage 9.5 adds a synchronous "wait until
    done" variant or eager completion in fast-path tests.
    """

    run_id:        str           = Field(..., description="Run identifier.")
    status:        str           = Field(..., description="Run lifecycle state at dispatch time — normally 'running'.")
    artifacts_dir: str           = Field(..., description="Resolved per-run artifacts root (populated immediately).")
    n_scenarios:   Optional[int] = Field(None, description="Number of scenarios scored (None until the worker completes).")
    n_clusters:    Optional[int] = Field(None, description="Number of clusters scored (None until the worker completes).")
    n_predictions: Optional[int] = Field(None, description="Total prediction cells (None until the worker completes).")
    manifest_path: Optional[str] = Field(None, description="Absolute path to manifest.json (None until run completes).")
    error:         Optional[str] = Field(None, description="Detail message; set only when status='failed'.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/status
# ──────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Lightweight current-state probe.

    Used by the UI between stages to gate the next button.  Designed
    to be cheap to poll (no I/O, no heavy serialisation).
    """

    has_active_run:   bool          = Field(..., description="True once /load has been called.")
    run_id:           Optional[str] = None
    ensemble_version: Optional[str] = None
    status:           Optional[str] = Field(None, description="Run lifecycle state (see services.inference_state.STATUS_*).")
    last_error:       Optional[str] = None
    n_events:         int           = Field(0, description="Total activity-log events emitted so far for this run.")
    created_at:       Optional[str] = None
    artifacts_dir:    Optional[str] = None
    manifest_path:    Optional[str] = None


# ──────────────────────────────────────────────────────────────────────
# GET /inference/events
# ──────────────────────────────────────────────────────────────────────

class EventsResponse(BaseModel):
    """Slice of the activity log for cursor-based polling.

    Pass ``next_cursor`` from the previous response as ``?cursor=`` on
    the next request to get only new events.
    """

    events:      List[EventModel] = Field(..., description="Events emitted since the cursor.")
    next_cursor: int              = Field(..., description="Total events emitted so far — pass on the next poll.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/manifest
# ──────────────────────────────────────────────────────────────────────

class ManifestResponse(BaseModel):
    """The run-level ``manifest.json`` written by ``post_infer``.

    Untyped on the values so the manifest schema can evolve without a
    contract churn — the dashboard reads keys it knows about and is
    permissive on the rest.  See
    :meth:`EnsembleInferencePipeline._write_run_manifest` for the
    canonical fields.
    """

    run_id:    str           = Field(..., description="Run identifier.")
    manifest:  Dict[str, Any] = Field(..., description="Raw manifest.json contents.")


# ══════════════════════════════════════════════════════════════════════
# Stage 10 — data-plane response models
#
# All models below describe responses served by the result-reading
# endpoints (`GET /inference/runs/...`).  They mirror the on-disk
# layout written by Stage 6 (`EnsembleInferencePipeline.post_infer`
# + `_post_infer_cluster`):
#
#   * Run-level summaries are LONG-format parquets → list-of-rows
#     responses (`PortfolioResponse`, `ClusterSummaryResponse`).
#   * Per-cluster trade-level outputs are WIDE-format parquets →
#     compact matrix response (`ClusterTradesResponse`) that the
#     dashboard's AG Grid can render directly.
# ══════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs
# ──────────────────────────────────────────────────────────────────────

class RunSummary(BaseModel):
    """Lightweight summary of one inference run on disk.

    Built by :meth:`InferenceResultReader.run_summary` from
    ``manifest.json``.  When the manifest doesn't exist yet (run is
    still in flight) ``status`` is ``'in_progress'`` and most
    fields are ``None`` / zero.
    """

    run_id:           str             = Field(..., description="Run identifier — ``<ensemble_version>__<UTC_yyyymmdd_HHMMSS>``.")
    ensemble_version: Optional[str]   = Field(None, description="Ensemble version this run scored against.")
    generated_at:     Optional[str]   = Field(None, description="UTC ISO timestamp when post_infer wrote the manifest.")
    n_scenarios:      int             = Field(0,    description="Scenarios scored in the run.")
    n_clusters:       int             = Field(0,    description="Clusters that produced artifacts.")
    status:           str             = Field("complete", description="``complete`` once manifest exists; ``in_progress`` otherwise.")
    manifest_path:    str             = Field(..., description="Absolute path to ``manifest.json`` on the server.")
    latency_seconds:  Optional[float] = Field(None, description="End-to-end run latency from the pipeline.")


class RunsListResponse(BaseModel):
    """Body of ``GET /inference/runs`` — every run discoverable on disk."""

    runs:  List[RunSummary] = Field(..., description="One entry per run, most recent first.")
    count: int              = Field(..., description="Total runs returned (== len(runs)).")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/portfolio
# ──────────────────────────────────────────────────────────────────────

class PortfolioRow(BaseModel):
    """One row of ``portfolio_predictions.parquet`` — one scenario.

    ``sum_pnl_*`` is the SUM of per-cluster ``sum_pnl_*`` for the
    same scenario, hence portfolio-level PnL in the requested space.
    ``n_clusters`` reports the number of clusters that contributed
    (so the dashboard can flag scenarios where some cluster was
    missing).
    """

    scenario_label:   str   = Field(..., description="Scenario identifier (row index of the parquet).")
    sum_pnl_scaled:   float = Field(..., description="Portfolio PnL in the model's scaled output space.")
    sum_pnl_original: float = Field(..., description="Portfolio PnL in the original (notional-restored) space.")
    n_clusters:       int   = Field(..., description="Distinct clusters that contributed to this scenario.")


class PortfolioResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/portfolio``."""

    run_id:      str                = Field(..., description="Run identifier.")
    n_scenarios: int                = Field(..., description="Total scenarios (== len(rows)).")
    rows:        List[PortfolioRow] = Field(..., description="One row per scenario.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

class ClusterSummaryRow(BaseModel):
    """One row of ``cluster_predictions.parquet`` — one cluster × one scenario.

    Mirrors the ``summary_df`` returned by
    :meth:`HybridGnnRnnInferencePipeline.transform_predictions`.
    All ``*_pnl_original`` stats are per-trade aggregates *within
    the cluster*; ``*_pnl_scaled`` exists only as a sanity check
    against the model's output space.
    """

    scenario_label:    str   = Field(..., description="Scenario identifier.")
    cluster_id:        str   = Field(..., description="Cluster identifier.")
    sum_pnl_scaled:    float = Field(..., description="Cluster PnL in scaled space.")
    sum_pnl_original:  float = Field(..., description="Cluster PnL in original space.")
    mean_pnl_original: float = Field(..., description="Mean per-trade PnL across the cluster, original space.")
    std_pnl_original:  float = Field(..., description="Std-dev per-trade PnL across the cluster, original space.")
    min_pnl_original:  float = Field(..., description="Min per-trade PnL within the cluster, original space.")
    max_pnl_original:  float = Field(..., description="Max per-trade PnL within the cluster, original space.")


class ClusterSummaryResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/clusters``."""

    run_id:      str                     = Field(..., description="Run identifier.")
    n_clusters:  int                     = Field(..., description="Distinct cluster IDs in the response.")
    n_scenarios: int                     = Field(..., description="Distinct scenarios in the response.")
    rows:        List[ClusterSummaryRow] = Field(..., description="One row per cluster × scenario.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/clusters/{cluster_id}/trades
# ──────────────────────────────────────────────────────────────────────

class ClusterTradesResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/clusters/{cid}/trades``.

    Returns the wide-format parquet as a (small) two-axis labelled
    matrix instead of a list of rows because:

    * Typical sizes are ``[n_scenarios=300, n_trades=100]`` — 30k
      floats; a long-format response would be 30k JSON objects, 5-10×
      heavier on the wire.
    * The dashboard's AG Grid renders an in-memory matrix directly.
    * The pivoting cost (long → wide for any other view, e.g.
      transposing to a histogram) is identical from this shape.
    """

    run_id:          str               = Field(..., description="Run identifier.")
    cluster_id:      str               = Field(..., description="Cluster identifier.")
    space:           str               = Field(..., description="Output space served — ``'scaled'`` or ``'original'``.")
    n_scenarios:     int               = Field(..., description="Scenarios scored (rows of ``values``).")
    n_trades:        int               = Field(..., description="Trades in the cluster (cols of ``values``).")
    scenario_labels: List[str]         = Field(..., description="Row labels of ``values`` — scenario IDs.")
    trade_ids:       List[str]         = Field(..., description="Column labels of ``values`` — canonical trade IDs.")
    values:          List[List[float]] = Field(..., description="2-D matrix ``[n_scenarios][n_trades]``.")


# ──────────────────────────────────────────────────────────────────────
# GET /inference/runs/{run_id}/validation
# GET /inference/runs/{run_id}/scenarios
# ──────────────────────────────────────────────────────────────────────

class ValidationSnapshotResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/validation``.

    Returns the ``ValidationReport`` that ``post_infer`` snapshotted
    into the manifest at write time.  ``snapshot`` is ``None`` for
    runs whose manifest didn't carry one (e.g. very old manifests).
    """

    run_id:   str                       = Field(..., description="Run identifier.")
    snapshot: Optional[Dict[str, Any]]  = Field(None, description="The ``ValidationReport.to_dict()`` payload at run time.")


class ScenariosSnapshotResponse(BaseModel):
    """Body of ``GET /inference/runs/{run_id}/scenarios``.

    Returns the ``LoadedScenariosReport`` that ``post_infer``
    snapshotted into the manifest.
    """

    run_id:   str                       = Field(..., description="Run identifier.")
    snapshot: Optional[Dict[str, Any]]  = Field(None, description="The ``LoadedScenariosReport.to_dict()`` payload at run time.")
