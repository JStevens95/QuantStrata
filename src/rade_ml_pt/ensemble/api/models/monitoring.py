"""Pydantic schemas for the ``/monitoring`` router.

Type-safe wire contracts for the five-button monitoring workflow:

    POST /monitoring/load          → LoadResponse
    POST /monitoring/scenarios     ← LoadScenariosRequest → LoadScenariosResponse
    POST /monitoring/validate                             → ValidateResponse
    POST /monitoring/run           ← RunRequest           → RunResponse
    POST /monitoring/promote                              → PromoteResponse
    GET  /monitoring/status                               → StatusResponse
    GET  /monitoring/events?cursor=N                      → EventsResponse
    GET  /monitoring/manifest                             → ManifestResponse
    GET  /monitoring/runs                                 → RunsListResponse
    GET  /monitoring/runs/{run_id}/manifest               → ManifestResponse
    GET  /monitoring/runs/{run_id}/drift_summary          → DriftSummaryResponse
    GET  /monitoring/runs/{run_id}/clusters               → ClusterSeverityResponse
    GET  /monitoring/runs/{run_id}/clusters/{cid}/drift   → ClusterDriftResponse

Most response models mirror the dataclass shapes the monitoring pipeline
already exposes (``MonitoringResult``, ``PromoteResult``) or the JSON
shapes its writers produce (``manifest.json``, ``drift_summary.json``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────
# Activity-log event (re-used from inference vocab)
# ──────────────────────────────────────────────────────────────────────

class MonitoringEventModel(BaseModel):
    """One activity-log event emitted by the monitoring pipeline.

    Wire shape identical to :data:`infer_events.ActivityEntry` — the
    monitoring pipeline emits through the same vocab.  Re-declared
    here so the ``/monitoring/events`` OpenAPI schema is self-
    contained (doesn't bleed inference model imports into the
    monitoring tab consumer).
    """

    id:     str            = Field(..., description="Hex UUID, set at event-construction time.")
    stage:  str            = Field(..., description="One of 'ingest', 'validate', 'inference', 'monitoring'.")
    phase:  str            = Field(..., description="Human-readable phase label.")
    status: str            = Field(..., description="One of 'ok', 'running', 'fail', 'pending'.")
    ts:     str            = Field(..., description="UTC ISO-8601 timestamp.")
    target: Optional[str]  = Field(None, description="Optional secondary label.")
    detail: Optional[str]  = Field(None, description="Optional detail / error text.")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/load
# ──────────────────────────────────────────────────────────────────────

class LoadResponse(BaseModel):
    """Result of the cold-load stage."""

    run_id:           str  = Field(..., description="Provisional run id (replaced when run() picks the canonical id).")
    ensemble_version: str  = Field(..., description="Resolved ensemble version.")
    n_clusters:       int  = Field(..., description="Number of clusters in the loaded ensemble.")
    status:           str  = Field(..., description="Run lifecycle state (one of 'loaded' / 'failed').")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/scenarios
# ──────────────────────────────────────────────────────────────────────

class LoadScenariosRequest(BaseModel):
    """Request body for ``POST /monitoring/scenarios``.

    Server-side path; file-upload (multipart) deferred — same contract
    decision as inference.
    """

    new_scenario_dir: str = Field(
        ...,
        description=(
            "Path on the API host to a folder containing the new-scenario "
            "shock CSVs.  Must be readable by the API process."
        ),
    )


class LoadScenariosResponse(BaseModel):
    """Result of parsing the scenario folder.

    Shape mirrors ``LoadedScenariosReport.to_dict()`` from the
    inference pipeline, since the underlying ``load_scenarios()`` is
    the same call the monitoring pipeline composes.
    """

    new_scenario_dir:  str       = Field(..., description="Resolved absolute path.")
    risk_factor_names: List[str] = Field(default_factory=list, description="RF stems parsed from the folder.")
    n_risk_factors:    int       = Field(0,  description="Risk-factor count surfaced by the shocks.")
    n_scenarios:       int       = Field(..., description="Total number of scenarios parsed.")
    scenario_labels:   List[str] = Field(..., description="List of scenario labels.")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/validate
# ──────────────────────────────────────────────────────────────────────

class ClusterRoutingDecisionModel(BaseModel):
    """One cluster's affected / unaffected decision.

    Re-declared here (rather than re-imported from inference models)
    so the monitoring schema is self-contained and OpenAPI consumers
    don't need to chase cross-router refs.  Field set mirrors
    :meth:`ClusterRoutingDecision.to_dict` exactly so the router can
    ``**decision.to_dict()`` directly into this model.
    """

    cluster_id:                 str       = Field(..., description="Cluster identifier.")
    is_affected:                bool      = Field(..., description="Whether this cluster needs a forward pass.")
    intersecting_risk_factors:  List[str] = Field(default_factory=list)
    n_elementary_trades:        int       = Field(0)
    n_target_trades:            int       = Field(0)
    missing_scenario_labels:    List[str] = Field(default_factory=list)


class ValidateResponse(BaseModel):
    """Result of running the validation report on the loaded scenarios."""

    ensemble_version:       str                                = Field(...)
    n_scenarios:            int                                = Field(...)
    scenario_labels:        List[str]                          = Field(...)
    cluster_decisions:      List[ClusterRoutingDecisionModel]  = Field(...)
    errors:                 List[str]                          = Field(default_factory=list)
    warnings:               List[str]                          = Field(default_factory=list)
    is_valid:               bool                               = Field(...)
    affected_cluster_ids:   List[str]                          = Field(default_factory=list)
    unaffected_cluster_ids: List[str]                          = Field(default_factory=list)
    affected_count:         int                                = Field(0)
    unaffected_count:       int                                = Field(0)


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/run
# ──────────────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    """Optional body for ``POST /monitoring/run``.

    All fields default to ``None`` — pass an explicit
    ``new_scenario_dir`` only if you skipped the staged
    ``/scenarios`` call (rare; UI always goes through the stages).
    """

    new_scenario_dir: Optional[str] = Field(
        None,
        description="Fallback scenario dir; usually omitted because /scenarios was already called.",
    )


class RunResponse(BaseModel):
    """Immediate response for ``POST /monitoring/run`` (non-blocking dispatch).

    Terminal metrics (``n_clusters``, ``severity``, ``manifest_path``,
    ...) are unknown at dispatch — they land on the state once the
    worker thread completes.  Clients poll ``GET /monitoring/status``
    + ``GET /monitoring/manifest`` to retrieve them.
    """

    run_id:        str = Field(..., description="Provisional run id at dispatch time.")
    status:        str = Field(..., description="Lifecycle state immediately after dispatch (typically 'running').")
    artifacts_dir: str = Field(..., description="Resolved per-run artifacts directory.")


# ──────────────────────────────────────────────────────────────────────
# POST /monitoring/promote
# ──────────────────────────────────────────────────────────────────────

class PromoteResponse(BaseModel):
    """Immediate response for ``POST /monitoring/promote`` (non-blocking).

    Same dispatch-time contract as ``RunResponse``: only the
    lifecycle status is known here; ``predictions_dir`` /
    ``n_clusters_predicted`` populate on the state once the worker
    finishes.
    """

    run_id: str = Field(..., description="Monitoring run id this promote is attached to.")
    status: str = Field(..., description="Lifecycle state after dispatch (typically 'promoting').")


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/status
# ──────────────────────────────────────────────────────────────────────

class StatusResponse(BaseModel):
    """Cheap status probe — drives the UI's next-button gate."""

    has_active_run:    bool          = Field(...)
    run_id:            Optional[str] = Field(None)
    ensemble_version:  Optional[str] = Field(None)
    status:            Optional[str] = Field(None)
    last_error:        Optional[str] = Field(None)
    n_events:          int           = Field(0)
    created_at:        Optional[str] = Field(None)
    artifacts_dir:     Optional[str] = Field(None)
    manifest_path:     Optional[str] = Field(None)
    predictions_dir:   Optional[str] = Field(None)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/events
# ──────────────────────────────────────────────────────────────────────

class EventsResponse(BaseModel):
    """Cursor-paginated activity log slice."""

    events:      List[MonitoringEventModel] = Field(...)
    next_cursor: int                        = Field(..., description="Pass back on the next poll to chain.")


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/manifest
# GET /monitoring/runs/{run_id}/manifest
# ──────────────────────────────────────────────────────────────────────

class ManifestResponse(BaseModel):
    """The full manifest JSON for a monitoring run.

    The inner ``manifest`` payload is left untyped (``Dict[str, Any]``)
    because the manifest schema is owned by the writer
    (:mod:`monitoring.writers.write_monitoring_manifest_json`) and we
    don't want to duplicate that schema in two places.  Consumers
    interested in field-level validation should reference
    :class:`MonitoringResult` / :class:`PromoteResult`.
    """

    run_id:   str            = Field(...)
    manifest: Dict[str, Any] = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs
# ──────────────────────────────────────────────────────────────────────

class RunSummary(BaseModel):
    """Lightweight summary for the run-history table.

    Reads exclusively from the manifest (small JSON), no parquets —
    safe to compute for hundreds of runs in a single
    ``GET /runs`` call.
    """

    run_id:                str            = Field(...)
    ensemble_version:      Optional[str]  = Field(None)
    status:                str            = Field(..., description="'in_progress' | 'complete' | 'promoted'")
    created_at:            Optional[str]  = Field(None)
    n_scenarios:           Optional[int]  = Field(None)
    n_clusters:            Optional[int]  = Field(None)
    n_clusters_affected:   Optional[int]  = Field(None)
    n_clusters_unaffected: Optional[int]  = Field(None)
    severity:              Optional[str]  = Field(None, description="Portfolio severity rollup.")
    mean_psi:              Optional[float] = Field(None)
    max_psi:               Optional[float] = Field(None)
    has_predictions:       bool           = Field(False, description="True iff promote-to-predictions has run.")


class RunsListResponse(BaseModel):
    """Result of ``GET /monitoring/runs`` — list every run on disk."""

    runs:  List[RunSummary] = Field(...)
    count: int              = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs/{run_id}/drift_summary
# ──────────────────────────────────────────────────────────────────────

class DriftSummaryResponse(BaseModel):
    """Portfolio-level drift KPIs.

    Inner ``summary`` mirrors
    :func:`monitoring.drift.build_portfolio_drift_summary`'s output —
    same untyped-dict reasoning as :class:`ManifestResponse`.
    """

    run_id:  str            = Field(...)
    summary: Dict[str, Any] = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs/{run_id}/clusters
# ──────────────────────────────────────────────────────────────────────

class ClusterSeverityRow(BaseModel):
    """One cluster's severity rollup (UI heatmap row)."""

    cluster_id: str             = Field(...)
    severity:   str             = Field(...)
    max_psi:    Optional[float] = Field(None)
    mean_psi:   Optional[float] = Field(None)
    n_features: Optional[int]   = Field(None)


class ClusterSeverityResponse(BaseModel):
    """Aggregate severity per cluster — sourced from drift_summary.json."""

    run_id: str                       = Field(...)
    rows:   List[ClusterSeverityRow]  = Field(...)
    count:  int                       = Field(...)


# ──────────────────────────────────────────────────────────────────────
# GET /monitoring/runs/{run_id}/clusters/{cluster_id}/drift
# ──────────────────────────────────────────────────────────────────────

class ClusterDriftRow(BaseModel):
    """One per-feature drift-table row."""

    cluster_id:    str             = Field(...)
    feature_name:  str             = Field(...)
    psi:           Optional[float] = Field(None)
    js_divergence: Optional[float] = Field(None)
    mean_shift:    Optional[float] = Field(None)
    std_ratio:     Optional[float] = Field(None)
    severity:      str             = Field(...)


class ClusterDriftResponse(BaseModel):
    """Long-format drift table for a single cluster."""

    run_id:     str                  = Field(...)
    cluster_id: str                  = Field(...)
    rows:       List[ClusterDriftRow] = Field(...)
    n_features: int                  = Field(...)


__all__ = [
    "MonitoringEventModel",
    "LoadResponse",
    "LoadScenariosRequest",
    "LoadScenariosResponse",
    "ClusterRoutingDecisionModel",
    "ValidateResponse",
    "RunRequest",
    "RunResponse",
    "PromoteResponse",
    "StatusResponse",
    "EventsResponse",
    "ManifestResponse",
    "RunSummary",
    "RunsListResponse",
    "DriftSummaryResponse",
    "ClusterSeverityRow",
    "ClusterSeverityResponse",
    "ClusterDriftRow",
    "ClusterDriftResponse",
]
