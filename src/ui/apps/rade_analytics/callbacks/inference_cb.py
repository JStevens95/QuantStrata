"""Inference Console callbacks — Stage 12 (API-driven, threaded ``/run``).

Wires the inference page's eight gestures to the
:mod:`~src.rade_ml_pt.ensemble.api.routers.inference` endpoints via
:class:`~src.ui.apps.rade_analytics.data.backend.RadeBackend`.

State machine
-------------

The page is a thin client over the API's run-state machine.  Each
gesture either advances the server-side state or polls it:

    on_mount     ─ url=/inference ──────────►  POST /load        (cold-load)
                                               │
                                               ▼
    on_upload    ─ Upload btn ──────────────►  POST /scenarios
                                               │
                                               ▼
    on_validate  ─ Validate btn ────────────►  POST /validate    (gates run)
                                               │
                                               ▼
    on_run       ─ Run btn ─────────────────►  POST /run         (async)
                                               │
                                               ▼  arms polling
    on_poll      ─ Interval tick ────┬──────►  GET  /events?cursor=N
                                     └──────►  GET  /status
                                               │
                                               ▼  terminal state
                                               hydrate run_meta_store
                                               │
                                               ▼
    hydrate_results  ─ run_meta_store ───────►  GET /runs/{id}/portfolio,
                                                GET /runs/{id}/clusters
                                                renders KPIs + figures + grid

Two ancillary callbacks decouple display from data flow:

    render_activity  ─ activity_log_store ──►  rebuilds the feed DOM
    on_row_select    ─ AG Grid click ───────►  writes selected_scenario_store

The polling pair (``polling_store`` + ``poll_interval``) is the only
piece of bespoke client state; everything else flows from server
state on demand.

Activity-log delivery
---------------------

Server events are emitted by :class:`EventCollector` on the pipeline
and exposed via ``GET /events?cursor=N``.  ``EventModel`` is shape-
compatible with the layout's ``ActivityEntry`` dict, so events flow
through with ``model_dump()`` and need no further mapping.  Local
events (e.g. *"Ensemble loaded"* on mount) use :func:`_local_event`
to keep the timestamp + UUID conventions consistent.

Page Contract anchors
---------------------

* §2.1 — capture/render split: capture writes Stores + side effects,
  render reads Stores + emits children/figures.  Both sections are
  registered through :func:`register`, mirroring ``overview_cb`` /
  ``portfolio_cb``.
* §3 Rule L4 — page identity is the URL pathname, not a per-page
  mount tripwire; ``on_mount`` keys on ``Input(SHELL_IDS["url"], …)``.
* §6 — long-running side effects (the ``/run`` dispatch) write the
  ``polling_store`` *before* the user navigates away; if they
  revisit, ``on_mount`` re-syncs via ``GET /status``.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html, no_update
from dash.exceptions import PreventUpdate
from dash_iconify import DashIconify

from ..figures.inference_charts import (
    build_chart_main,
    build_coverage_donut,
    build_sensitivity_heatmap,
    build_sensitivity_tornado,
    empty_sensitivity_heatmap,
    empty_sensitivity_tornado,
    build_risk_attribution_chart,
    build_stress_tails_chart,
    empty_coverage_donut,
    empty_pnl_distribution,
    empty_risk_attribution,
    empty_stress_tails,
)
from ..layouts.inference import INFERENCE_IDS, render_activity_entries
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER:        str = "—"
_INFERENCE_ROUTE:    str = "/inference"

# API run statuses that mean the worker thread has finished — the
# poll callback uses this to disarm the polling pair.
_TERMINAL_STATUSES: set[str] = {"complete", "failed"}

# Statuses the API reports while a run is in flight (worker thread
# alive).  Anything outside this set + the terminal set means the
# state machine is idle and the run button should be re-enabled.
_RUNNING_STATUS:     str = "running"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _local_event(
    stage:  str,
    phase:  str,
    *,
    status: str           = "ok",
    target: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct an activity-log entry locally.

    Used by callbacks that want to narrate a client-side gesture
    (e.g. *"Ensemble load requested"*) without waiting for the
    server's confirmation event.  Shape matches
    :class:`~src.rade_ml_pt.ensemble.api.models.inference.EventModel`
    so the local entry can sit alongside server events in the
    activity-log store without a mapping layer.
    """
    return {
        "id":     uuid.uuid4().hex,
        "stage":  stage,
        "phase":  phase,
        "status": status,
        "ts":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": target,
        "detail": detail,
    }


def _format_currency(value: Any, *, digits: int = 0) -> str:
    """Pretty-print a PnL-style number; ``—`` on missing / NaN.

    Default ``digits=0`` matches the Inference Console's KPI-tile
    convention (Phase 0.1) — callers that want sub-unit precision
    must pass ``digits=`` explicitly.
    """
    if value is None:
        return _PLACEHOLDER
    try:
        val = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:,.{digits}f}"


def _format_int(value: Any) -> str:
    if value is None:
        return _PLACEHOLDER
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return _PLACEHOLDER


def _status_icon(status: str) -> DashIconify:
    """Render an ingest-status icon matching the activity log's
    semantic-colour palette so the two feedback channels stay
    visually coherent."""
    mapping = {
        "ok":      ("tabler:circle-check",   "#34d399"),
        "fail":    ("tabler:circle-x",       "#fb7185"),
        "running": ("tabler:loader-2",       "#a78bfa"),
        "pending": ("tabler:circle-dashed",  "#94a3b8"),
    }
    icon, colour = mapping.get(status, mapping["pending"])
    return DashIconify(icon=icon, width=18, color=colour)


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────

def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Inference Console callback to ``app``.

    Mirrors the Page Contract §2 capture/render split — same shape
    every other page module follows (``overview_cb``, ``portfolio_cb``,
    ``cluster_deep_dive_cb``).
    """
    _register_capture(app, backend)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers
# ─────────────────────────────────────────────────────────────────────

def _register_capture(app: "Dash", backend: "RadeBackend") -> None:
    """Capture-side callbacks (gestures → Stores + API side effects)."""
    _register_on_mount(app, backend)
    _register_on_upload(app, backend)
    _register_on_validate(app, backend)
    _register_on_run(app, backend)
    _register_on_poll(app, backend)
    _register_on_row_select(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Render-side callbacks (Stores → DOM, no Store writes)."""
    _register_render_activity(app)
    _register_hydrate_results(app, backend)
    _register_hydrate_diagnostics(app, backend)
    _register_hydrate_sensitivity(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. on_mount — URL hits /inference → POST /load
# ═════════════════════════════════════════════════════════════════════

def _register_on_mount(app: "Dash", backend: "RadeBackend") -> None:
    """Cold-load the ensemble whenever the page first mounts.

    Idempotent on the server side: an already-loaded API silently
    replaces its state.  We still gate with a ``/status`` probe to
    avoid the full registry walk on every route flip.
    """

    @app.callback(
        Output(INFERENCE_IDS["subtitle"],          "children"),
        Output(INFERENCE_IDS["activity_log_store"], "data"),
        Output(INFERENCE_IDS["ingest_meta_store"],  "data"),
        Output(INFERENCE_IDS["run_meta_store"],     "data"),
        Output(INFERENCE_IDS["polling_store"],      "data"),
        Output(INFERENCE_IDS["poll_interval"],      "disabled"),
        Input(SHELL_IDS["url"],                     "pathname"),
        prevent_initial_call=False,
    )
    def _on_mount(
        pathname: Optional[str],
    ) -> Tuple[Any, List[Dict[str, Any]], Optional[Any], Optional[Any], Dict[str, Any], bool]:
        if pathname != _INFERENCE_ROUTE:
            raise PreventUpdate

        # Fresh page mount → reset every page Store to its initial
        # shape.  This is intentional: stale data from a prior visit
        # would otherwise leak into the new render before /status
        # has a chance to refresh it.
        polling_initial = {"armed": False, "run_id": None, "cursor": 0}

        # /status: cheap, no-side-effect probe.  Tells us whether
        # the API already has an active run we should adopt or
        # whether we need to do a cold /load.
        status_res = backend.inference_status()
        if status_res.ok and status_res.data is not None and status_res.data.has_active_run:
            status = status_res.data
            subtitle = (
                f"Run {status.run_id} · ensemble {status.ensemble_version} "
                f"· state '{status.status}'"
            )
            log: List[Dict[str, Any]] = [
                _local_event(
                    "inference",
                    "Resumed existing run",
                    target=status.run_id,
                    detail=f"State: {status.status}",
                )
            ]
            # Adopt the existing run_id so subsequent ``/events`` polls
            # advance the same cursor.  Arm only if the run is still
            # in flight; otherwise leave disarmed so the user must
            # explicitly re-arm via the Run button.
            armed = status.status == _RUNNING_STATUS
            polling_initial = {
                "armed":   armed,
                "run_id":  status.run_id,
                "cursor":  status.n_events or 0,
            }
            interval_disabled = not armed
            return subtitle, log, None, None, polling_initial, interval_disabled

        # No active run → cold-load.  Failures here are surfaced via
        # the activity log so the user sees *why* nothing else works.
        load_res = backend.inference_load()
        if not load_res.ok:
            logger.warning("/load failed: %s", load_res.error)
            return (
                "Failed to load ensemble — check API logs.",
                [_local_event(
                    "inference",
                    "Ensemble load failed",
                    status="fail",
                    detail=load_res.error,
                )],
                None,
                None,
                polling_initial,
                True,
            )

        load = load_res.data
        subtitle = (
            f"Loaded ensemble {load.ensemble_version} "
            f"({load.n_clusters} clusters) — upload scenarios to begin."
        )
        log = [_local_event(
            "inference",
            "Ensemble loaded",
            target=load.ensemble_version,
            detail=f"{load.n_clusters} clusters",
        )]
        return subtitle, log, None, None, polling_initial, True


# ═════════════════════════════════════════════════════════════════════
# 2. on_upload — Upload scenarios btn → POST /scenarios
# ═════════════════════════════════════════════════════════════════════

def _register_on_upload(app: "Dash", backend: "RadeBackend") -> None:
    """Submit the typed folder path to the API's scenario loader."""

    @app.callback(
        Output(INFERENCE_IDS["ingest_meta_store"],   "data", allow_duplicate=True),
        Output(INFERENCE_IDS["ingest_status"],       "children"),
        Output(INFERENCE_IDS["activity_log_store"],  "data", allow_duplicate=True),
        Output(INFERENCE_IDS["upload_scenarios_btn"], "loading"),
        Input(INFERENCE_IDS["upload_scenarios_btn"], "n_clicks"),
        State(INFERENCE_IDS["scenario_folder_path"], "value"),
        State(INFERENCE_IDS["activity_log_store"],   "data"),
        prevent_initial_call=True,
    )
    def _on_upload(
        n_clicks:    Optional[int],
        folder_path: Optional[str],
        log:         Optional[List[Dict[str, Any]]],
    ) -> Tuple[Any, Any, List[Dict[str, Any]], bool]:
        if not n_clicks:
            raise PreventUpdate

        log = list(log or [])

        # Defensive empty-input check.  The button is always
        # clickable but a no-text submission is a no-op.
        if not folder_path or not folder_path.strip():
            log.append(_local_event(
                "ingest",
                "Empty scenario folder path",
                status="fail",
                detail="Type a server-readable folder path before uploading.",
            ))
            return no_update, _status_icon("fail"), log, False

        log.append(_local_event(
            "ingest",
            "Uploading scenarios",
            status="running",
            target=folder_path,
        ))

        # The actual API call.  Synchronous on the server side
        # (parsing a folder of CSVs is fast), so the loading state
        # on the button only ever shows briefly.
        res = backend.inference_load_scenarios(folder_path)
        if not res.ok:
            logger.warning("/scenarios failed: %s", res.error)
            log.append(_local_event(
                "ingest",
                "Scenario load failed",
                status="fail",
                target=folder_path,
                detail=res.error,
            ))
            return no_update, _status_icon("fail"), log, False

        scenarios = res.data
        ingest_meta: Dict[str, Any] = {
            "source":            folder_path,
            "n_risk_factors":    scenarios.n_risk_factors,
            "n_scenarios":       scenarios.n_scenarios,
            "risk_factor_names": list(scenarios.risk_factor_names),
            "scenario_labels":   list(scenarios.scenario_labels),
            "completed_ts":      datetime.now(timezone.utc).isoformat(
                timespec="seconds",
            ),
        }
        log.append(_local_event(
            "ingest",
            "Scenarios ingested",
            target=folder_path,
            detail=(
                f"{scenarios.n_scenarios} scenarios across "
                f"{scenarios.n_risk_factors} risk factors"
            ),
        ))
        return ingest_meta, _status_icon("ok"), log, False


# ═════════════════════════════════════════════════════════════════════
# 3. on_validate — Validate-only btn → POST /validate
# ═════════════════════════════════════════════════════════════════════

def _register_on_validate(app: "Dash", backend: "RadeBackend") -> None:
    """Validate the loaded scenarios and gate the Run button on success."""

    @app.callback(
        Output(INFERENCE_IDS["manifest_preview_container"], "children"),
        Output(INFERENCE_IDS["run_btn"],                    "disabled"),
        Output(INFERENCE_IDS["activity_log_store"],         "data", allow_duplicate=True),
        Output(INFERENCE_IDS["validate_only_btn"],          "loading"),
        Input(INFERENCE_IDS["validate_only_btn"],           "n_clicks"),
        State(INFERENCE_IDS["activity_log_store"],          "data"),
        prevent_initial_call=True,
    )
    def _on_validate(
        n_clicks: Optional[int],
        log:      Optional[List[Dict[str, Any]]],
    ) -> Tuple[List[Any], bool, List[Dict[str, Any]], bool]:
        if not n_clicks:
            raise PreventUpdate

        log = list(log or [])
        log.append(_local_event("validate", "Validation requested", status="running"))

        res = backend.inference_validate()
        if not res.ok:
            logger.warning("/validate failed: %s", res.error)
            log.append(_local_event(
                "validate",
                "Validation failed",
                status="fail",
                detail=res.error,
            ))
            return (
                [_manifest_card_error(res.error or "Unknown error")],
                True,
                log,
                False,
            )

        report = res.data
        is_valid = bool(report.is_valid)
        log.append(_local_event(
            "validate",
            "Validation complete" if is_valid else "Validation failed",
            status="ok" if is_valid else "fail",
            detail=(
                f"{report.affected_count} affected / "
                f"{report.unaffected_count} unaffected clusters"
            ),
        ))
        return (
            _manifest_card_for_validation(report),
            not is_valid,
            log,
            False,
        )


def _manifest_card_for_validation(report: Any) -> List[Any]:
    """Render a compact validation summary into the manifest card."""
    rows: List[Any] = [
        html.Div(
            f"Ensemble · {report.ensemble_version}",
            className="text-xs font-semibold text-slate-200",
        ),
        html.Div(
            f"{report.n_scenarios} scenarios across "
            f"{len(report.cluster_decisions)} clusters",
            className="text-xs text-slate-400",
        ),
        html.Div(
            f"Affected: {report.affected_count} · "
            f"Unaffected: {report.unaffected_count} · "
            f"Cheap path: {'yes' if report.cheap_path_used else 'no'}",
            className="text-xs text-slate-400",
        ),
    ]
    if report.errors:
        rows.append(html.Div(
            "Errors:",
            className="text-xs font-semibold text-rose-300 mt-2",
        ))
        for err in report.errors:
            rows.append(html.Div(f"• {err}", className="text-xs text-rose-300"))
    if report.warnings:
        rows.append(html.Div(
            "Warnings:",
            className="text-xs font-semibold text-amber-300 mt-2",
        ))
        for warn in report.warnings:
            rows.append(html.Div(f"• {warn}", className="text-xs text-amber-300"))
    return rows


def _manifest_card_error(message: str) -> Any:
    """Render an error state in the manifest card."""
    return html.Div(
        className="flex flex-col gap-1 text-rose-300",
        children=[
            html.Div("Validation failed", className="text-xs font-semibold"),
            html.Div(message, className="text-xs"),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# 4. on_run — Run btn → POST /run + arm polling
# ═════════════════════════════════════════════════════════════════════

def _register_on_run(app: "Dash", backend: "RadeBackend") -> None:
    """Dispatch the inference run and arm the polling pair.

    The API's ``/run`` returns immediately with ``status='running'``;
    the worker thread does the actual forward pass.  We mark the
    polling store armed and disable=False the interval so the next
    tick (1Hz later) starts draining ``/events``.
    """

    @app.callback(
        Output(INFERENCE_IDS["polling_store"],        "data", allow_duplicate=True),
        Output(INFERENCE_IDS["poll_interval"],        "disabled", allow_duplicate=True),
        Output(INFERENCE_IDS["activity_log_store"],   "data", allow_duplicate=True),
        Output(INFERENCE_IDS["run_btn"],              "loading"),
        Input(INFERENCE_IDS["run_btn"],               "n_clicks"),
        State(INFERENCE_IDS["polling_store"],         "data"),
        State(INFERENCE_IDS["activity_log_store"],    "data"),
        prevent_initial_call=True,
    )
    def _on_run(
        n_clicks:   Optional[int],
        polling:    Optional[Dict[str, Any]],
        log:        Optional[List[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], bool, List[Dict[str, Any]], bool]:
        if not n_clicks:
            raise PreventUpdate

        log = list(log or [])
        log.append(_local_event("inference", "Run dispatched", status="running"))

        res = backend.inference_start()
        if not res.ok:
            logger.warning("/run failed: %s", res.error)
            log.append(_local_event(
                "inference",
                "Run dispatch failed",
                status="fail",
                detail=res.error,
            ))
            return polling or {"armed": False, "run_id": None, "cursor": 0}, True, log, False

        run = res.data
        # Cursor=0 — the worker thread may have already emitted a
        # few events before our HTTP response landed; the first
        # /events poll picks those up.
        new_polling = {
            "armed":  True,
            "run_id": run.run_id,
            "cursor": 0,
        }
        # Interval ``disabled=False`` flips polling on; the next
        # ``on_poll`` tick (≤1s later) will start the drain.
        return new_polling, False, log, True


# ═════════════════════════════════════════════════════════════════════
# 5. on_poll — Interval tick → GET /events + GET /status
# ═════════════════════════════════════════════════════════════════════

def _register_on_poll(app: "Dash", backend: "RadeBackend") -> None:
    """Drain events + watch for terminal state once per interval tick.

    Heart of the live-update story.  Runs at 1 Hz when armed.  Two
    invariants worth highlighting:

    * ``polling_store.cursor`` is the single source of truth for
      pagination.  We never trust the interval's ``n_intervals``.
    * Terminal-state hydration (writing ``run_meta_store``) happens
      *here*, not in :func:`_on_run`, because only the poll knows
      when the worker thread actually finishes.
    """

    @app.callback(
        Output(INFERENCE_IDS["activity_log_store"],  "data", allow_duplicate=True),
        Output(INFERENCE_IDS["polling_store"],       "data", allow_duplicate=True),
        Output(INFERENCE_IDS["poll_interval"],       "disabled", allow_duplicate=True),
        Output(INFERENCE_IDS["run_meta_store"],      "data", allow_duplicate=True),
        Output(INFERENCE_IDS["run_btn"],             "loading", allow_duplicate=True),
        Input(INFERENCE_IDS["poll_interval"],        "n_intervals"),
        State(INFERENCE_IDS["polling_store"],        "data"),
        State(INFERENCE_IDS["activity_log_store"],   "data"),
        prevent_initial_call=True,
    )
    def _on_poll(
        n_intervals: Optional[int],
        polling:     Optional[Dict[str, Any]],
        log:         Optional[List[Dict[str, Any]]],
    ) -> Tuple[Any, Any, Any, Any, Any]:
        polling = polling or {"armed": False, "run_id": None, "cursor": 0}
        if not polling.get("armed"):
            # Defence-in-depth: the interval should be disabled when
            # disarmed, but a stale tick can still land in-flight.
            # PreventUpdate is the cheapest way to short-circuit
            # without firing six no-ops on every output.
            raise PreventUpdate

        cursor = int(polling.get("cursor", 0))
        log = list(log or [])

        # 1. Drain new events.  EventModel is shape-compatible with
        #    the layout's activity-entry contract, so model_dump()
        #    is a direct passthrough.
        events_res = backend.inference_events(cursor=cursor)
        if events_res.ok and events_res.data is not None:
            new_events = events_res.data.events
            for ev in new_events:
                log.append(ev.model_dump())
            cursor = int(events_res.data.next_cursor)

        # 2. Probe terminal state.  We do this every tick (even if
        #    no new events landed) so a fast run that completes
        #    between ticks still gets hydrated promptly.
        status_res = backend.inference_status()
        if not status_res.ok or status_res.data is None:
            # Transport hiccup — keep polling, the next tick may
            # succeed.  Don't disarm here; that would silently
            # strand the page on a temporary network blip.
            logger.debug("/status hiccup: %s", status_res.error)
            return (
                log,
                {**polling, "cursor": cursor},
                no_update,
                no_update,
                no_update,
            )

        status = status_res.data
        if status.status in _TERMINAL_STATUSES:
            # Terminal state — disarm, snapshot run_meta_store, and
            # stop the interval.  The hydrate_results callback fires
            # off run_meta_store and pulls the data-plane artifacts.
            run_meta = {
                "run_id":           status.run_id,
                "ensemble_version": status.ensemble_version,
                "status":           status.status,
                "manifest_path":    status.manifest_path,
                "completed_ts":     datetime.now(timezone.utc).isoformat(
                    timespec="seconds",
                ),
            }
            disarmed = {**polling, "armed": False, "cursor": cursor}
            return log, disarmed, True, run_meta, False

        # Still running — bump the cursor, keep polling.
        return (
            log,
            {**polling, "cursor": cursor},
            no_update,
            no_update,
            no_update,
        )


# ═════════════════════════════════════════════════════════════════════
# 6. render_activity — activity_log_store → activity-feed DOM
# ═════════════════════════════════════════════════════════════════════

def _register_render_activity(app: "Dash") -> None:
    """Rebuild the activity feed whenever its Store mutates.

    The activity log is the user-visible heart of the page during a
    run — every other callback advances it indirectly via
    ``allow_duplicate=True`` writes; this single render callback
    converts the store payload to DOM children.
    """

    @app.callback(
        Output(INFERENCE_IDS["activity_log_container"], "children"),
        Input(INFERENCE_IDS["activity_log_store"],      "data"),
        prevent_initial_call=False,
    )
    def _render(entries: Optional[List[Dict[str, Any]]]) -> List[Any]:
        return render_activity_entries(entries)


# ═════════════════════════════════════════════════════════════════════
# 7. hydrate_results — run_meta_store → KPIs + figures + AG Grid
# ═════════════════════════════════════════════════════════════════════

def _register_hydrate_results(app: "Dash", backend: "RadeBackend") -> None:
    """Pull data-plane artifacts when a run completes and paint the page.

    Fires on:
      * ``run_meta_store`` write (terminal-state hydration from
        :func:`_on_poll`), and
      * each of the three segmented controls so users can switch
        chart modes without re-running.

    Stage 12 still uses :mod:`figures.inference_charts`'s empty-state
    builders for the three figures; Stage 13 swaps them in place.
    Everything else (KPIs, AG Grid, stress mini-KPIs) is live data.
    """

    @app.callback(
        # KPI strip
        Output(INFERENCE_IDS["kpi_scenarios_value"], "children"),
        Output(INFERENCE_IDS["kpi_clusters_value"],  "children"),
        Output(INFERENCE_IDS["kpi_latency_value"],   "children"),
        Output(INFERENCE_IDS["kpi_portfolio_value"], "children"),
        # Three figures
        Output(INFERENCE_IDS["chart_main"],              "figure"),
        Output(INFERENCE_IDS["risk_attribution_chart"],  "figure"),
        Output(INFERENCE_IDS["stress_tails_chart"],      "figure"),
        # Stress mini-KPIs
        Output(INFERENCE_IDS["stress_kpi_var"],   "children"),
        Output(INFERENCE_IDS["stress_kpi_cvar"],  "children"),
        Output(INFERENCE_IDS["stress_kpi_worst"], "children"),
        # AG Grid rows
        Output(INFERENCE_IDS["scenario_results_grid"], "rowData"),
        # Triggers
        Input(INFERENCE_IDS["run_meta_store"],            "data"),
        Input(INFERENCE_IDS["chart_view_mode"],           "value"),
        Input(INFERENCE_IDS["risk_attribution_breakdown"], "value"),
        Input(INFERENCE_IDS["stress_tails_mode"],         "value"),
        prevent_initial_call=False,
    )
    def _hydrate(
        run_meta:   Optional[Dict[str, Any]],
        chart_mode: Optional[str],
        risk_axis:  Optional[str],
        stress_mode: Optional[str],
    ) -> Tuple[Any, ...]:
        # Stage 13: ``chart_mode``, ``risk_axis``, and ``stress_mode``
        # now drive the three figure dispatchers in
        # :mod:`figures.inference_charts`.  Each dispatcher falls
        # back to the matching empty figure when no data is loaded,
        # so the "no run yet" branch below can use the static
        # empty builders without dispatching.

        empty_returns: Tuple[Any, ...] = (
            _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
            empty_pnl_distribution(),
            empty_risk_attribution(),
            empty_stress_tails(),
            _stress_kpi_block("VaR (95%)",  _PLACEHOLDER),
            _stress_kpi_block("CVaR (95%)", _PLACEHOLDER),
            _stress_kpi_block("Worst loss", _PLACEHOLDER),
            [],
        )

        # No completed run yet — keep the empty-state painting in
        # place but don't waste an API call.
        if not run_meta or run_meta.get("status") != "complete":
            return empty_returns

        run_id = run_meta.get("run_id")
        if not run_id:
            return empty_returns

        # Portfolio + cluster summary parquets.  Both are tiny
        # (one row per scenario / cluster×scenario) and cached at
        # the backend layer, so re-firing on segmented-control flips
        # is cheap.
        portfolio_res = backend.inference_portfolio_df(run_id)
        clusters_res  = backend.inference_clusters_df(run_id)

        if not portfolio_res.ok or portfolio_res.data is None:
            logger.warning("portfolio fetch failed: %s", portfolio_res.error)
            return empty_returns

        portfolio_df = portfolio_res.data
        clusters_df  = (
            clusters_res.data
            if clusters_res.ok and clusters_res.data is not None
            else pd.DataFrame()
        )

        # --- KPI strip ---------------------------------------------------
        n_scenarios = int(len(portfolio_df))
        n_clusters  = int(
            clusters_df["cluster_id"].nunique()
            if not clusters_df.empty and "cluster_id" in clusters_df.columns
            else 0
        )
        portfolio_total = float(
            portfolio_df["sum_pnl_original"].sum()
            if "sum_pnl_original" in portfolio_df.columns
            else 0.0
        )
        # Latency comes from the manifest — fetch it lazily here
        # rather than passing through ``run_meta`` so a refresh of
        # the manifest after the run (rare but possible) is picked
        # up automatically.  The same manifest fetch supplies the
        # canonical expected-cluster list for the Phase 0.2 zero-pad
        # in :func:`build_risk_attribution_chart` below.
        manifest_res = backend.inference_run_manifest(run_id)
        latency_s    = (
            manifest_res.data.manifest.get("latency_seconds")
            if manifest_res.ok and manifest_res.data is not None else None
        )
        latency_txt  = (
            f"{float(latency_s):.0f}s"
            if latency_s is not None else _PLACEHOLDER
        )

        # Expected-cluster list for the risk-attribution chart's
        # defensive zero-pad.  ``validation.cluster_decisions`` is
        # populated for every router cluster (affected + unaffected)
        # so it is the canonical source of "what should appear".
        #
        # The same ``cluster_decisions`` list is also handed to the
        # Phase 1a.2 RF-attribution builder (Approach B) — it needs
        # the per-cluster ``intersecting_risk_factors`` payload to
        # split each cluster's signed PnL across its RFs.
        expected_cluster_ids: List[str] = []
        cluster_decisions: List[Dict[str, Any]] = []
        if manifest_res.ok and manifest_res.data is not None:
            validation_block = (
                manifest_res.data.manifest.get("validation") or {}
            )
            for decision in validation_block.get("cluster_decisions") or []:
                if not isinstance(decision, dict):
                    continue
                cluster_decisions.append(decision)
                cid = decision.get("cluster_id")
                if cid:
                    expected_cluster_ids.append(str(cid))

        # --- Stress mini-KPIs --------------------------------------------
        # VaR / CVaR on the portfolio's sum_pnl_original — quick
        # quantile arithmetic, no extra API roundtrip.
        var_txt   = _PLACEHOLDER
        cvar_txt  = _PLACEHOLDER
        worst_txt = _PLACEHOLDER
        if "sum_pnl_original" in portfolio_df.columns and not portfolio_df.empty:
            pnl = portfolio_df["sum_pnl_original"].astype(float)
            var_value  = float(pnl.quantile(0.05))
            cvar_value = float(pnl[pnl <= var_value].mean()) if (pnl <= var_value).any() else var_value
            worst_value = float(pnl.min())
            var_txt   = _format_currency(var_value)
            cvar_txt  = _format_currency(cvar_value)
            worst_txt = _format_currency(worst_value)

        # --- AG Grid rows -----------------------------------------------
        # Map portfolio rows onto the column defs the layout
        # declares: ``scenario_id`` / ``cluster`` / ``predicted`` /
        # ``p95_band`` / ``confidence``.  Two columns have no
        # natural source from portfolio summary (``p95_band``,
        # ``confidence``); we leave them None so the grid renders
        # them as the ``—`` placeholder defined in the column
        # ``valueFormatter``.
        row_data: List[Dict[str, Any]] = [
            {
                "scenario_id": str(row.get("scenario_label", "")),
                "cluster":     f'{int(row.get("n_clusters", 0))} clusters',
                "predicted":   float(row.get("sum_pnl_original", 0.0)),
                "p95_band":    None,
                "confidence":  None,
            }
            for _, row in portfolio_df.iterrows()
        ]

        # Stage 13 — dispatch each figure on its segmented-control
        # mode.  Builders defensively fall back to the matching
        # empty figure if the upstream frame is missing the
        # expected column, so we don't pre-branch here.
        chart_main_fig       = build_chart_main(portfolio_df, chart_mode)
        risk_attribution_fig = build_risk_attribution_chart(
            clusters_df,
            risk_axis,
            expected_cluster_ids=expected_cluster_ids or None,
            cluster_decisions=cluster_decisions or None,
        )
        stress_tails_fig     = build_stress_tails_chart(portfolio_df, stress_mode)

        return (
            _format_int(n_scenarios),
            _format_int(n_clusters),
            latency_txt,
            _format_currency(portfolio_total),
            chart_main_fig,
            risk_attribution_fig,
            stress_tails_fig,
            _stress_kpi_block("VaR (95%)",  var_txt),
            _stress_kpi_block("CVaR (95%)", cvar_txt),
            _stress_kpi_block("Worst loss", worst_txt),
            row_data,
        )


def _stress_kpi_block(label: str, value: str) -> List[Any]:
    """Re-render one stress-tile body (label + value).

    Kept in this module rather than ``layouts/inference.py`` because
    the original layout helper builds the *outer* tile div; this
    helper builds only the children that the callback owns.
    """
    return [
        html.Div(label, className="rade-stress-mini-label"),
        html.Div(value, className="rade-stress-mini-value"),
    ]


# ═════════════════════════════════════════════════════════════════════
# 8. hydrate_diagnostics — run_meta_store → run-info / routing / donut /
#    validation report  (Phase 0.3)
# ═════════════════════════════════════════════════════════════════════

def _register_hydrate_diagnostics(app: "Dash", backend: "RadeBackend") -> None:
    """Hydrate every container on the Diagnostics sub-tab.

    Single manifest fetch feeds four outputs:

    * **Run info** — six mini-tiles (run_id / ensemble / status /
      generated_at / input_mode / latency).
    * **Routing matrix** — per-cluster decision table.
    * **Coverage donut** — affected vs unaffected counts.
    * **Validation report** — errors + warnings, or the "no issues"
      green badge.

    Fires on ``run_meta_store`` writes, same trigger as
    :func:`_register_hydrate_results`.  Pre-run state shows the
    layout-baked placeholders, so we only need to return the empty
    tuple for the *no-data* branch.
    """

    @app.callback(
        Output(INFERENCE_IDS["diagnostics_run_info"],         "children"),
        Output(INFERENCE_IDS["diagnostics_routing_table"],    "children"),
        Output(INFERENCE_IDS["diagnostics_coverage_donut"],   "figure"),
        Output(INFERENCE_IDS["diagnostics_validation_card"],  "children"),
        Input(INFERENCE_IDS["run_meta_store"],                "data"),
        prevent_initial_call=False,
    )
    def _hydrate_diag(
        run_meta: Optional[Dict[str, Any]],
    ) -> Tuple[List[Any], Any, go.Figure, List[Any]]:
        empty_returns: Tuple[List[Any], Any, go.Figure, List[Any]] = (
            _diag_run_info_placeholder(),
            _diag_routing_placeholder(),
            empty_coverage_donut(),
            _diag_validation_placeholder(),
        )

        if not run_meta or run_meta.get("status") != "complete":
            return empty_returns

        run_id = run_meta.get("run_id")
        if not run_id:
            return empty_returns

        manifest_res = backend.inference_run_manifest(run_id)
        if not manifest_res.ok or manifest_res.data is None:
            logger.warning(
                "Diagnostics hydrate: manifest fetch failed for %s: %s",
                run_id, manifest_res.error,
            )
            return empty_returns

        manifest = manifest_res.data.manifest or {}
        validation_block = manifest.get("validation") or {}
        decisions: List[Dict[str, Any]] = list(
            validation_block.get("cluster_decisions") or []
        )
        errors: List[str]   = list(validation_block.get("errors")   or [])
        warnings: List[str] = list(validation_block.get("warnings") or [])

        affected   = sum(1 for d in decisions if d.get("is_affected"))
        unaffected = sum(1 for d in decisions if not d.get("is_affected"))

        return (
            _diag_run_info_tiles(
                run_id   = str(run_id),
                manifest = manifest,
                run_meta = run_meta,
            ),
            _diag_routing_table(decisions),
            build_coverage_donut(affected, unaffected),
            _diag_validation_card_body(errors, warnings),
        )


# ═════════════════════════════════════════════════════════════════════
# 9. hydrate_sensitivity — run_meta_store → heatmap + tornado
#    (Phase 1a.1)
# ═════════════════════════════════════════════════════════════════════

def _register_hydrate_sensitivity(app: "Dash", backend: "RadeBackend") -> None:
    """Hydrate the Sensitivity sub-tab on every completed run.

    Two outputs, both derived from the per-cluster summary DataFrame:

    * **Heatmap** — cluster × scenario signed PnL grid via
      :func:`build_sensitivity_heatmap`.
    * **Tornado** — per-cluster σ across scenarios via
      :func:`build_sensitivity_tornado`.

    Fires on ``run_meta_store`` writes, same trigger as
    :func:`_register_hydrate_diagnostics`.  The clusters DataFrame is
    cached at the backend layer (``bind_inference_run_clusters_df``)
    so the duplicated fetch with the main hydrate callback is a fast
    in-memory hit, not a duplicated API roundtrip.

    Pre-run state returns the empty figures so the layout-baked
    placeholders are replaced cleanly even when the user navigates
    away and back to the Inference page.
    """

    @app.callback(
        Output(INFERENCE_IDS["sensitivity_heatmap"], "figure"),
        Output(INFERENCE_IDS["sensitivity_tornado"], "figure"),
        Input(INFERENCE_IDS["run_meta_store"], "data"),
        prevent_initial_call=False,
    )
    def _hydrate_sensitivity(
        run_meta: Optional[Dict[str, Any]],
    ) -> Tuple[go.Figure, go.Figure]:
        empty_returns: Tuple[go.Figure, go.Figure] = (
            empty_sensitivity_heatmap(),
            empty_sensitivity_tornado(),
        )

        if not run_meta or run_meta.get("status") != "complete":
            return empty_returns

        run_id = run_meta.get("run_id")
        if not run_id:
            return empty_returns

        clusters_res = backend.inference_clusters_df(run_id)
        if not clusters_res.ok or clusters_res.data is None:
            logger.warning(
                "Sensitivity hydrate: clusters_df fetch failed for %s: %s",
                run_id, clusters_res.error,
            )
            return empty_returns

        clusters_df = clusters_res.data
        if clusters_df is None or clusters_df.empty:
            return empty_returns

        return (
            build_sensitivity_heatmap(clusters_df),
            build_sensitivity_tornado(clusters_df),
        )


# ─────────────────────────────────────────────────────────────────────
# Diagnostics — placeholder / render helpers
# ─────────────────────────────────────────────────────────────────────

def _diag_mini_tile(label: str, value: Any, *, mono: bool = False) -> Any:
    """One label/value mini-tile (run-info strip).

    ``mono`` toggles the JetBrains-Mono class on the value so IDs and
    versions read alignedly with the rest of the dashboard.
    """
    value_cls = "rade-stress-mini-value"
    if mono:
        value_cls = f"{value_cls} font-mono"
    return html.Div(
        className="rade-stress-mini-kpi",
        children=[
            html.Div(label.upper(), className="rade-stress-mini-label"),
            html.Div(value,         className=value_cls),
        ],
    )


def _diag_run_info_placeholder() -> List[Any]:
    """Six em-dash tiles for the pre-run state."""
    return [
        _diag_mini_tile(label, _PLACEHOLDER, mono=True)
        for label in (
            "Run ID", "Ensemble", "Status",
            "Generated at", "Input mode", "Latency",
        )
    ]


def _diag_run_info_tiles(
    *,
    run_id:   str,
    manifest: Dict[str, Any],
    run_meta: Dict[str, Any],
) -> List[Any]:
    """Six populated tiles for the run-info strip."""
    status = str(
        run_meta.get("status")
        or manifest.get("status")
        or _PLACEHOLDER
    )
    return [
        _diag_mini_tile("Run ID", run_id, mono=True),
        _diag_mini_tile(
            "Ensemble",
            str(manifest.get("ensemble_version") or _PLACEHOLDER),
            mono=True,
        ),
        _diag_mini_tile("Status", _diag_status_pill(status)),
        _diag_mini_tile(
            "Generated at",
            str(manifest.get("generated_at") or _PLACEHOLDER),
            mono=True,
        ),
        _diag_mini_tile(
            "Input mode",
            str(manifest.get("input_mode") or _PLACEHOLDER),
        ),
        _diag_mini_tile(
            "Latency",
            _diag_format_latency(manifest.get("latency_seconds")),
            mono=True,
        ),
    ]


def _diag_status_pill(status: str) -> Any:
    """Coloured pill for the run status — green=complete, rose=failed."""
    is_complete = status.lower() == "complete"
    fill_cls = (
        "bg-emerald-500/15 text-emerald-300 border border-emerald-500/40"
        if is_complete
        else "bg-rose-500/15 text-rose-300 border border-rose-500/40"
    )
    return html.Span(
        status.capitalize(),
        className=(
            "inline-block px-2 py-0.5 rounded-md text-xs " + fill_cls
        ),
    )


def _diag_format_latency(value: Any) -> str:
    """Format ``latency_seconds`` → ``"12s"`` / ``"—"``."""
    if value is None:
        return _PLACEHOLDER
    try:
        return f"{float(value):.0f}s"
    except (TypeError, ValueError):
        return _PLACEHOLDER


def _diag_routing_placeholder() -> Any:
    """Single-line placeholder shown until a run completes."""
    return html.Div(
        "Awaiting run — routing matrix will appear here.",
        className="text-xs text-slate-500 italic px-2 py-3",
    )


def _diag_routing_table(decisions: List[Dict[str, Any]]) -> Any:
    """HTML table of cluster routing decisions.

    Plain ``html.Table`` (no AG Grid) because the matrix is small,
    read-only, and the styling matches our slate/violet palette
    closer than the AG Grid Quartz theme.  Empty rows render a
    centred italic note so the user can tell "no data" from
    "data not yet fetched".
    """
    if not decisions:
        return _diag_routing_placeholder()

    header = html.Thead(html.Tr(className="rade-diag-header", children=[
        html.Th("Cluster",          className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Path",             className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Intersecting RFs", className="text-left  px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Missing labels",   className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
        html.Th("Targets",          className="text-right px-2 py-1.5 text-xs font-semibold text-slate-300"),
    ]))

    body_rows: List[Any] = []
    for d in decisions:
        cid              = str(d.get("cluster_id") or "")
        is_affected      = bool(d.get("is_affected", False))
        intersecting     = list(d.get("intersecting_risk_factors") or [])
        missing          = list(d.get("missing_scenario_labels")   or [])
        # ``ClusterRoutingDecision`` exposes both elementary and target
        # counts; surface targets since that is the cluster's effective
        # output universe (and the matching field shown in run-summary
        # cards elsewhere in the page).
        n_target_trades  = int(d.get("n_target_trades") or 0)

        body_rows.append(html.Tr(
            className="rade-diag-row border-t border-slate-800",
            children=[
                html.Td(cid,
                    className="rade-grid-mono px-2 py-1.5 text-xs text-violet-300",
                ),
                html.Td(_diag_path_pill(is_affected),
                    className="px-2 py-1.5",
                ),
                html.Td(_diag_format_rf_list(intersecting),
                    className="px-2 py-1.5 text-xs text-slate-300",
                ),
                html.Td(_diag_format_missing_count(len(missing)),
                    className="px-2 py-1.5 text-xs text-right",
                ),
                html.Td(f"{n_target_trades:,}",
                    className=(
                        "rade-grid-mono px-2 py-1.5 text-xs "
                        "text-slate-300 text-right"
                    ),
                ),
            ],
        ))

    return html.Table(
        className="w-full text-xs",
        children=[header, html.Tbody(body_rows)],
    )


def _diag_path_pill(is_affected: bool) -> Any:
    """Coloured pill for the routing path — violet=affected, slate=cheap."""
    if is_affected:
        return html.Span(
            "Affected",
            className=(
                "inline-block px-2 py-0.5 rounded-md text-xs "
                "bg-violet-500/15 text-violet-300 border border-violet-500/40"
            ),
        )
    return html.Span(
        "Cheap path",
        className=(
            "inline-block px-2 py-0.5 rounded-md text-xs "
            "bg-slate-700/40 text-slate-400 border border-slate-700"
        ),
    )


def _diag_format_rf_list(rfs: List[str]) -> Any:
    """Truncate long RF lists to ``first_3 (+N more)`` for readability."""
    if not rfs:
        return html.Span(_PLACEHOLDER, className="text-slate-600")
    if len(rfs) <= 3:
        return ", ".join(rfs)
    return f"{', '.join(rfs[:3])} (+{len(rfs) - 3} more)"


def _diag_format_missing_count(n: int) -> Any:
    """Show 0 as a quiet slate count; non-zero as a rose-tinted alert."""
    if n == 0:
        return html.Span("0", className="rade-grid-mono text-slate-500")
    return html.Span(
        str(n),
        className=(
            "inline-block px-2 py-0.5 rounded-md rade-grid-mono "
            "bg-rose-500/15 text-rose-300 border border-rose-500/40"
        ),
    )


def _diag_validation_placeholder() -> List[Any]:
    """Single-line placeholder shown until a run completes."""
    return [
        html.Div("Validation report",
                 className="text-xs font-semibold text-slate-200"),
        html.Div(
            "Awaiting run — errors / warnings will appear here once "
            "the run completes.",
            className="text-xs text-slate-500 italic",
        ),
    ]


def _diag_validation_card_body(
    errors:   List[str],
    warnings: List[str],
) -> List[Any]:
    """Render the validation card.

    Two states:

    * **No issues** — single green badge ("Validation passed.")
    * **Errors / warnings present** — summary line + bulleted lists,
      rose for errors and amber for warnings.
    """
    if not errors and not warnings:
        return [
            html.Div(
                className="flex items-center gap-2",
                children=[
                    DashIconify(icon="tabler:circle-check",
                                width=18, color="#34d399"),
                    html.Div(
                        "Validation passed — no errors, no warnings.",
                        className="text-xs text-emerald-300",
                    ),
                ],
            ),
        ]

    summary_parts: List[str] = []
    if errors:
        summary_parts.append(
            f"{len(errors)} error{'s' if len(errors) != 1 else ''}"
        )
    if warnings:
        summary_parts.append(
            f"{len(warnings)} warning{'s' if len(warnings) != 1 else ''}"
        )

    children: List[Any] = [
        html.Div("Validation report",
                 className="text-xs font-semibold text-slate-200"),
        html.Div(" · ".join(summary_parts),
                 className="text-xs text-slate-400"),
    ]
    if errors:
        children.append(html.Div(
            "Errors:",
            className="text-xs font-semibold text-rose-300 mt-2",
        ))
        for err in errors:
            children.append(html.Div(
                f"• {err}",
                className="text-xs text-rose-300",
            ))
    if warnings:
        children.append(html.Div(
            "Warnings:",
            className="text-xs font-semibold text-amber-300 mt-2",
        ))
        for warn in warnings:
            children.append(html.Div(
                f"• {warn}",
                className="text-xs text-amber-300",
            ))
    return children


# ═════════════════════════════════════════════════════════════════════
# 9. on_row_select — AG Grid row click → selected_scenario_store
# ═════════════════════════════════════════════════════════════════════

def _register_on_row_select(app: "Dash") -> None:
    """Capture the active scenario when the user clicks a grid row.

    Stage 12 only writes the store; chart-level filtering keys off
    this in Stage 13 once the figure builders are real.
    """

    @app.callback(
        Output(INFERENCE_IDS["selected_scenario_store"], "data"),
        Input(INFERENCE_IDS["scenario_results_grid"],    "selectedRows"),
        prevent_initial_call=True,
    )
    def _on_row_select(
        selected: Optional[List[Dict[str, Any]]],
    ) -> Optional[str]:
        if not selected:
            return None
        # Single-row selection (``rowSelection='single'`` per layout)
        # — first entry is the only one we care about.
        row = selected[0]
        return row.get("scenario_id")


__all__ = ["register"]
