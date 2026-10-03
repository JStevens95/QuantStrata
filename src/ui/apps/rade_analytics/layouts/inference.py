"""Inference Console page layout — **scenario ingestion** path (V2).

This iteration upgrades the original "scenario folder ingest" mock with
two product-driven additions:

1.  **Activity log** in the INPUT column — a streaming, status-coded
    feed that narrates *every* lifecycle event (file loaded, scenarios
    ingested into a risk factor, model loaded, forward pass, predictions
    ready, …) with green-tick / red-cross / pulsing-circle icons.

2.  **Two new analytics tabs** in the RESULTS column — *Risk
    attribution* (P&L breakdown by cluster · risk factor · trade type)
    and *Stress & tails* (VaR, CVaR, worst-N).  These sit alongside the
    existing **Charts**, **Sensitivity** and **Diagnostics** tabs.

Row map
-------

* **Row 0** — ``dcc.Store`` mount tripwire (Page Contract §3 Rule L4)
  *plus* four data Stores driving the page state machine: activity
  log, ingest meta, run meta, selected scenario.

* **Row 1** — Title + subtitle.  Ensemble **version** and **train /
  val / test** split remain in the chrome topbar (:data:`TOPBAR_IDS`)
  — not duplicated here — so this page stays aligned with the rest of
  Rade.

* **Row 2** — Scenario folder sourcing bar:

    * ``dmc.TextInput`` — paste a server-accessible folder path.

    * ``dcc.Upload`` — ``multiple=True``, wraps a *Browse files* button
      so analysts can pick many scenario files at once.  True
      **folder** selection (``webkitdirectory``) is not exposed by
      ``dcc.Upload`` in Dash 3.4 — use the pasted path for recursive
      directory ingest on the server, or extend with a small
      clientside bundle later.

    * *Upload scenarios* ``dmc.Button`` — exposes ``loading=`` /
      ``loaderProps`` at ``False``; the hydrate callback toggles
      ``loading``, then swaps the ingest status slot for an emerald
      tick when ingest completes.

    * Ingest feedback slot ``#inference-ingest-status`` — empty shell
      the callback swaps to ``tabler:circle-check``, error icon, etc.

* **Row 3** — Two-column **input** workspace, equal-height via
  CSS-grid ``items-stretch`` + ``h-full`` cards:

    * **Left · Scenario bundle / manifest card** — scrollable
      manifest preview of the ingested bundle (filenames, scenario
      counts, risk-factor coverage, validity).  Footer holds
      **Validate only** and **Run** (solid violet) actions.  The
      Run button is **disabled by default** and is unlocked only
      after :func:`callbacks.inference_cb._on_validate` reports a
      successful validation — this is what keeps users from
      dispatching a run on an un-validated bundle.

    * **Right · Activity log card** — scrollable feed bound to
      :data:`INFERENCE_IDS["activity_log_store"]`.  Each entry is
      a dict (see *Callback contract* below) and the renderer
      emits one ``rade-activity-row`` per entry.

* **Row 4** — Full-width **results** panel: four KPI tiles
  (sparkline slots via ``sparkline_id``); ``dmc.Tabs``
  (**Charts · Sensitivity · Risk attribution · Stress & tails ·
  Diagnostics**).

      * **Charts** holds the segmented control (Distribution /
        Timeseries / *More · TBD*) and the main ``dcc.Graph``
        (:class:`~components.chart_container.ChartContainer`).

      * **Risk attribution** holds a *breakdown* segmented control
        (Cluster / Risk factor / Trade type) and an attribution chart
        (bar / treemap depending on the breakdown).

      * **Stress & tails** holds a 3-mini-KPI strip (VaR · CVaR ·
        Worst loss) and a tail-view chart (percentile fan / worst-N
        sparkline / tail histogram).

  Below the tabs sit *Save run* / *Publish* / *Export*, then the
  ``rade-inference-results-grid`` AG Grid (single-row select) for
  per-scenario aggregates.  Selecting a row fills
  :data:`INFERENCE_IDS["selected_scenario_store"]`, which the
  Stage-2 chart callbacks key off to filter every chart above.

Callback contract (Stage 2)
---------------------------

The page is intentionally **stateful via Stores** so callbacks can
remain pure functions.  Use these as the contract:

* :data:`INFERENCE_IDS["activity_log_store"]` — ``List[ActivityEntry]``.
  Append-only.  Each entry::

      {
          "id":     "<uuid>",          # stable React key
          "stage":  "ingest|validate|inference",
          "phase":  "File loaded",     # human readable
          "target": "scenario_001.json",  # optional
          "status": "ok|fail|running|pending",
          "ts":     "2028-04-01T12:30:01Z",
          "detail": "Optional error / extra detail string",
      }

  See :func:`render_activity_entries` — exposed publicly so
  callbacks can rebuild the rendered list cheaply each time the
  store mutates.

* :data:`INFERENCE_IDS["ingest_meta_store"]` — ``IngestMeta | None``::

      {
          "source":     "/data/.../sim_2028q3" | "uploaded",
          "files":      [
              {"name": "...", "scenarios": 100, "risk_factors": [...],
               "valid": true, "errors": []},
              ...
          ],
          "started_ts": "...",
          "completed_ts": "...",
          "n_files":    3,
          "n_scenarios": 300,
      }

* :data:`INFERENCE_IDS["run_meta_store"]` — ``RunMeta | None`` —
  populated when the ensemble inference run completes::

      {
          "run_id":      "<uuid>",
          "started_ts":  "...",
          "completed_ts": "...",
          "elapsed_ms":  1234,
          "n_clusters":  4,
          "n_trades":    120,
          "kpi": {"scenarios": 300, "clusters": 4,
                  "avg_inference_ms": 4.2, "portfolio_pnl": 12345.6},
      }

* :data:`INFERENCE_IDS["selected_scenario_store"]` — ``str | None``
  scenario id; row click on the results grid sets this; chart
  callbacks observe it and slice the underlying frame.

Beyond V2, roadmap analytics (add as additional tabs / accordions):

    * Cluster routing matrix — ensemble coverage vs fallback buckets.

    * Train-baseline Δ overlay — run distribution vs frozen eval split.

    * Per-member latency waterfall — exposes which cluster dominates
      wall-clock during batch runs.

    * Comparison tab — scenario-A vs scenario-B side-by-side.

Page Contract anchors
---------------------

* §3 Rule L3 — ids only through :data:`INFERENCE_IDS`.
* §3 Rule L4 — ``mount_signal`` Store + ``rade-page`` root class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

import dash_mantine_components as dmc
from dash import dcc, html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard
from ..figures.inference_charts import (
    empty_coverage_donut,
    empty_pnl_distribution,
    empty_risk_attribution,
    empty_sensitivity_heatmap,
    empty_sensitivity_tornado,
    empty_stress_tails,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────


INFERENCE_IDS: Dict[str, str] = {
    "root":                       "inference-root",
    "mount_signal":               "inference-mount-signal",
    "subtitle":                   "inference-subtitle",

    # Row 0 — page-level data Stores driving the state machine.
    "activity_log_store":         "inference-activity-log-store",
    "ingest_meta_store":          "inference-ingest-meta-store",
    "run_meta_store":             "inference-run-meta-store",
    "selected_scenario_store":    "inference-selected-scenario-store",

    # Row 0 — polling primitives (Stage 12 / 14).  ``polling_store``
    # carries the live run cursor + run_id + armed flag so the poll
    # callback can advance ``/events`` page-by-page without flicker;
    # ``poll_interval`` is disabled at mount and flipped on by the
    # ``on_run`` callback the moment a run is dispatched.
    "polling_store":              "inference-polling-store",
    "poll_interval":              "inference-poll-interval",

    # Row 2 — scenario folder ingestion.
    "scenario_folder_path":       "inference-scenario-folder-path",
    "scenario_folder_upload":     "inference-scenario-folder-upload",
    "upload_scenarios_btn":       "inference-upload-scenarios-btn",
    "ingest_status":              "inference-ingest-status",

    # Row 3 · left — INPUT column.
    "activity_log_container":     "inference-activity-log-container",
    "manifest_preview_container": "inference-manifest-preview-container",
    "validate_only_btn":          "inference-validate-only-btn",
    "run_btn":                    "inference-run-btn",

    # Row 3 · right — KPI strip.
    "kpi_scenarios":              "inference-kpi-scenarios",
    "kpi_scenarios_value":        "inference-kpi-scenarios-value",
    "kpi_scenarios_spark":        "inference-kpi-scenarios-spark",
    "kpi_clusters":               "inference-kpi-clusters",
    "kpi_clusters_value":         "inference-kpi-clusters-value",
    "kpi_clusters_spark":         "inference-kpi-clusters-spark",
    "kpi_latency":                "inference-kpi-latency",
    "kpi_latency_value":          "inference-kpi-latency-value",
    "kpi_latency_spark":          "inference-kpi-latency-spark",
    "kpi_portfolio":              "inference-kpi-portfolio",
    "kpi_portfolio_value":        "inference-kpi-portfolio-value",
    "kpi_portfolio_spark":        "inference-kpi-portfolio-spark",

    # Row 3 · right — analytics tabs (5 tabs).
    "analytics_tabs":             "inference-analytics-tabs",

    # Charts tab — segmented mode + main graph.
    "chart_view_mode":            "inference-chart-view-mode",
    "chart_main":                 "inference-chart-main",

    # Risk-attribution tab — breakdown axis + chart.
    "risk_attribution_breakdown": "inference-risk-attribution-breakdown",
    "risk_attribution_chart":     "inference-risk-attribution-chart",

    # Stress & tails tab — mini KPI strip + chart-mode + chart.
    "stress_tails_mode":          "inference-stress-tails-mode",
    "stress_tails_chart":         "inference-stress-tails-chart",
    "stress_kpi_var":             "inference-stress-kpi-var",
    "stress_kpi_cvar":            "inference-stress-kpi-cvar",
    "stress_kpi_worst":           "inference-stress-kpi-worst",

    # Diagnostics tab (Phase 0.3) — run-info strip + routing matrix +
    # coverage donut + validation report.  All four containers are
    # hydrated from a single manifest fetch in
    # ``_register_hydrate_diagnostics``.
    "diagnostics_run_info":         "inference-diagnostics-run-info",
    "diagnostics_routing_table":    "inference-diagnostics-routing-table",
    "diagnostics_coverage_donut":   "inference-diagnostics-coverage-donut",
    "diagnostics_validation_card":  "inference-diagnostics-validation-card",

    # Sensitivity tab (Phase 1a.1) — cluster × scenario heatmap +
    # per-cluster volatility tornado.  Both hydrated from the
    # cluster-summary DataFrame in
    # ``_register_hydrate_sensitivity``.
    "sensitivity_heatmap":          "inference-sensitivity-heatmap",
    "sensitivity_tornado":          "inference-sensitivity-tornado",

    # Row 3 · right — run footer actions.
    "save_run_as":                "inference-save-run-as",
    "publish_btn":                "inference-publish-btn",
    "export_json_btn":            "inference-export-json-btn",
    "export_csv_btn":             "inference-export-csv-btn",

    # Row 3 · right — results grid (row click → filters charts Stage 2).
    "scenario_results_grid":      "inference-scenario-results-grid",
}


_PLACEHOLDER = "—"


_V1_SUBTITLE = (
    "Price the book under freshly ingested scenarios — preview "
    "(no inference executor connected)"
)


_FOOTER_CAPTION = (
    "Runs hashed + persisted under inference_runs/ · audit hooks from "
    "Governance (Stage 2)"
)


# ─────────────────────────────────────────────────────────────────────
# Activity log — public render helper used by both layout + callbacks.
# ─────────────────────────────────────────────────────────────────────


_STATUS_ICON = {
    "ok":      ("tabler:circle-check",   "rade-activity-icon--ok"),
    "fail":    ("tabler:circle-x",       "rade-activity-icon--fail"),
    "running": ("tabler:loader-2",       "rade-activity-icon--running"),
    "pending": ("tabler:circle-dashed",  "rade-activity-icon--pending"),
}


_STAGE_LABEL = {
    "ingest":    "Ingest",
    "validate":  "Validate",
    "inference": "Inference",
}


def _activity_row(entry: Dict[str, Any]) -> html.Div:
    """Render one activity feed entry into a ``rade-activity-row``."""
    status = str(entry.get("status", "pending"))
    icon_name, icon_class = _STATUS_ICON.get(status, _STATUS_ICON["pending"])
    stage = str(entry.get("stage", "ingest"))
    phase = str(entry.get("phase", "—"))
    target = entry.get("target")
    detail = entry.get("detail")
    ts = str(entry.get("ts", ""))

    body_children: List[Any] = [
        html.Span(_STAGE_LABEL.get(stage, stage.title()),
                  className="rade-activity-stage"),
        html.Span(phase, className="rade-activity-phase"),
    ]
    if target:
        body_children.append(
            html.Span(target, className="rade-activity-target font-mono")
        )
    if detail:
        body_children.append(
            html.Div(detail, className="rade-activity-detail")
        )

    return html.Div(
        className="rade-activity-row",
        children=[
            html.Div(
                className=f"rade-activity-icon {icon_class}",
                children=DashIconify(icon=icon_name, width=16),
            ),
            html.Div(className="rade-activity-body", children=body_children),
            html.Span(ts, className="rade-activity-ts font-mono"),
        ],
    )


def render_activity_entries(
    entries: Optional[Sequence[Dict[str, Any]]],
) -> List[Any]:
    """Render an activity log store payload into row children.

    Intentionally exposed for Stage-2 callbacks: pass the current
    contents of :data:`INFERENCE_IDS["activity_log_store"]` and return
    the children of :data:`INFERENCE_IDS["activity_log_container"]`.
    Callbacks should append entries to the store, then call this with
    the *full* list — the diff is cheap enough at typical bundle sizes
    (<200 entries per run).
    """
    if not entries:
        return [
            html.Div(
                className="rade-activity-empty",
                children=[
                    DashIconify(
                        icon="tabler:wave-square",
                        width=18,
                        className="text-slate-600",
                    ),
                    html.Div(
                        "Activity feed will populate as scenarios are "
                        "uploaded, ingested, validated and priced.",
                        className="text-xs text-slate-500 leading-snug",
                    ),
                ],
            ),
        ]
    return [_activity_row(e) for e in entries]


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Scenario ingestion bar
# ─────────────────────────────────────────────────────────────────────


def _scenario_ingestion_bar() -> html.Div:
    path_field = dmc.TextInput(
        id=INFERENCE_IDS["scenario_folder_path"],
        label="Scenario folder path",
        description="Filesystem path reachable by the app server",
        placeholder="/data/market/scenarios/sim_2028q3",
        size="sm",
        radius="md",
        classNames={"input": "font-mono text-xs"},
        style={"flex": "1 1 280px", "minWidth": "260px"},
    )

    browse = dcc.Upload(
        id=INFERENCE_IDS["scenario_folder_upload"],
        multiple=True,
        style={
            "display":        "inline-block",
            "border":         "none",
            "padding":        0,
            "margin":         0,
            "background":     "transparent",
            "cursor":         "pointer",
            "verticalAlign":  "middle",
        },
        children=dmc.Button(
            children="Browse files",
            variant="default",
            size="sm",
            radius="md",
            leftSection=DashIconify(icon="tabler:folder-open", width=14),
        ),
    )

    upload_trigger = dmc.Button(
        id=INFERENCE_IDS["upload_scenarios_btn"],
        children="Upload scenarios",
        variant="filled",
        color="violet",
        size="sm",
        radius="md",
        leftSection=DashIconify(icon="tabler:cloud-upload", width=16),
        loading=False,
        loaderProps={"type": "oval", "size": "xs"},
        n_clicks=0,
    )

    ingest_status = html.Div(
        id=INFERENCE_IDS["ingest_status"],
        title="Shows a green tick once scenarios are ingested",
        className=(
            "flex items-center justify-center w-10 h-10 "
            "rounded-md border border-slate-700 bg-slate-950/40"
        ),
        children=[],
    )

    return html.Div(
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div(
                "Scenario source",
                className="text-sm font-semibold text-slate-200",
            ),
            html.Div(
                # ``flex-wrap`` + ``gap-y-4`` so the path field stretches
                # horizontally on wide viewports but the trio of buttons
                # drops below it cleanly on narrow ones.
                className="flex flex-row flex-wrap items-end gap-3 gap-y-4",
                children=[
                    path_field,
                    browse,
                    upload_trigger,
                    ingest_status,
                ],
            ),
            html.Div(
                "Browser multi-file pick gathers individual files; paste a "
                "folder path above for full-tree ingest on the server.",
                className="text-xs text-slate-500 leading-snug max-w-2xl",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 · left — INPUT column (Activity log + Manifest)
# ─────────────────────────────────────────────────────────────────────


def _activity_log_card() -> html.Div:
    # ``h-full`` on the card + ``flex-1`` on the scroll-area makes the
    # card stretch to the grid row's height (set by the taller of
    # manifest / activity-log via CSS-grid ``items-stretch``).  The
    # inner ``overflow-y-auto`` keeps the entry list scrollable rather
    # than overflowing the card.
    return html.Div(
        className="rade-card flex flex-col gap-2 min-w-0 h-full",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Div(
                        "Activity log",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Live — append-only",
                        className=(
                            "text-[11px] uppercase tracking-wide "
                            "text-slate-500"
                        ),
                    ),
                ],
            ),
            html.Div(
                id=INFERENCE_IDS["activity_log_container"],
                className=(
                    "rade-activity-log flex-1 overflow-y-auto rounded-md "
                    "border border-slate-800 bg-slate-950/40 p-3"
                ),
                style={"minHeight": "220px"},
                children=render_activity_entries(None),
            ),
        ],
    )


def _manifest_card() -> html.Div:
    # ``flex-1`` on the manifest preview lets the card stretch to the
    # row's height (driven by the taller of manifest / activity-log)
    # while keeping the inner content scrollable.
    manifest_box = html.Div(
        id=INFERENCE_IDS["manifest_preview_container"],
        className=(
            "flex-1 overflow-y-auto rounded-md border border-slate-800 "
            "bg-slate-950/40 p-3 text-xs text-slate-400 leading-relaxed"
        ),
        style={"minHeight": "180px"},
        children=[
            html.Div(
                "No scenarios ingested yet.",
                className="text-slate-500",
            ),
            html.Div(
                className="flex flex-col gap-1 mt-3 text-slate-500",
                children=[
                    html.Div("• Upload populates filenames, horizons, shocks."),
                    html.Div("• Validate only checks manifest + shock schema."),
                    html.Div("• Run executes ensemble inference on the bundle."),
                ],
            ),
        ],
    )

    actions = dmc.Group(
        gap="sm",
        grow=True,
        className="w-full mt-3",
        children=[
            dmc.Button(
                id=INFERENCE_IDS["validate_only_btn"],
                children="Validate only",
                variant="default",
                size="sm",
                radius="md",
                flex=1,
                leftSection=DashIconify(icon="tabler:checks", width=16),
                n_clicks=0,
            ),
            # Run is the page's primary action — solid violet (rather
            # than a violet→cyan gradient that read as cyan in practice)
            # so it visually commands the user's attention.  Locked by
            # default; ``_on_validate`` flips ``disabled=False`` only
            # after the validate call returns ``is_valid=True``.
            dmc.Button(
                id=INFERENCE_IDS["run_btn"],
                children="Run",
                color="violet",
                variant="filled",
                size="sm",
                radius="md",
                flex=1,
                leftSection=DashIconify(icon="tabler:player-play", width=16),
                disabled=True,
                n_clicks=0,
            ),
        ],
    )

    return html.Div(
        className="rade-card flex flex-col gap-2 min-w-0 h-full",
        children=[
            html.Div(
                className="flex items-center justify-between",
                children=[
                    html.Div(
                        "Scenario bundle",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    html.Div(
                        "Manifest preview",
                        className=(
                            "text-[11px] uppercase tracking-wide "
                            "text-slate-500"
                        ),
                    ),
                ],
            ),
            manifest_box,
            actions,
        ],
    )


def _row_input_workspace() -> html.Div:
    """Row 3 — two-column input workspace.

    Manifest card on the left, activity log on the right, **equal
    height** via CSS-grid ``items-stretch`` + ``h-full`` on each card.
    The ``lg:grid-cols-2`` two-up only kicks in at the ``lg``
    breakpoint; below that the cards stack vertically so the page
    stays readable on narrow viewports.

    Order is *manifest first, activity log second* because the
    primary user action lives in the manifest card (Validate, Run).
    The activity log is read-only confirmation of what those actions
    triggered.
    """
    return html.Div(
        className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-stretch",
        children=[
            html.Div(
                className="flex flex-col min-w-0",
                children=[_manifest_card()],
            ),
            html.Div(
                className="flex flex-col min-w-0",
                children=[_activity_log_card()],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 · right — RESULTS column
# ─────────────────────────────────────────────────────────────────────


def _kpi_row() -> html.Div:
    return html.Div(
        className="grid grid-cols-4 gap-3",
        children=[
            KpiCard(
                label="Scenarios priced",
                value=_PLACEHOLDER,
                icon="tabler:stack-pop",
                card_id=INFERENCE_IDS["kpi_scenarios"],
                value_id=INFERENCE_IDS["kpi_scenarios_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_scenarios_spark"],
            ),
            KpiCard(
                label="Clusters touched",
                value=_PLACEHOLDER,
                icon="tabler:hierarchy",
                card_id=INFERENCE_IDS["kpi_clusters"],
                value_id=INFERENCE_IDS["kpi_clusters_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_clusters_spark"],
            ),
            KpiCard(
                label="Avg inference",
                value=_PLACEHOLDER,
                icon="tabler:gauge",
                card_id=INFERENCE_IDS["kpi_latency"],
                value_id=INFERENCE_IDS["kpi_latency_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_latency_spark"],
            ),
            KpiCard(
                label="Portfolio P&L (est)",
                value=_PLACEHOLDER,
                icon="tabler:trending-up",
                card_id=INFERENCE_IDS["kpi_portfolio"],
                value_id=INFERENCE_IDS["kpi_portfolio_value"],
                sparkline_data=None,
                sparkline_id=INFERENCE_IDS["kpi_portfolio_spark"],
            ),
        ],
    )


def _placeholder_panel(title: str, body: str) -> html.Div:
    # Dashed-border + arbitrary min-height live in inline ``style``
    # because ``border-dashed`` and ``min-h-[200px]`` aren't in the
    # compiled ``rade.css`` utility bundle.
    return html.Div(
        className=(
            "flex flex-col gap-2 justify-center px-6 py-6 rounded-md "
            "border border-slate-800"
        ),
        style={
            "minHeight":   "200px",
            "borderStyle": "dashed",
            "background":  "rgba(15, 23, 42, 0.45)",
        },
        children=[
            html.Div(title, className="text-xs font-semibold text-slate-300"),
            html.Div(body, className="text-xs text-slate-500 max-w-xl"),
        ],
    )


def _charts_tab_body() -> html.Div:
    mode_toggle = dmc.SegmentedControl(
        id=INFERENCE_IDS["chart_view_mode"],
        value="distribution",
        size="xs",
        color="violet",
        radius="md",
        mb="xs",
        data=[
            {"label": "Distribution", "value": "distribution"},
            {"label": "Timeseries",   "value": "timeseries"},
            {"label": "More · TBD",   "value": "overlay"},
        ],
    )

    chart_card = ChartContainer(
        title="Aggregate book response",
        subtitle=(
            "Distribution / timeseries / overlay — callback swaps figure "
            "on ``chart_main`` while honouring segmented state and the "
            "active row in the scenario grid."
        ),
        graph_id=INFERENCE_IDS["chart_main"],
        figure=empty_pnl_distribution(),
        height=300,
        actions=[
            html.Div(
                className="flex flex-wrap justify-end",
                children=[mode_toggle],
            ),
        ],
    )

    return html.Div(className="flex flex-col gap-2", children=[chart_card])


def _sensitivity_tab_body() -> html.Div:
    """Sensitivity sub-tab — cluster × scenario heatmap + volatility tornado.

    Two cards, both hydrated from the cluster-summary DataFrame in
    :func:`callbacks.inference_cb._register_hydrate_sensitivity`:

    1. **Cluster × Scenario heatmap** — colour-coded grid of
       ``sum_pnl_original``.  Vertical streaks reveal scenarios that
       moved many clusters in concert; horizontal streaks reveal
       structurally biased clusters.  Symmetric diverging palette so
       the neutral colour always lands at zero.
    2. **Per-cluster volatility tornado** — clusters ranked by
       ``std(sum_pnl_original)`` across scenarios.  A cluster with
       low σ is "insensitive" (mostly the same PnL across scenarios);
       high σ means "highly sensitive to scenario choice".

    Phase 4 will replace the cluster-level tornado with true RF-level
    elasticities (∂PnL/∂shock per risk factor) once the trade-attribute
    API ships.
    """
    heatmap_card = ChartContainer(
        title="Cluster × Scenario sensitivity",
        subtitle=(
            "Signed P&L for every cluster × scenario combination — "
            "diverging palette pivots at zero, so cool colours = "
            "losses and warm colours = gains regardless of magnitude."
        ),
        graph_id=INFERENCE_IDS["sensitivity_heatmap"],
        figure=empty_sensitivity_heatmap(),
        height=420,
    )

    tornado_card = ChartContainer(
        title="Per-cluster sensitivity (σ across scenarios)",
        subtitle=(
            "Standard deviation of cluster P&L across scenarios — a "
            "cluster-level proxy for *how much* the scenario set moves "
            "the cluster, regardless of direction.  Top-N + Other."
        ),
        graph_id=INFERENCE_IDS["sensitivity_tornado"],
        figure=empty_sensitivity_tornado(),
        height=300,
    )

    return html.Div(
        className="flex flex-col gap-3",
        children=[heatmap_card, tornado_card],
    )


def _risk_attribution_tab_body() -> html.Div:
    """Risk-attribution view — bar/treemap by selected breakdown axis."""
    breakdown = dmc.SegmentedControl(
        id=INFERENCE_IDS["risk_attribution_breakdown"],
        value="cluster",
        size="xs",
        color="violet",
        radius="md",
        mb="xs",
        data=[
            {"label": "By cluster",      "value": "cluster"},
            {"label": "By risk factor",  "value": "risk_factor"},
            {"label": "By trade type",   "value": "trade_type"},
        ],
    )

    chart_card = ChartContainer(
        title="P&L attribution",
        subtitle=(
            "Signed contribution of each bucket to portfolio P&L for the "
            "current scenario set — bar when grouped one-deep, treemap "
            "when nested (Stage 2 swaps ``figure`` shape via callback)."
        ),
        graph_id=INFERENCE_IDS["risk_attribution_chart"],
        figure=empty_risk_attribution(),
        height=300,
        actions=[
            html.Div(
                className="flex flex-wrap justify-end",
                children=[breakdown],
            ),
        ],
    )

    return html.Div(className="flex flex-col gap-2", children=[chart_card])


def _stress_tails_tab_body() -> html.Div:
    """Stress / tail-risk view — mini KPI strip + chart-mode toggle."""
    mini_kpis = html.Div(
        className="grid grid-cols-3 gap-3",
        children=[
            html.Div(
                id=INFERENCE_IDS["stress_kpi_var"],
                className="rade-stress-mini-kpi",
                children=[
                    html.Div("VaR (95%)", className="rade-stress-mini-label"),
                    html.Div(_PLACEHOLDER, className="rade-stress-mini-value"),
                ],
            ),
            html.Div(
                id=INFERENCE_IDS["stress_kpi_cvar"],
                className="rade-stress-mini-kpi",
                children=[
                    html.Div("CVaR (95%)", className="rade-stress-mini-label"),
                    html.Div(_PLACEHOLDER, className="rade-stress-mini-value"),
                ],
            ),
            html.Div(
                id=INFERENCE_IDS["stress_kpi_worst"],
                className="rade-stress-mini-kpi",
                children=[
                    html.Div("Worst loss", className="rade-stress-mini-label"),
                    html.Div(_PLACEHOLDER, className="rade-stress-mini-value"),
                ],
            ),
        ],
    )

    mode_toggle = dmc.SegmentedControl(
        id=INFERENCE_IDS["stress_tails_mode"],
        value="fan",
        size="xs",
        color="violet",
        radius="md",
        mb="xs",
        data=[
            {"label": "Percentile fan", "value": "fan"},
            {"label": "Tail histogram", "value": "tail"},
            {"label": "Worst N",        "value": "worst"},
        ],
    )

    chart_card = ChartContainer(
        title="Tail-risk view",
        subtitle=(
            "Percentile band over scenario index, tail histogram of "
            "scenario P&L, or top-N worst-loss scenarios — callback "
            "swaps figure shape on ``stress_tails_chart``."
        ),
        graph_id=INFERENCE_IDS["stress_tails_chart"],
        figure=empty_stress_tails(),
        height=260,
        actions=[
            html.Div(
                className="flex flex-wrap justify-end",
                children=[mode_toggle],
            ),
        ],
    )

    return html.Div(
        className="flex flex-col gap-3",
        children=[mini_kpis, chart_card],
    )


def _diagnostics_tab_body() -> html.Div:
    """Diagnostics sub-tab — run health, routing matrix, validation report.

    Four containers, all hydrated by a single manifest fetch in
    :func:`callbacks.inference_cb._register_hydrate_diagnostics`:

    1. **Run-info strip** — six mini-tiles surfacing the run-level
       metadata (run_id / ensemble_version / status / generated_at /
       input_mode / latency).
    2. **Routing matrix** — per-cluster routing decision · path ·
       intersecting RFs · missing labels · target count.  Read-only
       table so the user can audit *which* clusters took the cheap
       historical-lookup path and *why*.
    3. **Coverage donut** — affected vs unaffected counts in a single
       glance; orients the eye on whether the run was scenario-driven
       or history-driven before drilling into the matrix.
    4. **Validation report** — surfaces ``validation.errors`` and
       ``validation.warnings`` from the manifest so the user can see
       why the Run button was blocked (errors) or what to caveat
       (warnings).
    """
    run_info = html.Div(
        id=INFERENCE_IDS["diagnostics_run_info"],
        className=(
            "rade-card grid grid-cols-2 md:grid-cols-3 "
            "lg:grid-cols-6 gap-3"
        ),
        children=_diagnostics_run_info_placeholder(),
    )

    routing_table = html.Div(
        className="rade-card flex flex-col gap-2 min-w-0",
        children=[
            html.Div("Routing matrix",
                     className="text-xs font-semibold text-slate-200"),
            html.Div(
                "Per-cluster routing decision · path · intersecting "
                "risk factors · target trade count.",
                className="text-xs text-slate-500",
            ),
            html.Div(
                id=INFERENCE_IDS["diagnostics_routing_table"],
                className="overflow-y-auto min-w-0",
                style={"maxHeight": "320px"},
                children=_diagnostics_routing_placeholder(),
            ),
        ],
    )

    coverage_donut = html.Div(
        className="rade-card flex flex-col gap-2 min-w-0",
        children=[
            html.Div("Routing coverage",
                     className="text-xs font-semibold text-slate-200"),
            html.Div(
                "Affected (re-priced) vs unaffected (cheap-path) "
                "cluster counts.",
                className="text-xs text-slate-500",
            ),
            dcc.Graph(
                id=INFERENCE_IDS["diagnostics_coverage_donut"],
                figure=empty_coverage_donut(),
                config={"displayModeBar": False},
                style={"height": "240px"},
            ),
        ],
    )

    validation_card = html.Div(
        id=INFERENCE_IDS["diagnostics_validation_card"],
        className="rade-card flex flex-col gap-2 min-w-0",
        children=_diagnostics_validation_placeholder(),
    )

    return html.Div(
        className="flex flex-col gap-3",
        children=[
            run_info,
            html.Div(
                className="grid grid-cols-1 lg:grid-cols-3 gap-3",
                children=[
                    html.Div(className="lg:col-span-2 min-w-0",
                             children=[routing_table]),
                    html.Div(className="lg:col-span-1 min-w-0",
                             children=[coverage_donut]),
                ],
            ),
            validation_card,
        ],
    )


def _diagnostics_run_info_placeholder() -> List[Any]:
    """Six em-dash tiles shown until ``run_meta_store`` hydrates."""
    labels = [
        "Run ID", "Ensemble", "Status",
        "Generated at", "Input mode", "Latency",
    ]
    return [
        html.Div(
            className="rade-stress-mini-kpi",
            children=[
                html.Div(label.upper(),
                         className="rade-stress-mini-label"),
                html.Div(_PLACEHOLDER,
                         className="rade-stress-mini-value"),
            ],
        )
        for label in labels
    ]


def _diagnostics_routing_placeholder() -> Any:
    """Single-line placeholder shown until a run completes."""
    return html.Div(
        "Awaiting run — routing matrix will appear here.",
        className="text-xs text-slate-500 italic px-2 py-3",
    )


def _diagnostics_validation_placeholder() -> List[Any]:
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


def _analytics_tabs_block() -> dmc.Tabs:
    charts_tab      = _charts_tab_body()
    sensitivity_tab = _sensitivity_tab_body()
    risk_attr_tab   = _risk_attribution_tab_body()
    stress_tab      = _stress_tails_tab_body()
    diagnostics_tab = _diagnostics_tab_body()

    return dmc.Tabs(
        id=INFERENCE_IDS["analytics_tabs"],
        value="charts",
        color="violet",
        variant="outline",
        radius="md",
        className="w-full inference-analytics-tabs",
        children=[
            dmc.TabsList(
                grow=True,
                children=[
                    dmc.TabsTab(
                        value="charts",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-histogram", width=14,
                        ),
                        children="Charts",
                    ),
                    dmc.TabsTab(
                        value="sensitivity",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-arcs", width=14,
                        ),
                        children="Sensitivity",
                    ),
                    dmc.TabsTab(
                        value="risk_attribution",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-treemap", width=14,
                        ),
                        children="Risk attribution",
                    ),
                    dmc.TabsTab(
                        value="stress_tails",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:chart-area-line", width=14,
                        ),
                        children="Stress & tails",
                    ),
                    dmc.TabsTab(
                        value="diagnostics",
                        flex=1,
                        leftSection=DashIconify(
                            icon="tabler:wave-sine", width=14,
                        ),
                        children="Diagnostics",
                    ),
                ],
            ),
            dmc.TabsPanel(value="charts",           pt="sm", pb=0,
                          children=charts_tab),
            dmc.TabsPanel(value="sensitivity",      pt="sm",
                          children=sensitivity_tab),
            dmc.TabsPanel(value="risk_attribution", pt="sm",
                          children=risk_attr_tab),
            dmc.TabsPanel(value="stress_tails",     pt="sm",
                          children=stress_tab),
            dmc.TabsPanel(value="diagnostics",      pt="sm",
                          children=diagnostics_tab),
        ],
    )


def _scenario_results_footer() -> html.Div:
    note = html.Div(
        "Selecting a scenario / trade row filters charts above — Stage 2",
        className="text-[11px] text-slate-500",
    )

    actions = html.Div(
        className="flex flex-row flex-wrap items-end gap-2",
        children=[
            dmc.TextInput(
                id=INFERENCE_IDS["save_run_as"],
                placeholder="Save run as…",
                size="xs",
                radius="md",
                style={"flex": "2 1 180px"},
            ),
            dmc.Button(
                id=INFERENCE_IDS["publish_btn"],
                children="Publish to Governance",
                color="gray",
                variant="filled",
                size="xs",
                radius="md",
                leftSection=DashIconify(icon="tabler:clipboard-check", width=14),
            ),
            dmc.Button(
                id=INFERENCE_IDS["export_json_btn"],
                children="Export JSON",
                variant="default",
                size="xs",
                radius="md",
                leftSection=DashIconify(icon="tabler:brand-json", width=14),
            ),
            dmc.Button(
                id=INFERENCE_IDS["export_csv_btn"],
                children="Export CSV",
                variant="default",
                size="xs",
                radius="md",
                leftSection=DashIconify(icon="tabler:download", width=14),
            ),
        ],
    )

    return html.Div(
        className="flex flex-col gap-3",
        children=[
            html.Div(
                "Scenario / trade results",
                className="text-sm font-semibold text-slate-200",
            ),
            actions,
            note,
        ],
    )


def _results_ag_grid_defs() -> List[Dict[str, Any]]:
    conf_rules = {
        "rade-pill rade-pill--approved": (
            "params.value === 'High' || params.value === 'high'"
        ),
        "rade-pill rade-pill--pending": (
            "params.value === 'Medium' || params.value === 'medium'"
        ),
        "rade-pill rade-pill--archived": (
            "params.value === 'Low' || params.value === 'low'"
        ),
    }

    fmt_cap = (
        "params.value ? "
        "params.value.charAt(0).toUpperCase() + params.value.slice(1) "
        ": '—'"
    )

    return [
        {
            "field":      "scenario_id",
            "headerName": "Scenario",
            "pinned":     "left",
            "minWidth":   120,
            "cellClass":  "rade-grid-mono",
        },
        {
            "field":      "cluster",
            "headerName": "Cluster",
            "minWidth":   100,
            "cellClass":  "rade-grid-mono",
        },
        {
            "field":      "predicted",
            "headerName": "Predicted",
            "type":       "numericColumn",
            "minWidth":   100,
            "valueFormatter": {
                "function": (
                    "params.value == null ? '—' : "
                    "Number(params.value).toPrecision(5)"
                ),
            },
        },
        {"field": "p95_band", "headerName": "P95 band",
         "type":  "numericColumn", "minWidth": 100},
        {
            "field":          "confidence",
            "headerName":     "Confidence",
            "minWidth":       110,
            "cellClassRules": conf_rules,
            "valueFormatter": {"function": fmt_cap},
        },
    ]


def _results_panel() -> html.Div:
    defs = _results_ag_grid_defs()
    return html.Div(
        # ``rade-card`` already supplies padding, border + radius;
        # don't double-pad with an extra ``p-5`` here.
        className="rade-card flex flex-col gap-4 min-w-0",
        children=[
            html.Div(
                "Results",
                className="text-sm font-semibold text-slate-200",
            ),
            _kpi_row(),
            html.Div(
                children=[_analytics_tabs_block()],
                className="w-full",
            ),
            _scenario_results_footer(),
            AgGridTable(
                grid_id=INFERENCE_IDS["scenario_results_grid"],
                row_data=[],
                column_defs=defs,
                grid_options={
                    "pagination": True,
                    "paginationPageSize": 12,
                    "paginationPageSizeSelector": [10, 25, 50, 100],
                    "rowHeight": 38,
                    "headerHeight": 38,
                    "animateRows": False,
                    "suppressCellFocus": True,
                    "rowSelection": "single",
                    "domLayout": "normal",
                },
                height=280,
                className="rade-inference-results-grid",
            ),
        ],
    )


def _row_results_workspace() -> html.Div:
    """Row 4 — full-width results panel.

    KPIs + analytics tabs + scenario grid all live inside the
    existing :func:`_results_panel` card.  Wrapping it in a single
    full-width column (no grid) gives the results breathing room —
    the previous 3/5 column allocation was visibly squished on wide
    monitors.
    """
    return html.Div(
        className="flex flex-col min-w-0",
        children=[_results_panel()],
    )


def _page_footer_line() -> html.Div:
    return html.Div(
        className="flex justify-end mt-8",
        children=[html.Span(_FOOTER_CAPTION, className="text-xs text-slate-500")],
    )


def _page_stores() -> List[Any]:
    """All ``dcc.Store`` mounts driving the page state machine.

    Kept as a list so ``build_inference`` can splat them at the top of
    the page tree without tracking individual ids.

    The ``polling_store`` + ``poll_interval`` pair powers the live
    activity-log drain documented in :func:`callbacks.inference_cb.on_poll`:
    ``on_run`` flips ``polling_store.data['armed']`` to ``True`` and
    enables ``poll_interval``; each interval tick reads the current
    cursor, calls ``/events?cursor=N`` + ``/status``, advances the
    cursor, and disarms the polling pair when the API reports a
    terminal status.  Decoupling the cursor from the interval lets a
    completed run pause polling without losing its place if the user
    re-arms (e.g. for re-runs).
    """
    return [
        dcc.Store(
            id=INFERENCE_IDS["mount_signal"],
            data=True,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["activity_log_store"],
            data=[],
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["ingest_meta_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["run_meta_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["selected_scenario_store"],
            data=None,
            storage_type="memory",
        ),
        dcc.Store(
            id=INFERENCE_IDS["polling_store"],
            # Initial shape mirrors the on-the-wire contract the
            # poll callback expects.  ``armed=False`` means the
            # interval will short-circuit even if its disabled flag
            # ever drifts out of sync (defence in depth).
            data={"armed": False, "run_id": None, "cursor": 0},
            storage_type="memory",
        ),
        dcc.Interval(
            id=INFERENCE_IDS["poll_interval"],
            # 1 Hz is a comfortable trade-off between perceived
            # liveness (one new line per second when the worker is
            # busy) and server load (the events endpoint is cheap;
            # /status is a single dict read).  The interval stays
            # disabled until ``on_run`` arms it — pages that never
            # dispatch a run pay zero polling cost.
            interval=1000,
            n_intervals=0,
            disabled=True,
        ),
    ]


def build_inference(*, session: Optional["Session"] = None) -> html.Div:
    """Compose the Inference Console tree (V2 — empty layout, no callbacks).

    ``session`` follows Page Contract §2.1 (uniform ``build_*`` signature).
    Chrome seeds version + split via ``TOPBAR_IDS``; no per-page hydration
    reads today — reserved for ingest-to-session persistence.
    """
    del session  # unused until capture callbacks hydrate inference state

    return html.Div(
        id=INFERENCE_IDS["root"],
        className="rade-page rade-inference flex flex-col gap-5",
        children=[
            *_page_stores(),
            html.Div(
                className="flex flex-col gap-6",
                children=[
                    html.Div(
                        className="flex flex-col gap-1",
                        children=[
                            html.Div(
                                className="rade-page-title",
                                children="Inference Console",
                            ),
                            html.Span(
                                _V1_SUBTITLE,
                                id=INFERENCE_IDS["subtitle"],
                                className="text-xs text-slate-500",
                            ),
                            html.Span(
                                "Active ensemble version + split controlled "
                                "from the top bar.",
                                className="text-[11px] text-slate-600",
                            ),
                        ],
                    ),
                    _scenario_ingestion_bar(),
                    _row_input_workspace(),
                    _row_results_workspace(),
                    _page_footer_line(),
                ],
            ),
        ],
    )


__all__ = [
    "INFERENCE_IDS",
    "build_inference",
    "render_activity_entries",
]
