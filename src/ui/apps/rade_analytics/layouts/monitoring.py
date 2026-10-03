"""Monitoring page layout.

Mirrors ``docs/platform_designs/rade_model_monitoring.png`` region-for-
region:

* **Row 0** — invisible mount tripwire (Page Contract §3 Rule L4) so
  the Stage-2 render callback can fire on layout mount without
  racing the DOM swap.
* **Row 1** — Header band: page title + subtitle on the left, four KPI
  chips spanning the row (Baseline-vs-Live KS, PSI, Rolling MAE 7d,
  Alerts Open).
* **Row 2** — Two-column row: *Residual Drift — last 30 days* line
  chart on the left (~2/3 width) and *Feature Drift (PSI)* horizontal
  bar chart on the right (~1/3 width).
* **Row 3** — Two-column row: *Active Alerts* AG Grid (~1/2 width) and
  *Latency Histogram* (~1/2 width).
* **Row 4** — Footer caption: data-source disclosure.

V1 status — Option A (empty layout, no callbacks)
-------------------------------------------------
This module ships the full layout with **empty defaults everywhere**:

* KPI values default to em-dash placeholders.
* Charts mount with the matching ``empty_*`` figure from
  :mod:`figures.monitoring_charts` (each carries an "Awaiting …"
  annotation hinting at what the populated chart will show).
* The Active Alerts grid mounts with ``rowData=[]``.

No callbacks are registered today (see ``callbacks/__init__.py``) so
no API request fires when the page is navigated to.  When the V1
snapshot-mode callbacks land they will overwrite each region via
``Output`` — no layout change required.

Page Contract anchors
---------------------
* §3 Rule L1 — every primitive a callback might target has a stable
  id via :data:`MONITORING_IDS`.
* §3 Rule L3 — no hardcoded id strings outside this dict.
* §3 Rule L4 — :data:`MONITORING_IDS["mount_signal"]` is a Store with
  ``data=True`` so render callbacks can fire after the DOM swap.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from dash import dcc, html

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard
from ..figures.monitoring_charts import (
    empty_feature_drift_psi,
    empty_latency_histogram,
    empty_residual_drift,
)

if TYPE_CHECKING:
    from ..data.session import Session


# ─────────────────────────────────────────────────────────────────────
# Stable id contract — every component a callback might target lives
# here so callbacks never hardcode strings (Page Contract §3 Rule L3).
# ─────────────────────────────────────────────────────────────────────


MONITORING_IDS: Dict[str, str] = {
    "root":                     "monitoring-root",

    # Mount tripwire — Page Contract §3 Rule L4.
    "mount_signal":             "monitoring-mount-signal",

    # Row 1 — Header subtitle (carries the "updated N min ago" suffix
    # once the snapshot callback comes online).
    "subtitle":                 "monitoring-subtitle",

    # Row 1 — KPI chips.  Each card carries a separate value id so
    # callbacks can target the value text without rebuilding chrome.
    "kpi_ks":                   "monitoring-kpi-ks",
    "kpi_ks_value":             "monitoring-kpi-ks-value",
    "kpi_psi":                  "monitoring-kpi-psi",
    "kpi_psi_value":            "monitoring-kpi-psi-value",
    "kpi_rolling_mae":          "monitoring-kpi-rolling-mae",
    "kpi_rolling_mae_value":    "monitoring-kpi-rolling-mae-value",
    "kpi_alerts_open":          "monitoring-kpi-alerts-open",
    "kpi_alerts_open_value":    "monitoring-kpi-alerts-open-value",

    # Row 2 — Drift charts.
    "residual_drift_chart":     "monitoring-residual-drift-chart",
    "feature_drift_chart":      "monitoring-feature-drift-chart",

    # Row 3 — Alerts grid + Latency histogram.
    "alerts_grid":              "monitoring-alerts-grid",
    "latency_histogram_chart":  "monitoring-latency-histogram-chart",
}


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────


# Placeholder used for any KPI value that doesn't have a callback
# overwriting it yet.  Matches the convention used on every other
# Rade page (Overview / Governance / Cluster Deep-Dive).
_PLACEHOLDER = "—"


# Subtitle wording for V1 — honest about the empty state.  The
# snapshot-mode callback (Option B) replaces this with
# ``"Train vs Test drift snapshot — updated <N> min ago"``.
_V1_SUBTITLE = (
    "Drift & residual surveillance — preview "
    "(no live telemetry connected)"
)


# Footer caption — surface the producer-side reality so reviewers
# don't infer anything about live monitoring from a static layout.
_FOOTER_CAPTION = (
    "Monitoring data sourced from production logging pipeline; "
    "live feed wires up in Stage 2."
)


# ─────────────────────────────────────────────────────────────────────
# Row 1 — Header band (title + subtitle + KPI strip)
# ─────────────────────────────────────────────────────────────────────


def _kpi_strip() -> html.Div:
    """Four KPI cards across Row 1.

    Values are em-dashes at build time and overwritten by the
    snapshot render callback once it lands.  Card icons stay static.
    """
    return html.Div(
        className="grid grid-cols-4 gap-4",
        children=[
            KpiCard(
                label="Baseline vs Live KS",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_ks"],
                value_id=MONITORING_IDS["kpi_ks_value"],
                icon="tabler:wave-square",
            ),
            KpiCard(
                label="PSI (Population Stability)",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_psi"],
                value_id=MONITORING_IDS["kpi_psi_value"],
                icon="tabler:chart-histogram",
            ),
            KpiCard(
                label="Rolling MAE (7d)",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_rolling_mae"],
                value_id=MONITORING_IDS["kpi_rolling_mae_value"],
                icon="tabler:trending-down",
            ),
            KpiCard(
                label="Alerts Open",
                value=_PLACEHOLDER,
                card_id=MONITORING_IDS["kpi_alerts_open"],
                value_id=MONITORING_IDS["kpi_alerts_open_value"],
                icon="tabler:bell-ringing",
            ),
        ],
    )


def _row_header() -> html.Div:
    """Row 1 — page title + subtitle on the left, KPI strip below.

    The subtitle's id is part of the contract so the snapshot
    callback can drop a real "updated N min ago" suffix into it
    without rebuilding chrome.
    """
    return html.Div(
        className="flex flex-col gap-3",
        children=[
            html.Div(
                className=(
                    "flex items-end justify-between gap-4 flex-wrap"
                ),
                children=[
                    html.Div(
                        className="flex flex-col gap-1",
                        children=[
                            html.Div(
                                "Production Monitoring",
                                className="rade-page-title",
                            ),
                            html.Div(
                                _V1_SUBTITLE,
                                id=MONITORING_IDS["subtitle"],
                                className="text-xs text-slate-500",
                            ),
                        ],
                    ),
                ],
            ),
            _kpi_strip(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Drift charts (residual line + feature PSI bar)
# ─────────────────────────────────────────────────────────────────────


def _row_drift_charts() -> html.Div:
    """Row 2 — residual drift line (~2/3) + feature drift PSI bar (~1/3).

    Layout uses a 3-column CSS grid: residual drift spans 2 cols,
    feature drift takes the remaining 1.  Mirrors the
    Overview / Cluster-Deep-Dive split conventions so visual rhythm
    stays consistent across pages.
    """
    return html.Div(
        className="grid grid-cols-3 gap-4 items-stretch",
        children=[
            html.Div(
                className="col-span-2",
                children=[
                    ChartContainer(
                        title="Residual Drift — last 30 days",
                        subtitle=(
                            "Median residual + P5–P95 envelope · "
                            "anomaly markers"
                        ),
                        graph_id=MONITORING_IDS["residual_drift_chart"],
                        figure=empty_residual_drift(),
                        height=320,
                    ),
                ],
            ),
            html.Div(
                className="col-span-1",
                children=[
                    ChartContainer(
                        title="Feature Drift (PSI)",
                        subtitle="Top 10 features · severity-coloured",
                        graph_id=MONITORING_IDS["feature_drift_chart"],
                        figure=empty_feature_drift_psi(),
                        height=320,
                    ),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Active alerts + Latency histogram
# ─────────────────────────────────────────────────────────────────────


# Column defs for the Active Alerts grid.  Severity column is wired
# to ``cellClassRules`` against the existing ``rade-pill`` palette
# (rose / amber / slate via reused ``rejected`` / ``pending`` /
# ``archived`` variants) so when the alerts callback comes online
# styling lights up automatically — no second CSS pass.
_ALERTS_COLUMN_DEFS: List[Dict[str, Any]] = [
    {
        "field":      "alert_id",
        "headerName": "Alert ID",
        "minWidth":   140,
        "pinned":     "left",
        "cellClass":  "rade-grid-mono",
    },
    {
        "field":      "cluster",
        "headerName": "Cluster",
        "minWidth":   110,
    },
    {
        "field":      "type",
        "headerName": "Type",
        "minWidth":   120,
    },
    {
        "field":      "severity",
        "headerName": "Severity",
        "minWidth":   120,
        "cellClassRules": {
            # Until severity-specific pills land, reuse the existing
            # governance palette: critical → rose, warn → amber,
            # info → slate.  Colour intent stays correct; the class
            # names are reused not duplicated.
            "rade-pill rade-pill--rejected": (
                "params.value === 'critical' || params.value === 'rose'"
            ),
            "rade-pill rade-pill--pending": (
                "params.value === 'warn' || params.value === 'amber'"
            ),
            "rade-pill rade-pill--archived": (
                "params.value === 'info' || params.value === 'slate'"
            ),
        },
        "valueFormatter": {
            "function": (
                "params.value ? "
                "params.value.charAt(0).toUpperCase() + params.value.slice(1)"
                " : '—'"
            ),
        },
    },
    {
        "field":      "opened",
        "headerName": "Opened",
        "minWidth":   140,
        "valueFormatter": {
            "function": (
                "params.value ? "
                "new Date(params.value).toLocaleDateString('en-GB', "
                "{day:'2-digit', month:'short', year:'numeric'}) "
                ": '—'"
            ),
        },
    },
    {
        "field":      "owner",
        "headerName": "Owner",
        "minWidth":   140,
    },
]


def _row_alerts_and_latency() -> html.Div:
    """Row 3 — active alerts grid (~1/2) + latency histogram (~1/2).

    Both children share equal width so the page reads as two paired
    cards, matching the design.  When the snapshot callback lands
    the latency histogram becomes a synthesised gamma distribution
    until inference telemetry exists; alerts stay empty until the
    Stage-2 alerts producer ships.
    """
    return html.Div(
        className="grid grid-cols-2 gap-4 items-stretch",
        children=[
            html.Div(
                className="rade-card flex flex-col gap-3",
                children=[
                    html.Div(
                        "Active Alerts",
                        className="text-sm font-semibold text-slate-200",
                    ),
                    AgGridTable(
                        grid_id=MONITORING_IDS["alerts_grid"],
                        row_data=[],
                        column_defs=_ALERTS_COLUMN_DEFS,
                        grid_options={
                            "pagination": False,
                            "rowHeight": 36,
                            "headerHeight": 38,
                            "animateRows": False,
                            "suppressCellFocus": True,
                            "domLayout": "autoHeight",
                        },
                        height=240,
                        className="rade-monitoring-alerts-grid",
                    ),
                ],
            ),
            ChartContainer(
                title="Latency Histogram",
                subtitle="Inference latency · P50 / P95 / P99",
                graph_id=MONITORING_IDS["latency_histogram_chart"],
                figure=empty_latency_histogram(),
                height=260,
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Footer caption
# ─────────────────────────────────────────────────────────────────────


def _row_footer() -> html.Div:
    """Producer-disclosure caption.

    Rendered as plain text right-aligned underneath Row 3, matching
    the design's tiny grey footer line.  Stays static — it's not
    user-data; the wording becomes "monitoring-data-source-truthful"
    once Stage 2 wires up production logging.
    """
    return html.Div(
        className="flex justify-end",
        children=[
            html.Div(
                _FOOTER_CAPTION,
                className="text-xs text-slate-500",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def build_monitoring(*, session: Optional["Session"] = None) -> html.Div:
    """Build the full Monitoring page tree.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but unused today — the page
    has no per-user persisted state.  Reserved so adding e.g. a
    ``monitoring_split_filter`` field to ``Session`` later is a
    one-line layout change.
    """
    del session  # unused today; reserved for forward-compat

    return html.Div(
        id=MONITORING_IDS["root"],
        className="rade-page",
        children=[
            # Mount tripwire — Page Contract §3 Rule L4.
            dcc.Store(
                id=MONITORING_IDS["mount_signal"],
                data=True,
                storage_type="memory",
            ),
            _row_header(),
            _row_drift_charts(),
            _row_alerts_and_latency(),
            _row_footer(),
        ],
    )


__all__ = [
    "MONITORING_IDS",
    "build_monitoring",
]
