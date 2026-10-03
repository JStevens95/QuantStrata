"""Overview / landing page layout.

Visually mirrors ``docs/platform_designs/rade_landing_dashboard.png``
region-for-region:

* **KPI strip** (row 1)  — 5 ``KpiCard``s: MAE, RMSE, active clusters,
  total trades, inference latency.
* **Charts row** (row 2) — left: portfolio PnL line chart (wide);
  right: cluster-health heatmap (narrow).
* **Insights row** (row 3) — three equal-width cards: top performers
  (``AgGridTable``), attention required (status-chip list), recent
  activity (timestamped feed).
* **Quick actions** (row 4) — run inference / compare versions /
  download report / open AI assistant.

At the Phase-D skeleton stage, **all data is hardcoded** so the layout
can be visually verified without the API running.  Phase D.3 (the
callback) keeps the layout untouched and only updates outputs keyed on
the ids exported in :data:`OVERVIEW_IDS`.

Design spec anchors
-------------------
* §4 Page template — p-6 container, 20 px vertical rhythm.
* §6 Components — KpiCard / ChartContainer / AgGridTable compositions.
* §7 Palette — deltas & chips follow the emerald/amber/rose scale.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import html
from dash_iconify import DashIconify

from ..components.ag_grid_table import AgGridTable
from ..components.chart_container import ChartContainer
from ..components.kpi_card import KpiCard

if TYPE_CHECKING:
    from ..data.session import Session


# Every dynamic id in the overview page lives here so the Phase D.3
# callback never hardcodes strings.  Add new ids by feature-name and
# keep the suffix convention: ``-value``, ``-chart``, ``-grid``.
OVERVIEW_IDS = {
    # Root container (useful for scroll anchors / page-level tests).
    "root":               "overview-root",

    # KPI strip — ``id`` on the card, ``value_id`` on the figure span.
    "kpi_mae":            "overview-kpi-mae",
    "kpi_mae_value":      "overview-kpi-mae-value",
    "kpi_rmse":           "overview-kpi-rmse",
    "kpi_rmse_value":     "overview-kpi-rmse-value",
    "kpi_clusters":       "overview-kpi-clusters",
    "kpi_clusters_value": "overview-kpi-clusters-value",
    "kpi_trades":         "overview-kpi-trades",
    "kpi_trades_value":   "overview-kpi-trades-value",
    "kpi_latency":        "overview-kpi-latency",
    "kpi_latency_value":  "overview-kpi-latency-value",

    # Charts + tables.
    "portfolio_chart":    "overview-portfolio-chart",
    "cluster_heatmap":    "overview-cluster-heatmap",
    "top_performers":     "overview-top-performers-grid",
    "attention_list":     "overview-attention-list",
    "activity_feed":      "overview-activity-feed",

    # Quick-action buttons — callbacks will wire these in Phase F.
    "qa_run_inference":   "overview-qa-run-inference",
    "qa_compare_versions":"overview-qa-compare-versions",
    "qa_download_report": "overview-qa-download-report",
    "qa_open_assistant":  "overview-qa-open-assistant",
}


# ─────────────────────────────────────────────────────────────────────
# Placeholder data — deleted wholesale in Phase D.3 when the callback
# takes over.  Values mirror the numbers shown on the mock.
# ─────────────────────────────────────────────────────────────────────


_KPI_PLACEHOLDER = {
    "mae":      {"value": "1.55",  "delta": "-8%",   "tone": "negative"},
    "rmse":     {"value": "10.69", "delta": "-3%",   "tone": "negative"},
    "clusters": {"value": "12",    "delta": None,    "tone": "neutral"},
    "trades":   {"value": "2,500", "delta": None,    "tone": "neutral"},
    "latency":  {"value": "0.23s", "delta": "ok",    "tone": "positive"},
}

# 20-cell health grid — 5 cols × 4 rows — status keyed to match the
# mock's colour pattern roughly (majority ok, a few amber / err).
_HEATMAP_PLACEHOLDER: List[str] = [
    "ok",   "warn", "ok",   "ok",   "ok",
    "warn", "ok",   "ok",   "warn", "ok",
    "ok",   "ok",   "warn", "ok",   "err",
    "ok",   "warn", "ok",   "err",  "ok",
]

_TOP_PERFORMERS: List[Dict[str, Any]] = [
    {"name": "Cluster 1", "pnl": "+222.57%", "predicted": "+88%"},
    {"name": "Cluster 2", "pnl": "+92.07%",  "predicted": "-30%"},
    {"name": "Cluster 3", "pnl": "+63.59%",  "predicted": "-20%"},
]

_ATTENTION_ITEMS: List[Dict[str, str]] = [
    {"label": "Items 1", "status": "amber",   "chip": "amber"},
    {"label": "Items 2", "status": "flagged", "chip": "flagged"},
    {"label": "Items 3", "status": "amber",   "chip": "amber"},
]

_ACTIVITY_FEED: List[Dict[str, str]] = [
    {"time": "2h ago", "body": "Predicted",      "meta": "8 hrs ago"},
    {"time": "1h ago", "body": "Added",          "meta": "10 days ago"},
    {"time": "1h ago", "body": "Added",          "meta": "10 days ago"},
]


# ─────────────────────────────────────────────────────────────────────
# Row builders
# ─────────────────────────────────────────────────────────────────────


def _kpi_strip() -> html.Div:
    """Row 1 — five KPI cards in a 5-column grid."""
    kpis = [
        KpiCard(
            card_id=OVERVIEW_IDS["kpi_mae"],
            value_id=OVERVIEW_IDS["kpi_mae_value"],
            label="Portfolio MAE",
            value=_KPI_PLACEHOLDER["mae"]["value"],
            delta=_KPI_PLACEHOLDER["mae"]["delta"],
            delta_tone=_KPI_PLACEHOLDER["mae"]["tone"],
        ),
        KpiCard(
            card_id=OVERVIEW_IDS["kpi_rmse"],
            value_id=OVERVIEW_IDS["kpi_rmse_value"],
            label="Portfolio RMSE",
            value=_KPI_PLACEHOLDER["rmse"]["value"],
            delta=_KPI_PLACEHOLDER["rmse"]["delta"],
            delta_tone=_KPI_PLACEHOLDER["rmse"]["tone"],
        ),
        KpiCard(
            card_id=OVERVIEW_IDS["kpi_clusters"],
            value_id=OVERVIEW_IDS["kpi_clusters_value"],
            label="Active Clusters",
            value=_KPI_PLACEHOLDER["clusters"]["value"],
        ),
        KpiCard(
            card_id=OVERVIEW_IDS["kpi_trades"],
            value_id=OVERVIEW_IDS["kpi_trades_value"],
            label="Total Trades",
            value=_KPI_PLACEHOLDER["trades"]["value"],
        ),
        KpiCard(
            card_id=OVERVIEW_IDS["kpi_latency"],
            value_id=OVERVIEW_IDS["kpi_latency_value"],
            label="Inference Latency",
            value=_KPI_PLACEHOLDER["latency"]["value"],
            delta=_KPI_PLACEHOLDER["latency"]["delta"],
            delta_tone=_KPI_PLACEHOLDER["latency"]["tone"],
            icon="tabler:circle-check",
        ),
    ]
    return html.Div(className="grid grid-cols-5 gap-4", children=kpis)


def _portfolio_figure() -> go.Figure:
    """Placeholder purple area + dashed target line matching the mock."""
    x = list(range(36))
    # Gentle upward-trending curve with one mid dip.
    y_pred = [
        1.00, 1.02, 1.05, 1.08, 1.12, 1.15, 1.18, 1.23, 1.28,
        1.30, 1.32, 1.38, 1.45, 1.52, 1.58, 1.62, 1.60, 1.55,
        1.50, 1.48, 1.52, 1.60, 1.70, 1.78, 1.85, 1.92, 1.98,
        2.05, 2.12, 2.18, 2.22, 2.28, 2.34, 2.40, 2.48, 2.55,
    ]
    y_target = [0.95 + 0.045 * i for i in range(36)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x, y=y_pred,
            mode="lines",
            name="Predicted PnL",
            line={"color": "#8b5cf6", "width": 2.5},
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.18)",
            hovertemplate="%{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=y_target,
            mode="lines",
            name="Target PnL",
            line={"color": "#cbd5e1", "width": 1.5, "dash": "dash"},
            hovertemplate="%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin={"l": 32, "r": 16, "t": 8, "b": 32},
        font={"family": "Inter, system-ui, sans-serif", "color": "#cbd5e1", "size": 11},
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
        },
        yaxis={
            "gridcolor": "rgba(30, 41, 59, 0.6)",
            "zeroline": False,
            "tickfont": {"color": "#64748b"},
        },
        legend={
            "orientation": "h",
            "y": 1.08, "x": 1, "xanchor": "right",
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"color": "#94a3b8", "size": 11},
        },
        hovermode="x unified",
    )
    return fig


def _cluster_heatmap() -> html.Div:
    """Right-hand ``Cluster Health`` panel — 5×4 grid of coloured cells."""
    cells = [
        html.Div(
            className=f"rade-heatmap-cell rade-heatmap-cell--{status}",
            title=f"Cluster {i + 1}: {status}",
        )
        for i, status in enumerate(_HEATMAP_PLACEHOLDER)
    ]
    return html.Div(
        id=OVERVIEW_IDS["cluster_heatmap"],
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div("Cluster Health", className="text-sm font-semibold text-slate-200"),
            html.Div(className="rade-cluster-heatmap", children=cells),
        ],
    )


def _charts_row() -> html.Div:
    """Row 2 — portfolio PnL (wide) + cluster health (narrow)."""
    return html.Div(
        className="grid grid-cols-3 gap-4",
        children=[
            html.Div(
                className="col-span-2",
                children=ChartContainer(
                    title="Portfolio PnL — All Clusters",
                    graph_id=OVERVIEW_IDS["portfolio_chart"],
                    figure=_portfolio_figure(),
                    height=280,
                ),
            ),
            html.Div(className="col-span-1", children=_cluster_heatmap()),
        ],
    )


def _top_performers_grid() -> html.Div:
    """Bottom-left card — top-N cluster leaderboard."""
    return html.Div(
        id=OVERVIEW_IDS["top_performers"],
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div("Top Performers", className="rade-list-title"),
            html.Div(
                className="rade-list-header",
                children=[
                    html.Span("Name"),
                    html.Span("PnL / Predicted"),
                ],
            ),
            *[
                html.Div(
                    className="rade-list-row",
                    children=[
                        html.Span(row["name"], className="rade-list-row-label"),
                        html.Span(
                            children=[
                                html.Span(
                                    row["pnl"],
                                    className="rade-delta-pos",
                                ),
                                html.Span(" "),
                                html.Span(
                                    row["predicted"],
                                    className=(
                                        "rade-delta-pos"
                                        if row["predicted"].startswith("+")
                                        else "rade-delta-neg"
                                    ),
                                ),
                            ],
                        ),
                    ],
                )
                for row in _TOP_PERFORMERS
            ],
        ],
    )


def _attention_list() -> html.Div:
    """Bottom-middle card — items needing review."""
    return html.Div(
        id=OVERVIEW_IDS["attention_list"],
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div("Attention Required", className="rade-list-title"),
            html.Div(
                className="rade-list-header",
                children=[
                    html.Span("Item"),
                    html.Span("Status / Flagged"),
                ],
            ),
            *[
                html.Div(
                    className="rade-list-row",
                    children=[
                        html.Span(row["label"], className="rade-list-row-label"),
                        html.Span(
                            row["status"],
                            className=f"rade-chip rade-chip--{row['chip']}",
                        ),
                    ],
                )
                for row in _ATTENTION_ITEMS
            ],
        ],
    )


def _activity_feed() -> html.Div:
    """Bottom-right card — recent platform activity."""
    return html.Div(
        id=OVERVIEW_IDS["activity_feed"],
        className="rade-card flex flex-col gap-3",
        children=[
            html.Div("Recent Activity", className="rade-list-title"),
            html.Div(
                className="rade-feed",
                children=[
                    html.Div(
                        className="rade-feed-row",
                        children=[
                            html.Span(row["time"], className="rade-feed-time"),
                            html.Div(
                                className="rade-feed-body",
                                children=[
                                    html.Span(row["body"]),
                                    html.Span(row["meta"], className="rade-feed-body-meta"),
                                ],
                            ),
                        ],
                    )
                    for row in _ACTIVITY_FEED
                ],
            ),
        ],
    )


def _insights_row() -> html.Div:
    """Row 3 — three equal-width insights cards."""
    return html.Div(
        className="grid grid-cols-3 gap-4",
        children=[
            _top_performers_grid(),
            _attention_list(),
            _activity_feed(),
        ],
    )


def _quick_actions() -> html.Div:
    """Row 4 — global call-to-actions pinned to the bottom of the page."""
    return html.Div(
        className="rade-quick-actions",
        children=[
            html.Div("Quick Actions", className="rade-quick-actions-label"),
            html.Div(
                className="rade-quick-actions-group",
                children=[
                    dmc.Button(
                        id=OVERVIEW_IDS["qa_run_inference"],
                        children="Run Inference",
                        color="violet",
                        variant="filled",
                        size="xs",
                        leftSection=DashIconify(icon="tabler:player-play", width=14),
                    ),
                    dmc.Button(
                        id=OVERVIEW_IDS["qa_compare_versions"],
                        children="Compare Versions",
                        color="gray",
                        variant="subtle",
                        size="xs",
                        leftSection=DashIconify(icon="tabler:git-compare", width=14),
                    ),
                    dmc.Button(
                        id=OVERVIEW_IDS["qa_download_report"],
                        children="Download Report",
                        color="gray",
                        variant="subtle",
                        size="xs",
                        leftSection=DashIconify(icon="tabler:file-download", width=14),
                    ),
                    dmc.Button(
                        id=OVERVIEW_IDS["qa_open_assistant"],
                        children="Open AI Assistant",
                        color="gray",
                        variant="subtle",
                        size="xs",
                        leftSection=DashIconify(icon="tabler:sparkles", width=14),
                    ),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def build_overview(*, session: Optional["Session"] = None) -> html.Div:
    """Build the full overview page layout.

    Returned tree is pure layout — every id that a callback will
    target is listed in :data:`OVERVIEW_IDS`.  The skeleton ships with
    hardcoded placeholder data so the page renders correctly without
    an API; Phase D.3's callback overwrites every KPI value, chart
    figure and list row on first pathname tick.

    The ``session`` kwarg is accepted for uniformity with every other
    page builder (Page Contract §2.1) but is currently unused — the
    overview callback hydrates dynamic state from the API on each
    pathname tick, and there's nothing on this page to seed from
    session at build time.
    """
    del session  # unused today; reserved for forward-compat
    return html.Div(
        id=OVERVIEW_IDS["root"],
        className="rade-page",
        children=[
            _kpi_strip(),
            _charts_row(),
            _insights_row(),
            _quick_actions(),
        ],
    )


# Keep AgGridTable importable even though the skeleton draws the
# leaderboard with hand-authored rows — Phase D.3 may swap the "top
# performers" panel for a proper AgGrid once row counts grow.
_ = AgGridTable


__all__ = [
    "OVERVIEW_IDS",
    "build_overview",
]
