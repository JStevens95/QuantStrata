"""Evaluation → Portfolio sub-tab layout.

Phase E.1.  Five-row vertical rhythm:

    Row 1 · KPI strip          (MAE, RMSE, Hit %, Coverage)   ← aggregate
    Row 2 · Portfolio PnL      (predicted vs actual)          ← aggregate
    Row 3 · Error over time    (rolling mean + ±1σ band)      ← aggregate
    ── Error Analysis ──── [ Break down by: None ▼ ] [× clear]
    Row 4 · Residual violin     |    Predicted vs Actual scatter
    Row 5 · Leaderboard         (AgGrid, populated when grouped)

Rows 1-3 never react to the group-by control — they always reflect the
filter bar's WHERE clause applied to the ensemble aggregate.  Rows 4-5
are the "faceted" zone where the group-by kicks in.

The click-to-focus UX lives on the scatter card header: when the user
clicks a grouped point the ``groupby_clear_btn``-style chip becomes
the focus chip ("Focused: Rates [× Show all]").  Double-click on the
plot *or* the chip's close button clears it.  State is in
``session.evaluation.portfolio_scatter_focus``.

Callbacks live in :mod:`..callbacks.portfolio_cb`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_mantine_components as dmc
from dash import html
from dash_iconify import DashIconify

from ...components.ag_grid_table import AgGridTable
from ...components.chart_container import ChartContainer
from ...components.kpi_card import KpiCard
from ...data.session import EVALUATION_PORTFOLIO_GROUP_BY, Session


# Every dynamic id on the portfolio sub-tab lives here so callbacks
# never hardcode strings.  Keep the naming prefix-consistent:
# ``eval-portfolio-{component}``.
PORTFOLIO_IDS: Dict[str, str] = {
    "root":                 "eval-portfolio-root",

    # Row 1 — KPI strip values.  Card containers carry stable card_ids
    # so future deltas can target them without rebuilding the whole
    # strip.
    "kpi_mae_card":         "eval-portfolio-kpi-mae-card",
    "kpi_mae_value":        "eval-portfolio-kpi-mae-value",
    "kpi_rmse_card":        "eval-portfolio-kpi-rmse-card",
    "kpi_rmse_value":       "eval-portfolio-kpi-rmse-value",
    "kpi_hit_rate_card":    "eval-portfolio-kpi-hit-rate-card",
    "kpi_hit_rate_value":   "eval-portfolio-kpi-hit-rate-value",
    "kpi_coverage_card":    "eval-portfolio-kpi-coverage-card",
    "kpi_coverage_value":   "eval-portfolio-kpi-coverage-value",

    # Row 2
    "pnl_chart":            "eval-portfolio-pnl-chart",
    # Row 3
    "error_ts_chart":       "eval-portfolio-error-ts-chart",

    # Error-analysis divider + break-down control
    "analysis_divider":     "eval-portfolio-analysis-divider",
    "groupby_select":       "eval-portfolio-groupby-select",
    "groupby_clear_btn":    "eval-portfolio-groupby-clear-btn",
    "groupby_count_label":  "eval-portfolio-groupby-count-label",

    # Row 4
    "residual_violin":      "eval-portfolio-residual-violin",
    "pred_actual_scatter":  "eval-portfolio-pred-actual-scatter",

    # Focus chip (scatter card header) — shown only when focus is set.
    "focus_chip_container": "eval-portfolio-focus-chip-container",
    "focus_chip_label":     "eval-portfolio-focus-chip-label",
    "focus_chip_clear_btn": "eval-portfolio-focus-chip-clear-btn",

    # Row 5
    "leaderboard_card":      "eval-portfolio-leaderboard-card",
    "leaderboard_grid":      "eval-portfolio-leaderboard-grid",
    "leaderboard_grid_wrap": "eval-portfolio-leaderboard-grid-wrap",
    "leaderboard_empty":     "eval-portfolio-leaderboard-empty",
    "leaderboard_header":    "eval-portfolio-leaderboard-header",
}


# Human-readable labels for the break-down select.  Keep in sync with
# :data:`EVALUATION_PORTFOLIO_GROUP_BY` — the assert below catches
# drift at import time.
_GROUP_BY_LABELS: Dict[str, str] = {
    "desk":        "Desk",
    "product":     "Product",
    "currency":    "Currency",
    "asset_class": "Asset class",
    "cluster":     "Cluster",
}

assert set(_GROUP_BY_LABELS.keys()) == set(EVALUATION_PORTFOLIO_GROUP_BY), (
    "Portfolio group-by labels out of sync with session whitelist"
)


def _groupby_options() -> List[Dict[str, str]]:
    """Build the ``dmc.Select.data`` payload preserving the canonical
    order defined by :data:`EVALUATION_PORTFOLIO_GROUP_BY`."""
    return [
        {"value": key, "label": _GROUP_BY_LABELS[key]}
        for key in EVALUATION_PORTFOLIO_GROUP_BY
    ]


# ─────────────────────────────────────────────────────────────────────
# Public builder
# ─────────────────────────────────────────────────────────────────────


def build_portfolio(*, session: Optional[Session] = None) -> html.Div:
    """Assemble the Portfolio sub-tab tree (pure layout, no callbacks).

    Parameters
    ----------
    session
        Optional :class:`Session` whose
        :attr:`Session.evaluation.portfolio_group_by` seeds the
        break-down :class:`dmc.Select` value at build time.  Page
        Contract §3 Rule L1: initial UI state is sourced from session
        at build time, eliminating the historical hydration callback
        (and the URL-trigger race conditions it created).  ``None``
        falls back to a fresh :class:`Session` so unit tests + preview
        scripts that don't thread session through still render a
        sensible default (Break-down = None).
    """
    sess = session or Session()
    group_by = sess.evaluation.portfolio_group_by

    return html.Div(
        id=PORTFOLIO_IDS["root"],
        className="rade-evaluation-subtab flex flex-col gap-4",
        children=[
            _row_kpis(),
            _row_pnl_chart(),
            _row_error_timeseries(),
            _row_analysis_divider(group_by=group_by),
            _row_faceted_charts(),
            _row_leaderboard(),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 1 — KPIs
# ─────────────────────────────────────────────────────────────────────


def _row_kpis() -> html.Div:
    return html.Div(
        className="grid grid-cols-4 gap-3",
        children=[
            KpiCard(
                label="MAE",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_mae_card"],
                value_id=PORTFOLIO_IDS["kpi_mae_value"],
                icon="tabler:arrow-narrow-down",
            ),
            KpiCard(
                label="RMSE",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_rmse_card"],
                value_id=PORTFOLIO_IDS["kpi_rmse_value"],
                icon="tabler:square-root",
            ),
            KpiCard(
                label="Hit rate",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_hit_rate_card"],
                value_id=PORTFOLIO_IDS["kpi_hit_rate_value"],
                icon="tabler:target",
            ),
            KpiCard(
                label="Coverage",
                value="—",
                card_id=PORTFOLIO_IDS["kpi_coverage_card"],
                value_id=PORTFOLIO_IDS["kpi_coverage_value"],
                icon="tabler:chart-bar",
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 2 — Portfolio PnL
# ─────────────────────────────────────────────────────────────────────


def _row_pnl_chart() -> html.Div:
    return ChartContainer(
        title="Portfolio PnL",
        subtitle="Predicted vs actual across the active split",
        graph_id=PORTFOLIO_IDS["pnl_chart"],
        height=300,
    )


# ─────────────────────────────────────────────────────────────────────
# Row 3 — Error over time
# ─────────────────────────────────────────────────────────────────────


def _row_error_timeseries() -> html.Div:
    return ChartContainer(
        title="Error over time",
        subtitle="Rolling absolute error with ±1σ band",
        graph_id=PORTFOLIO_IDS["error_ts_chart"],
        height=260,
    )


# ─────────────────────────────────────────────────────────────────────
# Error-analysis divider row — section heading + break-down control
# ─────────────────────────────────────────────────────────────────────


def _row_analysis_divider(*, group_by: Optional[str] = None) -> html.Div:
    """Visual divider that also hosts the Group-by control.

    Everything below this row responds to the control; everything above
    stays aggregate.  The caption on the left anchors the user's mental
    model for what the control affects.

    Parameters
    ----------
    group_by
        Initial value for the break-down :class:`dmc.Select`.  Sourced
        from ``session.evaluation.portfolio_group_by`` at build time
        (Page Contract §3 Rule L1).  Also drives the Clear button's
        initial visibility — when no break-down is set the button is
        hidden so the row reads "Break down by [None ▼]" cleanly; the
        ``_render_grouped`` callback later re-asserts visibility via
        the same style attribute, so this is the single-paint
        first-frame state.
    """
    clear_btn_style = (
        {"display": "inline-flex"} if group_by else {"display": "none"}
    )

    return html.Div(
        id=PORTFOLIO_IDS["analysis_divider"],
        className="rade-section-divider flex items-center gap-3",
        children=[
            html.Div(
                className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-slate-400",
                children=[
                    DashIconify(icon="tabler:chart-dots-2", width=14),
                    html.Span("Error analysis"),
                ],
            ),
            html.Div(className="flex-1 h-px bg-slate-800"),
            html.Div(
                className="flex items-center gap-2",
                children=[
                    html.Span(
                        "Break down by",
                        className="text-xs text-slate-400",
                    ),
                    dmc.Select(
                        id=PORTFOLIO_IDS["groupby_select"],
                        placeholder="None",
                        data=_groupby_options(),
                        value=group_by,
                        clearable=True,
                        searchable=False,
                        size="xs",
                        radius="sm",
                        className="rade-portfolio-groupby-select",
                        style={"minWidth": "160px"},
                    ),
                    html.Span(
                        id=PORTFOLIO_IDS["groupby_count_label"],
                        className="text-xs text-slate-500",
                        children="",
                    ),
                    dmc.Button(
                        id=PORTFOLIO_IDS["groupby_clear_btn"],
                        children="Clear",
                        leftSection=DashIconify(icon="tabler:x", width=12),
                        size="xs",
                        variant="subtle",
                        color="gray",
                        style=clear_btn_style,
                    ),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 4 — Residual violin + Pred-vs-Actual scatter
# ─────────────────────────────────────────────────────────────────────


def _row_faceted_charts() -> html.Div:
    return html.Div(
        className="grid grid-cols-2 gap-3",
        children=[
            ChartContainer(
                title="Residual distribution",
                subtitle="Pred − actual",
                graph_id=PORTFOLIO_IDS["residual_violin"],
                height=360,
            ),
            ChartContainer(
                title="Predicted vs actual",
                subtitle="Dashed line is the identity",
                graph_id=PORTFOLIO_IDS["pred_actual_scatter"],
                height=360,
                actions=[_focus_chip()],
                config={"doubleClick": "reset"},
            ),
        ],
    )


def _focus_chip() -> html.Div:
    """Focus-state chip shown in the scatter card header.

    Hidden by default (``display: none``); the callback swaps the
    inline style + label text when the user clicks a grouped point.
    """
    return html.Div(
        id=PORTFOLIO_IDS["focus_chip_container"],
        className="rade-focus-chip flex items-center gap-1",
        style={"display": "none"},
        children=[
            DashIconify(icon="tabler:focus-2", width=12, className="text-violet-400"),
            html.Span(
                "Focused: —",
                id=PORTFOLIO_IDS["focus_chip_label"],
                className="text-xs text-slate-300",
            ),
            html.Button(
                "× Show all",
                id=PORTFOLIO_IDS["focus_chip_clear_btn"],
                className="rade-focus-chip-close",
                **{"aria-label": "Clear scatter focus"},
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────
# Row 5 — Leaderboard (populated when grouped)
# ─────────────────────────────────────────────────────────────────────


def _row_leaderboard() -> html.Div:
    """Leaderboard card.

    Always mounted so the AgGrid instance (and its state — column
    sorts, filters, page size) survives group-by toggling.  The empty
    state div is layered on top and hidden by the callback once data
    lands; this keeps the DOM structure stable for testing.
    """
    header = html.Div(
        id=PORTFOLIO_IDS["leaderboard_header"],
        className="flex items-center justify-between",
        children=[
            html.Div(
                className="flex flex-col",
                children=[
                    html.Div("Leaderboard", className="text-sm font-semibold text-slate-200"),
                    html.Div(
                        "Pick a break-down dimension to compare contributors.",
                        className="text-xs text-slate-500",
                    ),
                ],
            ),
        ],
    )

    empty_state = html.Div(
        id=PORTFOLIO_IDS["leaderboard_empty"],
        className="rade-list-empty flex flex-col items-center justify-center gap-2 py-8",
        children=[
            DashIconify(icon="tabler:table-off", width=22, className="text-slate-600"),
            html.Div(
                "Pick a break-down dimension above to compare contributors "
                "by desk, product, currency, asset class or cluster.",
                className="text-xs text-slate-500 text-center max-w-sm",
            ),
        ],
    )

    grid = AgGridTable(
        grid_id=PORTFOLIO_IDS["leaderboard_grid"],
        column_defs=_initial_column_defs(),
        row_data=[],
        height=320,
        className="rade-portfolio-leaderboard",
    )

    # Wrap grid in a div we can show/hide — AgGrid's own container is
    # tricky to style inline.
    grid_wrapper = html.Div(
        id=PORTFOLIO_IDS["leaderboard_grid_wrap"],
        className="rade-portfolio-leaderboard-grid-wrap",
        style={"display": "none"},
        children=grid,
    )

    return html.Div(
        id=PORTFOLIO_IDS["leaderboard_card"],
        className="rade-card flex flex-col gap-3",
        children=[header, empty_state, grid_wrapper],
    )


def _initial_column_defs() -> List[Dict[str, Any]]:
    """Bootstrap columnDefs used before the first callback fires.

    The column set is rewritten dynamically by the callback once a
    group-by dimension is picked (the first column's header switches
    between "Desk", "Product", "Currency", …) but keeping a sensible
    default here means the card renders tidily during the brief
    no-data state.
    """
    return [
        {"field": "group_label", "headerName": "Break-down", "flex": 2, "minWidth": 140},
        {"field": "mae",         "headerName": "MAE",        "flex": 1, "type": "numericColumn"},
        {"field": "rmse",        "headerName": "RMSE",       "flex": 1, "type": "numericColumn"},
        {"field": "hit_rate",    "headerName": "Hit %",      "flex": 1, "type": "numericColumn"},
        {"field": "contribution", "headerName": "Contribution", "flex": 1, "type": "numericColumn"},
        {"field": "n_clusters",  "headerName": "Clusters",   "flex": 1, "type": "numericColumn"},
    ]


__all__ = ["PORTFOLIO_IDS", "build_portfolio"]
