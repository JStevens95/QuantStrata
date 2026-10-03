"""Overview page callbacks — live-wiring for the ``/`` route.

Phase D.3 of the Rade UI build.  Replaces the placeholder constants
baked into :mod:`..layouts.overview` with real data fetched through
:class:`RadeBackend` whenever:

* the user lands on ``/`` (first route resolution or any URL change
  back to the overview),
* the split toggle in the topbar flips (e.g. test → val),
* the active ensemble version changes in session.

Wiring scope (user-approved option (i))
---------------------------------------

Everything that has a concrete backend source is live-wired:

* KPI cards — MAE, RMSE, active clusters, total trades
  (from ``ensemble_metrics`` and ``clusters`` endpoints).
* Portfolio PnL figure — predicted vs actual line chart for the
  active split (from ``portfolio`` endpoint).
* Cluster-health heatmap — per-cluster p95 absolute-error terciles
  mapped to ``ok / warn / err`` (from ``per-member-metrics`` endpoint).
* Top-performers card — 3 clusters with the lowest MAE on the
  active split (also ``per-member-metrics``).

Intentionally **not wired** (placeholder text retained in the layout,
future phase):

* Inference-latency KPI — no backend source today.
* Attention-required + Recent-activity cards — governance / event-feed
  endpoints don't exist yet.

Failure behaviour
-----------------

Every fetch is wrapped in :class:`BackendResult`.  On error the
callback falls back to ``"—"`` for scalar values, an empty figure for
the chart, and empty cell/row lists for the tables — never a
stack-trace — and logs a warning.  The user still sees a usable
page with the parts that did succeed.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import dash_mantine_components as dmc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate

from ..components.topbar import TOPBAR_IDS
from ..data.session import Session
from ..figures._theme import (
    is_all_nan,
    pnl_axis_title,
    pnl_hover_format,
    sort_chronologically,
)
from ..layouts.overview import OVERVIEW_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Styling + helpers
# ─────────────────────────────────────────────────────────────────────

_PLACEHOLDER = "—"


def _fmt_num(x: Any, *, digits: int = 2) -> str:
    """Pretty-print a numeric value, ``—`` when missing / NaN."""
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:,.{digits}f}"


def _fmt_int(x: Any) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = int(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    return f"{val:,}"


def _empty_portfolio_figure(message: str) -> go.Figure:
    """Figure shown when the portfolio endpoint yields no data."""
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin={"l": 32, "r": 16, "t": 8, "b": 32},
        font={"family": "Inter, system-ui, sans-serif", "color": "#cbd5e1", "size": 11},
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[
            {
                "text":      message,
                "showarrow": False,
                "font":      {"color": "#64748b", "size": 13},
                "xref":      "paper",
                "yref":      "paper",
                "x":         0.5,
                "y":         0.5,
            }
        ],
    )
    return fig


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every overview callback to ``app``.

    Mirrors the Page Contract §2 capture/render split — every other
    page module follows the same shape (``portfolio_cb``,
    ``cluster_deep_dive_cb``).  The two section helpers are the only
    top-level symbols a reader should need to scan to understand the
    page's wiring.

    Parameters
    ----------
    app
        The Dash app returned by :func:`rade_analytics.app.create_app`.
    backend
        Shared :class:`RadeBackend` — all data fetches go through here.
    """
    _register_capture(app)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers — capture / render split (Page Contract §2)
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    """Attach the capture-side callbacks (input gestures → session).

    Capture callbacks are forbidden from doing UI rendering;  they
    write only to :class:`Session` (via the session-store).  Backend
    access is also forbidden here — capture is pure transformation
    of user input into session state.

    The split-toggle sync lives here because it's a topbar widget
    consumed by every page; we register the writer alongside the
    Overview callbacks because Overview is the first / canonical
    consumer of ``session.split``.
    """
    _register_split_sync(app)
    _register_pnl_space_sync(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Attach the render-side callbacks (state → DOM, no session writes).

    Render callbacks are forbidden from writing to the session-store
    (it would create input→input chains that cascade across every
    page).  They consume URL + session-store as Inputs / States, do
    backend lookups via ``backend``, and emit values + figures into
    the page's components.
    """
    _register_overview_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# 1. Render — url/session → KPI values + chart + heatmap + top-performers
# ─────────────────────────────────────────────────────────────────────


def _register_overview_render(app: "Dash", backend: "RadeBackend") -> None:
    """Main overview render callback.

    Fires on URL changes and session-store writes.  We guard inside
    the callback so re-routes away from ``/`` don't waste API calls.
    """

    @app.callback(
        Output(OVERVIEW_IDS["kpi_mae_value"],      "children"),
        Output(OVERVIEW_IDS["kpi_rmse_value"],     "children"),
        Output(OVERVIEW_IDS["kpi_clusters_value"], "children"),
        Output(OVERVIEW_IDS["kpi_trades_value"],   "children"),
        Output(OVERVIEW_IDS["portfolio_chart"],    "figure"),
        Output(OVERVIEW_IDS["cluster_heatmap"],    "children"),
        Output(OVERVIEW_IDS["top_performers"],     "children"),
        Input(SHELL_IDS["url"],                    "pathname"),
        Input(SHELL_IDS["session_store"],          "data"),
        # Page Contract §4 Rule C5 — explicit opt-in.  The user may
        # land directly on ``/overview`` (URL share, browser back),
        # in which case the implicit initial pathname callback is the
        # only trigger that will paint the KPIs / chart / heatmap.
        # The pathname guard inside the body keeps cross-page hits
        # cheap (a single equality check).
        prevent_initial_call=False,
    )
    def _render(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, str, go.Figure, List[Any], List[Any]]:
        # Only spend API budget when the user is actually on the
        # Overview page.  When session_store fires from e.g. the
        # evaluation filter bar we bail without fetching anything.
        if pathname != "/overview":
            raise PreventUpdate

        session = Session.from_store(session_data)
        split  = session.split
        # Phase 3.4 — the portfolio chart's PnL units track the
        # topbar toggle.  KPI cards (MAE / RMSE) stay in scaled
        # space because the underlying ensemble_metrics parquet
        # only carries scaled metrics today; revisit if/when an
        # original-space metrics writer ships.
        space = session.pnl_space

        mae_txt, rmse_txt = _compute_kpi_metrics(backend, split)
        cluster_count_txt, trade_count_txt = _compute_cluster_kpis(backend)
        portfolio_fig = _compute_portfolio_figure(backend, split, space)
        heatmap_children = _compute_heatmap_children(backend, split)
        top_perf_children = _compute_top_performers_children(backend, split)

        return (
            mae_txt,
            rmse_txt,
            cluster_count_txt,
            trade_count_txt,
            portfolio_fig,
            heatmap_children,
            top_perf_children,
        )


# ─────────────────────────────────────────────────────────────────────
# 2. Split toggle → session store
# ─────────────────────────────────────────────────────────────────────


def _register_split_sync(app: "Dash") -> None:
    """Persist the topbar split toggle into ``session.split``.

    Every split-scoped page (overview, evaluation, monitoring, …)
    reads from the session store, so writing here is enough — we
    don't need per-page wiring.  Kept colocated with the overview
    callbacks because overview is the first consumer shipped.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(TOPBAR_IDS["split_toggle"],  "value"),
        State(SHELL_IDS["session_store"],  "data"),
        prevent_initial_call=True,
    )
    def _sync_split(
        value:        Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not value or value not in ("train", "val", "test"):
            raise PreventUpdate
        session = Session.from_store(session_data)
        if session.split == value:
            # No-op write would still fire every downstream Input(store).
            raise PreventUpdate
        return session.with_split(value).to_store()


# ─────────────────────────────────────────────────────────────────────
# 2b. PnL-space toggle → session store (Phase 3.3)
# ─────────────────────────────────────────────────────────────────────


def _register_pnl_space_sync(app: "Dash") -> None:
    """Persist the topbar PnL-space toggle into ``session.pnl_space``.

    Mirrors :func:`_register_split_sync` exactly — every space-aware
    page (overview, evaluation, cluster deep-dive, …) reads from the
    session store, so writing here is enough.  Colocated with the
    split sync because both are global presentation toggles; if either
    grows further options worth refactoring out, they can move into a
    dedicated ``shell_cb`` module together.

    Defensive parsing: any value outside the canonical
    ``PNL_SPACES`` whitelist short-circuits with ``PreventUpdate`` so
    a stale segmented control (e.g. from a future schema bump) never
    corrupts the session store.
    """
    from ..data.session import PNL_SPACES

    @app.callback(
        Output(SHELL_IDS["session_store"], "data", allow_duplicate=True),
        Input(TOPBAR_IDS["space_toggle"],  "value"),
        State(SHELL_IDS["session_store"],  "data"),
        prevent_initial_call=True,
    )
    def _sync_pnl_space(
        value:        Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not value or value not in PNL_SPACES:
            raise PreventUpdate
        session = Session.from_store(session_data)
        if session.pnl_space == value:
            # Idempotent no-op — avoid firing every downstream Input(store).
            raise PreventUpdate
        return session.with_pnl_space(value).to_store()


# ═════════════════════════════════════════════════════════════════════
# Fetch helpers — each returns a ready-to-render output slot.
# Keeping the backend access split into small functions makes the main
# callback readable and lets tests drive each slot independently.
# ═════════════════════════════════════════════════════════════════════


def _compute_kpi_metrics(
    backend: "RadeBackend",
    split: str,
) -> Tuple[str, str]:
    """Return (mae, rmse) display strings for the KPI strip."""
    res = backend.ensemble_metrics_df()
    if not res.ok or res.data is None or res.data.empty:
        if not res.ok:
            logger.warning("ensemble_metrics fetch failed: %s", res.error)
        return _PLACEHOLDER, _PLACEHOLDER

    df = res.data
    row = df[df["split"] == split]
    if row.empty:
        return _PLACEHOLDER, _PLACEHOLDER
    mae = row.iloc[0].get("mae")
    rmse = row.iloc[0].get("rmse")
    return _fmt_num(mae, digits=3), _fmt_num(rmse, digits=3)


def _compute_cluster_kpis(
    backend: "RadeBackend",
) -> Tuple[str, str]:
    """Return (active_clusters, total_trades) display strings."""
    res = backend.clusters_df()
    if not res.ok or res.data is None or res.data.empty:
        if not res.ok:
            logger.warning("clusters fetch failed: %s", res.error)
        return _PLACEHOLDER, _PLACEHOLDER

    df = res.data
    n_clusters = len(df)
    total_trades = int(df["n_trades"].fillna(0).sum()) if "n_trades" in df.columns else None
    return _fmt_int(n_clusters), _fmt_int(total_trades)


def _compute_portfolio_figure(
    backend: "RadeBackend",
    split:   str,
    space:   str,
) -> go.Figure:
    """Predicted-vs-actual portfolio PnL for the active split / PnL space.

    Phase 3.4 — ``space`` is threaded through to the API so the chart
    flips between z-space and notional units in lock-step with the
    topbar toggle.  When the user picks ``original`` but the underlying
    run has no scaler coverage, the API returns all-``null`` measures;
    we detect that here and render a dedicated empty-state rather than
    a flat-zero chart.
    """
    res = backend.portfolio_df(split, space=space)
    if not res.ok or res.data is None or res.data.empty:
        if not res.ok:
            logger.warning("portfolio fetch failed (space=%s): %s", space, res.error)
            return _empty_portfolio_figure(f"Portfolio unavailable: {res.error}")
        return _empty_portfolio_figure(f"No portfolio data for split={split}")

    # Detect the "user toggled to original on a run without scaler
    # coverage" case before rendering — Plotly happily draws lines
    # through gaps but the chart reads as a flat zero, which is
    # confusing.  ``is_all_nan`` is True only when *every* sample is
    # missing on both series; mixed-coverage runs still render with
    # the available points so the user can compare what's there.
    df = sort_chronologically(res.data)
    if (
        space == "original"
        and is_all_nan(df.get("predictions"))
        and is_all_nan(df.get("targets"))
    ):
        return _empty_portfolio_figure(
            "Original-space PnL is unavailable for this run "
            "(no scaler coverage). Toggle to Scaled to view this chart."
        )

    x_vals = df["scenario_label"] if "scenario_label" in df.columns else list(range(len(df)))

    hover_fmt = pnl_hover_format(space)
    fig = go.Figure()
    if "predictions" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df["predictions"],
                mode="lines",
                name="Predicted PnL",
                line={"color": "#8b5cf6", "width": 2.5},
                fill="tozeroy",
                fillcolor="rgba(139, 92, 246, 0.18)",
                hovertemplate=f"%{{y:{hover_fmt.lstrip(':')}}}<extra></extra>",
            )
        )
    if "targets" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=df["targets"],
                mode="lines",
                name="Actual PnL",
                line={"color": "#cbd5e1", "width": 1.5, "dash": "dash"},
                hovertemplate=f"%{{y:{hover_fmt.lstrip(':')}}}<extra></extra>",
            )
        )
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        margin={"l": 32, "r": 16, "t": 8, "b": 32},
        font={"family": "Inter, system-ui, sans-serif", "color": "#cbd5e1", "size": 11},
        xaxis={
            "showgrid":        False,
            "zeroline":        False,
            "showticklabels":  True,
            "tickfont":        {"color": "#64748b"},
        },
        yaxis={
            # Space-aware Y-axis title — "PnL (scaled)" vs "PnL
            # (notional)" — keeps the toggle's effect unambiguous on
            # the chart itself, not just the topbar.
            "title": {
                "text": pnl_axis_title(space),
                "font": {"color": "#94a3b8", "size": 11},
            },
            "gridcolor": "rgba(30, 41, 59, 0.6)",
            "zeroline":  False,
            "tickfont":  {"color": "#64748b"},
        },
        legend={
            "orientation": "h",
            "y": 1.08, "x": 1, "xanchor": "right",
            "bgcolor": "rgba(0, 0, 0, 0)",
            "font": {"color": "#94a3b8", "size": 11},
        },
        hovermode="x unified",
        # Page Contract §6 — keyed on ``split + space`` so the user's
        # zoom / pan survives session-store churn that doesn't affect
        # the displayed data domain, but resets when either the split
        # flips test↔val or the units change scaled↔original (data
        # domain genuinely shifts in both cases).
        uirevision=f"{split}:{space}",
    )
    return fig


def _compute_heatmap_children(
    backend: "RadeBackend",
    split: str,
) -> List[Any]:
    """Children list for the cluster-health card (title + cell grid).

    Each cell is wrapped in a :class:`dmc.Tooltip` whose label is a
    three-block summary of the cluster:

    * **Header** — the ``cluster_id``.
    * **Attributes** — every column from :meth:`RadeBackend.clusters_df`
      other than ``cluster_id``, including ``n_trades`` and any
      ensemble-specific keys (asset_class, currency_code, desk,
      product_code, …).
    * **Stats** — the metrics from :meth:`RadeBackend.per_member_metrics_df`
      for the active split (mae, rmse, p95 / p99 absolute error,
      n_scenarios, n_targets).

    The merge between the two frames is a left join on ``cluster_id``,
    so a cluster missing from the metadata feed still renders with its
    metrics block.  When the metadata fetch fails outright we fall
    back to a stats-only tooltip rather than blank-out the heatmap.
    """
    title = html.Div(
        "Cluster Health",
        className="text-sm font-semibold text-slate-200",
    )

    metrics_res = backend.per_member_metrics_df(split=split)
    if not metrics_res.ok or metrics_res.data is None or metrics_res.data.empty:
        if not metrics_res.ok:
            logger.warning("per-member-metrics fetch failed: %s", metrics_res.error)
        return [title, _heatmap_empty()]

    metrics_df = metrics_res.data
    if "p95_ae" not in metrics_df.columns or "cluster_id" not in metrics_df.columns:
        return [title, _heatmap_empty()]

    # Tercile-by-p95 classification — cheap and schema-free.  Future
    # phase can replace this with a governance-defined status column.
    metrics_df = (
        metrics_df.dropna(subset=["p95_ae"])
        .sort_values("p95_ae")
        .reset_index(drop=True)
    )
    if metrics_df.empty:
        return [title, _heatmap_empty()]

    # Merge cluster metadata for the tooltip body.  Failure here is
    # non-fatal — we still render the heatmap with metric-only tooltips.
    clusters_res = backend.clusters_df()
    if clusters_res.ok and clusters_res.data is not None and not clusters_res.data.empty:
        merged = metrics_df.merge(
            clusters_res.data, on="cluster_id", how="left",
        )
        attr_cols = [
            c for c in clusters_res.data.columns if c != "cluster_id"
        ]
    else:
        if not clusters_res.ok:
            logger.warning("clusters fetch failed: %s", clusters_res.error)
        merged = metrics_df
        attr_cols = []

    n = len(merged)
    lo = n // 3
    hi = (2 * n) // 3
    cells: List[Any] = []
    for i, row in merged.iterrows():
        if i < lo:
            status = "ok"
        elif i < hi:
            status = "warn"
        else:
            status = "err"
        cells.append(
            dmc.Tooltip(
                label=_heatmap_tooltip_label(row, attr_cols),
                multiline=True,
                w=240,
                position="left",
                withArrow=True,
                color="dark.7",
                # ``transitionProps`` tightens the show/hide so users
                # can scrub across cells without animations stacking.
                transitionProps={"transition": "fade", "duration": 80},
                children=html.Div(
                    className=f"rade-heatmap-cell rade-heatmap-cell--{status}",
                ),
            )
        )

    return [
        title,
        html.Div(className="rade-cluster-heatmap", children=cells),
    ]


def _heatmap_tooltip_label(
    row: "pd.Series",
    attr_cols: List[str],
) -> html.Div:
    """Three-block tooltip body: header, attributes, stats.

    Pure presentation helper — every dynamic value goes through
    :func:`_fmt_num` / :func:`_fmt_int` so missing data renders as
    ``—`` rather than ``nan``.  The styling uses the same emerald /
    rose / slate scale as the rest of the dashboard so the tooltip
    feels like an extension of the card.
    """
    cluster_id = str(row.get("cluster_id", "—"))

    header = html.Div(
        cluster_id,
        className="text-xs font-semibold text-slate-100",
        style={"marginBottom": "6px"},
    )

    attr_rows: List[Any] = []
    for col in attr_cols:
        val = row.get(col)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        if col == "n_trades":
            display = _fmt_int(val)
        else:
            display = str(val)
        attr_rows.append(_kv_row(col.replace("_", " "), display))

    stat_rows = [
        _kv_row("MAE",         _fmt_num(row.get("mae"),         digits=4)),
        _kv_row("RMSE",        _fmt_num(row.get("rmse"),        digits=4)),
        _kv_row("p95 |err|",   _fmt_num(row.get("p95_ae"),      digits=4)),
        _kv_row("p99 |err|",   _fmt_num(row.get("p99_ae"),      digits=4)),
        _kv_row("Scenarios",   _fmt_int(row.get("n_scenarios"))),
        _kv_row("Targets",     _fmt_int(row.get("n_targets"))),
    ]

    children: List[Any] = [header]
    if attr_rows:
        children.append(html.Div(children=attr_rows))
        children.append(_tooltip_divider())
    children.append(html.Div(children=stat_rows))

    return html.Div(
        children=children,
        style={
            "fontFamily": "Inter, system-ui, sans-serif",
            "fontSize": "11px",
            "lineHeight": "1.4",
        },
    )


def _kv_row(label: str, value: str) -> html.Div:
    """One ``key: value`` row inside a tooltip block."""
    return html.Div(
        children=[
            html.Span(
                label,
                style={"color": "#94a3b8", "textTransform": "capitalize"},
            ),
            html.Span(
                value,
                style={
                    "color":      "#e2e8f0",
                    "fontFamily": '"JetBrains Mono", monospace',
                    "fontSize":   "10.5px",
                },
            ),
        ],
        style={
            "display":        "flex",
            "justifyContent": "space-between",
            "gap":            "0.75rem",
        },
    )


def _tooltip_divider() -> html.Div:
    """Hairline divider between attribute / stat blocks in the tooltip."""
    return html.Div(
        style={
            "height":          "1px",
            "backgroundColor": "#334155",
            "margin":          "6px 0",
        },
    )


def _heatmap_empty() -> html.Div:
    return html.Div(
        "No per-cluster metrics available.",
        className="text-xs text-slate-500",
    )


def _compute_top_performers_children(
    backend: "RadeBackend",
    split: str,
    *,
    top_n: int = 3,
) -> List[Any]:
    """Children list for the top-performers card (title + rows)."""
    header = [
        html.Div("Top Performers", className="rade-list-title"),
        html.Div(
            className="rade-list-header",
            children=[
                html.Span("Cluster"),
                html.Span("MAE / Scenarios"),
            ],
        ),
    ]

    res = backend.per_member_metrics_df(split=split)
    if not res.ok or res.data is None or res.data.empty:
        if not res.ok:
            logger.warning("per-member-metrics fetch failed: %s", res.error)
        return [
            *header,
            html.Div(
                "No per-cluster metrics available.",
                className="text-xs text-slate-500 py-2",
            ),
        ]

    df = res.data
    if "mae" not in df.columns or "cluster_id" not in df.columns:
        return [
            *header,
            html.Div(
                "Per-cluster metrics schema missing mae/cluster_id.",
                className="text-xs text-slate-500 py-2",
            ),
        ]

    df = df.dropna(subset=["mae"]).sort_values("mae").head(top_n)
    if df.empty:
        return [
            *header,
            html.Div(
                "No clusters with finite MAE.",
                className="text-xs text-slate-500 py-2",
            ),
        ]

    rows: List[Any] = []
    for _, row in df.iterrows():
        mae_txt = _fmt_num(row.get("mae"), digits=4)
        n_scen = row.get("n_scenarios")
        n_scen_txt = _fmt_int(n_scen) if n_scen is not None else _PLACEHOLDER
        rows.append(
            html.Div(
                className="rade-list-row",
                children=[
                    html.Span(str(row["cluster_id"]), className="rade-list-row-label"),
                    html.Span(
                        children=[
                            html.Span(mae_txt, className="rade-delta-pos"),
                            html.Span(" "),
                            html.Span(
                                f"n={n_scen_txt}",
                                className="text-xs text-slate-500",
                            ),
                        ],
                    ),
                ],
            )
        )

    return [*header, *rows]


__all__ = ["register"]
