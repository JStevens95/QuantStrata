"""Evaluation → Cluster Deep-Dive sub-tab callbacks (Phase E.5 hybrid).

Page Contract structure
-----------------------
The public surface is a single :func:`register` that delegates to two
section helpers, matching Page Contract §2 (capture / render split):

* :func:`_register_capture` — user-input gestures and cross-page
  navigation → :class:`Session` writes (no UI side-effects).
* :func:`_register_render`  — derived state → DOM updates (no
  :class:`Session` writes, except for the bootstrap capture-edge
  described below).

Capture (5 callbacks)
~~~~~~~~~~~~~~~~~~~~~
* ``_sync_selection``               — cluster picker, trades-grid row
  click, clear-chip button → ``deep_dive_cluster_id`` /
  ``deep_dive_selected_trade_id``.
* ``_sync_curve_metrics``           — overlay-metric chip group
  ``value`` → ``deep_dive_curve_metrics``.
* ``_sync_elementary_selection``    — Elementary PnL Explorer
  ``selectedRows`` and reset button → ``deep_dive_elementary_trade_ids``.
* ``_navigate_to_trade_graph``      — "Trade-Graph" button on this
  page → ``/evaluation/trade-graph`` with the active cluster pinned.
* ``_navigate_from_trade_graph``    — "Open in Cluster Deep Dive" on
  the Trade-Graph tab → ``/evaluation/cluster`` with the trade-
  graph's cluster + selected trade copied into the deep-dive slots.

Render (8 callbacks)
~~~~~~~~~~~~~~~~~~~~
* ``_bootstrap``                    — mount-signal-triggered fetch of
  the cluster ``Select.data``; coalesces the URL ``?cid=`` deep-link
  + fresh-user defaults into the same round-trip.
* ``_render_attributes``            — session → cluster-attributes
  card body, graph-statistics card body, "Trade-Graph" button enabled
  state, ``store_trade_types`` (the canonical trade-type map +
  ordered ``target_ids`` / ``elementary_ids`` lists for downstream
  callbacks).
* ``_render_kpis``                  — session → 4 KPI values
  (MAE / RMSE / P95 / P99) + 4 sparkline figures showing the per-trade
  distribution shape across the cluster.
* ``_render_timeseries``            — session → cluster portfolio
  chart (predicted vs target line) + residual-over-time chart.
* ``_render_training_curves``       — session → training-curves
  figure + chip group children + chip empty-state + curve-metrics
  store.
* ``_render_grids``                 — session + store_trade_types →
  Trade-Level Metrics grid (all trades) and Elementary PnL Explorer
  grid (filtered to elementary).  Also re-asserts the elementary
  ``selectedRows`` from session so deep-links / browser-back paint
  correctly.
* ``_render_per_trade_detail``      — session + store_trade_types →
  Row 3 wrapper visibility + selected-trade chip label + per-trade
  residual histogram + bias-vs-magnitude scatter.  Lazy-loads the
  per-cluster predictions NPZ on demand via ``backend.predictions``.
* ``_render_elementary_pnl``        — session → empty-state vs chart
  visibility on Row 4 right + elementary-pnl multi-line chart figure.

Data contract
-------------
Trade-type classification comes from the trade-graph endpoint and is
shared across the page via ``store_trade_types``::

    {
        "types":          {trade_id: "target" | "elementary"},
        "target_ids":     [tid, …],     # NPZ column ordering
        "elementary_ids": [tid, …],
    }

Why the ordered ``target_ids`` list?  The predictions NPZ is shaped
``(n_scenarios, n_target_trades_in_cluster)``; column ``i`` corresponds
to ``target_ids[i]``.  Threading the list through the store means the
per-trade detail callback never re-fetches the trade graph just to
recover ordering — the cached fetch in :func:`_render_attributes` is
the single source of truth.

Every fetch goes through :class:`RadeBackend`; the cache layer there
coalesces duplicate requests within a render tick.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import parse_qs

import dash_mantine_components as dmc
import numpy as np
import pandas as pd
from dash import Input, Output, State, ctx, html, no_update
from dash.exceptions import PreventUpdate

from ..data.result_helpers import figure_with_fallback
from ..data.session import Session
from ..figures import (
    elementary_pnl_multiline,
    empty_figure,
    error_over_time,
    per_trade_bias_scatter,
    per_trade_residual_histogram,
    portfolio_pnl,
    training_curves_chart,
)
from ..layouts.evaluation.cluster_deep_dive import CLUSTER_DEEP_DIVE_IDS
from ..layouts.evaluation.trade_graph import TRADE_GRAPH_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import RadeBackend


logger = logging.getLogger(__name__)


_DEEP_DIVE_PATH   = "/evaluation/cluster"
_TRADE_GRAPH_PATH = "/evaluation/trade-graph"
_PLACEHOLDER      = "—"

# Visual order in the Cluster Attributes card.  Matches the mock
# screenshot top-down.  Each entry is ``(label, candidate_columns)``
# — the first candidate column present in the clusters_df row wins,
# so we silently degrade when the ensemble's attribute schema drops
# an entry without leaving a phantom row in the UI.
_ATTRIBUTE_ROWS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Asset Class",  ("asset_class", "AssetClassCode")),
    ("Currency",     ("currency_code", "CurrencyCode", "currency")),
    ("Desk",         ("desk", "DeskCode", "desk_code")),
    ("Product",      ("product_code", "ProductCode", "product")),
    ("N Trades",     ("n_trades",)),
    ("N Scenarios", ("n_scenarios",)),
)

# Graph-stats rows.  The first three carry real values from
# ``trade_graph.stats``; the last two are deferred and always render
# as ``—`` so the card visual structure matches the mock.
_GRAPH_STATS_ROWS: Tuple[Tuple[str, Optional[str]], ...] = (
    ("Nodes",           "n_nodes"),
    ("Edges",           "n_edges"),
    ("Density",         "density"),
    ("Avg Degree",      None),
    ("Avg Path Length", None),
)


# ─────────────────────────────────────────────────────────────────────
# Formatters
# ─────────────────────────────────────────────────────────────────────


def _fmt_float(x: Optional[float], *, precision: int = 4) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:.{precision}f}"


def _fmt_int(x: Any) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = int(x)
    except (TypeError, ValueError):
        try:
            val = int(float(x))
        except (TypeError, ValueError):
            return _PLACEHOLDER
    return f"{val:,}"


def _fmt_density(x: Optional[float]) -> str:
    if x is None:
        return _PLACEHOLDER
    try:
        val = float(x)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val):
        return _PLACEHOLDER
    return f"{val:.3f}"


def _nullable_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(fv):
        return None
    return fv


def _parse_cid_from_search(search: Optional[str]) -> Optional[str]:
    """Extract the ``?cid=`` query param from the URL search string."""
    if not search:
        return None
    try:
        params = parse_qs(search.lstrip("?"))
    except (ValueError, TypeError):
        return None
    values = params.get("cid")
    if not values:
        return None
    candidate = values[0]
    return candidate if isinstance(candidate, str) and candidate else None


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Cluster-Deep-Dive sub-tab callback to ``app``."""
    _register_capture(app)
    _register_render(app, backend)


def _register_capture(app: "Dash") -> None:
    """Capture-side callbacks (input gestures → session writes only)."""
    _register_sync_selection(app)
    _register_sync_curve_metrics(app)
    _register_sync_elementary_selection(app)
    _register_navigate_to_trade_graph(app)
    _register_navigate_from_trade_graph(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Render-side callbacks (state → DOM, no session writes except bootstrap)."""
    _register_bootstrap(app, backend)
    _register_render_attributes(app, backend)
    _register_render_kpis(app, backend)
    _register_render_timeseries(app, backend)
    _register_render_training_curves(app, backend)
    _register_render_grids(app, backend)
    _register_render_per_trade_detail(app, backend)
    _register_render_elementary_select(app)
    _register_render_elementary_pnl(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Bootstrap — populate Select option list + resolve initial cluster
# ═════════════════════════════════════════════════════════════════════


def _register_bootstrap(app: "Dash", backend: "RadeBackend") -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["cluster_select"], "data"),
        Output(CLUSTER_DEEP_DIVE_IDS["cluster_select"], "value",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],              "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],    "data"),
        State(SHELL_IDS["url"],                         "search"),
        State(SHELL_IDS["url"],                         "pathname"),
        State(SHELL_IDS["session_store"],               "data"),
        prevent_initial_call="initial_duplicate",
    )
    def _bootstrap(
        _mount_signal: Any,
        search:        Optional[str],
        pathname:      Optional[str],
        session_data:  Optional[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, str]], Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        res = backend.clusters_df()
        if not res.ok or res.data is None or res.data.empty:
            return [], no_update, no_update

        session = Session.from_store(session_data)
        df = res.data
        options = [
            {"value": cid, "label": cid}
            for cid in sorted(df["cluster_id"].unique())
        ]
        valid_ids = {o["value"] for o in options}

        layout_seed = (
            session.evaluation.deep_dive_cluster_id or session.cluster_id
        )
        url_override = _parse_cid_from_search(search)
        canonical = (
            url_override
            or session.evaluation.deep_dive_cluster_id
            or session.cluster_id
            or (options[0]["value"] if options else None)
        )
        if canonical not in valid_ids:
            canonical = options[0]["value"] if options else None

        if canonical == layout_seed:
            return options, no_update, no_update

        session.evaluation.deep_dive_cluster_id = canonical
        return options, canonical, session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 2. Sync picker / grid-click / clear-btn → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_selection(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                       "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["cluster_select"],           "value"),
        Input(CLUSTER_DEEP_DIVE_IDS["trades_grid"],              "cellClicked"),
        Input(CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],                        "data"),
        prevent_initial_call=True,
    )
    def _sync(
        cluster:      Optional[str],
        cell_clicked: Optional[Dict[str, Any]],
        clear_clicks: Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation
        changed = False

        if trigger == CLUSTER_DEEP_DIVE_IDS["cluster_select"]:
            new_cluster = cluster if cluster else None
            if ev.deep_dive_cluster_id != new_cluster:
                ev.deep_dive_cluster_id = new_cluster
                # Cluster change invalidates per-trade + per-scenario
                # selections — those ids almost certainly don't live
                # in the new cluster.  Overlay-metric chips also reset
                # because metric availability is per-cluster.
                ev.deep_dive_selected_trade_id = None
                ev.deep_dive_elementary_trade_ids = []
                ev.deep_dive_curve_metrics = []
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["trades_grid"]:
            trade_id = _trade_id_from_cell(cell_clicked)
            if trade_id and ev.deep_dive_selected_trade_id != trade_id:
                ev.deep_dive_selected_trade_id = trade_id
                changed = True

        elif trigger == CLUSTER_DEEP_DIVE_IDS["selected_trade_clear_btn"]:
            if clear_clicks and ev.deep_dive_selected_trade_id is not None:
                ev.deep_dive_selected_trade_id = None
                changed = True

        if not changed:
            raise PreventUpdate
        return session.to_store()


def _trade_id_from_cell(cell_data: Optional[Dict[str, Any]]) -> Optional[str]:
    """Pull ``trade_id`` out of AgGrid's ``cellClicked`` payload."""
    if not cell_data:
        return None
    row = cell_data.get("data") or {}
    tid = row.get("trade_id")
    return tid if isinstance(tid, str) and tid else None


# ═════════════════════════════════════════════════════════════════════
# 3. Overlay-metric chip group → session.deep_dive_curve_metrics
# ═════════════════════════════════════════════════════════════════════


def _register_sync_curve_metrics(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["session_store"],                          "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],  "value"),
        State(SHELL_IDS["session_store"],                           "data"),
        prevent_initial_call=True,
    )
    def _sync(
        selected:     Optional[List[str]],
        session_data: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        session = Session.from_store(session_data)
        normalised = _normalise_str_list(selected)
        if session.evaluation.deep_dive_curve_metrics == normalised:
            raise PreventUpdate
        session.evaluation.deep_dive_curve_metrics = normalised
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 4. Elementary explorer multi-select / reset → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_elementary_selection(app: "Dash") -> None:
    """MultiSelect picks / Reset button → session.

    Capture-only callback (Page Contract §4 Rule C2).  The Reset
    button clears the picker by writing an empty list to session;
    the picker's own ``value`` is then re-asserted by
    ``_register_render_elementary_select`` on the resulting
    session-store update.
    """
    @app.callback(
        Output(SHELL_IDS["session_store"],                       "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["elementary_select"],        "value"),
        Input(CLUSTER_DEEP_DIVE_IDS["elementary_reset_btn"],     "n_clicks"),
        State(SHELL_IDS["session_store"],                        "data"),
        prevent_initial_call=True,
    )
    def _sync(
        selected_values: Optional[List[str]],
        n_clicks:        Optional[int],
        session_data:    Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        ev = session.evaluation

        if trigger == CLUSTER_DEEP_DIVE_IDS["elementary_reset_btn"]:
            if not n_clicks:
                raise PreventUpdate
            if not ev.deep_dive_elementary_trade_ids:
                raise PreventUpdate
            ev.deep_dive_elementary_trade_ids = []
            return session.to_store()

        # MultiSelect ``value`` is already a list of selected ids in
        # click order — normalise to dedupe / drop empties only.
        new_ids = _normalise_str_list(selected_values)
        if sorted(ev.deep_dive_elementary_trade_ids) == sorted(new_ids):
            raise PreventUpdate
        # Preserve picker click order — the multi-line chart respects
        # the order trade ids are passed to it (matches legend order).
        ev.deep_dive_elementary_trade_ids = new_ids
        return session.to_store()


def _normalise_str_list(raw: Optional[Sequence[Any]]) -> List[str]:
    """Stable, deduped list of non-empty strings."""
    if not isinstance(raw, (list, tuple)):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for v in raw:
        if not isinstance(v, str) or not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


# ═════════════════════════════════════════════════════════════════════
# 5. Render — Cluster Attributes + Graph Stats + nav button + store
# ═════════════════════════════════════════════════════════════════════


def _register_render_attributes(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["attributes_body"],      "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["graph_stats_body"],     "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"], "disabled"),
        Output(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],    "data"),
        # mount_signal lives inside the same layout chunk as the
        # outputs, so it can't race the router's content swap (Page
        # Contract §3 Rule L4).  pathname remains a State-side gate to
        # silence callbacks that would otherwise fire when the
        # session-store mutation happens on another sub-tab.
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],          "data"),
        Input(SHELL_IDS["session_store"],                     "data"),
        State(SHELL_IDS["url"],                               "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Tuple[List[Any], List[Any], bool, Dict[str, Any]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id

        if not cluster_id:
            return (
                _attribute_rows(None),
                _graph_stats_rows(None),
                True,
                _empty_trade_types_store(),
            )

        clusters_res = backend.clusters_df(cluster_id=cluster_id)
        attrs_row: Optional[Dict[str, Any]] = None
        if (
            clusters_res.ok
            and clusters_res.data is not None
            and not clusters_res.data.empty
        ):
            attrs_row = clusters_res.data.iloc[0].to_dict()

        graph_res = backend.trade_graph(cluster_id=cluster_id)
        stats: Optional[Dict[str, Any]] = None
        store_payload = _empty_trade_types_store()
        if graph_res.ok and graph_res.data is not None:
            tg = graph_res.data
            stats = tg.stats.model_dump() if tg.stats else None
            types_map: Dict[str, str] = {}
            target_ids: List[str] = []
            elementary_ids: List[str] = []
            for node in tg.nodes:
                tid = str(node.trade_id)
                ttype = str(node.trade_type)
                types_map[tid] = ttype
                if ttype == "target":
                    target_ids.append(tid)
                elif ttype == "elementary":
                    elementary_ids.append(tid)
            store_payload = {
                "types":          types_map,
                "target_ids":     target_ids,
                "elementary_ids": elementary_ids,
                "cluster_id":     cluster_id,
            }

        # n_trades / n_scenarios fall back to the trades_df shape if
        # they aren't carried on clusters_df (older parquets).  The
        # cheap fetch is cached so this doesn't add a round-trip on
        # the steady-state path.
        if attrs_row is None:
            attrs_row = {}
        if "n_trades" not in attrs_row or attrs_row.get("n_trades") is None:
            attrs_row["n_trades"] = (
                len(store_payload["types"])
                if store_payload["types"] else None
            )

        return (
            _attribute_rows(attrs_row),
            _graph_stats_rows(stats),
            False,
            store_payload,
        )


def _empty_trade_types_store() -> Dict[str, Any]:
    return {
        "types":          {},
        "target_ids":     [],
        "elementary_ids": [],
        "cluster_id":     None,
    }


def _attribute_rows(row: Optional[Dict[str, Any]]) -> List[Any]:
    """Render each (label, value) row — placeholder ``—`` on missing data."""
    children: List[Any] = []
    for label, candidates in _ATTRIBUTE_ROWS:
        value: Optional[Any] = None
        if row:
            for col in candidates:
                if col in row and row[col] is not None:
                    candidate = row[col]
                    if not (isinstance(candidate, float) and pd.isna(candidate)):
                        value = candidate
                        break
        if isinstance(value, float):
            display = _fmt_float(value, precision=2)
        elif isinstance(value, (int,)) and value is not None:
            display = _fmt_int(value)
        elif label in ("N Trades", "N Scenarios"):
            display = _fmt_int(value) if value is not None else _PLACEHOLDER
        elif value is None:
            display = _PLACEHOLDER
        else:
            display = str(value)
        children.append(_kv_row(label, display))
    return children


def _graph_stats_rows(stats: Optional[Dict[str, Any]]) -> List[Any]:
    """Render Graph Statistics body — Nodes / Edges / Density real,
    Avg Degree / Avg Path Length deferred (always ``—``).
    """
    children: List[Any] = []
    for label, key in _GRAPH_STATS_ROWS:
        if key is None or stats is None:
            children.append(_kv_row(label, _PLACEHOLDER))
            continue
        value = stats.get(key)
        if key == "density":
            display = _fmt_density(_nullable_float(value))
        else:
            display = _fmt_int(value)
        children.append(_kv_row(label, display))
    return children


def _kv_row(label: str, value: str) -> html.Div:
    return html.Div(
        className="flex items-center justify-between text-xs",
        children=[
            html.Span(label, className="text-slate-400"),
            html.Span(value, className="text-slate-100 font-medium"),
        ],
    )


# ═════════════════════════════════════════════════════════════════════
# 6. Render — Cluster Metrics KPI grid (4 KPIs + 4 sparklines)
# ═════════════════════════════════════════════════════════════════════


_KPI_CONFIG: Tuple[Tuple[str, str, str, str], ...] = (
    # (kpi_key, candidate column,         value_id key,             spark_id key)
    ("MAE",      "mae",            "kpi_mae_value",   "kpi_mae_spark"),
    ("RMSE",     "rmse",           "kpi_rmse_value",  "kpi_rmse_spark"),
    ("P95",      "p95_ae",         "kpi_p95_value",   "kpi_p95_spark"),
    ("P99",      "p99_ae",         "kpi_p99_value",   "kpi_p99_spark"),
)


def _register_render_kpis(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_mae_value"],   "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_rmse_value"],  "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p95_value"],   "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p99_value"],   "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_mae_spark"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_rmse_spark"],  "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p95_spark"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["kpi_p99_spark"],   "figure"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],     "data"),
        Input(SHELL_IDS["session_store"],                "data"),
        State(SHELL_IDS["url"],                          "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Tuple[Any, ...]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cluster_id:
            blank_spark = _sparkline_payload([])
            return (
                _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
                blank_spark, blank_spark, blank_spark, blank_spark,
            )

        # KPI headline values come from per_member_metrics (cluster
        # aggregate); sparklines come from trades_df (per-trade
        # distribution across the cluster's trades).
        member_res = backend.per_member_metrics_df(
            split=session.split, cluster_id=cluster_id,
        )
        member_row: Dict[str, Any] = {}
        if (
            member_res.ok
            and member_res.data is not None
            and not member_res.data.empty
        ):
            member_row = member_res.data.iloc[0].to_dict()

        trades_res = backend.trades_df(session.split, cluster_id=cluster_id)
        trades_df: Optional[pd.DataFrame] = (
            trades_res.data
            if trades_res.ok and trades_res.data is not None
            else None
        )

        values: List[str] = []
        sparks: List[Dict[str, Any]] = []
        for _, col, _vid, _sid in _KPI_CONFIG:
            headline = _nullable_float(member_row.get(col))
            # If the per-member parquet doesn't carry the column,
            # fall back to the median of the trades_df column so
            # the user still sees a number rather than ``—``.
            if headline is None and trades_df is not None and col in trades_df.columns:
                series = trades_df[col].dropna()
                if not series.empty:
                    headline = float(series.median())
            values.append(_fmt_float(headline))

            spark_data: List[float] = []
            if trades_df is not None and col in trades_df.columns:
                series = trades_df[col].dropna().astype(float)
                if not series.empty:
                    spark_data = series.tolist()
            sparks.append(_sparkline_payload(spark_data))

        return (
            values[0], values[1], values[2], values[3],
            sparks[0], sparks[1], sparks[2], sparks[3],
        )


def _sparkline_payload(data: Sequence[float]) -> Dict[str, Any]:
    """Tiny line trace with no chrome — matches ``KpiCard``'s built-in
    ``_sparkline_figure`` helper so the visual baseline is consistent."""
    if not data:
        return {
            "data": [],
            "layout": {
                "xaxis":         {"visible": False},
                "yaxis":         {"visible": False},
                "margin":        {"l": 0, "r": 0, "t": 0, "b": 0},
                "paper_bgcolor": "rgba(0,0,0,0)",
                "plot_bgcolor":  "rgba(0,0,0,0)",
                "showlegend":    False,
                "height":        36,
            },
        }
    return {
        "data": [
            {
                "type": "scatter",
                "mode": "lines",
                "x":    list(range(len(data))),
                "y":    list(data),
                "line": {"color": "#94a3b8", "width": 1.4},
                "hoverinfo": "skip",
            }
        ],
        "layout": {
            "xaxis":         {"visible": False},
            "yaxis":         {"visible": False},
            "margin":        {"l": 0, "r": 0, "t": 0, "b": 0},
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor":  "rgba(0,0,0,0)",
            "showlegend":    False,
            "height":        36,
        },
    }


# ═════════════════════════════════════════════════════════════════════
# 7. Render — Cluster Portfolio + Residual-over-time
# ═════════════════════════════════════════════════════════════════════


def _register_render_timeseries(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["portfolio_chart"],   "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["residual_ts_chart"], "figure"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],       "data"),
        Input(SHELL_IDS["session_store"],                  "data"),
        State(SHELL_IDS["url"],                            "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Tuple[Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see its portfolio."),
                empty_figure("Pick a cluster to see its residual-over-time."),
            )

        # Phase 3.4 — flip the cluster portfolio + residual-over-time
        # charts in lock-step with the topbar Scaled / Original toggle.
        # The shared figure builders surface the units in the y-axis
        # title + hover format; the uirevision key bakes ``space`` in
        # so toggling units resets zoom but split / cluster changes
        # alone don't.
        space  = session.pnl_space
        res = backend.cluster_timeseries_df(
            session.split, cluster_id=cluster_id, space=space,
        )
        ui_key = f"{session.split}::{cluster_id}:{space}"
        return (
            figure_with_fallback(
                res,
                on_ok=lambda df: portfolio_pnl(df, uirevision_key=ui_key, space=space),
                empty_msg="No timeseries data for this cluster.",
            ),
            figure_with_fallback(
                res,
                on_ok=lambda df: error_over_time(df, uirevision_key=ui_key, space=space),
                empty_msg="No residual data for this cluster.",
            ),
        )


# ═════════════════════════════════════════════════════════════════════
# 8. Render — Training curves (figure + chip group + empty msg + store)
# ═════════════════════════════════════════════════════════════════════


def _register_render_training_curves(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chart"],       "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],  "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_group"],  "value"),
        Output(CLUSTER_DEEP_DIVE_IDS["training_curves_chip_empty"],  "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["store_curve_metrics"],         "data"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],                 "data"),
        Input(SHELL_IDS["session_store"],                            "data"),
        State(SHELL_IDS["url"],                                      "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Tuple[Any, List[Any], List[str], Dict[str, str], List[str]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id

        empty_chip_hidden  = {"display": "none"}
        empty_chip_visible = {"display": "inline"}

        if not cluster_id:
            return (
                empty_figure("Pick a cluster to see its training curves."),
                [],
                [],
                empty_chip_visible,
                [],
            )

        res = backend.training_curves_df(cluster_id=cluster_id)
        if not res.ok or res.data is None or res.data.empty:
            err = res.error if not res.ok else "no data"
            logger.info(
                "training_curves fetch failed / empty for cluster %s: %s",
                cluster_id, err,
            )
            return (
                empty_figure("No training curves staged for this cluster."),
                [],
                [],
                empty_chip_visible,
                [],
            )

        df = res.data
        available = list(df.attrs.get("metrics") or [])
        persisted = list(session.evaluation.deep_dive_curve_metrics)
        validated = [m for m in persisted if m in available]

        chip_children = [
            dmc.Chip(m, value=m, size="xs", variant="outline")
            for m in available
        ]
        fig = training_curves_chart(
            df,
            selected_metrics=validated,
            available_metrics=available,
            uirevision_key=cluster_id,
        )
        empty_style = empty_chip_hidden if available else empty_chip_visible
        return fig, chip_children, validated, empty_style, available


# ═════════════════════════════════════════════════════════════════════
# 9. Render — Trade-Level Metrics + Elementary Explorer grids
# ═════════════════════════════════════════════════════════════════════


_TRADES_GRID_COLUMN_HEADERS: Dict[str, str] = {
    "trade_id":       "Trade",
    "trade_type":     "Type",
    "mae":            "MAE",
    "mse":            "MSE",
    "rmse":           "RMSE",
    "max_ae":         "Max |err|",
    "p95_ae":         "P95 |err|",
    "p99_ae":         "P99 |err|",
    "mean_residual":  "Mean resid.",
    "std_residual":   "Std resid.",
    "n_scenarios":    "Scenarios",
}
_TRADES_GRID_PRIORITY: Tuple[str, ...] = (
    "trade_id", "trade_type",
    "mae", "rmse", "p95_ae", "p99_ae",
    "mean_residual", "std_residual", "n_scenarios",
)


def _register_render_grids(
    app: "Dash", backend: "RadeBackend",
) -> None:
    """Trade-Level Metrics grid render — targets-only by data model.

    Elementary trades are picked via the ``MultiSelect`` dropdown in
    Row 4 (see ``_register_render_elementary_select``) — they are not
    surfaced in this grid because the /trades endpoint serves only
    target-trade metrics (elementaries have no predictions, hence no
    MAE / RMSE / residual to display).  The earlier callback shape that
    drove both grids has been split: this one feeds the trades grid,
    ``_register_render_elementary_select`` feeds the dropdown options.
    """
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid"], "rowData"),
        Output(CLUSTER_DEEP_DIVE_IDS["trades_grid"], "columnDefs"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],      "data"),
        Input(SHELL_IDS["session_store"],                 "data"),
        Input(CLUSTER_DEEP_DIVE_IDS["store_trade_types"], "data"),
        State(SHELL_IDS["url"],                           "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:        Any,
        session_data:  Optional[Dict[str, Any]],
        types_payload: Optional[Dict[str, Any]],
        pathname:      Optional[str],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        trade_columns = _trades_grid_column_defs_initial()

        if not cluster_id:
            return [], trade_columns

        res = backend.trades_df(session.split, cluster_id=cluster_id)
        if not res.ok or res.data is None or res.data.empty:
            return [], trade_columns

        df = res.data.copy()
        # ``store_trade_types`` always emits ``trade_id`` as ``str`` (see
        # ``TradeGraphNode``) — but the parquet behind ``trades_df`` may
        # carry it as ``int64`` (legacy export) or ``object`` with stray
        # whitespace.  Pandas ``.map`` is dtype-strict, so without an
        # explicit cast the ``trade_type`` column lookup silently misses
        # every row and the grid shows "unknown" for every trade.
        # Normalise once here so the lookup against ``types_map`` hits.
        if "trade_id" in df.columns:
            df["trade_id"] = df["trade_id"].astype(str).str.strip()

        types_map: Dict[str, str] = {}
        if isinstance(types_payload, dict):
            types_map = {
                str(k).strip(): str(v)
                for k, v in (types_payload.get("types") or {}).items()
            }

        df["trade_type"] = (
            df["trade_id"].map(types_map).fillna("unknown")
            if "trade_id" in df.columns
            else "unknown"
        )

        trades_columns = _trades_grid_column_defs(df)
        trades_rows = df[
            [c["field"] for c in trades_columns if c["field"] in df.columns]
        ].to_dict(orient="records")
        return trades_rows, trades_columns


def _trades_grid_column_defs_initial() -> List[Dict[str, Any]]:
    """Stable column defs for the empty-state trades grid."""
    return [
        {"field": "trade_id",      "headerName": "Trade",        "flex": 2, "minWidth": 160},
        {"field": "trade_type",    "headerName": "Type",         "flex": 1, "minWidth": 100},
        {"field": "mae",           "headerName": "MAE",          "flex": 1, "type": "numericColumn"},
        {"field": "rmse",          "headerName": "RMSE",         "flex": 1, "type": "numericColumn"},
        {"field": "p95_ae",        "headerName": "P95 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "p99_ae",        "headerName": "P99 |err|",    "flex": 1, "type": "numericColumn"},
        {"field": "mean_residual", "headerName": "Mean resid.",  "flex": 1, "type": "numericColumn"},
        {"field": "std_residual",  "headerName": "Std resid.",   "flex": 1, "type": "numericColumn"},
        {"field": "n_scenarios",   "headerName": "Scenarios",    "flex": 1, "type": "numericColumn"},
    ]


def _trades_grid_column_defs(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Dynamic columnDefs from the available ``trades_df`` columns."""
    numeric_cols = {
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col].dtype)
    }
    ordered_cols: List[str] = []
    seen: set = set()
    for col in _TRADES_GRID_PRIORITY:
        if col in df.columns and col not in seen:
            ordered_cols.append(col)
            seen.add(col)
    for col in df.columns:
        if col in seen or col in ("cluster_id", "split"):
            continue
        ordered_cols.append(col)
        seen.add(col)

    defs: List[Dict[str, Any]] = []
    for col in ordered_cols:
        col_def: Dict[str, Any] = {
            "field":      str(col),
            "headerName": _TRADES_GRID_COLUMN_HEADERS.get(col, str(col)),
        }
        if col == "trade_id":
            col_def.update({"flex": 2, "minWidth": 160})
        elif col == "trade_type":
            col_def.update({"flex": 1, "minWidth": 100})
        else:
            col_def["flex"] = 1
            if col in numeric_cols:
                col_def["type"] = "numericColumn"
                col_def["valueFormatter"] = {
                    "function": "d3.format('.4~g')(params.value)",
                }
        defs.append(col_def)
    return defs


# ═════════════════════════════════════════════════════════════════════
# 9b. Render — Elementary MultiSelect dropdown (options + value)
# ═════════════════════════════════════════════════════════════════════


def _register_render_elementary_select(app: "Dash") -> None:
    """Populate the elementary-trade MultiSelect ``data`` and ``value``.

    Sources both outputs off ``store_trade_types`` (the canonical
    ``{trade_id: "target" | "elementary"}`` map produced by
    ``_register_render_attributes``):

    * ``data`` — every elementary trade in the active cluster, sorted
      alphabetically so the dropdown's display order is stable across
      reloads / cluster switches.
    * ``value`` — re-asserted from the session's
      ``deep_dive_elementary_trade_ids``, intersected with the cluster's
      elementary set so a stale selection from a previous cluster
      gracefully prunes itself rather than throwing a "value not in
      data" warning at runtime.

    Capture (Page Contract §4 Rule C2) is delegated to
    ``_register_sync_elementary_selection``; this callback is pure
    render.  The chart render
    (``_register_render_elementary_pnl``) is decoupled from the picker
    via session, so this callback intentionally does not write back
    to the store.
    """
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_select"],   "data"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_select"],   "value"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],         "data"),
        Input(SHELL_IDS["session_store"],                    "data"),
        Input(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],    "data"),
        State(SHELL_IDS["url"],                              "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:        Any,
        session_data:  Optional[Dict[str, Any]],
        types_payload: Optional[Dict[str, Any]],
        pathname:      Optional[str],
    ) -> Tuple[List[Dict[str, str]], List[str]]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)

        elementary_set: set = set()
        if isinstance(types_payload, dict):
            types_map = (types_payload.get("types") or {})
            elementary_set = {
                str(tid).strip()
                for tid, ttype in types_map.items()
                if str(ttype).lower().strip() == "elementary"
            }

        options = [
            {"value": tid, "label": tid}
            for tid in sorted(elementary_set)
        ]

        # Intersect the session selection with the cluster's actual
        # elementary set so stale ids (left over from another cluster)
        # don't trigger Dash's "value not in options" client-side
        # warning.  Preserve user-visible order otherwise.
        session_selection = list(
            session.evaluation.deep_dive_elementary_trade_ids or []
        )
        value = [tid for tid in session_selection if tid in elementary_set]
        return options, value


# ═════════════════════════════════════════════════════════════════════
# 10. Render — Per-trade detail (Row 3, collapse-on-no-selection)
# ═════════════════════════════════════════════════════════════════════


def _register_render_per_trade_detail(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["row3_wrapper"],            "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["selected_trade_label"],    "children"),
        Output(CLUSTER_DEEP_DIVE_IDS["per_trade_residual_hist"], "figure"),
        Output(CLUSTER_DEEP_DIVE_IDS["per_trade_bias_scatter"],  "figure"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],             "data"),
        Input(SHELL_IDS["session_store"],                        "data"),
        Input(CLUSTER_DEEP_DIVE_IDS["store_trade_types"],        "data"),
        State(SHELL_IDS["url"],                                  "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:        Any,
        session_data:  Optional[Dict[str, Any]],
        types_payload: Optional[Dict[str, Any]],
        pathname:      Optional[str],
    ) -> Tuple[Dict[str, Any], str, Any, Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        selected_trade_id = session.evaluation.deep_dive_selected_trade_id

        hidden_style  = {"display": "none"}
        visible_style = {"display": "grid"}

        if not selected_trade_id or not cluster_id:
            return hidden_style, "Trade: —", no_update, no_update

        types_map: Dict[str, str] = {}
        target_ids: List[str] = []
        if isinstance(types_payload, dict):
            types_map = dict(types_payload.get("types") or {})
            target_ids = list(types_payload.get("target_ids") or [])

        trade_type = types_map.get(selected_trade_id, "unknown")
        chip_label = f"Trade: {selected_trade_id}  ·  Type: {trade_type}"

        # Residuals only make sense for *target* trades — the model
        # doesn't predict elementary trades (those are inputs).  Show
        # the row, but render an explanatory empty figure so the user
        # learns *why* the chart is blank rather than seeing a stale
        # plot or the row silently snapping shut.
        if trade_type != "target":
            placeholder = empty_figure(
                "Residual diagnostics are only available for target trades."
            )
            return visible_style, chip_label, placeholder, placeholder

        if not target_ids:
            placeholder = empty_figure(
                "Trade-graph payload missing target ordering — "
                "re-run the eval pipeline to stage trade_universe.json."
            )
            return visible_style, chip_label, placeholder, placeholder

        try:
            target_idx = target_ids.index(selected_trade_id)
        except ValueError:
            placeholder = empty_figure(
                f"Trade {selected_trade_id!r} not found in this cluster's "
                "target ordering."
            )
            return visible_style, chip_label, placeholder, placeholder

        # Phase 3.4 — fetch the NPZ that matches the topbar's PnL-space
        # toggle.  The histogram + bias-scatter inherit those units
        # implicitly (numpy slices, no axis labels rendered today).
        # When original-space data isn't on disk for this run the API
        # returns 404 → we surface a targeted empty-state instead of
        # silently falling back to scaled.
        space = session.pnl_space
        pred_res = backend.predictions(
            cluster_id=cluster_id, split=session.split, space=space,
        )
        if not pred_res.ok or pred_res.data is None:
            err = pred_res.error or "predictions fetch failed"
            if space == "original":
                msg = (
                    f"Original-space predictions are unavailable for this "
                    f"cluster/split (no scaler coverage). Toggle to Scaled "
                    f"to view this chart.  [{err}]"
                )
            else:
                msg = f"Could not load per-scenario predictions ({err})."
            placeholder = empty_figure(msg)
            return visible_style, chip_label, placeholder, placeholder

        npz: Dict[str, np.ndarray] = pred_res.data
        predictions_arr = np.asarray(npz.get("predictions"))
        targets_arr     = np.asarray(npz.get("targets"))
        if (
            predictions_arr.size == 0
            or targets_arr.size == 0
            or predictions_arr.ndim != 2
            or targets_arr.shape != predictions_arr.shape
        ):
            placeholder = empty_figure(
                "Predictions NPZ is empty or malformed for this cluster."
            )
            return visible_style, chip_label, placeholder, placeholder

        if target_idx >= predictions_arr.shape[1]:
            placeholder = empty_figure(
                f"Target index {target_idx} out of range for predictions "
                f"shape {predictions_arr.shape}."
            )
            return visible_style, chip_label, placeholder, placeholder

        pred_slice   = predictions_arr[:, target_idx]
        target_slice = targets_arr[:, target_idx]
        # uirevision key folds in ``space`` so toggling units resets the
        # histogram / scatter zoom (the binning is unit-dependent), but
        # an unrelated session change leaves the user's zoom intact.
        ui_key = f"{cluster_id}::{session.split}::{selected_trade_id}::{space}"

        histogram = per_trade_residual_histogram(
            pred_slice, target_slice,
            trade_id=selected_trade_id,
            uirevision_key=ui_key,
        )
        scatter = per_trade_bias_scatter(
            pred_slice, target_slice,
            trade_id=selected_trade_id,
            uirevision_key=ui_key,
        )
        return visible_style, chip_label, histogram, scatter


# ═════════════════════════════════════════════════════════════════════
# 11. Render — Elementary PnL multi-line (Row 4 right)
# ═════════════════════════════════════════════════════════════════════


def _register_render_elementary_pnl(
    app: "Dash", backend: "RadeBackend",
) -> None:
    @app.callback(
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_pnl_empty"],      "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart_card"], "style"),
        Output(CLUSTER_DEEP_DIVE_IDS["elementary_pnl_chart"],      "figure"),
        Input(CLUSTER_DEEP_DIVE_IDS["mount_signal"],               "data"),
        Input(SHELL_IDS["session_store"],                          "data"),
        State(SHELL_IDS["url"],                                    "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def _render(
        _mount:       Any,
        session_data: Optional[Dict[str, Any]],
        pathname:     Optional[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Any]:
        if pathname != _DEEP_DIVE_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cluster_id = session.evaluation.deep_dive_cluster_id or session.cluster_id
        selected_ids = list(session.evaluation.deep_dive_elementary_trade_ids or [])

        empty_style_visible = {
            "display": "flex",
            "flex":    "1 1 auto",
            "min-height": "160px",
        }
        empty_style_hidden  = {"display": "none"}

        if not cluster_id or not selected_ids:
            return (
                empty_style_visible,
                {"display": "none"},
                empty_figure(
                    "Pick one or more elementary trades to plot their PnL."
                ),
            )

        res = backend.elementary_pnl_df(
            cluster_id=cluster_id, trade_ids=selected_ids,
        )
        if not res.ok:
            err = res.error or "elementary-pnl fetch failed"
            return (
                empty_style_visible,
                {"display": "none"},
                empty_figure(f"Could not load elementary PnL ({err})."),
            )
        df = res.data
        if df is None or df.empty:
            return (
                empty_style_visible,
                {"display": "none"},
                empty_figure(
                    "Selected elementary trades have no PnL data on this split."
                ),
            )

        ui_key = f"{cluster_id}::{','.join(sorted(selected_ids))}"
        fig = elementary_pnl_multiline(df, uirevision_key=ui_key)
        return (
            empty_style_hidden,
            {"display": "block"},
            fig,
        )


# ═════════════════════════════════════════════════════════════════════
# 12. Cross-page navigation — to / from Trade-Graph sub-tab
# ═════════════════════════════════════════════════════════════════════


def _register_navigate_to_trade_graph(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["url"],                             "pathname",
               allow_duplicate=True),
        Output(SHELL_IDS["url"],                             "search",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],                   "data",
               allow_duplicate=True),
        Input(CLUSTER_DEEP_DIVE_IDS["open_trade_graph_btn"], "n_clicks"),
        State(SHELL_IDS["session_store"],                    "data"),
        prevent_initial_call=True,
    )
    def _navigate(
        n_clicks:     Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cid = session.evaluation.deep_dive_cluster_id or session.cluster_id
        if not cid:
            raise PreventUpdate

        session.evaluation.trade_graph_cluster_id = cid
        session.evaluation.trade_graph_selected_trade_id = None

        return _TRADE_GRAPH_PATH, "", session.to_store()


def _register_navigate_from_trade_graph(app: "Dash") -> None:
    @app.callback(
        Output(SHELL_IDS["url"],                          "pathname",
               allow_duplicate=True),
        Output(SHELL_IDS["url"],                          "search",
               allow_duplicate=True),
        Output(SHELL_IDS["session_store"],                "data",
               allow_duplicate=True),
        Input(TRADE_GRAPH_IDS["selected_deep_dive_btn"],  "n_clicks"),
        State(SHELL_IDS["session_store"],                 "data"),
        prevent_initial_call=True,
    )
    def _navigate(
        n_clicks:     Optional[int],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, Dict[str, Any]]:
        if not n_clicks:
            raise PreventUpdate

        session = Session.from_store(session_data)
        cid = (
            session.evaluation.trade_graph_cluster_id
            or session.cluster_id
        )
        if not cid:
            raise PreventUpdate

        session.evaluation.deep_dive_cluster_id = cid
        session.evaluation.deep_dive_selected_trade_id = (
            session.evaluation.trade_graph_selected_trade_id
        )

        return _DEEP_DIVE_PATH, f"?cid={cid}", session.to_store()


__all__ = ["register"]
