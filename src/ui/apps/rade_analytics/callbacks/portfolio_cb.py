"""Evaluation → Portfolio sub-tab callbacks (Phase E.1).

Page Contract structure
-----------------------
The public surface is a single :func:`register` that delegates to two
section helpers, matching Page Contract §2 (capture / render split):

* :func:`_register_capture` — user-input gestures →
  :class:`Session` writes (no UI side-effects).
* :func:`_register_render` — derived state → DOM updates (no
  :class:`Session` writes).

Capture
~~~~~~~
1. ``_sync_groupby``          — Select + Clear → ``session.evaluation
   .portfolio_group_by`` (and clears any orphaned scatter_focus).
2. ``_sync_scatter_focus``    — scatter ``clickData`` / ``relayoutData``
   (double-click reset) / focus-chip close button →
   ``session.evaluation.portfolio_scatter_focus``.

Render
~~~~~~
3. ``_render_aggregate``      — KPIs + PnL + error-over-time.  Always
   reflects the **aggregate** (filter-narrowed) view; never reacts to
   the group-by control.
4. ``_render_grouped``        — Violin + scatter + leaderboard + focus-
   chip visibility.  The one big render that redraws whenever split,
   filters, group-by or scatter-focus changes.

Initial UI state (no hydration callbacks)
-----------------------------------------
The historic ``_hydrate_groupby`` callback (URL-trigger pushing
``session.evaluation.portfolio_group_by`` back into the
``groupby_select.value``) has been removed: ``build_portfolio(*,
session=...)`` now seeds the Select's ``value`` directly at layout
build time.  The session is threaded through
``evaluation_cb._sync_from_url`` → ``build_subtab_content`` →
``build_portfolio``, so every sub-tab re-mount gets the live session
without an extra round-trip.  See Page Contract §3 Rule L1.

Data plumbing
-------------
Callbacks fetch data through :class:`RadeBackend` only — no direct
``RadeApiClient`` access.  Two code paths for aggregate metrics:

* **Unfiltered fast path** — ``portfolio_df(split)`` returns the
  pre-aggregated ensemble PnL and errors; MAE/RMSE come from
  ``ensemble_metrics_df(split)``.
* **Filtered path** — ``cluster_timeseries_df(split)`` + ``clusters_df``
  join, filter on cluster attributes, then aggregate per scenario
  client-side.

For the grouped view we always go through the filtered path (even
without filters) because the per-cluster residuals are needed to colour
the violin + scatter and populate the leaderboard.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, no_update
from dash.exceptions import PreventUpdate

from ..data.session import EvaluationFilters, Session
from ..figures import (
    empty_figure,
    error_over_time,
    pred_actual_scatter,
    portfolio_pnl,
    residual_violin,
)
from ..figures._theme import pnl_axis_title
from ..layouts.evaluation.portfolio import PORTFOLIO_IDS
from ..layouts.shell import SHELL_IDS

if TYPE_CHECKING:
    from dash import Dash

    from ..data.backend import BackendResult, RadeBackend


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────

_PORTFOLIO_PATH = "/evaluation/portfolio"
_PLACEHOLDER = "—"

# Session → clusters_df column mapping.  The filter-bar uses the short
# names (``currency``, ``product``); the clusters DataFrame may surface
# either the canonical API schema names (``currency_code``,
# ``product_code``) or the short-form names (``currency``, ``product``)
# depending on which attribute keys the active ensemble uses.  Each
# entry below is an ordered tuple of candidate column names — the
# resolver picks the first one present in the DataFrame.  Keep
# canonical names first.
_GROUP_BY_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "desk":        ("desk",),
    "product":     ("product_code", "product", "product_id"),
    "currency":    ("currency_code", "currency", "ccy"),
    "asset_class": ("asset_class", "assetclass", "asset"),
    "cluster":     ("cluster_id",),
}


def _resolve_group_column(
    group_by:    Optional[str],
    df_columns:  Sequence[str],
) -> Optional[str]:
    """Pick the first candidate column for ``group_by`` present in
    ``df_columns``.  Returns ``None`` if no candidate matches —
    callers treat that as "no groups available"."""
    if group_by is None:
        return None
    columns = set(df_columns)
    for candidate in _GROUP_BY_CANDIDATES.get(group_by, ()):
        if candidate in columns:
            return candidate
    return None

_GROUP_BY_LABEL: Dict[str, str] = {
    "desk":        "Desk",
    "product":     "Product",
    "currency":    "Currency",
    "asset_class": "Asset class",
    "cluster":     "Cluster",
}

_CLEAR_BTN_VISIBLE = {}                 # inherit default layout style
_CLEAR_BTN_HIDDEN = {"display": "none"}

_FOCUS_CHIP_VISIBLE_STYLE: Dict[str, Any] = {"display": "inline-flex"}
_FOCUS_CHIP_HIDDEN_STYLE:  Dict[str, Any] = {"display": "none"}


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def register(app: "Dash", backend: "RadeBackend") -> None:
    """Attach every Portfolio-sub-tab callback to ``app``.

    Mirrors the Page Contract §2 capture/render split — every other
    page module follows the same shape.  The two section helpers are
    the only top-level symbols a reader should need to scan to
    understand the page's wiring.
    """
    _register_capture(app)
    _register_render(app, backend)


# ─────────────────────────────────────────────────────────────────────
# Section dispatchers — capture / render split (Page Contract §2)
# ─────────────────────────────────────────────────────────────────────


def _register_capture(app: "Dash") -> None:
    """Attach the capture-side callbacks (input gestures → session).

    Capture callbacks are forbidden from doing UI rendering;  they
    write only to :class:`Session` (via the session-store).  The
    downstream render callbacks then react to the resulting
    session-store change.  Backend access is also forbidden here —
    capture is pure transformation of user input into session state.
    """
    _register_sync_groupby(app)
    _register_sync_scatter_focus(app)


def _register_render(app: "Dash", backend: "RadeBackend") -> None:
    """Attach the render-side callbacks (state → DOM, no session writes).

    Render callbacks are forbidden from writing to the session-store
    (it would create input→input chains that cascade across every
    page).  They consume URL + session-store as Inputs / States, do
    backend lookups via ``backend``, and emit figures / text into the
    page's components.
    """
    _register_render_aggregate(app, backend)
    _register_render_grouped(app, backend)


# ═════════════════════════════════════════════════════════════════════
# 1. Group-by Select + Clear → session
# ═════════════════════════════════════════════════════════════════════


def _register_sync_groupby(app: "Dash") -> None:
    """Persist the break-down Select into session state.

    Fires when the user picks a dimension OR clicks the Clear button.
    Also clears ``scatter_focus`` whenever the group-by dimension
    changes — a focus value on desk ``"Alpha"`` is meaningless once
    the user switches to grouping by product.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"],         "data", allow_duplicate=True),
        Input(PORTFOLIO_IDS["groupby_select"],     "value"),
        Input(PORTFOLIO_IDS["groupby_clear_btn"],  "n_clicks"),
        State(SHELL_IDS["session_store"],          "data"),
        prevent_initial_call=True,
    )
    def _sync_groupby(
        selected_value: Optional[str],
        clear_clicks:   Optional[int],
        session_data:   Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        current = session.evaluation.portfolio_group_by

        if trigger == PORTFOLIO_IDS["groupby_clear_btn"]:
            if not clear_clicks:
                raise PreventUpdate
            new_value: Optional[str] = None
        else:
            # Normalise empty string / falsy values to None so the
            # session representation stays canonical.
            new_value = selected_value if selected_value else None

        if new_value == current and session.evaluation.portfolio_scatter_focus is None:
            # Nothing to do — avoid a redundant store write that would
            # fan out to every downstream render callback.
            raise PreventUpdate

        session.evaluation.portfolio_group_by = new_value
        # Changing / clearing the dimension invalidates any focus.
        session.evaluation.portfolio_scatter_focus = None
        return session.to_store()


# ═════════════════════════════════════════════════════════════════════
# 2. Scatter clickData / dblclick / chip-close → session focus value
# ═════════════════════════════════════════════════════════════════════


def _register_sync_scatter_focus(app: "Dash") -> None:
    """Click-to-focus / double-click-to-reset on the grouped scatter.

    Single-click on a point → write the point's group label into
    session so :func:`_render_grouped` can redraw the scatter with that
    subset.  Double-click on the plot (Plotly's native axes-reset
    gesture) fires ``relayoutData`` with ``xaxis.autorange: True`` —
    we treat that as a focus clear.  The chip's × button is the
    explicit escape hatch for users who haven't zoomed.
    """

    @app.callback(
        Output(SHELL_IDS["session_store"],                    "data", allow_duplicate=True),
        Input(PORTFOLIO_IDS["pred_actual_scatter"],           "clickData"),
        Input(PORTFOLIO_IDS["pred_actual_scatter"],           "relayoutData"),
        Input(PORTFOLIO_IDS["focus_chip_clear_btn"],          "n_clicks"),
        State(SHELL_IDS["session_store"],                     "data"),
        prevent_initial_call=True,
    )
    def _sync_scatter_focus(
        click_data:    Optional[Dict[str, Any]],
        relayout_data: Optional[Dict[str, Any]],
        clear_clicks:  Optional[int],
        session_data:  Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        trigger = ctx.triggered_id
        if trigger is None:
            raise PreventUpdate

        session = Session.from_store(session_data)
        current = session.evaluation.portfolio_scatter_focus

        # Focus is only meaningful while a group-by is active — cheap
        # guard against a stray click landing during the brief interval
        # between a "group-by cleared" session write and the scatter
        # re-render.
        if session.evaluation.portfolio_group_by is None:
            if current is None:
                raise PreventUpdate
            session.evaluation.portfolio_scatter_focus = None
            return session.to_store()

        new_focus: Optional[str] = current

        if trigger == PORTFOLIO_IDS["focus_chip_clear_btn"]:
            if not clear_clicks:
                raise PreventUpdate
            new_focus = None

        elif trigger == PORTFOLIO_IDS["pred_actual_scatter"]:
            trigger_prop = ctx.triggered[0]["prop_id"].split(".")[-1] if ctx.triggered else ""
            if trigger_prop == "clickData":
                new_focus = _extract_group_from_click(click_data)
                if new_focus is None:
                    raise PreventUpdate
            elif trigger_prop == "relayoutData":
                if not _is_autorange_reset(relayout_data):
                    raise PreventUpdate
                new_focus = None
            else:
                raise PreventUpdate

        if new_focus == current:
            raise PreventUpdate

        session.evaluation.portfolio_scatter_focus = new_focus
        return session.to_store()


def _extract_group_from_click(
    click_data: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Pull the group label out of a scatter ``clickData`` payload.

    Returns ``None`` for unclickable points (the identity line, or
    aggregate-mode points whose customdata[0] is empty).
    """
    if not click_data:
        return None
    points = click_data.get("points") or []
    if not points:
        return None
    cd = points[0].get("customdata")
    if not cd or not isinstance(cd, (list, tuple)) or not cd:
        return None
    value = cd[0]
    if not isinstance(value, str) or not value:
        return None
    return value


def _is_autorange_reset(relayout_data: Optional[Dict[str, Any]]) -> bool:
    """Plotly's double-click reset emits ``xaxis.autorange: True`` etc."""
    if not relayout_data:
        return False
    return any(
        key.endswith(".autorange") and value is True
        for key, value in relayout_data.items()
    )


# ═════════════════════════════════════════════════════════════════════
# 3. Aggregate render — KPIs + PnL + error over time
# ═════════════════════════════════════════════════════════════════════


def _register_render_aggregate(app: "Dash", backend: "RadeBackend") -> None:
    """The top three rows: always show the aggregate / filter-narrowed view."""

    @app.callback(
        Output(PORTFOLIO_IDS["kpi_mae_value"],      "children"),
        Output(PORTFOLIO_IDS["kpi_rmse_value"],     "children"),
        Output(PORTFOLIO_IDS["kpi_hit_rate_value"], "children"),
        Output(PORTFOLIO_IDS["kpi_coverage_value"], "children"),
        Output(PORTFOLIO_IDS["pnl_chart"],          "figure"),
        Output(PORTFOLIO_IDS["error_ts_chart"],     "figure"),
        Input(SHELL_IDS["url"],                     "pathname"),
        Input(SHELL_IDS["session_store"],           "data"),
        # Page Contract §4 Rule C5 — explicit opt-in.  Direct entry on
        # ``/evaluation/portfolio`` (URL share, refresh) must paint
        # KPIs + charts on first paint; the layout-time render only
        # has placeholder strings / empty figures.  The pathname guard
        # in the body keeps non-Portfolio sub-tab hits cheap.
        prevent_initial_call=False,
    )
    def _render_aggregate(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[str, str, str, str, go.Figure, go.Figure]:
        if pathname != _PORTFOLIO_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split   = session.split
        filters = session.evaluation.filters
        # Phase 3.4 — every numeric in this section (KPIs + charts)
        # is in whichever PnL space the topbar toggle is on.  The
        # backend caches per (split, space) so flipping the toggle is
        # a cheap re-fetch (or an instant cache hit after first paint).
        space   = session.pnl_space

        portfolio_df = _aggregate_portfolio_frame(backend, split, filters, space)
        if portfolio_df is None or portfolio_df.empty:
            empty_fig = empty_figure("No portfolio data for the active filter set.")
            return (
                _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER,
                empty_fig,
                empty_figure("No error-over-time data for the active filter set."),
            )

        mae_txt, rmse_txt, hit_txt, cov_txt = _kpi_strings(portfolio_df, space)
        # Page Contract §6 — keyed on (split, space) so re-renders
        # driven by filter changes (same split / space) preserve the
        # user's zoom / pan / legend toggles, while flipping either
        # split or PnL units resets UI state to the new data domain.
        uirev = f"{split}:{space}"
        pnl_fig = portfolio_pnl(portfolio_df, uirevision_key=uirev, space=space)
        err_fig = error_over_time(portfolio_df, uirevision_key=uirev, space=space)

        return mae_txt, rmse_txt, hit_txt, cov_txt, pnl_fig, err_fig


# ═════════════════════════════════════════════════════════════════════
# 4. Grouped render — violin + scatter + leaderboard + focus chip
# ═════════════════════════════════════════════════════════════════════


def _register_render_grouped(app: "Dash", backend: "RadeBackend") -> None:
    """Rows 4-5 plus every group-by / focus-driven visibility switch."""

    @app.callback(
        Output(PORTFOLIO_IDS["residual_violin"],         "figure"),
        Output(PORTFOLIO_IDS["pred_actual_scatter"],     "figure"),
        Output(PORTFOLIO_IDS["leaderboard_grid"],        "rowData"),
        Output(PORTFOLIO_IDS["leaderboard_grid"],        "columnDefs"),
        Output(PORTFOLIO_IDS["leaderboard_empty"],       "style"),
        Output(PORTFOLIO_IDS["leaderboard_grid_wrap"],   "style"),
        Output(PORTFOLIO_IDS["leaderboard_header"],      "children"),
        Output(PORTFOLIO_IDS["focus_chip_container"],    "style"),
        Output(PORTFOLIO_IDS["focus_chip_label"],        "children"),
        Output(PORTFOLIO_IDS["groupby_clear_btn"],       "style"),
        Output(PORTFOLIO_IDS["groupby_count_label"],     "children"),
        Input(SHELL_IDS["url"],                          "pathname"),
        Input(SHELL_IDS["session_store"],                "data"),
        # Page Contract §4 Rule C5 — explicit opt-in.  Same rationale
        # as ``_render_aggregate`` above: rows 4–5 (violin / scatter /
        # leaderboard / focus chip) must paint on direct page entry.
        prevent_initial_call=False,
    )
    def _render_grouped(
        pathname:     Optional[str],
        session_data: Optional[Dict[str, Any]],
    ) -> Tuple[Any, ...]:
        if pathname != _PORTFOLIO_PATH:
            raise PreventUpdate

        session = Session.from_store(session_data)
        split       = session.split
        filters     = session.evaluation.filters
        group_by    = session.evaluation.portfolio_group_by
        focus_group = session.evaluation.portfolio_scatter_focus
        # Phase 3.4 — violin / scatter / leaderboard all derive from
        # per-cluster predictions / targets, so they inherit the
        # PnL-units toggle automatically once we fetch in the right
        # space.  Axis-title overrides below make the units explicit.
        space       = session.pnl_space

        # Pull the per-cluster frame used by violin + scatter +
        # leaderboard.  Single fetch reused three times, keyed on
        # (split, space, cluster_id filter) via the backend cache.
        per_cluster = _per_cluster_frame(backend, split, filters, space)
        available_groups: List[str] = []
        group_column_name: Optional[str] = None
        if group_by is not None and per_cluster is not None and not per_cluster.empty:
            group_column_name = _resolve_group_column(group_by, per_cluster.columns)
            if group_column_name is not None:
                available_groups = sorted(
                    {str(v) for v in per_cluster[group_column_name].dropna().unique()}
                )

        # Defensive — a stale focus value that no longer matches any
        # present group has to be ignored *visually* even though we
        # leave it in session (the next sync will clean it up).
        effective_focus = (
            focus_group if focus_group in available_groups else None
        )

        # Phase 3.4 — space-aware axis titles for violin + scatter so
        # the units the user sees match the toggle they flipped.
        residual_y_title = pnl_axis_title(space, prefix="Residual (pred − actual)")
        scatter_x_title  = pnl_axis_title(space, prefix="Predicted")
        scatter_y_title  = pnl_axis_title(space, prefix="Actual")

        # ── Violin + scatter ────────────────────────────────────────
        if per_cluster is None or per_cluster.empty:
            violin_fig = empty_figure("No per-cluster residuals for this filter set.")
            scatter_fig = empty_figure("No per-cluster pairs for this filter set.")
        else:
            residuals = (
                per_cluster["predictions"] - per_cluster["targets"]
            ).to_numpy(dtype=float)
            hover_labels = per_cluster.get(
                "cluster_id",
                pd.Series([""] * len(per_cluster)),
            ).astype(str).tolist()

            if group_by is None or group_column_name is None or group_column_name not in per_cluster.columns:
                violin_fig = residual_violin(residuals, y_axis_title=residual_y_title)
                scatter_fig = pred_actual_scatter(
                    per_cluster["predictions"].to_numpy(dtype=float),
                    per_cluster["targets"].to_numpy(dtype=float),
                    hover_labels=hover_labels,
                    x_axis_title=scatter_x_title,
                    y_axis_title=scatter_y_title,
                )
            else:
                group_vals = per_cluster[group_column_name].astype(str).tolist()
                violin_fig = residual_violin(
                    residuals,
                    group_values=group_vals,
                    group_order=available_groups,
                    group_label=_GROUP_BY_LABEL.get(group_by, group_by),
                    y_axis_title=residual_y_title,
                )
                scatter_fig = pred_actual_scatter(
                    per_cluster["predictions"].to_numpy(dtype=float),
                    per_cluster["targets"].to_numpy(dtype=float),
                    group_values=group_vals,
                    group_order=available_groups,
                    focus_group=effective_focus,
                    hover_labels=hover_labels,
                    x_axis_title=scatter_x_title,
                    y_axis_title=scatter_y_title,
                )

        # ── Leaderboard ─────────────────────────────────────────────
        if group_by is None:
            leaderboard_rows: List[Dict[str, Any]] = []
            column_defs = _leaderboard_column_defs(group_by=None)
            empty_state_style: Dict[str, Any] = {}
            grid_wrap_style:   Dict[str, Any] = {"display": "none"}
        elif per_cluster is None or per_cluster.empty or group_column_name not in (per_cluster.columns if per_cluster is not None else []):
            leaderboard_rows = []
            column_defs = _leaderboard_column_defs(group_by=group_by)
            empty_state_style = {}
            grid_wrap_style = {"display": "none"}
        else:
            leaderboard_rows = _leaderboard_rows(
                per_cluster,
                group_column=group_column_name,
            )
            column_defs = _leaderboard_column_defs(group_by=group_by)
            empty_state_style = {"display": "none"}
            grid_wrap_style = {}

        # ── Header caption + visibility widgets ────────────────────
        header_children = _leaderboard_header_children(
            group_by=group_by,
            n_groups=len(available_groups),
            n_clusters=0 if per_cluster is None else len(per_cluster.drop_duplicates("cluster_id")) if "cluster_id" in (per_cluster.columns if per_cluster is not None else []) else 0,
        )
        focus_chip_style = (
            _FOCUS_CHIP_VISIBLE_STYLE if effective_focus is not None
            else _FOCUS_CHIP_HIDDEN_STYLE
        )
        focus_chip_label = f"Focused: {effective_focus}" if effective_focus else "Focused: —"

        clear_btn_style = (
            _CLEAR_BTN_VISIBLE if group_by is not None else _CLEAR_BTN_HIDDEN
        )
        if group_by is None:
            count_label = ""
        elif not available_groups:
            count_label = "no groups"
        else:
            count_label = f"{len(available_groups)} group{'' if len(available_groups) == 1 else 's'}"

        return (
            violin_fig,
            scatter_fig,
            leaderboard_rows,
            column_defs,
            empty_state_style,
            grid_wrap_style,
            header_children,
            focus_chip_style,
            focus_chip_label,
            clear_btn_style,
            count_label,
        )


# ═════════════════════════════════════════════════════════════════════
# Data plumbing helpers
# ═════════════════════════════════════════════════════════════════════


def _aggregate_portfolio_frame(
    backend: "RadeBackend",
    split:   str,
    filters: EvaluationFilters,
    space:   str = "scaled",
) -> Optional[pd.DataFrame]:
    """Return the portfolio-level timeseries for KPIs + PnL + error-ts.

    * **No filters** → portfolio_df(split, space=...) as-is (fastest path).
    * **Filtered**   → join clusters_df, mask by attribute filters,
      then aggregate per-cluster timeseries into a portfolio rollup.

    Date filters are applied last, regardless of path.  ``space``
    (Phase 3.4) is threaded all the way down to both the portfolio and
    the cluster-timeseries fetch so the resulting frame is internally
    consistent — sums of original-space cluster series produce
    original-space portfolio totals.
    """
    df: Optional[pd.DataFrame]

    if filters.is_empty() or _only_date_filter(filters):
        res = backend.portfolio_df(split, space=space)
        if not _ok(res):
            logger.warning(
                "portfolio_df(%s, space=%s) failed: %s",
                split, space, res.error,  # type: ignore[union-attr]
            )
            return None
        df = res.data.copy()  # type: ignore[union-attr]
    else:
        per_cluster = _per_cluster_frame(backend, split, filters, space)
        if per_cluster is None or per_cluster.empty:
            return None
        df = _aggregate_clusters_to_portfolio(per_cluster)

    if df is None or df.empty:
        return None
    return _apply_date_filter(df, filters)


def _per_cluster_frame(
    backend: "RadeBackend",
    split:   str,
    filters: EvaluationFilters,
    space:   str = "scaled",
) -> Optional[pd.DataFrame]:
    """Cluster-level timeseries joined with cluster attributes.

    Always goes through the cluster endpoint even when no filters are
    active because the grouped render needs per-cluster residuals to
    populate the leaderboard.  ``space`` (Phase 3.4) selects the PnL
    units of the predictions / targets columns.

    Returns ``None`` on backend failure; never raises.
    """
    res_clusters = backend.clusters_df()
    if not _ok(res_clusters):
        logger.warning("clusters_df failed: %s", res_clusters.error)  # type: ignore[union-attr]
        return None
    clusters = res_clusters.data  # type: ignore[union-attr]
    if clusters is None or clusters.empty:
        return None

    masked = _apply_attribute_filters(clusters, filters)
    if masked.empty:
        # All clusters filtered away — return an empty frame with the
        # right columns so the caller can still fall through cleanly.
        return masked

    res_ts = backend.cluster_timeseries_df(split, space=space)
    if not _ok(res_ts):
        logger.warning(
            "cluster_timeseries_df(%s, space=%s) failed: %s",
            split, space, res_ts.error,  # type: ignore[union-attr]
        )
        return None
    ts = res_ts.data  # type: ignore[union-attr]
    if ts is None or ts.empty:
        return ts

    allowed_ids = set(masked["cluster_id"].astype(str))
    ts_filtered = ts[ts["cluster_id"].astype(str).isin(allowed_ids)].copy()

    # Left-join every cluster attribute column onto the timeseries so
    # the group-by / scatter colouring resolver can find a match
    # regardless of whether the ensemble's attributes use canonical
    # names (``product_code``) or short-form names (``product``).  We
    # union the canonical schema names with every candidate alias the
    # group-by resolver knows about, then keep only those actually
    # present in ``masked``.
    candidate_cols: set[str] = {"desk", "product_code", "currency_code", "asset_class"}
    for aliases in _GROUP_BY_CANDIDATES.values():
        candidate_cols.update(aliases)
    candidate_cols.discard("cluster_id")
    attr_cols = [c for c in candidate_cols if c in masked.columns]
    if attr_cols:
        ts_filtered = ts_filtered.merge(
            masked[["cluster_id", *attr_cols]],
            on="cluster_id",
            how="left",
        )

    return _apply_date_filter(ts_filtered, filters)


def _aggregate_clusters_to_portfolio(per_cluster: pd.DataFrame) -> pd.DataFrame:
    """Sum predictions / targets across clusters per scenario to
    reconstruct a portfolio-level timeseries from the per-cluster frame."""
    if per_cluster.empty:
        return per_cluster

    group_keys = [c for c in ("scenario_idx", "scenario_label") if c in per_cluster.columns]
    if not group_keys:
        group_keys = ["scenario_idx"] if "scenario_idx" in per_cluster.columns else ["scenario_label"]

    agg = (
        per_cluster.groupby(group_keys, as_index=False)
        .agg(
            predictions=("predictions", "sum"),
            targets=("targets", "sum"),
        )
    )
    agg["error"] = agg["predictions"] - agg["targets"]
    agg["abs_error"] = agg["error"].abs()
    agg["squared_error"] = agg["error"] ** 2
    return agg


def _apply_attribute_filters(
    clusters: pd.DataFrame,
    filters:  EvaluationFilters,
) -> pd.DataFrame:
    """Apply the non-date portions of the filter bar to a clusters_df.

    Each filter dimension resolves through ``_resolve_group_column`` so
    we honour whichever attribute column actually exists in this
    ensemble (``product_code`` vs ``product`` vs ``product_id`` …).
    """
    df = clusters
    for dim_key, selected in (
        ("asset_class", filters.asset_class),
        ("currency",    filters.currency),
        ("desk",        filters.desk),
        ("product",     filters.product),
    ):
        if not selected:
            continue
        column = _resolve_group_column(dim_key, df.columns)
        if column is None:
            continue
        df = df[df[column].astype(str).isin(selected)]
    return df


def _apply_date_filter(
    df:       pd.DataFrame,
    filters:  EvaluationFilters,
) -> pd.DataFrame:
    """Apply ``date_from`` / ``date_to`` against ``scenario_label``.

    ``scenario_label`` is an ISO-8601 string in every ensemble shipped
    to date, so lexical comparison is correct.  Falls through
    untouched on any DataFrame that lacks the column.
    """
    if "scenario_label" not in df.columns:
        return df
    out = df
    if filters.date_from:
        out = out[out["scenario_label"].astype(str) >= str(filters.date_from)]
    if filters.date_to:
        out = out[out["scenario_label"].astype(str) <= str(filters.date_to)]
    return out


def _only_date_filter(filters: EvaluationFilters) -> bool:
    """Is the only non-empty filter the date range?"""
    return (
        not filters.asset_class
        and not filters.currency
        and not filters.desk
        and not filters.product
        and (filters.date_from is not None or filters.date_to is not None)
    )


def _ok(res: "BackendResult[Any]") -> bool:
    return bool(getattr(res, "ok", False)) and getattr(res, "data", None) is not None


# ═════════════════════════════════════════════════════════════════════
# KPI + leaderboard computations
# ═════════════════════════════════════════════════════════════════════


def _kpi_strings(
    portfolio_df: pd.DataFrame,
    space:        str = "scaled",
) -> Tuple[str, str, str, str]:
    """Compute display strings for the four aggregate KPIs.

    Phase 3.4 — ``space`` drives the MAE / RMSE digit count.  Scaled
    values are dimensionless and tiny, so 4 decimals reads well;
    original-space values are notional currency amounts where 0
    decimals with thousands separators is the right choice.  Hit-rate
    and coverage are dimensionless percentages — unaffected by
    ``space``.
    """
    if portfolio_df is None or portfolio_df.empty:
        return _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER, _PLACEHOLDER

    abs_err = (
        portfolio_df["abs_error"]
        if "abs_error" in portfolio_df.columns
        else (portfolio_df["predictions"] - portfolio_df["targets"]).abs()
    ).astype(float)
    sq_err = (
        portfolio_df["squared_error"]
        if "squared_error" in portfolio_df.columns
        else (portfolio_df["predictions"] - portfolio_df["targets"]) ** 2
    ).astype(float)

    mae_val = float(abs_err.mean()) if not abs_err.empty else float("nan")
    rmse_val = float(np.sqrt(sq_err.mean())) if not sq_err.empty else float("nan")

    hit_mask = np.sign(portfolio_df["predictions"]) == np.sign(portfolio_df["targets"])
    n_total = int(hit_mask.size)
    hit_pct = float(hit_mask.mean() * 100.0) if n_total > 0 else float("nan")

    # Coverage = % of scenarios with finite predictions + targets values.
    finite_mask = (
        portfolio_df["predictions"].notna()
        & portfolio_df["targets"].notna()
        & np.isfinite(portfolio_df["predictions"])
        & np.isfinite(portfolio_df["targets"])
    )
    coverage_pct = float(finite_mask.mean() * 100.0) if n_total > 0 else float("nan")

    pnl_digits = 0 if space == "original" else 4
    return (
        _fmt_num(mae_val, digits=pnl_digits),
        _fmt_num(rmse_val, digits=pnl_digits),
        _fmt_pct(hit_pct),
        _fmt_pct(coverage_pct),
    )


def _leaderboard_rows(
    per_cluster:   pd.DataFrame,
    *,
    group_column:  str,
) -> List[Dict[str, Any]]:
    """Aggregate per-cluster residuals into per-group leaderboard rows.

    Contribution is each group's share of the *portfolio absolute
    error*, which is a more faithful attribution than share of row
    count when groups differ in size.
    """
    if per_cluster is None or per_cluster.empty or group_column not in per_cluster.columns:
        return []

    df = per_cluster.copy()
    df["_abs_err"] = (df["predictions"] - df["targets"]).abs()
    df["_sq_err"]  = (df["predictions"] - df["targets"]) ** 2
    df["_hit"]     = (np.sign(df["predictions"]) == np.sign(df["targets"])).astype(float)

    agg = df.groupby(group_column, dropna=False).agg(
        mae=("_abs_err", "mean"),
        rmse_sq=("_sq_err", "mean"),
        hit_rate=("_hit", "mean"),
        contribution_total=("_abs_err", "sum"),
        n_clusters=("cluster_id", "nunique"),
    ).reset_index()

    total_abs = float(agg["contribution_total"].sum())
    if total_abs <= 0:
        agg["contribution"] = 0.0
    else:
        agg["contribution"] = agg["contribution_total"] / total_abs * 100.0

    agg["rmse"] = np.sqrt(agg["rmse_sq"].astype(float))
    agg["hit_rate"] = agg["hit_rate"].astype(float) * 100.0

    # Sort by contribution descending — traders see the biggest movers
    # first and can still resort via the grid's column headers.
    agg = agg.sort_values("contribution", ascending=False)

    rows: List[Dict[str, Any]] = []
    for _, r in agg.iterrows():
        rows.append(
            {
                "group_label":   "" if pd.isna(r[group_column]) else str(r[group_column]),
                "mae":           _round(r["mae"], 4),
                "rmse":          _round(r["rmse"], 4),
                "hit_rate":      _round(r["hit_rate"], 1),
                "contribution":  _round(r["contribution"], 1),
                "n_clusters":    int(r["n_clusters"]),
            }
        )
    return rows


def _leaderboard_column_defs(*, group_by: Optional[str]) -> List[Dict[str, Any]]:
    """Dynamic column headers — the first column's label tracks group-by."""
    first_label = _GROUP_BY_LABEL.get(group_by or "", "Break-down") if group_by else "Break-down"
    return [
        {"field": "group_label", "headerName": first_label, "flex": 2, "minWidth": 140},
        {
            "field": "mae",
            "headerName": "MAE",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.4f')(params.value)"},
        },
        {
            "field": "rmse",
            "headerName": "RMSE",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.4f')(params.value)"},
        },
        {
            "field": "hit_rate",
            "headerName": "Hit %",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.1f')(params.value)"},
        },
        {
            "field": "contribution",
            "headerName": "Contribution %",
            "flex": 1,
            "type": "numericColumn",
            "valueFormatter": {"function": "d3.format(',.1f')(params.value)"},
        },
        {"field": "n_clusters", "headerName": "Clusters", "flex": 1, "type": "numericColumn"},
    ]


def _leaderboard_header_children(
    *,
    group_by:   Optional[str],
    n_groups:   int,
    n_clusters: int,
) -> List[Any]:
    """Reactive title + caption for the leaderboard card."""
    from dash import html  # local import — keeps the module tree shallow

    if group_by is None:
        title = "Leaderboard"
        subtitle = "Pick a break-down dimension to compare contributors."
    else:
        title = f"Leaderboard — by {_GROUP_BY_LABEL.get(group_by, group_by)}"
        subtitle = (
            f"{n_groups} group{'' if n_groups == 1 else 's'} "
            f"covering {n_clusters} cluster{'' if n_clusters == 1 else 's'} "
            "under the active filter set."
        )

    return [
        html.Div(
            className="flex flex-col",
            children=[
                html.Div(title,    className="text-sm font-semibold text-slate-200"),
                html.Div(subtitle, className="text-xs text-slate-500"),
            ],
        ),
    ]


# ─────────────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────────────


def _fmt_num(value: Any, *, digits: int = 2) -> str:
    if value is None:
        return _PLACEHOLDER
    try:
        val = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val) or not np.isfinite(val):
        return _PLACEHOLDER
    return f"{val:,.{digits}f}"


def _fmt_pct(value: Any) -> str:
    if value is None:
        return _PLACEHOLDER
    try:
        val = float(value)
    except (TypeError, ValueError):
        return _PLACEHOLDER
    if pd.isna(val) or not np.isfinite(val):
        return _PLACEHOLDER
    return f"{val:,.1f}%"


def _round(value: Any, digits: int) -> Optional[float]:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(val) or not np.isfinite(val):
        return None
    return round(val, digits)


# ─────────────────────────────────────────────────────────────────────
# Silence IDE "unused" lints on helpers imported only for side-effects
# ─────────────────────────────────────────────────────────────────────

_ = no_update   # re-exported to satisfy the "imported but unused" linter
                # — some future callbacks in this module may need it.


__all__ = ["register"]
