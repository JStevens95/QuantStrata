"""
Callbacks for Tab 2 — Evaluation (all 5 sub-tabs).

Handles sub-tab routing and per-sub-tab data loading / figure building.
Uses the shared slicing pattern: catalogue filter → column indices →
store slice → aggregate → plot.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    EVAL_SUB_PORTFOLIO,
    EVAL_SUB_DESK,
    EVAL_SUB_PRODUCT,
    EVAL_SUB_CCY,
    EVAL_SUB_CLUSTER,
    EVAL_GROUP_COLUMNS,
    METRIC_DISPLAY_NAMES,
)
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.timeseries import pnl_timeseries, overlaid_group_timeseries
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram, violin_overlay
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.tables import percentile_table_data, worst_scenarios_data


def register(app):
    """Register Evaluation tab callbacks on *app*."""

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("eval-sub-tab-content", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_eval_sub_tab(sub_tab: str):
        """Swap sub-tab content based on selection."""
        if sub_tab == EVAL_SUB_PORTFOLIO:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.portfolio import layout
            return layout()
        elif sub_tab == EVAL_SUB_DESK:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_desk import layout
            return layout()
        elif sub_tab == EVAL_SUB_PRODUCT:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_product import layout
            return layout()
        elif sub_tab == EVAL_SUB_CCY:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_ccy import layout
            return layout()
        elif sub_tab == EVAL_SUB_CLUSTER:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_cluster import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── Filter visibility & options ─────────────────────────────
    _FILTER_ROW = {"display": "flex", "alignItems": "center", "marginRight": "20px"}
    _HIDDEN = {"display": "none"}

    @app.callback(
        Output("eval-filter-desk", "style"),
        Output("eval-filter-product", "style"),
        Output("eval-filter-ccy", "style"),
        Output("eval-filter-cluster", "style"),
        Input("eval-sub-tabs", "value"),
    )
    def toggle_filter_visibility(sub_tab):
        return (
            _FILTER_ROW if sub_tab == EVAL_SUB_DESK else _HIDDEN,
            _FILTER_ROW if sub_tab == EVAL_SUB_PRODUCT else _HIDDEN,
            _FILTER_ROW if sub_tab == EVAL_SUB_CCY else _HIDDEN,
            _FILTER_ROW if sub_tab == EVAL_SUB_CLUSTER else _HIDDEN,
        )

    @app.callback(
        Output("eval-desk-filter-desk", "options"),
        Output("eval-product-filter-product_type", "options"),
        Output("eval-ccy-filter-ccy", "options"),
        Output("eval-cluster-cluster-dropdown", "options"),
        Output("eval-cluster-cluster-dropdown", "value"),
        Input("eval-sub-tabs", "value"),
    )
    def populate_filter_options(_sub_tab):
        """Populate all filter dropdown options from the trade catalogue."""
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session

        catalogue = get_trade_catalogue()
        session = get_session()

        def _opts(logical_col):
            actual = EVAL_GROUP_COLUMNS.get(logical_col, logical_col)
            if catalogue is not None and actual in catalogue.columns:
                vals = sorted(catalogue[actual].dropna().unique().tolist())
                return [{"label": v, "value": v} for v in vals]
            return []

        cluster_opts, default_cluster = [], None
        if session and session.config:
            attrs = session.cluster_attributes
            for cid in session.config.cluster_ids:
                if attrs and cid in attrs:
                    parts = [f"{k}={v}" for k, v in attrs[cid].items() if v is not None]
                    label = f"{cid}  ({', '.join(parts)})" if parts else cid
                else:
                    label = cid
                cluster_opts.append({"label": label, "value": cid})
            default_cluster = session.config.cluster_ids[0] if session.config.cluster_ids else None

        return _opts("desk"), _opts("product_type"), _opts("ccy"), cluster_opts, default_cluster

    # ── Portfolio sub-tab ─────────────────────────────────────────
    @app.callback(
        Output("eval-portfolio-ts", "children"),
        Output("eval-portfolio-scatter", "children"),
        Output("eval-portfolio-residual", "children"),
        Output("eval-portfolio-percentile", "children"),
        Output("eval-portfolio-worst", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
    )
    def update_portfolio(split: str, sub_tab: str):
        """Build Portfolio sub-tab visuals."""
        if sub_tab != EVAL_SUB_PORTFOLIO:
            return no_update, no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store

        store = get_prediction_store(split)
        if store is None:
            msg = html.Div("No prediction data available for this split.")
            return msg, msg, msg, msg, msg

        portfolio_preds = store.predictions.sum(axis=1)
        portfolio_targets = store.targets.sum(axis=1)

        ts_fig = dcc.Graph(
            figure=pnl_timeseries(portfolio_preds, portfolio_targets,
                                  title=f"Portfolio PnL — {split.capitalize()}"),
            config={"displayModeBar": False},
        )
        scatter_fig = dcc.Graph(
            figure=pred_vs_target_scatter(portfolio_preds, portfolio_targets,
                                          title=f"Pred vs Target — {split.capitalize()}"),
            config={"displayModeBar": False},
        )
        residual_fig = dcc.Graph(
            figure=residual_histogram(portfolio_preds, portfolio_targets,
                                      title=f"Residual Distribution — {split.capitalize()}"),
            config={"displayModeBar": False},
        )

        pct_cols, pct_rows = percentile_table_data(portfolio_preds, portfolio_targets)
        pct_table = metric_table(pct_cols, pct_rows, "eval-portfolio-pct-table", height="220px")

        worst_cols, worst_rows = worst_scenarios_data(portfolio_preds, portfolio_targets)
        worst_table = metric_table(worst_cols, worst_rows, "eval-portfolio-worst-table", height="400px")

        return ts_fig, scatter_fig, residual_fig, pct_table, worst_table

    # ── Generic group-by sub-tab builder ──────────────────────────
    def _build_group_view(
        split: str,
        group_col: str,
        selected_values: Optional[List[str]],
        id_prefix: str,
    ):
        """
        Shared logic for By Desk / By Product / By CCY sub-tabs.

        Filters the trade catalogue by *group_col*, slices the prediction
        store, aggregates per group, and builds the standard chart set
        including a scatter grid (small multiples).

        Returns (timeseries, boxplot, scatter_grid, table).
        """
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS, TEXT_SECONDARY

        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        if store is None or catalogue is None or catalogue.empty:
            msg = html.Div("No data available.")
            return msg, msg, msg, msg

        if group_col not in catalogue.columns:
            msg = html.Div(f"Column '{group_col}' not found in trade catalogue.")
            return msg, msg, msg, msg

        groups = sorted(catalogue[group_col].dropna().unique().tolist())
        if selected_values:
            groups = [g for g in groups if g in selected_values]

        group_preds_dict = {}
        group_targets_dict = {}
        group_residuals = {}
        table_rows = []
        for grp in groups:
            mask = catalogue[group_col] == grp
            col_idx = np.where(mask.values)[0]
            if len(col_idx) == 0:
                continue
            p = store.predictions[:, col_idx].sum(axis=1)
            t = store.targets[:, col_idx].sum(axis=1)
            group_preds_dict[grp] = p
            group_targets_dict[grp] = t
            group_residuals[grp] = p - t
            table_rows.append({
                "group": grp,
                "n_trades": int(len(col_idx)),
                "mae": float(np.mean(np.abs(p - t))),
                "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            })

        ts_fig = dcc.Graph(
            figure=overlaid_group_timeseries(
                group_preds_dict,
                title=f"PnL by {group_col.replace('_', ' ').title()} — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )
        box_fig = dcc.Graph(
            figure=violin_overlay(
                group_residuals,
                title=f"Residual Distribution by {group_col.replace('_', ' ').title()}",
            ),
            config={"displayModeBar": False},
        )

        # Scatter grid: small multiples (G6)
        n_groups = len(group_preds_dict)
        scatter_grid = html.Div()
        if n_groups > 0:
            import plotly.graph_objects as go
            ncols = min(n_groups, 4)
            nrows = (n_groups + ncols - 1) // ncols
            titles = [t[:25] + "..." if len(t) > 25 else t for t in group_preds_dict.keys()]
            fig = make_subplots(rows=nrows, cols=ncols,
                                subplot_titles=titles,
                                vertical_spacing=0.12,
                                horizontal_spacing=0.08)
            for i, (grp, p) in enumerate(group_preds_dict.items()):
                t = group_targets_dict[grp]
                r, c = i // ncols + 1, i % ncols + 1
                fig.add_trace(go.Scattergl(
                    x=t, y=p, mode="markers",
                    marker=dict(size=2, color=CHART_COLORS[i % len(CHART_COLORS)], opacity=0.5),
                    showlegend=False,
                ), row=r, col=c)
                vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
                fig.add_trace(go.Scattergl(
                    x=[vmin, vmax], y=[vmin, vmax], mode="lines",
                    line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
                    showlegend=False,
                ), row=r, col=c)
            fig.update_layout(height=280 * nrows, title="Pred vs Target — Small Multiples")
            scatter_grid = dcc.Graph(figure=fig, config={"displayModeBar": False})

        col_defs = [
            {"field": "group", "headerName": group_col.replace("_", " ").title()},
            {"field": "n_trades", "headerName": "# Trades"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "rmse", "headerName": "RMSE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        table = metric_table(col_defs, table_rows, f"{id_prefix}-metrics-table", height="300px")

        return ts_fig, box_fig, scatter_grid, table

    # ── By Desk ───────────────────────────────────────────────────
    @app.callback(
        Output("eval-desk-timeseries", "children"),
        Output("eval-desk-boxplot", "children"),
        Output("eval-desk-scatter-grid", "children"),
        Output("eval-desk-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-desk-filter-desk", "value"),
    )
    def update_desk(split, sub_tab, selected_desks):
        if sub_tab != EVAL_SUB_DESK:
            return no_update, no_update, no_update, no_update
        return _build_group_view(split, EVAL_GROUP_COLUMNS.get("desk", "desk"), selected_desks, "eval-desk")

    # ── By Product ────────────────────────────────────────────────
    @app.callback(
        Output("eval-product-timeseries", "children"),
        Output("eval-product-boxplot", "children"),
        Output("eval-product-scatter-grid", "children"),
        Output("eval-product-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-product-filter-product_type", "value"),
    )
    def update_product(split, sub_tab, selected_products):
        if sub_tab != EVAL_SUB_PRODUCT:
            return no_update, no_update, no_update, no_update
        return _build_group_view(split, EVAL_GROUP_COLUMNS.get("product_type", "product_type"), selected_products, "eval-product")

    # ── By CCY ────────────────────────────────────────────────────
    @app.callback(
        Output("eval-ccy-timeseries", "children"),
        Output("eval-ccy-boxplot", "children"),
        Output("eval-ccy-scatter-grid", "children"),
        Output("eval-ccy-correlation", "children"),
        Output("eval-ccy-table-container", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-ccy-filter-ccy", "value"),
    )
    def update_ccy(split, sub_tab, selected_ccys):
        if sub_tab != EVAL_SUB_CCY:
            return no_update, no_update, no_update, no_update, no_update

        ccy_col = EVAL_GROUP_COLUMNS.get("ccy", "ccy")
        ts_fig, box_fig, scatter_grid, table = _build_group_view(split, ccy_col, selected_ccys, "eval-ccy")

        # Cross-currency residual correlation heatmap
        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY

        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        corr_fig = html.Div("Insufficient data for correlation.")

        if store is not None and ccy_col in catalogue.columns:
            ccys = sorted(catalogue[ccy_col].dropna().unique().tolist())
            if selected_ccys:
                ccys = [c for c in ccys if c in selected_ccys]

            residual_matrix = {}
            for c in ccys:
                mask = catalogue[ccy_col] == c
                idx = np.where(mask.values)[0]
                if len(idx) > 0:
                    p = store.predictions[:, idx].sum(axis=1)
                    t = store.targets[:, idx].sum(axis=1)
                    residual_matrix[c] = p - t

            if len(residual_matrix) > 1:
                import pandas as pd
                df = pd.DataFrame(residual_matrix)
                corr = df.corr()
                fig = go.Figure(go.Heatmap(
                    z=corr.values,
                    x=corr.columns.tolist(),
                    y=corr.index.tolist(),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=np.round(corr.values, 2).astype(str),
                    texttemplate="%{text}",
                ))
                fig.update_layout(title="Cross-CCY Residual Correlation", height=400)
                corr_fig = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return ts_fig, box_fig, scatter_grid, corr_fig, table

    # ── By Cluster ────────────────────────────────────────────────
    @app.callback(
        Output("eval-cluster-scatter", "children"),
        Output("eval-cluster-timeseries", "children"),
        Output("eval-cluster-violin", "children"),
        Output("eval-cluster-heatmap", "children"),
        Output("eval-cluster-trade-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-cluster-cluster-dropdown", "value"),
    )
    def update_by_cluster(split, sub_tab, cluster_id):
        _nu5 = (no_update,) * 5
        if sub_tab != EVAL_SUB_CLUSTER:
            return _nu5

        if isinstance(cluster_id, list):
            cluster_id = cluster_id[0] if cluster_id else None
        if not cluster_id and cluster_id != 0:
            return _nu5

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue

        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        if store is None or catalogue is None or catalogue.empty:
            msg = html.Div("No prediction data.")
            return msg, msg, msg, msg, msg

        cluster_id_str = str(cluster_id)
        mask = catalogue["cluster_id"] == cluster_id_str
        col_idx = np.where(mask.values)[0]
        if len(col_idx) == 0:
            msg = html.Div(f"No trades found for cluster {cluster_id}.")
            return msg, msg, msg, msg, msg

        preds = store.predictions[:, col_idx]
        targets = store.targets[:, col_idx]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            targets = targets.reshape(-1, 1)
        trade_ids = [str(store.trade_ids[int(i)]) for i in col_idx]

        cluster_pred = preds.sum(axis=1)
        cluster_target = targets.sum(axis=1)

        scatter = dcc.Graph(
            figure=pred_vs_target_scatter(
                cluster_pred, cluster_target,
                title=f"Pred vs Target — {cluster_id} ({split.capitalize()})",
            ),
            config={"displayModeBar": False},
        )

        ts = dcc.Graph(
            figure=pnl_timeseries(
                cluster_pred, cluster_target,
                title=f"PnL Timeseries — {cluster_id} ({split.capitalize()})",
            ),
            config={"displayModeBar": False},
        )

        trade_residuals = {}
        for j, tid in enumerate(trade_ids):
            trade_residuals[tid] = preds[:, j] - targets[:, j]
        violin_fig = violin_overlay(trade_residuals, title="Per-Trade Residual Distribution")
        violin = dcc.Graph(figure=violin_fig, config={"displayModeBar": False})

        residuals = preds - targets
        max_scenarios = 500
        heatmap_data = residuals[:max_scenarios] if residuals.shape[0] > max_scenarios else residuals
        hm_fig = go.Figure(go.Heatmap(
            z=heatmap_data, x=trade_ids,
            y=list(range(heatmap_data.shape[0])),
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="Residual"),
        ))
        hm_fig.update_layout(
            title=f"Per-Trade Residual Heatmap — {cluster_id}",
            xaxis_title="Trade", yaxis_title="Scenario",
            height=max(300, min(600, 2 * heatmap_data.shape[0])),
        )
        heatmap = dcc.Graph(figure=hm_fig, config={"displayModeBar": False})

        col_defs = [
            {"field": "trade_id", "headerName": "Trade ID"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "rmse", "headerName": "RMSE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "max_ae", "headerName": "Max AE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        rows = []
        for j, tid in enumerate(trade_ids):
            r = preds[:, j] - targets[:, j]
            rows.append({
                "trade_id": tid,
                "mae": float(np.mean(np.abs(r))),
                "rmse": float(np.sqrt(np.mean(r ** 2))),
                "max_ae": float(np.max(np.abs(r))),
            })
        table = metric_table(col_defs, rows, "eval-cluster-trade-metrics-table", height="350px")

        return scatter, ts, violin, heatmap, table
