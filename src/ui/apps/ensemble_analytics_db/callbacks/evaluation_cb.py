"""
Callbacks for Tab 2 — Evaluation.

All data comes from ``get_backend()`` — no ``EnsembleSession`` access.
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

    # ── Filter visibility & options ───────────────────────────────
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
        from src.ui.apps.ensemble_analytics_db.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend

        catalogue = get_trade_catalogue()
        backend = get_backend()

        def _opts(logical_col):
            actual = EVAL_GROUP_COLUMNS.get(logical_col, logical_col)
            if catalogue is not None and actual in catalogue.columns:
                vals = sorted(catalogue[actual].dropna().unique().tolist())
                return [{"label": v, "value": v} for v in vals]
            return []

        cluster_opts, default_cluster = [], None
        cluster_ids = backend.get_cluster_ids()
        cluster_attrs = backend.get_cluster_attributes()
        for cid in cluster_ids:
            if cluster_attrs and cid in cluster_attrs:
                parts = [f"{k}={v}" for k, v in cluster_attrs[cid].items() if v is not None]
                label = f"{cid}  ({', '.join(parts)})" if parts else cid
            else:
                label = cid
            cluster_opts.append({"label": label, "value": cid})
        default_cluster = cluster_ids[0] if cluster_ids else None

        return _opts("desk"), _opts("product_type"), _opts("ccy"), cluster_opts, default_cluster

    # ── Portfolio sub-tab (pre-computed) ──────────────────────────
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
        if sub_tab != EVAL_SUB_PORTFOLIO:
            return no_update, no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.prediction_store import (
            get_portfolio_summary,
        )

        portfolio = get_portfolio_summary(split)
        if portfolio is None:
            msg = html.Div("No prediction data available for this split.")
            return msg, msg, msg, msg, msg

        portfolio_preds = portfolio["predictions"]
        portfolio_targets = portfolio["targets"]

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

    # ── Generic group-by sub-tab builder (pre-computed) ───────────
    def _build_group_view(
        split: str,
        group_col: str,
        selected_values: Optional[List[str]],
        id_prefix: str,
    ):
        """Build group sub-tab using pre-computed cluster summaries."""
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_summary
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS, TEXT_SECONDARY

        backend = get_backend()
        cluster_attrs = backend.get_cluster_attributes()
        cluster_data = get_cluster_summary(split)

        if not cluster_data:
            msg = html.Div("No data available.")
            return msg, msg, msg, msg

        groups: dict = {}
        for cid, ca in cluster_attrs.items():
            val = ca.get(group_col)
            if val is not None:
                groups.setdefault(str(val), []).append(cid)

        if selected_values:
            groups = {g: cids for g, cids in groups.items() if g in selected_values}

        group_preds_dict = {}
        group_targets_dict = {}
        group_residuals = {}
        table_rows = []

        for grp in sorted(groups.keys()):
            cids = groups[grp]
            combined_pred = None
            combined_tgt = None
            n_trades = 0
            for cid in cids:
                cd = cluster_data.get(cid)
                if cd is None:
                    continue
                p = cd["predictions"]
                t = cd["targets"]
                combined_pred = p if combined_pred is None else combined_pred + p
                combined_tgt = t if combined_tgt is None else combined_tgt + t
                n_trades += cluster_attrs.get(cid, {}).get("n_trades", 0)

            if combined_pred is None:
                continue

            group_preds_dict[grp] = combined_pred
            group_targets_dict[grp] = combined_tgt
            residual = combined_pred - combined_tgt
            group_residuals[grp] = residual
            table_rows.append({
                "group": grp,
                "n_trades": int(n_trades),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
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
        attr_key = EVAL_GROUP_COLUMNS.get("desk", "desk")
        return _build_group_view(split, attr_key, selected_desks, "eval-desk")

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
        attr_key = EVAL_GROUP_COLUMNS.get("product_type", "product")
        return _build_group_view(split, attr_key, selected_products, "eval-product")

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

        attr_key = EVAL_GROUP_COLUMNS.get("ccy", "ccy")
        ts_fig, box_fig, scatter_grid, table = _build_group_view(
            split, attr_key, selected_ccys, "eval-ccy",
        )

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_group_correlations

        corr_fig = html.Div("Insufficient data for correlation.")
        correlations = get_group_correlations(split)
        ccy_corr = correlations.get(attr_key)

        if ccy_corr and "columns" in ccy_corr and "values" in ccy_corr:
            cols = ccy_corr["columns"]
            vals = np.array(ccy_corr["values"])
            fig = go.Figure(go.Heatmap(
                z=vals, x=cols, y=cols,
                colorscale="RdBu_r", zmid=0,
                text=np.round(vals, 2).astype(str),
                texttemplate="%{text}",
            ))
            fig.update_layout(title="Cross-CCY Residual Correlation", height=400)
            corr_fig = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return ts_fig, box_fig, scatter_grid, corr_fig, table

    # ── By Cluster (on-demand single-cluster load) ────────────────
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
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_predictions
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend

        backend = get_backend()
        cluster_id_str = str(cluster_id)
        data = get_cluster_predictions(cluster_id_str, split)
        if data is None:
            msg = html.Div("No prediction data for this cluster.")
            return msg, msg, msg, msg, msg

        preds = data["predictions"]
        targets = data["targets"]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            targets = targets.reshape(-1, 1)

        trade_ids = [str(t) for t in backend.get_cluster_mapping().get(cluster_id_str, [])]
        trade_ids = trade_ids[:preds.shape[1]]

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
            if j < preds.shape[1]:
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
            if j < preds.shape[1]:
                r = preds[:, j] - targets[:, j]
                rows.append({
                    "trade_id": tid,
                    "mae": float(np.mean(np.abs(r))),
                    "rmse": float(np.sqrt(np.mean(r ** 2))),
                    "max_ae": float(np.max(np.abs(r))),
                })
        table = metric_table(col_defs, rows, "eval-cluster-trade-metrics-table", height="350px")

        return scatter, ts, violin, heatmap, table
