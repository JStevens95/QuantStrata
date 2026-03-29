"""
Callbacks for Tab 4 — Market Data (all 4 sub-tabs).

Handles sub-tab routing, cluster selector rendering, and per-sub-tab
data loading from ``load_cluster_market_data``.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    MD_SUB_RF_SUMMARY,
    MD_SUB_SHOCK_EXPLORER,
    MD_SUB_SCENARIO_HEATMAP,
    MD_SUB_DISTRIBUTION,
)
from src.ui.apps.ensemble_analytics.components.cluster_selector import cluster_selector
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.heatmaps import rf_scenario_heatmap
from src.ui.apps.ensemble_analytics.figures.distributions import violin_overlay, qq_plot
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def register(app):
    """Register Market Data tab callbacks on *app*."""

    # ── Cluster selector ──────────────────────────────────────────
    @app.callback(
        Output("md-cluster-selector-container", "children"),
        Input("main-tabs", "value"),
    )
    def render_md_cluster_selector(tab):
        if tab != "tab-market-data":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        return cluster_selector(
            session.config.cluster_ids,
            session.cluster_attributes,
            id_prefix="md",
        )

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("md-sub-tab-content", "children"),
        Input("md-sub-tabs", "value"),
    )
    def render_md_sub_tab(sub_tab):
        if sub_tab == MD_SUB_RF_SUMMARY:
            from src.ui.apps.ensemble_analytics.tabs.market_data.rf_summary import layout
            return layout()
        elif sub_tab == MD_SUB_SHOCK_EXPLORER:
            from src.ui.apps.ensemble_analytics.tabs.market_data.shock_explorer import layout
            return layout()
        elif sub_tab == MD_SUB_SCENARIO_HEATMAP:
            from src.ui.apps.ensemble_analytics.tabs.market_data.scenario_heatmap import layout
            return layout()
        elif sub_tab == MD_SUB_DISTRIBUTION:
            from src.ui.apps.ensemble_analytics.tabs.market_data.distribution import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── RF Summary ────────────────────────────────────────────────
    @app.callback(
        Output("md-rf-summary-table", "children"),
        Output("md-rf-coverage-heatmap", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_rf_summary(cluster_id, sub_tab):
        if sub_tab != MD_SUB_RF_SUMMARY or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session

        session = get_session()
        all_rfs = {}

        for cid in session.config.cluster_ids:
            mdata = get_market_data(cid)
            for asset, rfs in mdata.items():
                for rf_name in rfs:
                    all_rfs.setdefault(rf_name, set()).add(cid)

        col_defs = [
            {"field": "rf_name", "headerName": "Risk Factor"},
            {"field": "n_clusters", "headerName": "# Clusters"},
            {"field": "clusters", "headerName": "Clusters"},
        ]
        rows = [
            {
                "rf_name": rf,
                "n_clusters": len(cids),
                "clusters": ", ".join(sorted(cids)),
            }
            for rf, cids in sorted(all_rfs.items())
        ]
        table = metric_table(col_defs, rows, "md-rf-inventory-table", height="400px")

        import plotly.graph_objects as go
        cluster_ids = session.config.cluster_ids
        rf_names = sorted(all_rfs.keys())
        coverage = np.zeros((len(rf_names), len(cluster_ids)))
        for i, rf in enumerate(rf_names):
            for j, cid in enumerate(cluster_ids):
                if cid in all_rfs.get(rf, set()):
                    coverage[i, j] = 1.0

        fig = go.Figure(go.Heatmap(
            z=coverage, x=cluster_ids, y=rf_names,
            colorscale=[[0, "#161b22"], [1, "#58a6ff"]],
            showscale=False,
        ))
        fig.update_layout(title="RF Coverage Matrix", height=max(300, 18 * len(rf_names) + 100))
        heatmap = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return table, heatmap

    # ── Shock Explorer ────────────────────────────────────────────
    @app.callback(
        Output("md-shock-asset-selector", "children"),
        Output("md-shock-rf-selector", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def render_shock_selectors(cluster_id, sub_tab):
        if sub_tab != MD_SUB_SHOCK_EXPLORER or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        asset_names = sorted(mdata.keys())
        all_rfs = sorted({rf for rfs in mdata.values() for rf in rfs})

        asset_dd = html.Div([
            html.Label("Asset:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="md-shock-asset-dd",
                options=[{"label": a, "value": a} for a in asset_names],
                value=asset_names[0] if asset_names else None,
                clearable=False, style={"width": "300px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"})

        rf_dd = html.Div([
            html.Label("RF:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Dropdown(
                id="md-shock-rf-dd",
                options=[{"label": r, "value": r} for r in all_rfs],
                value=all_rfs[0] if all_rfs else None,
                clearable=False, style={"width": "300px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"})

        return asset_dd, rf_dd

    @app.callback(
        Output("md-shock-timeseries", "children"),
        Output("md-shock-distribution", "children"),
        Output("md-shock-stats", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-shock-asset-dd", "value"),
        Input("md-shock-rf-dd", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_shock_explorer(cluster_id, asset, rf, sub_tab):
        if sub_tab != MD_SUB_SHOCK_EXPLORER or not all([cluster_id, asset, rf]):
            return no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        shocks = mdata.get(asset, {}).get(rf)
        if shocks is None:
            msg = html.Div("No shock data for this asset/RF combination.")
            return msg, msg, msg

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE

        ts = go.Figure(go.Scattergl(
            x=np.arange(len(shocks)), y=shocks, mode="lines",
            line=dict(color=ACCENT_BLUE, width=1.5),
        ))
        ts.update_layout(title=f"{asset} — {rf} Shocks", xaxis_title="Scenario", yaxis_title="Shock", height=350)

        hist = go.Figure(go.Histogram(x=shocks, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8))
        hist.update_layout(title=f"{rf} Shock Distribution", height=350)

        stats_content = html.Div([
            html.Span(f"Mean: {shocks.mean():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Std: {shocks.std():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Min: {shocks.min():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Max: {shocks.max():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"N: {len(shocks)}", style={"fontSize": "13px"}),
        ])

        return (
            dcc.Graph(figure=ts, config={"displayModeBar": False}),
            dcc.Graph(figure=hist, config={"displayModeBar": False}),
            stats_content,
        )

    # ── Scenario Heatmap ──────────────────────────────────────────
    @app.callback(
        Output("md-heatmap-container", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_scenario_heatmap(cluster_id, sub_tab):
        if sub_tab != MD_SUB_SCENARIO_HEATMAP or not cluster_id:
            return no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        if not mdata:
            return html.Div("No market data available.")

        all_rfs = {}
        for asset, rfs in mdata.items():
            for rf_name, arr in rfs.items():
                all_rfs[f"{asset}/{rf_name}"] = arr

        if not all_rfs:
            return html.Div("No RF shocks found.")

        rf_names = sorted(all_rfs.keys())
        n_scenarios = min(arr.shape[0] for arr in all_rfs.values())
        matrix = np.column_stack([all_rfs[rf][:n_scenarios] for rf in rf_names])

        fig = rf_scenario_heatmap(rf_names, matrix, title=f"Cluster {cluster_id} — RF × Scenario")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    # ── Distribution ──────────────────────────────────────────────
    @app.callback(
        Output("md-dist-violin", "children"),
        Output("md-dist-qq", "children"),
        Output("md-dist-corr-heatmap", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_distribution(cluster_id, sub_tab):
        if sub_tab != MD_SUB_DISTRIBUTION or not cluster_id:
            return no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        if not mdata:
            msg = html.Div("No market data available.")
            return msg, msg, msg

        rf_arrays = {}
        for asset, rfs in mdata.items():
            for rf_name, arr in rfs.items():
                rf_arrays[f"{asset}/{rf_name}"] = arr

        if not rf_arrays:
            msg = html.Div("No RF shocks found.")
            return msg, msg, msg

        # Violin of first 10 RFs
        subset = dict(list(rf_arrays.items())[:10])
        violin_fig = dcc.Graph(
            figure=violin_overlay(subset, title="RF Shock Distributions"),
            config={"displayModeBar": False},
        )

        # QQ plot of the first RF
        first_rf = next(iter(rf_arrays.values()))
        qq_fig = dcc.Graph(
            figure=qq_plot(first_rf, title=f"QQ — {next(iter(rf_arrays.keys()))}"),
            config={"displayModeBar": False},
        )

        # Correlation heatmap
        import plotly.graph_objects as go
        import pandas as pd
        df = pd.DataFrame({k: v[:min(len(v) for v in rf_arrays.values())]
                           for k, v in list(rf_arrays.items())[:20]})
        corr = df.corr()
        heatmap_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
        ))
        heatmap_fig.update_layout(title="RF Correlation Matrix", height=500)
        corr_content = dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False})

        return violin_fig, qq_fig, corr_content
