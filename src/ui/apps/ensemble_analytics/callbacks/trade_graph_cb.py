"""
Callbacks for Tab 5 — Trade Graph Explorer (all 4 sub-tabs).

Handles sub-tab routing, cluster selector, cytoscape rendering,
adjacency analysis, node analytics, and cross-cluster comparison.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update
import dash_cytoscape as cyto

from src.ui.apps.ensemble_analytics.config import (
    TG_SUB_GRAPH_VIEW,
    TG_SUB_ADJACENCY,
    TG_SUB_NODE_ANALYTICS,
    TG_SUB_CROSS_CLUSTER,
)
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.network import build_cytoscape_elements
from src.ui.apps.ensemble_analytics.figures.heatmaps import adjacency_spy
from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY


def register(app):
    """Register Trade Graph tab callbacks on *app*."""

    # ── Persistent control population ────────────────────────────
    @app.callback(
        Output("tg-cluster-dropdown", "options"),
        Output("tg-cluster-dropdown", "value"),
        Input("main-tabs", "value"),
    )
    def populate_tg_cluster(tab):
        if tab != "tab-trade-graph":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        attrs = session.cluster_attributes
        opts = []
        for cid in session.config.cluster_ids:
            if attrs and cid in attrs:
                parts = [f"{k}={v}" for k, v in attrs[cid].items() if v is not None]
                label = f"{cid}  ({', '.join(parts)})" if parts else cid
            else:
                label = cid
            opts.append({"label": label, "value": cid})
        default = session.config.cluster_ids[0] if session.config.cluster_ids else None
        return opts, default

    _ROW_VISIBLE = {"display": "flex", "alignItems": "center", "marginRight": "20px"}
    _HIDDEN = {"display": "none"}

    @app.callback(
        Output("tg-graph-controls-layout", "style"),
        Output("tg-graph-controls-threshold", "style"),
        Input("tg-sub-tabs", "value"),
    )
    def toggle_tg_graph_controls(sub_tab):
        show = sub_tab == TG_SUB_GRAPH_VIEW
        return (
            _ROW_VISIBLE if show else _HIDDEN,
            {"width": "300px"} if show else _HIDDEN,
        )

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("tg-sub-tab-content", "children"),
        Input("tg-sub-tabs", "value"),
    )
    def render_tg_sub_tab(sub_tab):
        if sub_tab == TG_SUB_GRAPH_VIEW:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.graph_view import layout
            return layout()
        elif sub_tab == TG_SUB_ADJACENCY:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.adjacency_analysis import layout
            return layout()
        elif sub_tab == TG_SUB_NODE_ANALYTICS:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.node_analytics import layout
            return layout()
        elif sub_tab == TG_SUB_CROSS_CLUSTER:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.cross_cluster import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── Graph View (G12: interactive controls + detail panel) ─────
    @app.callback(
        Output("tg-graph-container", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
        Input("tg-layout-selector", "value"),
        Input("tg-weight-threshold", "value"),
    )
    def update_graph_view(cluster_id, sub_tab, layout_name, threshold):
        if sub_tab != TG_SUB_GRAPH_VIEW or not cluster_id:
            return no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data

        gdata = get_graph_data(cluster_id)
        graph_results = gdata.get("graph_results", {})
        trade_universe = gdata.get("trade_universe", {})

        indices = graph_results.get("sparse_indices")
        values = graph_results.get("sparse_values")
        all_ids = trade_universe.get("elementary_ids", []) + trade_universe.get("target_ids", [])
        target_set = set(trade_universe.get("target_ids", []))

        if indices is None or values is None or not all_ids:
            return html.Div("No graph data available.", style={"color": TEXT_SECONDARY})

        indices = np.array(indices)
        values = np.array(values)

        node_attrs = {}
        for tid in all_ids:
            node_attrs[tid] = {"trade_type": "target" if tid in target_set else "elementary"}

        elements = build_cytoscape_elements(
            all_ids, indices, values,
            node_attrs=node_attrs,
            weight_threshold=threshold or 0.01,
        )

        return cyto.Cytoscape(
            id="tg-cytoscape",
            elements=elements,
            layout={"name": layout_name or "cose", "animate": False},
            style={"width": "100%", "height": "550px", "backgroundColor": BG_CARD},
            stylesheet=[
                {
                    "selector": "node[trade_type='elementary']",
                    "style": {
                        "label": "data(label)", "font-size": "9px",
                        "color": TEXT_PRIMARY, "background-color": ACCENT_BLUE,
                        "width": 16, "height": 16,
                    },
                },
                {
                    "selector": "node[trade_type='target']",
                    "style": {
                        "label": "data(label)", "font-size": "9px",
                        "color": TEXT_PRIMARY, "background-color": "#d29922",
                        "width": 22, "height": 22,
                    },
                },
                {
                    "selector": "edge",
                    "style": {"width": 1, "line-color": "#30363d", "opacity": 0.6},
                },
                {
                    "selector": ":selected",
                    "style": {"background-color": "#f85149", "border-width": 2, "border-color": "#fff"},
                },
            ],
        )

    @app.callback(
        Output("tg-node-detail", "children"),
        Input("tg-cytoscape", "tapNodeData"),
    )
    def show_node_detail(node_data):
        """Display details for a clicked node (G12)."""
        if not node_data:
            return html.Div("Click a node to see details.", style={"color": TEXT_SECONDARY, "fontSize": "13px"})

        tid = node_data.get("id", "?")
        trade_type = node_data.get("trade_type", "unknown")
        items = [
            html.Div(f"Trade: {tid}", style={"fontSize": "15px", "fontWeight": "600"}),
            html.Div(f"Type: {trade_type}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}),
        ]
        for k, v in node_data.items():
            if k not in ("id", "label", "trade_type"):
                items.append(html.Div(f"{k}: {v}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}))
        return html.Div(items)

    # ── Adjacency Analysis ────────────────────────────────────────
    @app.callback(
        Output("tg-adj-stats", "children"),
        Output("tg-adj-weight-hist", "children"),
        Output("tg-adj-degree-dist", "children"),
        Output("tg-adj-spy", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
    )
    def update_adjacency(cluster_id, sub_tab):
        if sub_tab != TG_SUB_ADJACENCY or not cluster_id:
            return no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
        import plotly.graph_objects as go

        gdata = get_graph_data(cluster_id)
        gr = gdata.get("graph_results", {})
        indices = gr.get("sparse_indices")
        values = gr.get("sparse_values")
        shape = gr.get("sparse_shape", [0, 0])

        if indices is None or values is None:
            msg = html.Div("No adjacency data.", style={"color": TEXT_SECONDARY})
            return msg, msg, msg, msg

        indices = np.array(indices)
        values = np.array(values)
        n_nodes = shape[0] if shape[0] > 0 else max(indices.max() + 1, 1)
        nnz = len(values)
        density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0

        stats = html.Div([
            html.Span(f"Nodes: {n_nodes}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Edges: {nnz}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Density: {density:.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Mean weight: {values.mean():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Max weight: {values.max():.4f}", style={"fontSize": "13px"}),
        ])

        # Edge weight histogram
        weight_hist = go.Figure(go.Histogram(x=values, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8))
        weight_hist.update_layout(title="Edge Weight Distribution", height=350)

        # Degree distribution
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows = indices[0]
        else:
            rows = indices[:, 0]
        degrees = np.bincount(rows.astype(int), minlength=n_nodes)
        degree_hist = go.Figure(go.Histogram(x=degrees, nbinsx=40, marker_color=ACCENT_BLUE, opacity=0.8))
        degree_hist.update_layout(title="Degree Distribution", height=350)

        # Spy plot
        spy_fig = adjacency_spy(indices, values, list(shape), title=f"Adjacency — Cluster {cluster_id}")

        return (
            stats,
            dcc.Graph(figure=weight_hist, config={"displayModeBar": False}),
            dcc.Graph(figure=degree_hist, config={"displayModeBar": False}),
            dcc.Graph(figure=spy_fig, config={"displayModeBar": False}),
        )

    # ── Node Analytics ────────────────────────────────────────────
    @app.callback(
        Output("tg-node-degree-scatter", "children"),
        Output("tg-node-feature-table", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
    )
    def update_node_analytics(cluster_id, sub_tab):
        if sub_tab != TG_SUB_NODE_ANALYTICS or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        import plotly.graph_objects as go

        gdata = get_graph_data(cluster_id)
        gr = gdata.get("graph_results", {})
        tu = gdata.get("trade_universe", {})
        indices = gr.get("sparse_indices")
        shape = gr.get("sparse_shape", [0, 0])
        trade_ids = tu.get("target_ids", [])

        if indices is None or not trade_ids:
            msg = html.Div("Insufficient data.", style={"color": TEXT_SECONDARY})
            return msg, msg

        indices = np.array(indices)
        n_nodes = shape[0] if shape[0] > 0 else max(indices.max() + 1, 1)
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows = indices[0]
        else:
            rows = indices[:, 0]
        degrees = np.bincount(rows.astype(int), minlength=n_nodes)

        # Try to get per-trade MAE from prediction store
        store = get_prediction_store("test")
        catalogue = get_trade_catalogue()
        trade_mae = np.full(len(trade_ids), np.nan)

        if store is not None and not catalogue.empty:
            mask = catalogue["cluster_id"] == cluster_id
            col_idx = np.where(mask.values)[0]
            if len(col_idx) > 0:
                preds = store.predictions[:, col_idx]
                targets = store.targets[:, col_idx]
                trade_mae = np.mean(np.abs(preds - targets), axis=0)

        # Degree vs MAE scatter
        n_plot = min(len(degrees), len(trade_mae), len(trade_ids))
        fig = go.Figure(go.Scattergl(
            x=degrees[:n_plot], y=trade_mae[:n_plot],
            mode="markers", marker=dict(size=5, color=ACCENT_BLUE, opacity=0.7),
            text=trade_ids[:n_plot], hoverinfo="text+x+y",
        ))
        fig.update_layout(title="Degree vs MAE", xaxis_title="Node Degree", yaxis_title="MAE", height=400)

        # Node table
        col_defs = [
            {"field": "trade_id", "headerName": "Trade ID"},
            {"field": "degree", "headerName": "Degree"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        rows = [
            {"trade_id": trade_ids[i], "degree": int(degrees[i]) if i < len(degrees) else 0,
             "mae": float(trade_mae[i]) if i < len(trade_mae) else 0.0}
            for i in range(n_plot)
        ]
        table = metric_table(col_defs, rows, "tg-node-table", height="350px")

        return (
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            table,
        )

    # ── Cross-Cluster Comparison ──────────────────────────────────
    @app.callback(
        Output("tg-cross-cluster-table", "children"),
        Output("tg-cross-cluster-chart", "children"),
        Input("tg-sub-tabs", "value"),
    )
    def update_cross_cluster(sub_tab):
        if sub_tab != TG_SUB_CROSS_CLUSTER:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics.data.graph_data_loader import get_graph_data
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        import plotly.graph_objects as go

        session = get_session()
        rows = []
        for cid in session.config.cluster_ids:
            gdata = get_graph_data(cid)
            gr = gdata.get("graph_results", {})
            indices = gr.get("sparse_indices")
            values = gr.get("sparse_values")
            shape = gr.get("sparse_shape", [0, 0])

            if indices is not None and values is not None:
                n_nodes = shape[0] if shape[0] > 0 else 0
                nnz = len(np.array(values))
                density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
                rows.append({
                    "cluster_id": cid,
                    "n_nodes": n_nodes,
                    "n_edges": nnz,
                    "density": round(density, 6),
                    "mean_weight": round(float(np.mean(values)), 4),
                })
            else:
                rows.append({"cluster_id": cid, "n_nodes": 0, "n_edges": 0, "density": 0, "mean_weight": 0})

        col_defs = [
            {"field": "cluster_id", "headerName": "Cluster"},
            {"field": "n_nodes", "headerName": "Nodes"},
            {"field": "n_edges", "headerName": "Edges"},
            {"field": "density", "headerName": "Density"},
            {"field": "mean_weight", "headerName": "Mean Weight"},
        ]
        table = metric_table(col_defs, rows, "tg-cross-table", height="300px")

        # Density bar chart
        fig = go.Figure(go.Bar(
            x=[r["cluster_id"] for r in rows],
            y=[r["density"] for r in rows],
            marker_color=ACCENT_BLUE,
            text=[f"{r['density']:.4f}" for r in rows],
            textposition="auto",
        ))
        fig.update_layout(title="Graph Density by Cluster", yaxis_title="Density", height=350)

        return table, dcc.Graph(figure=fig, config={"displayModeBar": False})
