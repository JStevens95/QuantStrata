"""
Callbacks for Tab 3 — Cluster Deep Dive.

Populates the cluster selector on mount, then rebuilds all content
when the cluster or split changes.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.cluster_selector import cluster_selector
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY
from src.ui.apps.ensemble_analytics.theme.styles import CARD_HEADER_STYLE


def register(app):
    """Register Cluster Deep Dive callbacks on *app*."""

    @app.callback(
        Output("deep-dive-cluster-selector", "children"),
        Input("main-tabs", "value"),
    )
    def render_selector(tab):
        if tab != "tab-cluster-deep-dive":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        return cluster_selector(
            session.config.cluster_ids,
            session.cluster_attributes,
            id_prefix="deep-dive",
        )

    @app.callback(
        Output("deep-dive-header", "children"),
        Output("deep-dive-split-table", "children"),
        Output("deep-dive-convergence", "children"),
        Output("deep-dive-scatter", "children"),
        Output("deep-dive-residual", "children"),
        Output("deep-dive-scatter-matrix", "children"),
        Output("deep-dive-elementary", "children"),
        Output("deep-dive-config", "children"),
        Input("deep-dive-cluster-dropdown", "value"),
        Input("deep-dive-split-toggle", "value"),
    )
    def update_deep_dive(cluster_id, split):
        n_out = 8
        if not cluster_id:
            return (no_update,) * n_out

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_AMBER, CHART_COLORS
        from src.ui.apps.ensemble_analytics.figures.bar_charts import grouped_split_bar

        session = get_session()
        display = session.load_cluster_display(cluster_id)
        attrs = session.cluster_attributes.get(cluster_id, {})
        tu = display.trade_universe
        dc = display.eval_metrics.get("data_config", {})

        # ── Header card (G8-enriched) ────────────────────────────
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v)
        n_elem = len(tu.get("elementary_ids", []))
        n_target = len(tu.get("target_ids", session.config.cluster_mapping.get(cluster_id, [])))
        header = html.Div([
            html.Div(f"Cluster: {cluster_id}", style={"fontSize": "18px", "fontWeight": "700"}),
            html.Div(f"Version: {display.version}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}),
            html.Div(attr_str, style={"color": TEXT_SECONDARY, "fontSize": "13px"}) if attr_str else None,
            html.Div(
                f"n_elementary: {n_elem} · n_target: {n_target} · "
                f"seq_length: {dc.get('seq_length', '?')} · "
                f"transform: {dc.get('transform_type', '?')}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
        ])

        # ── Split comparison table + grouped bar ─────────────────
        ens_display = session.ensemble_display
        split_rows = []
        if ens_display:
            for s in ["train", "val", "test"]:
                pm = ens_display.per_member_metrics.get(s, {}).get(cluster_id, {})
                if pm:
                    row = {"split": s.capitalize()}
                    row.update(pm)
                    split_rows.append(row)

        split_col_defs = [{"field": "split", "headerName": "Split", "pinned": "left"}]
        if split_rows:
            for key in split_rows[0]:
                if key != "split":
                    split_col_defs.append({
                        "field": key,
                        "headerName": METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                    })
        split_table = metric_table(split_col_defs, split_rows, "deep-dive-split-comp-table", height="180px")

        # ── Training convergence (G8): load from saved plot PNGs ─
        convergence_content = html.Div("No convergence plot available.", style={"color": TEXT_SECONDARY})
        for plot_key, path_str in display.plot_paths.items():
            if "loss" in plot_key.lower() or "convergence" in plot_key.lower():
                from pathlib import Path
                if Path(path_str).exists():
                    import base64
                    with open(path_str, "rb") as f:
                        encoded = base64.b64encode(f.read()).decode()
                    convergence_content = html.Img(
                        src=f"data:image/png;base64,{encoded}",
                        style={"width": "100%", "maxHeight": "400px", "objectFit": "contain"},
                    )
                break

        # ── Scatter + residual ───────────────────────────────────
        store = get_prediction_store(split)
        catalogue = get_trade_catalogue()
        scatter_content = html.Div("No predictions available.")
        residual_content = html.Div("No predictions available.")
        scatter_matrix_content = html.Div("No predictions available.")
        elementary_content = html.Div("Elementary PnL data not available.", style={"color": TEXT_SECONDARY})

        if store is not None and not catalogue.empty:
            mask = catalogue["cluster_id"] == cluster_id
            col_idx = np.where(mask.values)[0]
            if len(col_idx) > 0:
                preds = store.predictions[:, col_idx]
                targets = store.targets[:, col_idx]
                trade_ids = [store.trade_ids[i] for i in col_idx]
                cluster_pred = preds.sum(axis=1)
                cluster_target = targets.sum(axis=1)

                scatter_content = dcc.Graph(
                    figure=pred_vs_target_scatter(cluster_pred, cluster_target,
                                                  title=f"{cluster_id} — {split.capitalize()}"),
                    config={"displayModeBar": False},
                )
                residual_content = dcc.Graph(
                    figure=residual_histogram(cluster_pred, cluster_target,
                                              title=f"Residuals — {split.capitalize()}"),
                    config={"displayModeBar": False},
                )

                # Per-Trade Scatter Matrix (G9): up to 6 trades
                n_show = min(6, len(trade_ids))
                if n_show > 0:
                    ncols = min(n_show, 3)
                    nrows = (n_show + ncols - 1) // ncols
                    fig = make_subplots(rows=nrows, cols=ncols,
                                        subplot_titles=trade_ids[:n_show])
                    for j in range(n_show):
                        r, c = j // ncols + 1, j % ncols + 1
                        fig.add_trace(go.Scattergl(
                            x=targets[:, j], y=preds[:, j], mode="markers",
                            marker=dict(size=2, color=CHART_COLORS[j % len(CHART_COLORS)], opacity=0.5),
                            showlegend=False,
                        ), row=r, col=c)
                    fig.update_layout(height=280 * nrows, title="Per-Trade Scatter (first 6)")
                    scatter_matrix_content = dcc.Graph(figure=fig, config={"displayModeBar": False})

        # ── Data config summary as structured table (G11) ────────
        if dc:
            config_rows = [{"parameter": k, "value": str(v)} for k, v in dc.items()]
            config_col_defs = [
                {"field": "parameter", "headerName": "Parameter"},
                {"field": "value", "headerName": "Value"},
            ]
            config_content = metric_table(config_col_defs, config_rows,
                                          "deep-dive-config-table", height="300px")
        else:
            config_content = html.Div("No data config available.", style={"color": TEXT_SECONDARY})

        return (header, split_table, convergence_content, scatter_content,
                residual_content, scatter_matrix_content, elementary_content,
                config_content)
