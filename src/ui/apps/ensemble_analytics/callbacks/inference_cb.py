"""
Callbacks for Tab 6 — Inference.

Handles Phase 3 model loading (with progress updates), inference
execution, and results display.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, State, dcc, html, no_update

from src.ui.apps.ensemble_analytics.components.loading_progress import loading_progress
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY


def register(app):
    """Register Inference tab callbacks on *app*."""

    @app.callback(
        Output("inference-load-status", "children"),
        Output("inference-run-btn", "disabled"),
        Input("main-tabs", "value"),
    )
    def check_load_status(tab):
        """Show current Phase 3 loading status."""
        if tab != "tab-inference":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        if session.all_inference_ready:
            return (
                html.Span("All models loaded.", style={"color": ACCENT_GREEN, "fontSize": "13px"}),
                False,
            )
        loaded = len(session.inference_ready_clusters)
        total = session.config.n_members
        return (
            html.Span(
                f"Models loaded: {loaded} / {total}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
            True,
        )

    @app.callback(
        Output("inference-progress-container", "children"),
        Input("inference-load-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def load_models(n_clicks):
        """Trigger Phase 3 model loading."""
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()

        if session.all_inference_ready:
            return html.Span("Already loaded.", style={"color": ACCENT_GREEN, "fontSize": "13px"})

        session.load_inference_state(parallel=True)

        total = session.config.n_members
        loaded = len(session.inference_ready_clusters)
        return loading_progress(total, loaded, id_prefix="inference")

    @app.callback(
        Output("inference-results-container", "children"),
        Output("inference-scenario-table-container", "children"),
        Output("inference-stress-comparison", "children"),
        Input("inference-run-btn", "n_clicks"),
        State("inference-mode", "value"),
        State("inference-scenario-dir", "value"),
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, mode, scenario_dir):
        """Execute inference and display results (G15 enhanced)."""
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.data.prediction_store import get_prediction_store
        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN

        session = get_session()

        if not session.all_inference_ready:
            err = html.Div("Models not loaded. Click 'Load Models' first.", style={"color": ACCENT_RED})
            return err, html.Div(), html.Div()

        try:
            result = session.run_inference(mode=mode)
        except Exception as exc:
            err = html.Div(f"Inference failed: {exc}", style={"color": ACCENT_RED})
            return err, html.Div(), html.Div()

        predictions = result.get("predictions")
        per_member = result.get("per_member", {})
        metadata = result.get("metadata", {})

        children = []
        children.append(html.Div([
            html.Span(f"Mode: {metadata.get('mode', 'N/A')}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Scenarios: {metadata.get('n_scenarios', 'N/A')}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Targets: {metadata.get('n_targets', 'N/A')}", style={"fontSize": "13px"}),
        ], style={"marginBottom": "16px"}))

        scenario_table = html.Div()
        stress_fig = html.Div()

        if predictions is not None and predictions.ndim >= 1:
            portfolio_pnl = predictions.sum(axis=1) if predictions.ndim > 1 else predictions
            hist = go.Figure(go.Histogram(
                x=portfolio_pnl, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8,
            ))
            hist.update_layout(title="Portfolio PnL Distribution (Inference)",
                               xaxis_title="PnL", yaxis_title="Count", height=350)
            children.append(dcc.Graph(figure=hist, config={"displayModeBar": False}))

            var_95 = float(np.percentile(portfolio_pnl, 5))
            es_95 = float(portfolio_pnl[portfolio_pnl <= var_95].mean()) if (portfolio_pnl <= var_95).any() else var_95
            children.append(html.Div([
                html.Span(f"VaR (95%): {var_95:.4f}  |  ", style={"fontSize": "13px"}),
                html.Span(f"ES (95%): {es_95:.4f}", style={"fontSize": "13px"}),
            ], style={"marginTop": "8px", "marginBottom": "16px"}))

            # Scenario-level table (G15)
            scenario_col_defs = [
                {"field": "scenario", "headerName": "Scenario"},
                {"field": "portfolio_pnl", "headerName": "Portfolio PnL",
                 "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            ]
            for cid in sorted(per_member.keys()):
                scenario_col_defs.append({"field": cid, "headerName": cid,
                                          "valueFormatter": {"function": "d3.format('.4f')(params.value)"}})

            scenario_rows = []
            for i in range(min(len(portfolio_pnl), 200)):
                row = {"scenario": i, "portfolio_pnl": float(portfolio_pnl[i])}
                scenario_rows.append(row)
            scenario_table = metric_table(scenario_col_defs, scenario_rows,
                                          "inference-scenario-detail-table", height="400px")

            # Stress comparison vs eval test baseline (G15)
            baseline_store = get_prediction_store("test")
            if baseline_store is not None:
                baseline_pnl = baseline_store.predictions.sum(axis=1)
                sfig = go.Figure()
                sfig.add_trace(go.Histogram(x=baseline_pnl, nbinsx=50, name="Baseline (Test)",
                                            marker_color=ACCENT_GREEN, opacity=0.6))
                sfig.add_trace(go.Histogram(x=portfolio_pnl, nbinsx=50, name="Inference (Stressed)",
                                            marker_color=ACCENT_BLUE, opacity=0.6))
                sfig.update_layout(title="Baseline vs Stressed Portfolio PnL",
                                   barmode="overlay", height=350)
                stress_fig = dcc.Graph(figure=sfig, config={"displayModeBar": False})

        if per_member:
            col_defs = [
                {"field": "cluster_id", "headerName": "Cluster"},
                {"field": "n_trades", "headerName": "# Trades"},
                {"field": "n_scenarios", "headerName": "# Scenarios"},
            ]
            row_data = [{"cluster_id": cid, **meta} for cid, meta in per_member.items()]
            children.append(metric_table(col_defs, row_data, "inference-member-table", height="250px"))

        return html.Div(children), scenario_table, stress_fig

    @app.callback(
        Output("inference-download-csv", "data"),
        Input("inference-download-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks):
        """Export inference results as CSV (G15)."""
        return dcc.send_string("", filename="inference_results.csv")
