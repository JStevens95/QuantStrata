"""
Callbacks for Tab 7 — Model Governance.

All data comes from ``get_backend()`` — no ``EnsembleSession`` access.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(app):
    """Register Governance tab callbacks on *app*."""
    from dash import Input, Output, dcc, html, no_update

    @app.callback(
        Output("governance-compare-version", "options"),
        Input("main-tabs", "value"),
    )
    def populate_compare_dropdown(tab):
        if tab != "tab-governance":
            return no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend

        backend = get_backend()
        current_ver = backend.get_ensemble_version()
        versions = backend.get_registry_versions()

        opts = []
        for v in versions:
            ver = v["version"] if isinstance(v, dict) else str(v)
            if ver == current_ver:
                continue
            if isinstance(v, dict):
                lbl = f"{ver}  ({v.get('n_members', '?')} clusters, {v.get('n_trades', '?')} trades)"
            else:
                lbl = ver
            opts.append({"label": lbl, "value": ver})
        return opts

    @app.callback(
        Output("governance-comparison-content", "children"),
        Input("governance-compare-version", "value"),
    )
    def compare_versions(compare_version):
        if not compare_version:
            return html.Div(
                "Select a version above to compare.",
                style={"color": "#8b949e", "fontSize": "13px"},
            )

        import json as _json
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
        from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY
        import plotly.graph_objects as go

        backend = get_backend()
        current_ver = backend.get_ensemble_version()
        current_metrics = backend.get_ensemble_metrics("test")
        if not current_metrics:
            return html.Div(
                "No test-split metrics for the current version.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        compare_metrics = backend.get_ensemble_metrics_for_version(compare_version)
        if compare_metrics is None:
            return html.Div(
                f"No test metrics found for version '{compare_version}'.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        rows = []
        all_keys = sorted(set(list(current_metrics.keys()) + list(compare_metrics.keys())))
        for mk in all_keys:
            cv = current_metrics.get(mk)
            ev = compare_metrics.get(mk)
            if cv is None or ev is None:
                continue
            try:
                cv_f = float(cv)
                ev_f = float(ev)
            except (TypeError, ValueError):
                continue
            delta = cv_f - ev_f
            pct = (delta / abs(ev_f) * 100) if ev_f != 0 else 0.0
            rows.append({
                "metric": mk.upper(),
                "current": round(cv_f, 6),
                "compare": round(ev_f, 6),
                "delta": round(delta, 6),
                "pct_change": round(pct, 2),
            })

        if not rows:
            return html.Div(
                "No overlapping numeric metrics found.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        col_defs = [
            {"field": "metric", "headerName": "Metric"},
            {"field": "current", "headerName": f"Current ({current_ver})"},
            {"field": "compare", "headerName": compare_version},
            {"field": "delta", "headerName": "Delta"},
            {"field": "pct_change", "headerName": "% Change"},
        ]
        table = metric_table(col_defs, rows, "governance-comparison-table", height="250px")

        labels = [r["metric"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"Current ({current_ver})",
            x=labels, y=[r["current"] for r in rows],
            marker_color="#58a6ff",
        ))
        fig.add_trace(go.Bar(
            name=compare_version,
            x=labels, y=[r["compare"] for r in rows],
            marker_color="#d29922",
        ))
        fig.update_layout(
            title="Metric Comparison (Test Split)",
            barmode="group", height=350,
            xaxis_title="Metric", yaxis_title="Value",
        )

        return html.Div([
            table,
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])
