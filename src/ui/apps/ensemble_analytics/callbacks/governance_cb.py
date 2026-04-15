"""
Callbacks for Tab 7 — Model Governance.

Version comparison: populate the dropdown when the tab opens, then build
a metric-delta table and grouped bar chart when the user selects a version.
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
        """Populate version comparison dropdown, excluding the active version."""
        if tab != "tab-governance":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session

        session = get_session()
        current_ver = session.ensemble_version
        versions = session._ens_registry.list_versions()

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
        """Show metric delta table + bar chart between current and selected version."""
        if not compare_version:
            return html.Div(
                "Select a version above to compare.",
                style={"color": "#8b949e", "fontSize": "13px"},
            )

        import json as _json
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
        from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY
        import plotly.graph_objects as go

        session = get_session()
        current_ens = session.ensemble_display
        if not current_ens:
            return html.Div("Current display data not loaded.")

        current_metrics = current_ens.ensemble_metrics.get("test", {})
        if not current_metrics:
            return html.Div(
                "No test-split metrics for the current version.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        if session.artifacts_dir is None:
            return html.Div("Artifacts directory not configured.")

        compare_dir = session.artifacts_dir / "ensemble" / compare_version / "evaluation"
        compare_path = compare_dir / "ensemble_metrics.json"
        if not compare_path.exists():
            compare_path = compare_dir / "ensemble_metrics_test.json"
        if not compare_path.exists():
            return html.Div(
                f"No test metrics found for version '{compare_version}'. "
                f"Expected at: {compare_dir}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        with open(compare_path) as f:
            compare_metrics = _json.load(f)

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
                "No overlapping numeric metrics found between the two versions.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        col_defs = [
            {"field": "metric", "headerName": "Metric"},
            {"field": "current", "headerName": f"Current ({session.ensemble_version})"},
            {"field": "compare", "headerName": compare_version},
            {"field": "delta", "headerName": "Delta"},
            {"field": "pct_change", "headerName": "% Change"},
        ]
        table = metric_table(col_defs, rows, "governance-comparison-table", height="250px")

        labels = [r["metric"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"Current ({session.ensemble_version})",
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
            barmode="group",
            height=350,
            xaxis_title="Metric",
            yaxis_title="Value",
        )

        return html.Div([
            table,
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])
