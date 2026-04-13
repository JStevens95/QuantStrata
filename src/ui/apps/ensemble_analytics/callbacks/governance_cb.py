"""
Callbacks for Tab 7 — Model Governance.

The governance tab is almost entirely static (populated at layout time).
This module is a placeholder for future dynamic features such as
version-comparison callbacks.
"""
from __future__ import annotations


def register(app):
    """Register Governance tab callbacks on *app*."""
    from dash import Input, Output, dcc, html, no_update

    @app.callback(
        Output("governance-compare-version", "options"),
        Input("main-tabs", "value"),
    )
    def populate_compare_dropdown(tab):
        """Populate version comparison dropdown from registry (G16)."""
        if tab != "tab-governance":
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        session = get_session()
        try:
            versions = session._ens_registry.list_versions()
            opts = []
            for v in versions:
                if isinstance(v, dict):
                    ver = v.get("version", str(v))
                    n = v.get("n_members", "?")
                    t = v.get("n_trades", "?")
                    opts.append({"label": f"{ver} ({n} clusters, {t} trades)", "value": ver})
                else:
                    opts.append({"label": str(v), "value": str(v)})
            return opts
        except Exception:
            return []

    @app.callback(
        Output("governance-comparison-content", "children"),
        Input("governance-compare-version", "value"),
    )
    def compare_versions(compare_version):
        """Show metric delta table between current and selected version (G16)."""
        if not compare_version:
            return html.Div("Select a version above to compare.", style={"color": "#8b949e", "fontSize": "13px"})

        import json as _json
        from pathlib import Path
        from src.ui.apps.ensemble_analytics.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
        import plotly.graph_objects as go

        session = get_session()
        current_ens = session.ensemble_display
        if not current_ens:
            return html.Div("Current display data not loaded.")

        current_metrics = current_ens.ensemble_metrics.get("test", {})

        compare_dir = session.artifacts_dir / "ensemble" / compare_version / "evaluation"
        compare_path = compare_dir / "ensemble_metrics.json"
        if not compare_path.exists():
            compare_path = compare_dir / "ensemble_metrics_test.json"
        if not compare_path.exists():
            return html.Div(f"No test metrics found for version {compare_version}.")

        with open(compare_path) as f:
            compare_metrics = _json.load(f)

        col_defs = [
            {"field": "metric", "headerName": "Metric"},
            {"field": "current", "headerName": f"Current", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "compare", "headerName": compare_version, "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "delta", "headerName": "Δ", "valueFormatter": {"function": "d3.format('+.4f')(params.value)"}},
            {"field": "pct_change", "headerName": "% Change", "valueFormatter": {"function": "d3.format('+.1f')(params.value) + '%'"}},
        ]
        rows = []
        all_keys = sorted(set(list(current_metrics.keys()) + list(compare_metrics.keys())))
        for mk in all_keys:
            cv = current_metrics.get(mk)
            ev = compare_metrics.get(mk)
            if cv is not None and ev is not None:
                delta = cv - ev
                pct = (delta / abs(ev) * 100) if ev != 0 else 0
                rows.append({"metric": mk.upper(), "current": cv, "compare": ev, "delta": delta, "pct_change": pct})

        table = metric_table(col_defs, rows, "governance-comparison-table", height="250px")

        labels = [r["metric"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Current", x=labels, y=[r["current"] for r in rows], marker_color="#58a6ff"))
        fig.add_trace(go.Bar(name=compare_version, x=labels, y=[r["compare"] for r in rows], marker_color="#d29922"))
        fig.update_layout(title="Metric Comparison", barmode="group", height=350)

        return html.Div([table, dcc.Graph(figure=fig, config={"displayModeBar": False})])
