"""
Callback registration hub.

Call ``register_all_callbacks(app)`` once at app creation to wire up
all tab callbacks and the main tab-routing callback.
"""
from __future__ import annotations

import dash
from dash import Input, Output, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    TAB_OVERVIEW,
    TAB_EVALUATION,
    TAB_CLUSTER_DEEP_DIVE,
    TAB_MARKET_DATA,
    TAB_TRADE_GRAPH,
    TAB_INFERENCE,
    TAB_GOVERNANCE,
)


def register_all_callbacks(app: dash.Dash) -> None:
    """
    Register every callback module and the top-level tab router.

    Parameters
    ----------
    app : dash.Dash
        The application instance.
    """
    # ── Main tab routing ──────────────────────────────────────────
    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab_id: str):
        """Swap the main content area based on the active tab."""
        if tab_id == TAB_OVERVIEW:
            from src.ui.apps.ensemble_analytics.tabs.overview import layout
            return layout()
        elif tab_id == TAB_EVALUATION:
            from src.ui.apps.ensemble_analytics.tabs.evaluation import layout
            return layout()
        elif tab_id == TAB_CLUSTER_DEEP_DIVE:
            from src.ui.apps.ensemble_analytics.tabs.cluster_deep_dive import layout
            return layout()
        elif tab_id == TAB_MARKET_DATA:
            from src.ui.apps.ensemble_analytics.tabs.market_data import layout
            return layout()
        elif tab_id == TAB_TRADE_GRAPH:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph import layout
            return layout()
        elif tab_id == TAB_INFERENCE:
            from src.ui.apps.ensemble_analytics.tabs.inference import layout
            return layout()
        elif tab_id == TAB_GOVERNANCE:
            from src.ui.apps.ensemble_analytics.tabs.governance import layout
            return layout()
        return html.Div("Tab not found.")

    # ── Version reload callback ───────────────────────────────────
    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        Input("ensemble-version-selector", "value"),
        prevent_initial_call=True,
    )
    def reload_version(version):
        """Reload session when the version dropdown changes."""
        if not version:
            return no_update
        from src.ui.apps.ensemble_analytics.data.session_manager import reload
        from src.ui.apps.ensemble_analytics.data.trade_catalogue import invalidate
        reload(version)
        invalidate()
        return render_tab(TAB_OVERVIEW)

    # ── Per-tab callbacks ─────────────────────────────────────────
    from src.ui.apps.ensemble_analytics.callbacks.overview_cb import register as reg_overview
    from src.ui.apps.ensemble_analytics.callbacks.evaluation_cb import register as reg_evaluation
    from src.ui.apps.ensemble_analytics.callbacks.cluster_deep_dive_cb import register as reg_deep_dive
    from src.ui.apps.ensemble_analytics.callbacks.market_data_cb import register as reg_market_data
    from src.ui.apps.ensemble_analytics.callbacks.trade_graph_cb import register as reg_trade_graph
    from src.ui.apps.ensemble_analytics.callbacks.inference_cb import register as reg_inference
    from src.ui.apps.ensemble_analytics.callbacks.governance_cb import register as reg_governance

    reg_overview(app)
    reg_evaluation(app)
    reg_deep_dive(app)
    reg_market_data(app)
    reg_trade_graph(app)
    reg_inference(app)
    reg_governance(app)
