"""
Callback registration hub for the pre-computed dashboard.

Mirrors ``ensemble_analytics.callbacks`` but uses the pre-computed
data layer.  Tab layouts are imported from the original app (shared UI).
"""
from __future__ import annotations

import logging
import time

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

logger = logging.getLogger(__name__)


def register_all_callbacks(app: dash.Dash) -> None:
    """Register every callback module and the top-level tab router."""

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab_id: str):
        """Swap the main content area based on the active tab."""
        t0 = time.perf_counter()
        if tab_id == TAB_OVERVIEW:
            from src.ui.apps.ensemble_analytics.tabs.overview import layout
            result = layout()
        elif tab_id == TAB_EVALUATION:
            from src.ui.apps.ensemble_analytics.tabs.evaluation import layout
            result = layout()
        elif tab_id == TAB_CLUSTER_DEEP_DIVE:
            from src.ui.apps.ensemble_analytics.tabs.cluster_deep_dive import layout
            result = layout()
        elif tab_id == TAB_MARKET_DATA:
            from src.ui.apps.ensemble_analytics.tabs.market_data import layout
            result = layout()
        elif tab_id == TAB_TRADE_GRAPH:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph import layout
            result = layout()
        elif tab_id == TAB_INFERENCE:
            from src.ui.apps.ensemble_analytics.tabs.inference import layout
            result = layout()
        elif tab_id == TAB_GOVERNANCE:
            from src.ui.apps.ensemble_analytics.tabs.governance import layout
            result = layout()
        else:
            result = html.Div("Tab not found.")
        logger.info("[CALLBACK] render_tab(%s): %.3fs", tab_id, time.perf_counter() - t0)
        return result

    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        Input("ensemble-version-selector", "value"),
        prevent_initial_call=True,
    )
    def reload_version(version):
        """Reload session when the version dropdown changes."""
        if not version:
            return no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import reload
        from src.ui.apps.ensemble_analytics_db.data.trade_catalogue import invalidate
        reload(version)
        invalidate()
        return render_tab(TAB_OVERVIEW)

    from src.ui.apps.ensemble_analytics_db.callbacks.overview_cb import register as reg_overview
    from src.ui.apps.ensemble_analytics_db.callbacks.evaluation_cb import register as reg_evaluation
    from src.ui.apps.ensemble_analytics_db.callbacks.cluster_deep_dive_cb import register as reg_deep_dive
    from src.ui.apps.ensemble_analytics_db.callbacks.market_data_cb import register as reg_market_data
    from src.ui.apps.ensemble_analytics_db.callbacks.trade_graph_cb import register as reg_trade_graph
    from src.ui.apps.ensemble_analytics_db.callbacks.inference_cb import register as reg_inference
    from src.ui.apps.ensemble_analytics_db.callbacks.governance_cb import register as reg_governance

    reg_overview(app)
    reg_evaluation(app)
    reg_deep_dive(app)
    reg_market_data(app)
    reg_trade_graph(app)
    reg_inference(app)
    reg_governance(app)
