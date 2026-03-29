"""
Tab 5 — Trade Graph Explorer sub-tab container.

Renders the sub-tab bar for Graph View, Adjacency Analysis,
Node Analytics, and Cross-Cluster.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import TG_SUB_ORDER, TG_SUB_GRAPH_VIEW


def layout() -> html.Div:
    """Build the Trade Graph tab layout with sub-tab navigation."""
    sub_tabs = dcc.Tabs(
        id="tg-sub-tabs",
        value=TG_SUB_GRAPH_VIEW,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in TG_SUB_ORDER
        ],
    )
    return html.Div([
        html.Div(id="tg-cluster-selector-container"),
        sub_tabs,
        html.Div(id="tg-sub-tab-content", style={"marginTop": "16px"}),
    ])
