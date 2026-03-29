"""
Tab 4 — Market Data sub-tab container.

Renders the sub-tab bar for RF Summary, Shock Explorer, Scenario
Heatmap, and Distribution.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import MD_SUB_ORDER, MD_SUB_RF_SUMMARY


def layout() -> html.Div:
    """Build the Market Data tab layout with sub-tab navigation."""
    sub_tabs = dcc.Tabs(
        id="md-sub-tabs",
        value=MD_SUB_RF_SUMMARY,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in MD_SUB_ORDER
        ],
    )
    return html.Div([
        html.Div(id="md-cluster-selector-container"),
        sub_tabs,
        html.Div(id="md-sub-tab-content", style={"marginTop": "16px"}),
    ])
