"""
Tab 4 — Market Data sub-tab container.

All interactive controls (cluster selector, shock asset/RF dropdowns)
live as persistent components.  A callback toggles visibility based on
the active sub-tab.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import MD_SUB_ORDER, MD_SUB_RF_SUMMARY
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def layout() -> html.Div:
    """Build the Market Data tab layout with sub-tab navigation."""
    _label = {
        "color": TEXT_SECONDARY, "fontSize": "12px",
        "marginRight": "6px", "fontWeight": "600",
    }
    _row = {"display": "flex", "alignItems": "center", "marginRight": "20px"}

    controls = html.Div(
        [
            html.Div(
                [html.Label("Cluster:", style=_label),
                 dcc.Dropdown(id="md-cluster-dropdown", clearable=False,
                              style={"width": "400px", "fontSize": "13px"})],
                style=_row,
            ),
            html.Div(
                [html.Label("Asset:", style=_label),
                 dcc.Dropdown(id="md-shock-asset-dd", clearable=False,
                              style={"width": "300px", "fontSize": "13px"})],
                id="md-shock-asset-wrapper", style={**_row, "display": "none"},
            ),
            html.Div(
                [html.Label("RF:", style=_label),
                 dcc.Dropdown(id="md-shock-rf-dd", clearable=False,
                              style={"width": "300px", "fontSize": "13px"})],
                id="md-shock-rf-wrapper", style={**_row, "display": "none"},
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"},
    )

    sub_tabs = dcc.Tabs(
        id="md-sub-tabs",
        value=MD_SUB_RF_SUMMARY,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in MD_SUB_ORDER
        ],
    )
    return html.Div([
        controls,
        sub_tabs,
        html.Div(id="md-sub-tab-content", style={"marginTop": "16px"}),
    ])
