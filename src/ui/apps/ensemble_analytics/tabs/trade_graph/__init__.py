"""
Tab 5 — Trade Graph Explorer sub-tab container.

All interactive controls (cluster selector, graph layout options, edge
threshold) live as persistent components.  A callback toggles
visibility of graph-view-specific controls based on the active sub-tab.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import TG_SUB_ORDER, TG_SUB_GRAPH_VIEW
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def layout() -> html.Div:
    """Build the Trade Graph tab layout with sub-tab navigation."""
    _label = {
        "color": TEXT_SECONDARY, "fontSize": "12px",
        "marginRight": "6px", "fontWeight": "600",
    }
    _row = {"display": "flex", "alignItems": "center", "marginRight": "20px"}

    controls = html.Div(
        [
            html.Div(
                [html.Label("Cluster:", style=_label),
                 dcc.Dropdown(id="tg-cluster-dropdown", clearable=False,
                              style={"width": "400px", "fontSize": "13px"})],
                style=_row,
            ),
            html.Div(
                [html.Label("Layout:", style=_label),
                 dcc.RadioItems(
                     id="tg-layout-selector",
                     options=[
                         {"label": "Force-directed", "value": "cose"},
                         {"label": "Circular", "value": "circle"},
                         {"label": "Grid", "value": "grid"},
                     ],
                     value="cose", inline=True,
                     labelStyle={"marginRight": "12px", "fontSize": "13px"},
                 )],
                id="tg-graph-controls-layout", style={**_row, "display": "none"},
            ),
            html.Div(
                [html.Label("Edge threshold:", style=_label),
                 dcc.Slider(id="tg-weight-threshold", min=0, max=1, step=0.01,
                            value=0.01,
                            marks={0: "0", 0.25: "0.25", 0.5: "0.5", 1: "1"},
                            tooltip={"placement": "bottom"})],
                id="tg-graph-controls-threshold",
                style={"width": "300px", "display": "none"},
            ),
            html.Div(
                [html.Label("Search:", style=_label),
                 dcc.Input(id="tg-search-box", type="text",
                           placeholder="Filter by trade ID...",
                           debounce=True,
                           style={"width": "200px", "fontSize": "13px"})],
                id="tg-graph-controls-search",
                style={**_row, "display": "none"},
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center",
               "marginBottom": "12px", "gap": "16px"},
    )

    sub_tabs = dcc.Tabs(
        id="tg-sub-tabs",
        value=TG_SUB_GRAPH_VIEW,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in TG_SUB_ORDER
        ],
    )
    return html.Div([
        controls,
        sub_tabs,
        html.Div(id="tg-sub-tab-content", style={"marginTop": "16px"}),
    ])
