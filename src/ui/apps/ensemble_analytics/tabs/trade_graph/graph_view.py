"""
Trade Graph sub-tab: Graph View.

Interactive cytoscape network visualisation of the trade adjacency
graph for a selected cluster.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Graph View sub-tab skeleton with interactive controls."""
    from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY

    controls = html.Div([
        html.Div([
            html.Label("Layout:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.RadioItems(
                id="tg-layout-selector",
                options=[
                    {"label": "Force-directed", "value": "cose"},
                    {"label": "Circular", "value": "circle"},
                    {"label": "Grid", "value": "grid"},
                ],
                value="cose", inline=True,
                labelStyle={"marginRight": "12px", "fontSize": "13px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginRight": "24px"}),
        html.Div([
            html.Label("Edge threshold:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Slider(id="tg-weight-threshold", min=0, max=1, step=0.01, value=0.01,
                       marks={0: "0", 0.25: "0.25", 0.5: "0.5", 1: "1"},
                       tooltip={"placement": "bottom"}),
        ], style={"width": "300px"}),
        html.Div([
            html.Label("Search:", style={"color": TEXT_SECONDARY, "fontSize": "12px", "marginRight": "6px"}),
            dcc.Input(id="tg-search-box", type="text", placeholder="trade_id...",
                      style={"width": "200px", "fontSize": "13px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center",
              "marginBottom": "12px", "gap": "16px"})

    return html.Div([
        controls,
        html.Div(id="tg-graph-container", style={**CARD_STYLE, "height": "600px"}),
        html.Div(id="tg-node-detail", style=CARD_STYLE),
    ])
