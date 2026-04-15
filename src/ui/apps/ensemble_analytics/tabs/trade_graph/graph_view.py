"""
Trade Graph sub-tab: Graph View.

Interactive cytoscape network visualisation of the trade adjacency
graph for a selected cluster.  Controls (layout, threshold) live
persistently in the parent tab layout.
"""
from __future__ import annotations

from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE
from src.ui.apps.ensemble_analytics.theme.colors import (
    ACCENT_BLUE,
    ACCENT_AMBER,
    TEXT_SECONDARY,
)

_CIRCLE = {
    "display": "inline-block",
    "width": "12px",
    "height": "12px",
    "borderRadius": "50%",
    "marginRight": "6px",
    "verticalAlign": "middle",
}

_BTN = {
    "padding": "4px 10px",
    "fontSize": "12px",
    "border": "1px solid #30363d",
    "borderRadius": "4px",
    "background": "#161b22",
    "color": "#e6edf3",
    "cursor": "pointer",
    "marginRight": "6px",
}


def layout() -> html.Div:
    """Build the Graph View sub-tab skeleton."""
    legend = html.Div(
        [
            html.Span(style={**_CIRCLE, "backgroundColor": ACCENT_AMBER}),
            html.Span("Target", style={"fontSize": "12px", "marginRight": "16px", "color": TEXT_SECONDARY}),
            html.Span(style={**_CIRCLE, "backgroundColor": ACCENT_BLUE}),
            html.Span("Elementary", style={"fontSize": "12px", "color": TEXT_SECONDARY}),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
    )

    zoom_controls = html.Div(
        [
            html.Button("Zoom In", id="tg-zoom-in", n_clicks=0, style=_BTN),
            html.Button("Zoom Out", id="tg-zoom-out", n_clicks=0, style=_BTN),
            html.Button("Reset", id="tg-zoom-reset", n_clicks=0, style=_BTN),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "8px"},
    )

    return html.Div([
        html.Div(
            [legend, zoom_controls],
            style={"display": "flex", "justifyContent": "space-between", "flexWrap": "wrap"},
        ),
        html.Div(id="tg-graph-container", style={**CARD_STYLE, "height": "600px"}),
        html.Div(id="tg-node-detail", style=CARD_STYLE),
    ])
