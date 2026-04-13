"""
Trade Graph sub-tab: Graph View.

Interactive cytoscape network visualisation of the trade adjacency
graph for a selected cluster.  Controls (layout, threshold) live
persistently in the parent tab layout.
"""
from __future__ import annotations

from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Graph View sub-tab skeleton."""
    return html.Div([
        html.Div(id="tg-graph-container", style={**CARD_STYLE, "height": "600px"}),
        html.Div(id="tg-node-detail", style=CARD_STYLE),
    ])
