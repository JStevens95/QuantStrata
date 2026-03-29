"""
Trade Graph sub-tab: Cross-Cluster Comparison.

Compare graph structure (density, degree stats) across all clusters.
"""
from __future__ import annotations

from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Cross-Cluster sub-tab skeleton."""
    return html.Div([
        html.Div("Graph Structure Comparison", style=SECTION_TITLE_STYLE),
        html.Div(id="tg-cross-cluster-table", style=CARD_STYLE),
        html.Div(id="tg-cross-cluster-chart", style=CARD_STYLE),
    ])
