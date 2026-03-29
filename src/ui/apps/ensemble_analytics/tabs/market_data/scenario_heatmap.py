"""
Market Data sub-tab: Scenario Heatmap.

RF x scenario heatmap for a selected cluster — shows the full shock
surface.
"""
from __future__ import annotations

from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Scenario Heatmap sub-tab skeleton."""
    return html.Div([
        html.Div(id="md-heatmap-container", style=CARD_STYLE),
    ])
