"""
Market Data sub-tab: RF Summary.

Portfolio-wide risk-factor inventory — counts and coverage matrix
across clusters.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the RF Summary sub-tab skeleton."""
    return html.Div([
        html.Div("Risk Factor Inventory", style=SECTION_TITLE_STYLE),
        html.Div(id="md-rf-summary-table", style=CARD_STYLE),
        html.Div("Cluster × RF Coverage", style=SECTION_TITLE_STYLE),
        html.Div(id="md-rf-coverage-heatmap", style=CARD_STYLE),
    ])
