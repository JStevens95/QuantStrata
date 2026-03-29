"""
Trade Graph sub-tab: Node Analytics.

Degree vs model error, node feature table.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Node Analytics sub-tab skeleton."""
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(id="tg-node-degree-scatter", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="tg-node-feature-table", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
    ])
