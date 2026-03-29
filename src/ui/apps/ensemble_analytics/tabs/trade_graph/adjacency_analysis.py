"""
Trade Graph sub-tab: Adjacency Analysis.

Graph statistics, edge weight histogram, degree distribution, and
adjacency spy plot.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Adjacency Analysis sub-tab skeleton."""
    return html.Div([
        html.Div(id="tg-adj-stats", style=CARD_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="tg-adj-weight-hist", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="tg-adj-degree-dist", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div("Adjacency Spy Plot", style=SECTION_TITLE_STYLE),
        html.Div(id="tg-adj-spy", style=CARD_STYLE),
    ])
