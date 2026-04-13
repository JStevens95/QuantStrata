"""
Market Data sub-tab: Shock Explorer.

Drill into RF shock time-series and distributions for a selected
cluster and asset.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE


def layout() -> html.Div:
    """Build the Shock Explorer sub-tab skeleton."""
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(id="md-shock-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="md-shock-distribution", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="md-shock-stats", style=CARD_STYLE),
    ])
