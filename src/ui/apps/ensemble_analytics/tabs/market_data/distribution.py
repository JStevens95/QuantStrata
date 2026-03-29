"""
Market Data sub-tab: Distribution.

Cross-cluster RF distribution comparison — QQ plots and correlation
matrices.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Distribution sub-tab skeleton."""
    return html.Div([
        html.Div("Cross-Cluster RF Comparison", style=SECTION_TITLE_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="md-dist-violin", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="md-dist-qq", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="md-dist-corr-heatmap", style=CARD_STYLE),
    ])
