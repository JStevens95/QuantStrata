"""
Evaluation sub-tab: By CCY.

Same pattern as By Desk, grouped by currency.  Includes an extra
cross-currency residual correlation heatmap.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By CCY sub-tab skeleton."""
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(id="eval-ccy-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-ccy-boxplot", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="eval-ccy-scatter-grid", style=CARD_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="eval-ccy-correlation", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-ccy-table-container", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
    ])
