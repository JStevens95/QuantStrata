"""
Evaluation sub-tab: By Desk.

Aggregates predictions and targets by desk attribute.  Shows overlaid
time-series, residual box plots, and a metrics table.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By Desk sub-tab skeleton."""
    return html.Div([
        html.Div(id="eval-desk-filter-bar"),
        dbc.Row([
            dbc.Col(html.Div(id="eval-desk-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-desk-boxplot", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="eval-desk-scatter-grid", style=CARD_STYLE),
        html.Div("Desk Metrics", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-desk-table", style=CARD_STYLE),
    ])
