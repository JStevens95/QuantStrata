"""
Evaluation sub-tab: Portfolio.

Full-book PnL analysis — predictions vs targets summed across all
trades.  Includes time-series overlay, scatter, residual distribution,
percentile table, and worst-scenario table.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the Portfolio sub-tab skeleton (populated by callback)."""
    return html.Div([
        # Time-series + scatter row
        dbc.Row([
            dbc.Col(html.Div(id="eval-portfolio-ts", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-portfolio-scatter", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        # Residual distribution + percentile table
        dbc.Row([
            dbc.Col(html.Div(id="eval-portfolio-residual", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-portfolio-percentile", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        # Worst scenarios table
        html.Div("Worst Scenarios", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-portfolio-worst", style=CARD_STYLE),
    ])
