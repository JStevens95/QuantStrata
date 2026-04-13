"""
Evaluation sub-tab: By Product.

Same pattern as By Desk, grouped by ``product_type`` / ``product_subtype``.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By Product sub-tab skeleton."""
    return html.Div([
        dbc.Row([
            dbc.Col(html.Div(id="eval-product-timeseries", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-product-boxplot", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div(id="eval-product-scatter-grid", style=CARD_STYLE),
        html.Div("Product Metrics", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-product-table", style=CARD_STYLE),
    ])
