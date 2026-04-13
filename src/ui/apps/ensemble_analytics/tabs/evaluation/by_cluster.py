"""
Evaluation sub-tab: By Cluster.

Per-cluster view with cluster dropdown, per-trade PnL heatmap,
scatter, trade-level metrics table, and violin overlay.
"""
from __future__ import annotations

from dash import html
import dash_bootstrap_components as dbc

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE


def layout() -> html.Div:
    """Build the By Cluster sub-tab skeleton."""
    return html.Div([
        html.Div(id="eval-cluster-heatmap", style=CARD_STYLE),
        dbc.Row([
            dbc.Col(html.Div(id="eval-cluster-scatter", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="eval-cluster-violin", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),
        html.Div("Trade-Level Metrics", style=SECTION_TITLE_STYLE),
        html.Div(id="eval-cluster-trade-table", style=CARD_STYLE),
    ])
