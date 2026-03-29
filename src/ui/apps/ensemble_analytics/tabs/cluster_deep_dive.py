"""
Tab 3 — Cluster Deep Dive layout.

Forensic single-cluster view with cluster selector, split comparison,
per-trade scatter, residual distribution, and config summary.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE
from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle


def layout() -> html.Div:
    """Build the Cluster Deep Dive tab skeleton."""
    return html.Div([
        html.Div(id="deep-dive-cluster-selector"),
        split_toggle(id_prefix="deep-dive", default="test"),

        html.Div(id="deep-dive-header", style=CARD_STYLE),

        html.Div("Split Comparison", style=SECTION_TITLE_STYLE),
        html.Div(id="deep-dive-split-table", style=CARD_STYLE),

        dbc.Row([
            dbc.Col(html.Div(id="deep-dive-convergence", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="deep-dive-scatter", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        dbc.Row([
            dbc.Col(html.Div(id="deep-dive-residual", style=CARD_STYLE), md=6),
            dbc.Col(html.Div(id="deep-dive-scatter-matrix", style=CARD_STYLE), md=6),
        ], className="g-3 mb-3"),

        html.Div("Elementary PnL Explorer", style=SECTION_TITLE_STYLE),
        html.Div(id="deep-dive-elementary", style=CARD_STYLE),

        html.Div("Configuration Summary", style=SECTION_TITLE_STYLE),
        html.Div(id="deep-dive-config", style=CARD_STYLE),
    ])
