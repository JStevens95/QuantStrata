"""
Tab 1 — Overview layout.

Renders the production-readiness dashboard: KPI cards, portfolio scatter,
member comparison bar, cluster heatmap, and sortable member table.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle
from src.ui.apps.ensemble_analytics.components.kpi_card import kpi_card
from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE
from src.ui.apps.ensemble_analytics.config import DEFAULT_SPLIT


def layout() -> html.Div:
    """
    Build the Overview tab layout.

    All dynamic content is rendered by callbacks via the placeholder
    ``id`` attributes.  This function builds the static skeleton.

    Returns
    -------
    html.Div
    """
    return html.Div([
        split_toggle(id_prefix="overview", default=DEFAULT_SPLIT),

        # KPI row (populated by callback)
        dbc.Row(id="overview-kpi-row", className="g-3 mb-4"),

        # Charts row
        dbc.Row([
            dbc.Col(
                html.Div(id="overview-scatter-container", style=CARD_STYLE),
                md=6,
            ),
            dbc.Col(
                html.Div(id="overview-bar-container", style=CARD_STYLE),
                md=6,
            ),
        ], className="g-3 mb-4"),

        # Cluster heatmap
        html.Div(
            "Cluster Performance",
            style=SECTION_TITLE_STYLE,
        ),
        html.Div(id="overview-heatmap-container", style=CARD_STYLE),

        # Member KPI table
        html.Div(
            "Member Metrics",
            style=SECTION_TITLE_STYLE,
        ),
        html.Div(id="overview-table-container", style=CARD_STYLE),
    ])
