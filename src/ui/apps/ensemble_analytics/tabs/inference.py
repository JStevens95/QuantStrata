"""
Tab 6 — Inference layout.

Provides model loading, scenario upload, inference execution, and
results display.  The only tab that triggers Phase 3.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.styles import CARD_STYLE, SECTION_TITLE_STYLE
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY, ACCENT_BLUE


def layout() -> html.Div:
    """Build the Inference tab layout."""
    return html.Div([
        # Phase 3 loading section
        html.Div([
            html.Div("Model Loading", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-load-status", style={"marginBottom": "12px"}),
            dbc.Button(
                "Load Models",
                id="inference-load-btn",
                color="primary",
                size="sm",
                style={"marginBottom": "16px"},
            ),
            html.Div(id="inference-progress-container"),
        ], style=CARD_STYLE),

        # Inference controls
        html.Div([
            html.Div("Run Inference", style=SECTION_TITLE_STYLE),
            dbc.Row([
                dbc.Col([
                    html.Label("Mode:", style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
                    dcc.Dropdown(
                        id="inference-mode",
                        options=[
                            {"label": "New Scenarios", "value": "new_scenarios"},
                            {"label": "New Trades (coming soon)", "value": "new_trades", "disabled": True},
                        ],
                        value="new_scenarios",
                        clearable=False,
                        style={"width": "250px"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Scenario Directory:", style={"color": TEXT_SECONDARY, "fontSize": "12px"}),
                    dcc.Input(
                        id="inference-scenario-dir",
                        type="text",
                        placeholder="/path/to/new_scenario_shocks/",
                        style={"width": "100%", "fontSize": "13px"},
                    ),
                ], width=6),
                dbc.Col([
                    html.Br(),
                    dbc.Button(
                        "Run",
                        id="inference-run-btn",
                        color="success",
                        size="sm",
                        disabled=True,
                    ),
                ], width=2, className="d-flex align-items-end"),
            ], className="mb-3"),
        ], style=CARD_STYLE),

        # Results
        html.Div([
            html.Div("Results", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-results-container"),
        ], style=CARD_STYLE),

        # Scenario-level table + export (G15)
        html.Div([
            html.Div("Scenario-Level Predictions", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-scenario-table-container"),
            dbc.Button("Download CSV", id="inference-download-csv-btn", size="sm",
                       color="secondary", className="me-2 mt-2"),
            dcc.Download(id="inference-download-csv"),
        ], style=CARD_STYLE),

        # Stress scenario comparison (G15)
        html.Div([
            html.Div("Stress Scenario Comparison", style=SECTION_TITLE_STYLE),
            html.Div(id="inference-stress-comparison"),
        ], style=CARD_STYLE),
    ])
