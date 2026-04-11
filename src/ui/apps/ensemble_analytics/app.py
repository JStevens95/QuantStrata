"""
Dash application factory for the Ensemble Analytics dashboard.

Creates the Dash app instance, builds the top-level layout (header +
tab container), registers all callbacks, and initialises the data layer.
"""
from __future__ import annotations

import logging
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import (
    APP_TITLE,
    TAB_ORDER,
    TAB_OVERVIEW,
)
from src.ui.apps.ensemble_analytics.theme.colors import BG_PRIMARY, TEXT_PRIMARY, ACCENT_BLUE
from src.ui.apps.ensemble_analytics.theme.styles import NAVBAR_STYLE, CONTAINER_STYLE
from src.ui.apps.ensemble_analytics.theme.plotly_template import PLOTLY_TEMPLATE

logger = logging.getLogger(__name__)


def create_app(
    registry_dir: str,
    artifacts_dir: str,
    version: str = "latest",
    debug: bool = False,
) -> dash.Dash:
    """
    Build and return the configured Dash application.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load on startup.
    debug : bool
        Enable Dash debug mode (hot-reload, verbose errors).

    Returns
    -------
    dash.Dash
        Fully configured application ready for ``app.run()``.
    """
    import plotly.io as pio
    pio.templates["ensemble_dark"] = PLOTLY_TEMPLATE
    pio.templates.default = "ensemble_dark"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title=APP_TITLE,
    )

    # ── Initialise data layer ─────────────────────────────────────
    from src.ui.apps.ensemble_analytics.data.session_manager import initialise
    initialise(registry_dir, artifacts_dir, version)

    # ── Build layout ──────────────────────────────────────────────
    app.layout = _build_layout(version)

    # ── Register callbacks ────────────────────────────────────────
    from src.ui.apps.ensemble_analytics.callbacks import register_all_callbacks
    register_all_callbacks(app)

    return app


def _build_layout(version: str) -> dbc.Container:
    """
    Assemble the top-level page layout.

    Structure: navbar → version store → tab bar → tab content container.
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session

    session = get_session()
    config = session.config

    # Build ensemble version options for the dropdown
    ens_registry = session._ens_registry
    available_versions = []
    try:
        available_versions = [
            {
                "label": f"{v['version']} ({v['n_members']} clusters, {v['n_trades']} trades)",
                "value": v["version"],
            }
            for v in ens_registry.list_versions()
        ]
    except Exception:
        available_versions = [{"label": version, "value": version}]

    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H4(
                                APP_TITLE,
                                className="mb-0",
                                style={"color": TEXT_PRIMARY, "fontWeight": "600"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ensemble-version-selector",
                                options=available_versions,
                                value=session.ensemble_version,
                                clearable=False,
                                style={
                                    "width": "340px",
                                    "backgroundColor": BG_PRIMARY,
                                    "color": TEXT_PRIMARY,
                                },
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span(
                                f"{config.n_members} clusters · "
                                f"{len(config.all_trade_ids)} trades",
                                style={"color": "#8b949e", "fontSize": "13px"},
                            ),
                            width="auto",
                            className="ms-3 d-flex align-items-center",
                        ),
                    ],
                    align="center",
                    className="g-3",
                ),
            ],
            fluid=True,
        ),
        style=NAVBAR_STYLE,
        dark=True,
    )

    tab_bar = dcc.Tabs(
        id="main-tabs",
        value=TAB_OVERVIEW,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in TAB_ORDER
        ],
        style={"borderBottom": "1px solid #30363d"},
    )

    return dbc.Container(
        [
            navbar,
            dcc.Store(id="active-split", data="test"),
            dcc.Store(id="active-cluster", data=None),
            html.Div(style={"height": "8px"}),
            tab_bar,
            html.Div(id="tab-content", style={"marginTop": "16px"}),
        ],
        fluid=True,
        style=CONTAINER_STYLE,
    )
