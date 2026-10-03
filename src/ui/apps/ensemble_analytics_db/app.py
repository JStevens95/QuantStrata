"""
Dash application factory for the pre-computed Ensemble Analytics dashboard.

Shares all UI code (tabs, figures, components, theme) with the original
``ensemble_analytics`` app but reads pre-computed aggregates from
``evaluation/`` via a ``DataBackend`` instead of the
``GlobalPredictionStore``.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

import pandas  # noqa: F401 — pre-load before callbacks to avoid import race
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

_DB_APP_TITLE = "Ensemble Analytics (DB) — Hybrid GNN-RNN"


def create_app(
    registry_dir: str = "",
    artifacts_dir: str = "",
    version: str = "latest",
    backend: str = "file",
    db_path: Optional[str] = None,
    api_url: Optional[str] = None,
    debug: bool = False,
) -> dash.Dash:
    """
    Build and return the DB-ready Dash application.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load on startup.
    backend : str
        Data backend: ``"file"``, ``"sqlite"``, ``"cache"``
        (in-memory via ArtifactCache), or ``"api"`` (remote FastAPI server).
    db_path : str or None
        Path to SQLite database (required when *backend* is ``"sqlite"``).
    api_url : str or None
        FastAPI server URL (required when *backend* is ``"api"``).
    debug : bool
        Enable Dash debug mode.

    Returns
    -------
    dash.Dash
        Fully configured application ready for ``app.run()``.
    """
    t0 = time.perf_counter()
    import plotly.io as pio
    pio.templates["ensemble_dark"] = PLOTLY_TEMPLATE
    pio.templates.default = "ensemble_dark"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title=_DB_APP_TITLE,
    )

    from src.ui.apps.ensemble_analytics_db.data.session_manager import initialise
    initialise(
        registry_dir, artifacts_dir, version,
        backend=backend, db_path=db_path, api_url=api_url,
    )
    logger.info("[STARTUP] initialise done: %.1fs", time.perf_counter() - t0)

    app.layout = _build_layout(version)
    logger.info("[STARTUP] layout built: %.1fs", time.perf_counter() - t0)

    from src.ui.apps.ensemble_analytics_db.callbacks import register_all_callbacks
    register_all_callbacks(app)
    logger.info("[STARTUP] create_app complete: %.1fs", time.perf_counter() - t0)

    return app


def _build_layout(version: str) -> dbc.Container:
    """Assemble the top-level page layout (shared with original app)."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend

    backend = get_backend()
    meta = backend.get_ensemble_version_meta()

    available_versions = []
    try:
        for v in backend.get_registry_versions():
            n_c = v.get("n_clusters", v.get("n_members", 0))
            n_t = v.get("n_trades", 0)
            available_versions.append({
                "label": f"{v['version']} ({n_c} clusters, {n_t} trades)",
                "value": v["version"],
            })
    except Exception:
        available_versions = [{"label": version, "value": version}]

    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H4(
                                _DB_APP_TITLE,
                                className="mb-0",
                                style={"color": TEXT_PRIMARY, "fontWeight": "600"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ensemble-version-selector",
                                options=available_versions,
                                value=backend.get_ensemble_version(),
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
                                f"{meta['n_clusters']} clusters · "
                                f"{meta['n_trades']} trades",
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
