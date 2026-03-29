"""
Phase 3 loading progress bar for the Inference tab.

Shows how many clusters have been loaded out of the total, with a
visual progress bar and status text.
"""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, TEXT_SECONDARY


def loading_progress(
    total: int,
    loaded: int,
    id_prefix: str = "inference",
) -> html.Div:
    """
    Build a loading progress indicator.

    Parameters
    ----------
    total : int
        Total number of clusters to load.
    loaded : int
        Number of clusters loaded so far.
    id_prefix : str
        Component ID prefix.

    Returns
    -------
    html.Div
    """
    pct = int((loaded / total) * 100) if total > 0 else 0

    return html.Div(
        [
            html.Div(
                f"Loading models: {loaded} / {total} clusters ({pct}%)",
                id=f"{id_prefix}-progress-text",
                style={"color": TEXT_SECONDARY, "fontSize": "13px", "marginBottom": "8px"},
            ),
            dbc.Progress(
                id=f"{id_prefix}-progress-bar",
                value=pct,
                max=100,
                color="info",
                style={"height": "8px"},
            ),
        ],
        id=f"{id_prefix}-progress-container",
    )
