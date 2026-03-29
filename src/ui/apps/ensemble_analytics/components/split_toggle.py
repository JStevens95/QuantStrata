"""
Split toggle component (Train / Val / Test radio buttons).

Local to each tab that needs it — not a global control.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import SPLITS, DEFAULT_SPLIT
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def split_toggle(id_prefix: str, default: str = DEFAULT_SPLIT) -> html.Div:
    """
    Build a compact horizontal radio-button group for split selection.

    Parameters
    ----------
    id_prefix : str
        Prefix for the component ID (e.g. ``"overview"`` produces
        ``"overview-split-toggle"``).
    default : str
        Initially selected split.

    Returns
    -------
    html.Div
    """
    return html.Div(
        [
            html.Label(
                "Split:",
                style={
                    "color": TEXT_SECONDARY,
                    "fontSize": "12px",
                    "marginRight": "8px",
                    "fontWeight": "600",
                },
            ),
            dcc.RadioItems(
                id=f"{id_prefix}-split-toggle",
                options=[{"label": s.capitalize(), "value": s} for s in SPLITS],
                value=default,
                inline=True,
                labelStyle={
                    "marginRight": "16px",
                    "fontSize": "13px",
                    "cursor": "pointer",
                },
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
    )
