"""
KPI card component.

Renders a compact metric card with a title, formatted value, and an
optional delta badge showing change direction with colour coding.
"""
from __future__ import annotations

from typing import Optional

import dash_bootstrap_components as dbc
from dash import html

from src.ui.apps.ensemble_analytics.theme.colors import (
    ACCENT_GREEN,
    ACCENT_RED,
    TEXT_SECONDARY,
)
from src.ui.apps.ensemble_analytics.theme.styles import (
    CARD_STYLE,
    CARD_HEADER_STYLE,
    KPI_VALUE_STYLE,
)


def kpi_card(
    title: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: Optional[str] = None,
    card_id: Optional[str] = None,
) -> dbc.Card:
    """
    Build a KPI display card.

    Parameters
    ----------
    title : str
        Metric label (e.g. ``"MAE"``).
    value : str
        Pre-formatted metric value (e.g. ``"0.0342"``).
    delta : str, optional
        Delta string (e.g. ``"+2.1%"``).
    delta_color : str, optional
        Override colour for the delta badge.  Defaults to green for
        values starting with ``"-"`` (improvement) and red otherwise.
    card_id : str, optional
        HTML id for callback targeting.

    Returns
    -------
    dbc.Card
    """
    children = [
        html.Div(title, style=CARD_HEADER_STYLE),
        html.Div(value, style=KPI_VALUE_STYLE),
    ]

    if delta is not None:
        if delta_color is None:
            delta_color = ACCENT_GREEN if delta.startswith("-") else ACCENT_RED
        children.append(
            html.Span(
                delta,
                style={
                    "fontSize": "12px",
                    "fontWeight": "600",
                    "color": delta_color,
                    "marginTop": "4px",
                    "display": "inline-block",
                },
            )
        )

    props = {"style": CARD_STYLE, "children": children}
    if card_id:
        props["id"] = card_id
    return dbc.Card(**props)
