"""
Tab 2 — Evaluation sub-tab container.

Renders the sub-tab bar and delegates to individual sub-tab layout
modules.  The active sub-tab content is swapped by a callback.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import EVAL_SUB_ORDER, EVAL_SUB_PORTFOLIO
from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle


def layout() -> html.Div:
    """
    Build the Evaluation tab layout with sub-tab navigation.

    Returns
    -------
    html.Div
    """
    sub_tabs = dcc.Tabs(
        id="eval-sub-tabs",
        value=EVAL_SUB_PORTFOLIO,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in EVAL_SUB_ORDER
        ],
    )

    return html.Div([
        split_toggle(id_prefix="eval", default="test"),
        sub_tabs,
        html.Div(id="eval-sub-tab-content", style={"marginTop": "16px"}),
    ])
