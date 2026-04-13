"""
Tab 2 — Evaluation sub-tab container.

All sub-tab layouts are rendered simultaneously (hidden by default)
so that filter dropdowns always exist in the DOM for Dash callbacks.
A callback toggles visibility based on the active sub-tab.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import (
    EVAL_SUB_ORDER,
    EVAL_SUB_PORTFOLIO,
    EVAL_SUB_DESK,
    EVAL_SUB_PRODUCT,
    EVAL_SUB_CCY,
    EVAL_SUB_CLUSTER,
)
from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle
from src.ui.apps.ensemble_analytics.tabs.evaluation.portfolio import layout as portfolio_layout
from src.ui.apps.ensemble_analytics.tabs.evaluation.by_desk import layout as desk_layout
from src.ui.apps.ensemble_analytics.tabs.evaluation.by_product import layout as product_layout
from src.ui.apps.ensemble_analytics.tabs.evaluation.by_ccy import layout as ccy_layout
from src.ui.apps.ensemble_analytics.tabs.evaluation.by_cluster import layout as cluster_layout

_SUB_TAB_IDS = [
    EVAL_SUB_PORTFOLIO,
    EVAL_SUB_DESK,
    EVAL_SUB_PRODUCT,
    EVAL_SUB_CCY,
    EVAL_SUB_CLUSTER,
]

_SUB_TAB_LAYOUTS = {
    EVAL_SUB_PORTFOLIO: portfolio_layout,
    EVAL_SUB_DESK: desk_layout,
    EVAL_SUB_PRODUCT: product_layout,
    EVAL_SUB_CCY: ccy_layout,
    EVAL_SUB_CLUSTER: cluster_layout,
}


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

    panels = []
    for tab_id in _SUB_TAB_IDS:
        is_default = tab_id == EVAL_SUB_PORTFOLIO
        panels.append(
            html.Div(
                _SUB_TAB_LAYOUTS[tab_id](),
                id=f"eval-panel-{tab_id}",
                style={"display": "block" if is_default else "none"},
            )
        )

    return html.Div([
        split_toggle(id_prefix="eval", default="test"),
        sub_tabs,
        html.Div(panels, id="eval-sub-tab-content", style={"marginTop": "16px"}),
    ])
