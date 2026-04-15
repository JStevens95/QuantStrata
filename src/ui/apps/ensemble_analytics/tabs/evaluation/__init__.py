"""
Tab 2 — Evaluation sub-tab container.

All filter dropdowns live in a persistent filter area that is always
in the DOM.  A callback toggles visibility of each filter based on the
active sub-tab, eliminating Dash lifecycle issues with missing IDs.
"""
from __future__ import annotations

from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import EVAL_SUB_ORDER, EVAL_SUB_PORTFOLIO
from src.ui.apps.ensemble_analytics.components.split_toggle import split_toggle
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


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

    _label = {
        "color": TEXT_SECONDARY, "fontSize": "12px",
        "marginRight": "6px", "fontWeight": "600",
    }
    _row = {"display": "flex", "alignItems": "center", "marginRight": "20px"}
    _dd = {"width": "260px", "fontSize": "13px"}

    filter_area = html.Div(
        [
            html.Div(
                [html.Label("Desk:", style=_label),
                 dcc.Dropdown(id="eval-desk-filter-desk", multi=True, style=_dd)],
                id="eval-filter-desk", style={**_row, "display": "none"},
            ),
            html.Div(
                [html.Label("Product:", style=_label),
                 dcc.Dropdown(id="eval-product-filter-product_type", multi=True, style=_dd)],
                id="eval-filter-product", style={**_row, "display": "none"},
            ),
            html.Div(
                [html.Label("Currency:", style=_label),
                 dcc.Dropdown(id="eval-ccy-filter-ccy", multi=True, style=_dd)],
                id="eval-filter-ccy", style={**_row, "display": "none"},
            ),
            html.Div(
                [html.Label("Cluster:", style=_label),
                 dcc.Dropdown(id="eval-cluster-cluster-dropdown", clearable=False,
                              style={**_dd, "width": "600px"})],
                id="eval-filter-cluster", style={**_row, "display": "none"},
            ),
        ],
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "12px"},
    )

    return html.Div([
        split_toggle(id_prefix="eval", default="test"),
        sub_tabs,
        filter_area,
        html.Div(id="eval-sub-tab-content", style={"marginTop": "16px"}),
    ])
