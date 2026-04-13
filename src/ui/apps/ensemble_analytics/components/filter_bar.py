"""
Multi-select filter bar for Evaluation sub-tabs.

Renders dropdowns for desk, product_type, and ccy columns from the
global trade catalogue.  Returns a mask-building function alongside
the layout.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def filter_bar(
    catalogue: pd.DataFrame,
    id_prefix: str,
    columns: Optional[List[str]] = None,
    catalogue_columns: Optional[Dict[str, str]] = None,
) -> html.Div:
    """
    Build a horizontal row of multi-select dropdowns.

    Parameters
    ----------
    catalogue : pd.DataFrame
        Global trade catalogue.
    id_prefix : str
        Component ID prefix (e.g. ``"eval-desk"``).
    columns : list of str, optional
        Logical column names used for component IDs.  Defaults to
        ``["desk", "product_type", "ccy"]``.
    catalogue_columns : dict, optional
        ``{logical_name: actual_catalogue_column}``.  When provided,
        the dropdown filters by the mapped column but keeps the
        logical name for the component ID.  If ``None``, logical names
        are used directly as catalogue columns.

    Returns
    -------
    html.Div
    """
    if columns is None:
        columns = ["desk", "product_type", "ccy"]
    if catalogue_columns is None:
        catalogue_columns = {}

    children = []
    for col in columns:
        actual_col = catalogue_columns.get(col, col)
        dropdown_id = f"{id_prefix}-filter-{col}"

        if actual_col in catalogue.columns:
            unique_vals = sorted(catalogue[actual_col].dropna().unique().tolist())
            children.append(
                html.Div(
                    [
                        html.Label(
                            actual_col.replace("_", " ").title() + ":",
                            style={
                                "color": TEXT_SECONDARY,
                                "fontSize": "12px",
                                "marginRight": "6px",
                                "fontWeight": "600",
                            },
                        ),
                        dcc.Dropdown(
                            id=dropdown_id,
                            options=[{"label": v, "value": v} for v in unique_vals],
                            multi=True,
                            placeholder=f"All {actual_col.replace('_', ' ').title()}s",
                            style={"width": "220px", "fontSize": "13px"},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "alignItems": "center",
                        "marginRight": "20px",
                    },
                )
            )
        else:
            children.append(
                dcc.Dropdown(id=dropdown_id, multi=True, style={"display": "none"})
            )

    return html.Div(
        children,
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "16px"},
    )
