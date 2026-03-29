"""
Multi-select filter bar for Evaluation sub-tabs.

Renders dropdowns for desk, product_type, and ccy columns from the
global trade catalogue.  Returns a mask-building function alongside
the layout.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd
from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def filter_bar(
    catalogue: pd.DataFrame,
    id_prefix: str,
    columns: Optional[List[str]] = None,
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
        Catalogue columns to create filters for.  Defaults to
        ``["desk", "product_type", "ccy"]`` (skipping any not present
        in the catalogue).

    Returns
    -------
    html.Div
    """
    if columns is None:
        columns = ["desk", "product_type", "ccy"]
    columns = [c for c in columns if c in catalogue.columns]

    children = []
    for col in columns:
        unique_vals = sorted(catalogue[col].dropna().unique().tolist())
        children.append(
            html.Div(
                [
                    html.Label(
                        col.replace("_", " ").title() + ":",
                        style={
                            "color": TEXT_SECONDARY,
                            "fontSize": "12px",
                            "marginRight": "6px",
                            "fontWeight": "600",
                        },
                    ),
                    dcc.Dropdown(
                        id=f"{id_prefix}-filter-{col}",
                        options=[{"label": v, "value": v} for v in unique_vals],
                        multi=True,
                        placeholder=f"All {col.replace('_', ' ').title()}s",
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

    return html.Div(
        children,
        style={"display": "flex", "flexWrap": "wrap", "marginBottom": "16px"},
    )
