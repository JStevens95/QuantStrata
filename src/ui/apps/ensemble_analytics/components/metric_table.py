"""
AG Grid metric table component.

Wraps ``dash_ag_grid.AgGrid`` with dark-theme defaults and optional
conditional formatting (colour-code cells by metric value).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import dash_ag_grid as dag

from src.ui.apps.ensemble_analytics.theme.colors import BG_CARD, TEXT_PRIMARY, BORDER


def metric_table(
    column_defs: List[Dict[str, Any]],
    row_data: List[Dict[str, Any]],
    table_id: str,
    height: str = "400px",
    sort_model: Optional[List[Dict[str, str]]] = None,
) -> dag.AgGrid:
    """
    Build a dark-themed AG Grid table.

    Parameters
    ----------
    column_defs : list of dict
        AG Grid column definitions.  Each dict must have at least
        ``"field"`` and ``"headerName"`` keys.
    row_data : list of dict
        Row records.
    table_id : str
        Dash component ID.
    height : str
        CSS height string.
    sort_model : list of dict, optional
        Initial sort (e.g. ``[{"colId": "mae", "sort": "asc"}]``).

    Returns
    -------
    dag.AgGrid
    """
    grid_options: Dict[str, Any] = {
        "animateRows": True,
        "pagination": False,
    }
    if sort_model:
        grid_options["sortModel"] = sort_model

    return dag.AgGrid(
        id=table_id,
        columnDefs=column_defs,
        rowData=row_data,
        defaultColDef={
            "sortable": True,
            "resizable": True,
            "filter": True,
        },
        dashGridOptions=grid_options,
        style={
            "height": height,
            "--ag-background-color": BG_CARD,
            "--ag-header-background-color": BG_CARD,
            "--ag-odd-row-background-color": BG_CARD,
            "--ag-row-hover-color": "#1c2333",
            "--ag-foreground-color": TEXT_PRIMARY,
            "--ag-border-color": BORDER,
            "--ag-header-foreground-color": TEXT_PRIMARY,
            "--ag-font-size": "13px",
        },
    )
