"""Rade-themed ``dash-ag-grid`` wrapper plus DataFrame helper.

Every tabular view in the app (trade catalogue, cluster leaderboard,
residual browser, governance evidence log) uses this wrapper so the
dark Alpine theme, row striping, pagination and sensible column
defaults live in exactly one place.

Design spec anchors
-------------------
* §6 Components — Grid is ``ag-theme-alpine-dark`` + ``rade-grid``
  (overrides in ``rade.css``).
* §8 Table defaults — 50 rows per page, 36 px row height, sortable +
  filterable + resizable by default.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import dash_ag_grid as dag

if TYPE_CHECKING:
    # Keep the runtime coupling soft; most pages will already have
    # imported pandas, so :func:`columns_from_dataframe` just duck-types
    # the dtype lookup.
    import pandas as pd


# Default ag-grid ``defaultColDef`` — caller may replace wholesale or
# merge field-by-field via the ``default_col_def`` kwarg.
_DEFAULT_COL_DEF: Dict[str, Any] = {
    "sortable": True,
    "filter": True,
    "resizable": True,
    "flex": 1,
    "minWidth": 100,
    "suppressMovable": False,
}


# Default ag-grid ``dashGridOptions`` — pagination on, Alpine spacing,
# no row animation (keeps large tables responsive).
_DEFAULT_GRID_OPTIONS: Dict[str, Any] = {
    "pagination": True,
    "paginationPageSize": 50,
    "paginationPageSizeSelector": [25, 50, 100, 200],
    "rowHeight": 36,
    "headerHeight": 38,
    "animateRows": False,
    "suppressCellFocus": True,
    "domLayout": "normal",
}


def AgGridTable(
    *,
    grid_id: str,
    row_data: Optional[List[Dict[str, Any]]] = None,
    column_defs: Optional[List[Dict[str, Any]]] = None,
    default_col_def: Optional[Dict[str, Any]] = None,
    grid_options: Optional[Dict[str, Any]] = None,
    height: int = 360,
    className: str = "",
    **ag_grid_kwargs: Any,
) -> dag.AgGrid:
    """Rade-themed AG Grid.

    Parameters
    ----------
    grid_id
        DOM id of the ``dag.AgGrid`` — target this from callbacks.
    row_data
        Initial rows as list-of-dicts, or ``None`` for an empty table
        (callback will backfill).
    column_defs
        List of ag-grid column definitions.  Use
        :func:`columns_from_dataframe` for quick defaults.
    default_col_def
        Override or extend :data:`_DEFAULT_COL_DEF`.  Keys you supply
        win; keys you omit fall through to defaults.
    grid_options
        Override or extend :data:`_DEFAULT_GRID_OPTIONS` the same way.
    height
        Outer card height in pixels.  Default 360.
    className
        Extra Tailwind classes appended to the grid's className.
    **ag_grid_kwargs
        Any other kwarg forwarded straight to ``dag.AgGrid`` (e.g.
        ``getRowId``, ``rowSelection``).
    """
    merged_col_def = {**_DEFAULT_COL_DEF, **(default_col_def or {})}
    merged_grid_options = {**_DEFAULT_GRID_OPTIONS, **(grid_options or {})}

    # Defensive merge: any duplicate keys in ``ag_grid_kwargs`` lose to
    # the wrapper's own values for ``className`` / ``style``.  Without
    # this guard a caller threading kwargs through ``**extra`` (or a
    # stale-bytecode build of an older ``AgGridTable`` signature) would
    # raise ``TypeError: AgGrid() got multiple values for keyword
    # argument 'className'`` from the line below.
    forwarded: Dict[str, Any] = {**ag_grid_kwargs}
    forwarded["className"] = (
        f"ag-theme-alpine-dark rade-grid {className}".strip()
    )
    forwarded.setdefault("style", {"height": f"{height}px"})

    return dag.AgGrid(
        id=grid_id,
        rowData=row_data or [],
        columnDefs=column_defs or [],
        defaultColDef=merged_col_def,
        dashGridOptions=merged_grid_options,
        **forwarded,
    )


def columns_from_dataframe(
    df: "pd.DataFrame",
    *,
    numeric_format: Optional[str] = None,
    header_overrides: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Generate minimal ag-grid ``columnDefs`` from a DataFrame.

    The result is deliberately plain: ``field``, ``headerName`` and
    ``type="numericColumn"`` for numeric dtypes.  Pages that need
    value formatters or cell renderers should build ``columnDefs``
    by hand — this helper is for prototypes and uniform tables.

    Parameters
    ----------
    df
        Source DataFrame — columns mapped one-to-one.
    numeric_format
        Optional d3-format spec (e.g. ``".4~g"``).  When set, numeric
        columns gain a ``valueFormatter`` using that spec.
    header_overrides
        Optional map ``{column_name: display_label}`` so you can
        rename e.g. ``"AssetClassCode" -> "Asset class"`` without
        touching the DataFrame.
    """
    import pandas as pd  # lazy import so the module is pandas-free

    header_overrides = header_overrides or {}
    defs: List[Dict[str, Any]] = []
    for col in df.columns:
        col_def: Dict[str, Any] = {
            "field": str(col),
            "headerName": header_overrides.get(col, str(col)),
        }
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            col_def["type"] = "numericColumn"
            if numeric_format:
                col_def["valueFormatter"] = {
                    "function": f"d3.format('{numeric_format}')(params.value)"
                }
        defs.append(col_def)
    return defs


__all__ = ["AgGridTable", "columns_from_dataframe"]
