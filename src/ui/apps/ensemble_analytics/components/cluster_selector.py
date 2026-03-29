"""
Cluster selector dropdown.

Shows cluster IDs with optional attribute labels (ccy, desk, product)
for quick identification.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from dash import dcc, html

from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def cluster_selector(
    cluster_ids: List[str],
    cluster_attrs: Optional[Dict[str, Dict[str, Any]]] = None,
    id_prefix: str = "cluster",
    multi: bool = False,
    default: Optional[str] = None,
) -> html.Div:
    """
    Build a cluster selection dropdown.

    Parameters
    ----------
    cluster_ids : list of str
        Available cluster IDs.
    cluster_attrs : dict, optional
        ``{cluster_id: {ccy: ..., desk: ..., product: ...}}``.
        When provided, labels show attributes alongside the ID.
    id_prefix : str
        Component ID prefix.
    multi : bool
        Allow multi-selection.
    default : str, optional
        Default selected cluster.  Falls back to the first ID.

    Returns
    -------
    html.Div
    """
    options = []
    for cid in cluster_ids:
        if cluster_attrs and cid in cluster_attrs:
            attrs = cluster_attrs[cid]
            parts = [f"{k}={v}" for k, v in attrs.items() if v is not None]
            label = f"{cid}  ({', '.join(parts)})" if parts else cid
        else:
            label = cid
        options.append({"label": label, "value": cid})

    return html.Div(
        [
            html.Label(
                "Cluster:",
                style={
                    "color": TEXT_SECONDARY,
                    "fontSize": "12px",
                    "marginRight": "8px",
                    "fontWeight": "600",
                },
            ),
            dcc.Dropdown(
                id=f"{id_prefix}-cluster-dropdown",
                options=options,
                value=default or (cluster_ids[0] if cluster_ids else None),
                multi=multi,
                clearable=False,
                style={"width": "400px", "fontSize": "13px"},
            ),
        ],
        style={"display": "flex", "alignItems": "center", "marginBottom": "16px"},
    )
