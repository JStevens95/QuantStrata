"""
Graph / network figure builders for ``dash-cytoscape``.

Returns element lists suitable for ``cyto.Cytoscape(elements=...)``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def build_cytoscape_elements(
    trade_ids: List[str],
    indices: np.ndarray,
    values: np.ndarray,
    node_attrs: Optional[Dict[str, Dict[str, Any]]] = None,
    weight_threshold: float = 0.0,
) -> List[Dict[str, Any]]:
    """
    Convert sparse adjacency to cytoscape element dicts.

    Parameters
    ----------
    trade_ids : list of str
        Node labels.
    indices : np.ndarray
        Shape ``[2, nnz]`` or ``[nnz, 2]`` — edge endpoints.
    values : np.ndarray
        Edge weights, shape ``[nnz]``.
    node_attrs : dict, optional
        ``{trade_id: {attr: val, ...}}``.  Extra attributes are added to
        each node's ``data`` dict (useful for colouring / sizing).
    weight_threshold : float
        Only include edges with ``abs(weight) > threshold``.

    Returns
    -------
    list of dict
        Cytoscape element dicts (nodes + edges).
    """
    elements: List[Dict[str, Any]] = []

    for tid in trade_ids:
        node_data: Dict[str, Any] = {"id": tid, "label": tid}
        if node_attrs and tid in node_attrs:
            node_data.update(node_attrs[tid])
        elements.append({"data": node_data})

    if indices.ndim == 2 and indices.shape[0] == 2:
        rows, cols = indices[0], indices[1]
    elif indices.ndim == 2 and indices.shape[1] == 2:
        rows, cols = indices[:, 0], indices[:, 1]
    else:
        return elements

    for i in range(len(values)):
        if abs(values[i]) <= weight_threshold:
            continue
        src = trade_ids[int(rows[i])] if int(rows[i]) < len(trade_ids) else str(int(rows[i]))
        tgt = trade_ids[int(cols[i])] if int(cols[i]) < len(trade_ids) else str(int(cols[i]))
        elements.append({
            "data": {
                "source": src,
                "target": tgt,
                "weight": float(values[i]),
            }
        })

    return elements


def ego_network(
    center_id: str,
    trade_ids: List[str],
    indices: np.ndarray,
    values: np.ndarray,
    hops: int = 1,
) -> List[Dict[str, Any]]:
    """
    Extract the ego (neighbourhood) subgraph around a single node.

    Parameters
    ----------
    center_id : str
        Focal node.
    trade_ids : list of str
        Node labels.
    indices : np.ndarray
        Sparse adjacency indices.
    values : np.ndarray
        Edge weights.
    hops : int
        Number of hops from the centre.

    Returns
    -------
    list of dict
        Cytoscape element dicts for the subgraph.
    """
    if center_id not in trade_ids:
        return []

    tid_to_idx = {tid: i for i, tid in enumerate(trade_ids)}
    center_idx = tid_to_idx[center_id]

    if indices.ndim == 2 and indices.shape[0] == 2:
        rows, cols = indices[0], indices[1]
    else:
        rows, cols = indices[:, 0], indices[:, 1]

    visited = {center_idx}
    frontier = {center_idx}
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            mask_src = rows == node
            mask_tgt = cols == node
            neighbours = set(cols[mask_src].tolist()) | set(rows[mask_tgt].tolist())
            next_frontier |= neighbours - visited
        visited |= next_frontier
        frontier = next_frontier

    visited_set = visited
    elements: List[Dict[str, Any]] = []
    for idx in visited_set:
        if idx < len(trade_ids):
            data: Dict[str, Any] = {"id": trade_ids[idx], "label": trade_ids[idx]}
            if idx == center_idx:
                data["is_center"] = True
            elements.append({"data": data})

    for i in range(len(values)):
        r, c = int(rows[i]), int(cols[i])
        if r in visited_set and c in visited_set:
            elements.append({
                "data": {
                    "source": trade_ids[r] if r < len(trade_ids) else str(r),
                    "target": trade_ids[c] if c < len(trade_ids) else str(c),
                    "weight": float(values[i]),
                }
            })

    return elements
