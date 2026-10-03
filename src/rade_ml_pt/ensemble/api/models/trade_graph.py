"""Pydantic response schemas for the ``/prism/v1/trade-graph`` endpoint.

Serves one cluster's full trade-level graph — the sparse adjacency
recorded in ``graph_results.joblib`` plus the ``target`` / ``elementary``
trade split from ``trade_universe.json`` (both staged into the eval
artefact bundle by :mod:`rade_ml_pt.pipelines.ensemble.eval`).

Layout
------
* Nodes are trades; each node carries its ``cluster_id`` plus a
  ``trade_type`` tag (``"target"`` / ``"elementary"``) driving the UI's
  node colouring (amber for target trades, violet for elementary).
* Edges carry the sparse adjacency weight verbatim.  Self-loops (``src``
  equals ``dst``) are dropped server-side — they're an implementation
  artefact of the graph builder and add no visual signal.

The wire format is deliberately lightweight — Cytoscape renders tens
of thousands of nodes comfortably, so we send one row per node / edge
with zero nested payloads.  Tool-tip details (attributes, metrics) are
looked up client-side via the existing trade-metrics endpoint.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TradeGraphNode(BaseModel):
    """One trade node in a cluster's graph."""

    trade_id: str = Field(..., description="Trade identifier (string).")
    cluster_id: str = Field(..., description="Parent cluster identifier.")
    trade_type: str = Field(
        ...,
        description=(
            "``target`` or ``elementary`` — classifies the trade for the "
            "UI's node colouring and the Selected-Trade card. Determined "
            "from ``target_ids`` / ``elementary_ids`` in the member's "
            "``trade_universe.json``."
        ),
    )


class TradeGraphEdge(BaseModel):
    """One undirected edge between two trades in the cluster graph."""

    source: str = Field(..., description="Source trade_id.")
    target: str = Field(..., description="Target trade_id.")
    weight: float = Field(
        ...,
        description="Adjacency entry from ``sparse_values`` — graph-builder units.",
    )


class TradeGraphStats(BaseModel):
    """Summary numbers computed alongside the node / edge lists.

    Mirrors :class:`GraphStatsRow` (B8) but is re-derived here from the
    same joblib so the UI never shows graph stats that disagree with
    the rendered nodes / edges (e.g. when an older B8 parquet is still
    sitting next to a freshly-staged joblib).
    """

    n_nodes: int
    n_edges: int
    density: float
    mean_weight: float


class TradeGraphResponse(BaseModel):
    """Full trade-graph payload for one cluster."""

    cluster_id: str
    n_target_trades: int = Field(
        ..., description="Count of ``target`` trades in this cluster.",
    )
    n_elementary_trades: int = Field(
        ..., description="Count of ``elementary`` trades in this cluster.",
    )
    stats: TradeGraphStats
    nodes: List[TradeGraphNode]
    edges: List[TradeGraphEdge]
    warnings: Optional[List[str]] = Field(
        default=None,
        description=(
            "Non-fatal issues encountered while building the payload "
            "(missing trade_universe.json, trade ids falling outside "
            "the target / elementary index, …).  ``None`` on a clean "
            "read."
        ),
    )


__all__ = [
    "TradeGraphEdge",
    "TradeGraphNode",
    "TradeGraphResponse",
    "TradeGraphStats",
]
