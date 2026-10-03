"""Pydantic response schemas for the ``/prism/v1/graph-stats`` endpoint.

Backs B8 in the PRISM eval contract — one row per cluster, carrying the
basic topology numbers extracted from each member's
``graph_results.joblib`` (nodes, edges, density, mean edge weight).
Ensemble-scoped (no split dimension).
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class GraphStatsRow(BaseModel):
    """Per-cluster graph topology summary."""

    cluster_id: str
    n_nodes: int = Field(..., description="Nodes in the member's trade graph.")
    n_edges: int = Field(..., description="Non-zero entries in the sparse adjacency.")
    density: float = Field(
        ...,
        description="``n_edges / n_nodes**2`` — 0 when the member has no graph.",
    )
    mean_weight: float = Field(
        ...,
        description="Mean of the sparse adjacency values; 0 when absent.",
    )


class GraphStatsResponse(BaseModel):
    n_clusters: int
    rows: List[GraphStatsRow]
