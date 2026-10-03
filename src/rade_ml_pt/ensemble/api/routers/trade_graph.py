"""``/prism/v1/trade-graph`` — per-cluster trade graph (nodes + edges).

Reads the sparse adjacency staged by the eval pipeline
(``members/{cluster_id}/graph_results.joblib``) and the adjacent
``trade_universe.json``.  Used by the Evaluation → Trade-Graph UI tab to
render a Cytoscape network and populate the right-hand metadata panels.

The endpoint is cluster-scoped because individual cluster payloads are
small (thousands of nodes / edges at most) but an ensemble-wide render
would be multi-MB and much less useful — the UI surfaces one cluster at
a time.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from src.rade_ml_pt.ensemble.api.dependencies import get_reader
from src.rade_ml_pt.ensemble.api.models.trade_graph import TradeGraphResponse
from src.rade_ml_pt.ensemble.api.services.reader import ArtifactReader

router = APIRouter(prefix="/prism/v1", tags=["trade-graph"])


@router.get("/trade-graph", response_model=TradeGraphResponse)
def get_trade_graph(
    cluster_id: str = Query(
        ...,
        description=(
            "Cluster to render.  Required — the endpoint is cluster-"
            "scoped because full-ensemble graphs aren't useful in the "
            "UI and quickly exceed sensible payload sizes."
        ),
    ),
    reader: ArtifactReader = Depends(get_reader),
) -> TradeGraphResponse:
    try:
        return reader.trade_graph(cluster_id=cluster_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
