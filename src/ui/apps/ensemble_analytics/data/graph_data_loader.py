"""
Cluster graph-data loader.

Delegates to ``EnsembleSession.load_cluster_graph_data`` which
handles caching internally.  This module provides a clean import
path for Trade Graph tab callbacks.
"""
from __future__ import annotations

from typing import Any, Dict


def get_graph_data(cluster_id: str) -> Dict[str, Any]:
    """
    Return graph adjacency and encoder feature data for one cluster.

    Returns
    -------
    dict
        Keys: ``graph_results``, ``encoder_results``, ``trade_universe``.
        Empty dict if joblib files are not available.
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    return get_session().load_cluster_graph_data(cluster_id)
