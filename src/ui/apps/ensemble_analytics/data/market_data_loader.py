"""
Cluster market-data loader.

Delegates to ``EnsembleSession.load_cluster_market_data`` which
handles caching internally.  This module provides a clean import
path for Market Data tab callbacks.
"""
from __future__ import annotations

from typing import Any, Dict


def get_market_data(cluster_id: str) -> Dict[str, Any]:
    """
    Return market / risk-factor shock data for one cluster.

    Returns
    -------
    dict
        ``{asset_name: {rf_name: np.ndarray}}``.
        Empty dict if ``cluster_assets.joblib`` is not available.
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    return get_session().load_cluster_market_data(cluster_id)
