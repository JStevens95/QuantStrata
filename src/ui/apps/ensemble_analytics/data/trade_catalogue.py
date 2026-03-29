"""
Cached global trade catalogue.

Builds the catalogue DataFrame once on first access and caches it
at module level.  The catalogue enables cross-cluster filtering by
desk, product, ccy, or any saved trade attribute.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

_cache: Optional[pd.DataFrame] = None


def get_trade_catalogue() -> pd.DataFrame:
    """
    Return the global trade catalogue DataFrame, building on first call.

    Returns
    -------
    pd.DataFrame
        Columns include ``trade_id``, ``cluster_id``, plus any
        per-trade attributes (``product_type``, ``ccy``, ``desk``, etc.)
        and cluster-level attributes.
    """
    global _cache
    if _cache is not None:
        return _cache

    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    _cache = get_session().build_global_trade_catalogue()
    return _cache


def invalidate() -> None:
    """Clear the cache (called on session reload)."""
    global _cache
    _cache = None
