"""
Cached ``GlobalPredictionStore`` access, one per split.

The store is built lazily by the session from per-member ``.npz`` files
and cached for the session lifetime.  This module provides a clean
import path for callbacks.
"""
from __future__ import annotations

from typing import Optional

from src.rade_ml_pt.ensemble.session import GlobalPredictionStore


def get_prediction_store(split: str = "test") -> Optional[GlobalPredictionStore]:
    """
    Return the prediction store for *split*, or ``None`` if unavailable.

    Parameters
    ----------
    split : str
        One of ``"test"``, ``"val"``, ``"train"``.

    Returns
    -------
    GlobalPredictionStore or None
    """
    from src.ui.apps.ensemble_analytics.data.session_manager import get_session
    return get_session().get_prediction_store(split)
