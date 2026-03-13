"""
Prediction aggregation strategies for the ensemble model.

Each function takes per-member predictions (keyed by cluster ID) and returns
a single combined array covering the full trade universe.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def concat_aggregate(
    member_predictions: Dict[str, np.ndarray],
    cluster_trade_indices: Dict[str, List[int]],
    n_total_targets: int,
    n_scenarios: Optional[int] = None,
) -> np.ndarray:
    """
    Concatenate disjoint cluster predictions into the full target array.

    Each cluster owns a non-overlapping slice of targets.  Predictions are
    placed at the correct column positions using *cluster_trade_indices*.

    Parameters
    ----------
    member_predictions : dict
        ``{cluster_id: array [n_scenarios, n_cluster_targets]}``.
    cluster_trade_indices : dict
        ``{cluster_id: [col_index_in_full_array, ...]}``.
    n_total_targets : int
        Total number of target columns in the combined output.
    n_scenarios : int or None
        Number of rows (scenarios).  Inferred from the first member if omitted.

    Returns
    -------
    np.ndarray, shape ``[n_scenarios, n_total_targets]``
    """
    if n_scenarios is None:
        first = next(iter(member_predictions.values()))
        n_scenarios = first.shape[0]

    combined = np.zeros((n_scenarios, n_total_targets), dtype=np.float32)

    for cid, preds in member_predictions.items():
        indices = cluster_trade_indices.get(cid, [])
        if len(indices) != preds.shape[-1]:
            raise ValueError(
                f"Cluster '{cid}': index count ({len(indices)}) != "
                f"prediction columns ({preds.shape[-1]})"
            )
        for local_col, global_col in enumerate(indices):
            combined[:, global_col] = preds[:, local_col]

    return combined


def weighted_mean_aggregate(
    member_predictions: Dict[str, np.ndarray],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Weighted average of overlapping member predictions.

    All members must produce arrays of the same shape.  Each member's
    predictions are scaled by its weight and normalised by the sum of
    weights.

    Parameters
    ----------
    member_predictions : dict
        ``{cluster_id: array [n_scenarios, n_targets]}``.
    weights : dict
        ``{cluster_id: float}`` — non-negative weights.

    Returns
    -------
    np.ndarray, same shape as member arrays.
    """
    total_weight = 0.0
    combined = None

    for cid, preds in member_predictions.items():
        w = weights.get(cid, 1.0)
        if combined is None:
            combined = np.zeros_like(preds, dtype=np.float64)
        combined += w * preds.astype(np.float64)
        total_weight += w

    if combined is None:
        raise ValueError("No member predictions provided.")

    if total_weight == 0.0:
        raise ValueError("Total weight is zero — all members have zero weight.")

    return (combined / total_weight).astype(np.float32)


AGGREGATION_REGISTRY = {
    "concat": concat_aggregate,
    "weighted_mean": weighted_mean_aggregate,
}


def get_aggregation_fn(name: str):
    """Look up an aggregation function by name."""
    fn = AGGREGATION_REGISTRY.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown aggregation '{name}'. "
            f"Available: {list(AGGREGATION_REGISTRY.keys())}"
        )
    return fn
