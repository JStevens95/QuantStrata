"""
Ensemble-level metric aggregation.

Computes per-member, ensemble-wide, and version-comparison metrics from
raw predictions and targets.  These metrics feed the UI dashboard.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_ensemble_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute aggregate metrics on the combined ensemble predictions.

    Parameters
    ----------
    predictions : np.ndarray, shape ``[n_scenarios, n_targets]``
    targets : np.ndarray, same shape

    Returns
    -------
    dict
        Keys: ``mae``, ``mse``, ``rmse``, ``max_ae``, ``p95_ae``, ``p99_ae``.
    """
    residuals = predictions - targets
    abs_res = np.abs(residuals)

    return {
        "mae": float(np.mean(abs_res)),
        "mse": float(np.mean(residuals ** 2)),
        "rmse": float(np.sqrt(np.mean(residuals ** 2))),
        "max_ae": float(np.max(abs_res)),
        "p95_ae": float(np.percentile(abs_res, 95)),
        "p99_ae": float(np.percentile(abs_res, 99)),
    }


def aggregate_member_metrics(
    per_member_results: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    """
    Roll up per-member metrics into ensemble-level summaries.

    Parameters
    ----------
    per_member_results : dict
        ``{cluster_id: {mae: float, mse: float, ...}}``.

    Returns
    -------
    dict
        ``mean``, ``std``, ``min``, ``max`` across members for each metric
        key, plus the per-member table.
    """
    if not per_member_results:
        return {}

    metric_keys = set()
    for m in per_member_results.values():
        metric_keys.update(m.keys())

    rollup: Dict[str, Any] = {"per_member": per_member_results}

    for key in sorted(metric_keys):
        values = [
            m[key] for m in per_member_results.values()
            if key in m and m[key] is not None
        ]
        if not values:
            continue
        arr = np.array(values)
        rollup[f"mean_{key}"] = float(np.mean(arr))
        rollup[f"std_{key}"] = float(np.std(arr))
        rollup[f"min_{key}"] = float(np.min(arr))
        rollup[f"max_{key}"] = float(np.max(arr))

    return rollup


def compute_per_member_metrics(
    member_predictions: Dict[str, np.ndarray],
    member_targets: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each member independently.

    Parameters
    ----------
    member_predictions : dict
        ``{cluster_id: array [n_scenarios, n_cluster_targets]}``.
    member_targets : dict
        ``{cluster_id: array}`` — matching targets.

    Returns
    -------
    dict
        ``{cluster_id: {mae, mse, rmse, n_targets, n_scenarios}}``.
    """
    results: Dict[str, Dict[str, float]] = {}

    for cid in member_predictions:
        preds = member_predictions[cid]
        tgts = member_targets.get(cid)
        if tgts is None:
            logger.warning("No targets for cluster '%s'; skipping metrics.", cid)
            continue

        residuals = preds - tgts
        abs_res = np.abs(residuals)

        results[cid] = {
            "mae": float(np.mean(abs_res)),
            "mse": float(np.mean(residuals ** 2)),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "n_targets": int(preds.shape[-1]) if preds.ndim > 1 else 1,
            "n_scenarios": int(preds.shape[0]),
        }

    return results


def build_version_comparison(
    metrics_a: Dict[str, float],
    metrics_b: Dict[str, float],
    version_a: str = "A",
    version_b: str = "B",
) -> Dict[str, Any]:
    """
    Side-by-side comparison of two ensemble versions.

    Returns
    -------
    dict
        ``{metric_name: {version_a: val, version_b: val, delta: val, improved: bool}}``.
    """
    all_keys = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))
    comparison: Dict[str, Any] = {}

    for key in all_keys:
        va = metrics_a.get(key)
        vb = metrics_b.get(key)

        entry: Dict[str, Any] = {version_a: va, version_b: vb}
        if va is not None and vb is not None:
            entry["delta"] = vb - va
            entry["improved"] = vb < va
        comparison[key] = entry

    return comparison


def build_trade_to_cluster_mapping(
    cluster_mapping: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Flatten cluster mapping to ``{trade_id: cluster_id}``.

    Convenience helper for the UI drill-down.
    """
    mapping: Dict[str, str] = {}
    for cid, tids in cluster_mapping.items():
        for tid in tids:
            mapping[tid] = cid
    return mapping
