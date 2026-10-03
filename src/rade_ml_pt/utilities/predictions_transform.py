"""Predictions post-processing — scaled ↔ original PnL space.

Single source of truth for the *scaled → original* conversion of cluster-
level model output.  The chain is two stateless steps applied to a
``[n_scenarios, n_trades]`` numpy matrix:

  1. **Inverse-scale**     — undo the standardisation applied during
                             training (z-space → p-space).  Handles the
                             case where the cluster carries a reduced /
                             expanded trade population vs the scaler.
  2. **Restore notional**  — re-apply per-trade ``NotionalSign`` so the
                             output values match the user's reporting
                             convention.

Why a neutral module
--------------------
Inference and evaluation are separate runtime processes with separate
data-loading conventions; coupling them at the data-structure level
(e.g. via an ``InferenceContext`` wrapper) is wrong.  Sharing the *math*
via a neutral utility keeps the two pipelines fully decoupled — neither
imports the other — while guaranteeing a single, well-tested definition
of "original-space PnL" across the codebase.

Both the inference pipeline
(:meth:`~src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer.HybridGnnRnnInferencePipeline.transform_predictions`)
and the ensemble eval pipeline
(:mod:`~src.rade_ml_pt.pipelines.ensemble.eval`) call into this module
so the dashboard's *scaled* / *original* toggle reads consistent
numbers regardless of which pipeline produced the artifact.

Public surface
--------------
* :func:`inverse_scale_predictions`  — one ``inverse_transform`` call
                                       with dimension-mismatch handling.
* :func:`apply_notional_signs`       — per-trade sign restoration on a
                                       wide PnL frame.
* :func:`build_scenario_summary`     — per-scenario aggregate stats.
* :func:`transform_to_original`      — end-to-end composite
                                       (preds_2d → 3 wide frames).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Step 1 — inverse-scale (z-space → p-space)
# ----------------------------------------------------------------------

def inverse_scale_predictions(
    preds:  np.ndarray,
    scaler: Any,
) -> Tuple[np.ndarray, List[str]]:
    """Invert standardisation on a ``[n_scenarios, n_trades]`` matrix.

    The scaler's ``feature_names_in_`` defines the canonical trade-id
    ordering for the cluster's output columns.  If the input width
    doesn't match the scaler width the function pads / slices safely:

    * ``n_trades == scaler_width`` — direct ``inverse_transform``.
    * ``n_trades <  scaler_width`` — zero-pad to the scaler width,
      inverse-transform, slice back.  Trailing columns are dropped.
    * ``n_trades >  scaler_width`` — inverse-transform the head, pass
      the tail through unchanged.  Unknown columns get
      ``unknown_{i}`` trade IDs.

    Parameters
    ----------
    preds : np.ndarray
        2-D, shape ``[n_scenarios, n_trades]``.
    scaler : object
        sklearn-compatible scaler with ``feature_names_in_`` and
        ``inverse_transform`` methods.  Must not be ``None``.

    Returns
    -------
    unscaled : np.ndarray
        Same shape as ``preds``, in original (post-``inverse_transform``)
        space.
    trade_ids : list of str
        Column labels for ``unscaled`` in the order matching the
        scaler's canonical training order, with ``unknown_{i}``
        fallbacks for any columns past the scaler width.

    Raises
    ------
    ValueError
        If ``preds`` is not 2-D.
    RuntimeError
        If ``scaler`` is ``None`` or has no ``feature_names_in_``.
    """
    arr = np.asarray(preds)
    if arr.ndim != 2:
        raise ValueError(
            f"inverse_scale_predictions: expected 2-D predictions "
            f"[n_scenarios, n_trades]; got shape {arr.shape}."
        )
    if scaler is None:
        raise RuntimeError(
            "inverse_scale_predictions: scaler is None — cannot invert "
            "standardisation."
        )

    trade_ids_canonical = list(getattr(scaler, "feature_names_in_", []))
    if not trade_ids_canonical:
        raise RuntimeError(
            "inverse_scale_predictions: scaler has no feature_names_in_ — "
            "refit on a DataFrame at training time so the canonical "
            "trade-id ordering is preserved on disk."
        )

    n_scenarios, n_trades = arr.shape
    n_scaler_features     = len(trade_ids_canonical)

    if n_trades == n_scaler_features:
        unscaled  = scaler.inverse_transform(arr)
        trade_ids = trade_ids_canonical

    elif n_trades < n_scaler_features:
        padded                = np.zeros((n_scenarios, n_scaler_features), dtype=arr.dtype)
        padded[:, :n_trades]  = arr
        unscaled_full         = scaler.inverse_transform(padded)
        unscaled              = unscaled_full[:, :n_trades]
        trade_ids             = trade_ids_canonical[:n_trades]

    else:  # n_trades > n_scaler_features
        head     = scaler.inverse_transform(arr[:, :n_scaler_features])
        unscaled = np.concatenate([head, arr[:, n_scaler_features:]], axis=1)
        trade_ids = (
            trade_ids_canonical
            + [f"unknown_{i}" for i in range(n_trades - n_scaler_features)]
        )

    return unscaled, trade_ids


# ----------------------------------------------------------------------
# Step 2 — restore per-trade notional sign
# ----------------------------------------------------------------------

def apply_notional_signs(
    pnl_df:            pd.DataFrame,
    target_attributes: Optional[Dict[str, Any]],
    *,
    key_field:  str  = "TradeKey",
    sign_field: str  = "NotionalSign",
    strict:     bool = True,
    cluster_id: Optional[str] = None,
) -> pd.DataFrame:
    """Restore per-trade ``NotionalSign`` on a wide PnL frame.

    Training stores PnL with ``|notional|`` (sign stripped) so trades of
    opposite direction can share a scaler.  At inference / eval time we
    re-apply each trade's sign so the output matches the user's reporting
    convention.

    Parameters
    ----------
    pnl_df : pd.DataFrame
        Wide PnL frame, shape ``[n_scenarios, n_trades]``; columns are
        trade IDs.  Typically the output of
        :func:`inverse_scale_predictions` wrapped in a DataFrame.
    target_attributes : dict or None
        Per-trade attribute table.  Must contain a trade-key column
        (``key_field`` or ``trade_id`` as fallback) and a ``sign_field``
        of the same length.
    key_field, sign_field : str
        Names of the attribute keys to read.  Defaults match the
        ``target_attributes.json`` convention written at training time.
    strict : bool, default True
        If ``True``, raise ``RuntimeError`` on any missing / misaligned
        attribute.  If ``False``, log a warning and return ``pnl_df``
        unchanged.  Inference and ensemble eval keep this ``True`` (a
        missing sign is a hard contract violation); only the legacy
        per-member eval lowers it for backward compatibility.
    cluster_id : str, optional
        Used only to enrich error / warning messages.

    Returns
    -------
    pd.DataFrame
        Same shape / index / columns as ``pnl_df``; each column
        multiplied by its ``NotionalSign``.
    """
    ctx_tag = f" (cluster '{cluster_id}')" if cluster_id else ""

    def _fail(msg: str) -> pd.DataFrame:
        full = f"apply_notional_signs{ctx_tag}: {msg}"
        if strict:
            raise RuntimeError(full)
        logger.warning("%s — returning PnL unchanged.", full)
        return pnl_df

    if target_attributes is None:
        return _fail("target_attributes is None")

    keys  = target_attributes.get(key_field) or target_attributes.get("trade_id")
    signs = target_attributes.get(sign_field)
    if keys is None or signs is None:
        return _fail(
            f"missing required attribute(s) {key_field!r} and/or "
            f"{sign_field!r}"
        )

    sign_series = pd.Series(list(signs), index=list(keys), dtype=np.float32)
    try:
        aligned = sign_series.reindex(pnl_df.columns)
    except Exception as exc:
        return _fail(
            f"could not align {sign_field!r} to PnL columns ({exc})"
        )

    if aligned.isna().any():
        missing = aligned[aligned.isna()].index.tolist()
        return _fail(
            f"{len(missing)} trade(s) missing {sign_field!r} "
            f"(first 5: {missing[:5]})"
        )

    return pnl_df.mul(aligned, axis=1)


# ----------------------------------------------------------------------
# Step 3 — per-scenario aggregate summary
# ----------------------------------------------------------------------

def build_scenario_summary(
    cluster_id:    str,
    scaled_wide:   pd.DataFrame,
    original_wide: pd.DataFrame,
) -> pd.DataFrame:
    """Per-scenario aggregate stats — one row per scenario.

    Cheap to compute here, cheap to stack across clusters downstream
    (``sum_pnl_*`` sums cleanly across clusters per scenario for the
    portfolio roll-up).

    Returned columns
    ----------------
    * ``scenario_label``     — from the wide frames' index
    * ``cluster_id``         — constant per call
    * ``sum_pnl_scaled``     — preds.sum(axis=1) in z-space
    * ``sum_pnl_original``   — preds.sum(axis=1) in original notional
    * ``mean_pnl_original``  — preds.mean(axis=1) in original notional
    * ``std_pnl_original``   — preds.std(axis=1) in original notional
    * ``min_pnl_original``   — preds.min(axis=1) in original notional
    * ``max_pnl_original``   — preds.max(axis=1) in original notional
    """
    scaled_vals   = scaled_wide.to_numpy()
    original_vals = original_wide.to_numpy()
    return pd.DataFrame({
        "scenario_label":    list(scaled_wide.index),
        "cluster_id":        cluster_id,
        "sum_pnl_scaled":    scaled_vals.sum(axis=1),
        "sum_pnl_original":  original_vals.sum(axis=1),
        "mean_pnl_original": original_vals.mean(axis=1),
        "std_pnl_original":  original_vals.std(axis=1),
        "min_pnl_original":  original_vals.min(axis=1),
        "max_pnl_original":  original_vals.max(axis=1),
    })


# ----------------------------------------------------------------------
# Composite — end-to-end scaled → original
# ----------------------------------------------------------------------

def transform_to_original(
    cluster_id:        str,
    cluster_preds:     np.ndarray,
    scenario_labels:   Optional[List[str]],
    scaler:            Any,
    target_attributes: Dict[str, Any],
    *,
    strict_notional:   bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """End-to-end scaled → original for a single cluster's predictions.

    Composes :func:`inverse_scale_predictions`, the wide-frame
    assembly step, and :func:`apply_notional_signs`, then materialises
    the per-scenario summary via :func:`build_scenario_summary`.  This
    is the public entry point both inference and eval call.

    Parameters
    ----------
    cluster_id : str
        Used to tag the summary frame and enrich error messages.
    cluster_preds : np.ndarray
        Raw model output for the cluster.  Shape
        ``[n_scenarios, n_trades]`` — rows are scenarios, columns are
        trades.  Mirrors the convention used everywhere in
        ``ensemble/aggregation``, ``ensemble/session``, and
        ``pipelines/ensemble/eval``.
    scenario_labels : list of str or None
        Per-row labels (length ``n_scenarios``); used as the row index
        of both wide frames and the ``scenario_label`` column of the
        summary frame.  ``None`` ⇒ positional ``0..n_scenarios-1``
        index.
    scaler : object
        sklearn-compatible scaler with ``feature_names_in_`` and
        ``inverse_transform``.  Loaded by the caller from the cluster's
        ``target_scaler.pkl`` under the registry — *not* wrapped in any
        context object.
    target_attributes : dict
        Per-trade attribute table loaded from ``target_attributes.json``.
        Must contain ``TradeKey`` (or ``trade_id``) and ``NotionalSign``
        lists.
    strict_notional : bool, default True
        Forwarded to :func:`apply_notional_signs`.  Inference and
        ensemble eval keep this ``True`` (missing sign ⇒ hard error);
        only legacy per-member eval lowers it.

    Returns
    -------
    scaled_wide : pd.DataFrame
        Shape ``[n_scenarios, n_trades]``.  Predictions in **scaled**
        (model-output) space.  Index = ``scenario_label``; columns =
        trade IDs in canonical scaler order.
    original_wide : pd.DataFrame
        Same shape / index / columns — inverse-scaled AND
        notional-sign-restored predictions in **original** notional
        space.
    summary_df : pd.DataFrame
        One row per scenario.  See :func:`build_scenario_summary`.

    Raises
    ------
    ValueError
        If ``cluster_preds`` is not 2-D or ``scenario_labels`` length
        doesn't match ``n_scenarios``.
    RuntimeError
        If ``scaler`` is ``None`` / missing ``feature_names_in_``, or
        ``strict_notional`` is ``True`` and ``target_attributes`` is
        missing / misaligned.
    """
    preds = np.asarray(cluster_preds)
    if preds.ndim != 2:
        raise ValueError(
            f"transform_to_original: expected 2-D predictions "
            f"[n_scenarios, n_trades]; got shape {preds.shape}."
        )
    n_scenarios, _ = preds.shape

    if scenario_labels is not None and len(scenario_labels) != n_scenarios:
        raise ValueError(
            f"transform_to_original: len(scenario_labels)="
            f"{len(scenario_labels)} != n_scenarios={n_scenarios}."
        )

    # Step 1: inverse-scale (z-space → p-space).
    unscaled, trade_ids = inverse_scale_predictions(preds, scaler)

    # Step 2: assemble wide frames keyed by scenario_label.
    scenario_index = (
        list(scenario_labels) if scenario_labels is not None
        else list(range(n_scenarios))
    )

    scaled_wide = pd.DataFrame(preds, index=scenario_index, columns=trade_ids)
    scaled_wide.index.name = "scenario_label"

    unscaled_wide = pd.DataFrame(unscaled, index=scenario_index, columns=trade_ids)
    unscaled_wide.index.name = "scenario_label"

    # Step 3: notional-sign restoration.
    original_wide = apply_notional_signs(
        unscaled_wide, target_attributes,
        strict     = strict_notional,
        cluster_id = cluster_id,
    )
    original_wide.index.name = "scenario_label"

    # Step 4: per-scenario summary for downstream stacking.
    summary_df = build_scenario_summary(cluster_id, scaled_wide, original_wide)

    return scaled_wide, original_wide, summary_df


__all__ = [
    "inverse_scale_predictions",
    "apply_notional_signs",
    "build_scenario_summary",
    "transform_to_original",
]
