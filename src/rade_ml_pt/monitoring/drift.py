"""Drift metrics: PSI, JS divergence, per-feature drift table.

Pure numerical helpers — no I/O.  Callers (eval pipeline, inference
pipeline, monitoring pipeline) are responsible for loading baseline
parquets and current feature matrices.

Schema and thresholds follow the design in
``docs/platform_designs/prism_retool_migration.md`` §11.15 / Phase 4
(E-series artifacts).  See ``monitoring.baselines`` for the writer side
(training-time histograms with persisted edges) and
``monitoring.loaders.load_baseline`` for decoding those parquets back
into NumPy-ready DataFrames.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


SCHEMA_VERSION: int = 1

# Industry-standard PSI severity thresholds.  Single source of truth
# for this module + every downstream consumer (UI colour scales, alert
# rules, etc.).  Boundaries are inclusive on the upper side
# (``psi == 0.10`` → ``"warn"``;  ``psi == 0.25`` → ``"critical"``).
PSI_WARN_THRESHOLD:     float = 0.10
PSI_CRITICAL_THRESHOLD: float = 0.25

# Sentinel labels returned by ``classify_severity`` so the UI can
# distinguish missing-data cells from genuinely-stable features.
SEVERITY_INFO:     str = "info"
SEVERITY_WARN:     str = "warn"
SEVERITY_CRITICAL: str = "critical"
SEVERITY_NO_DATA:  str = "no_data"


# ═════════════════════════════════════════════════════════════════════
# Single-feature primitives — PSI + JSD
# ═════════════════════════════════════════════════════════════════════

def population_stability_index(
    baseline_counts: np.ndarray,
    baseline_edges:  np.ndarray,
    current_values:  np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Population Stability Index between a baseline histogram and current obs.

    .. math::

        \\text{PSI} = \\sum_i (p_{\\text{curr},i} - p_{\\text{base},i})
                     \\times \\ln\\!\\left(\\frac{p_{\\text{curr},i}}
                                                 {p_{\\text{base},i}}\\right)

    Each bin proportion is smoothed by ``epsilon`` to avoid div-by-zero
    and ``log(0)`` when the baseline or current has empty bins.  Current
    observations that fall outside
    ``[baseline_edges[0], baseline_edges[-1]]`` are clipped into the
    outermost bin — this matches the PSI convention of "lump outliers
    into the tail" and prevents PSI from spuriously spiking just
    because today saw a value the baseline never did.

    Non-finite ``current_values`` (NaN / inf) are dropped before
    binning, so callers don't have to do that themselves.

    Parameters
    ----------
    baseline_counts
        Histogram counts at training time (length ``n_bins``).
    baseline_edges
        Histogram edges (length ``n_bins + 1``).
    current_values
        Today's raw observations.  No prior binning expected.
    epsilon
        Bin proportion floor to prevent log(0) / divide-by-zero.

    Returns
    -------
    float
        PSI value.  Returns ``np.nan`` when either side is degenerate
        (all-zero counts, mismatched shapes, empty current observations).
    """
    counts = np.asarray(baseline_counts, dtype=np.float64)
    edges  = np.asarray(baseline_edges,  dtype=np.float64)
    cur    = np.asarray(current_values,  dtype=np.float64)
    cur    = cur[np.isfinite(cur)]

    if counts.size == 0 or edges.size != counts.size + 1:
        return float("nan")
    if cur.size == 0:
        return float("nan")
    if counts.sum() <= 0:
        return float("nan")

    cur_clipped   = np.clip(cur, edges[0], edges[-1])
    cur_counts, _ = np.histogram(cur_clipped, bins=edges)
    if cur_counts.sum() <= 0:
        return float("nan")

    p_base = counts     / counts.sum()
    p_curr = cur_counts / cur_counts.sum()

    p_base = p_base + epsilon
    p_curr = p_curr + epsilon

    return float(np.sum((p_curr - p_base) * np.log(p_curr / p_base)))


def js_divergence(
    baseline_counts: np.ndarray,
    current_counts:  np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> float:
    """Jensen-Shannon divergence between two histograms (log base 2).

    .. math::

        \\text{JSD}(P \\| Q) = \\tfrac{1}{2}\\text{KL}(P \\| M)
                             + \\tfrac{1}{2}\\text{KL}(Q \\| M),
        \\quad M = \\tfrac{1}{2}(P + Q)

    Bounded ``[0, 1]`` when log base 2 is used (which is the convention
    used here); symmetric in ``P``, ``Q``.  Both inputs are normalised
    to probability mass functions and smoothed by ``epsilon`` to avoid
    ``log(0)`` on empty bins.

    Parameters
    ----------
    baseline_counts, current_counts
        Histogram counts.  Must share the same bin layout (same
        length) — the caller is responsible for ensuring they were
        binned against identical edges.
    epsilon
        Probability floor to prevent ``log(0)``.

    Returns
    -------
    float
        JSD value in ``[0, 1]``.  Returns ``np.nan`` on shape mismatch
        or degenerate inputs (either array all-zero).
    """
    p = np.asarray(baseline_counts, dtype=np.float64)
    q = np.asarray(current_counts,  dtype=np.float64)
    if p.shape != q.shape or p.size == 0:
        return float("nan")
    if p.sum() <= 0 or q.sum() <= 0:
        return float("nan")

    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    p_safe = p + epsilon
    q_safe = q + epsilon
    m_safe = m + epsilon

    kl_pm = float(np.sum(p_safe * np.log2(p_safe / m_safe)))
    kl_qm = float(np.sum(q_safe * np.log2(q_safe / m_safe)))
    return 0.5 * kl_pm + 0.5 * kl_qm


# ═════════════════════════════════════════════════════════════════════
# Severity classifier
# ═════════════════════════════════════════════════════════════════════

def classify_severity(psi: Optional[float]) -> str:
    """Map a PSI value to ``info`` / ``warn`` / ``critical`` / ``no_data``.

    ====================  ================  ============
    PSI range             Severity           UI colour
    ====================  ================  ============
    ``[0, 0.10)``         ``info``           green
    ``[0.10, 0.25)``      ``warn``           amber
    ``[0.25, ∞)``         ``critical``       red
    ``NaN / None / <0``   ``no_data``        grey
    ====================  ================  ============

    Non-numeric or otherwise unparseable inputs return ``no_data``
    rather than raising, so the classifier is safe to apply
    row-by-row across a DataFrame containing transient nulls.
    """
    if psi is None:
        return SEVERITY_NO_DATA
    try:
        v = float(psi)
    except (TypeError, ValueError):
        return SEVERITY_NO_DATA
    if not np.isfinite(v) or v < 0:
        return SEVERITY_NO_DATA
    if v < PSI_WARN_THRESHOLD:
        return SEVERITY_INFO
    if v < PSI_CRITICAL_THRESHOLD:
        return SEVERITY_WARN
    return SEVERITY_CRITICAL


# ═════════════════════════════════════════════════════════════════════
# Per-cluster drift table builder
# ═════════════════════════════════════════════════════════════════════

_DRIFT_TABLE_COLUMNS: Sequence[str] = (
    "cluster_id",
    "feature_name",
    "psi",
    "js_divergence",
    "mean_shift",
    "std_ratio",
    "severity",
)

_DRIFT_TABLE_DTYPES: Dict[str, Any] = {
    "cluster_id":    "string",
    "feature_name":  "string",
    "psi":           np.float32,
    "js_divergence": np.float32,
    "mean_shift":    np.float32,
    "std_ratio":     np.float32,
    "severity":      "string",
}


def build_drift_table(
    baseline_df:      pd.DataFrame,
    current_features: pd.DataFrame,
    *,
    cluster_id:       str,
    epsilon:          float = 1e-6,
) -> pd.DataFrame:
    """Compute drift for every feature in ``baseline_df`` vs ``current_features``.

    The baseline is the **anchor** — every feature it tracks gets a row
    in the output:

    * Features present in **both** baseline and current → PSI + JSD +
      mean_shift + std_ratio computed, severity classified.
    * Features in baseline but **missing / all-NaN** in current → row
      emitted with NaN metrics and ``severity = "no_data"`` so the UI
      heatmap shows a grey cell rather than collapsing the column.
    * Features in current but **not** in baseline → silently ignored
      (we cannot score drift without a baseline anchor).

    Output schema (long-format, one row per baseline feature):

    | column          | dtype     | meaning                              |
    |-----------------|-----------|--------------------------------------|
    | ``cluster_id``  | string    | owning cluster                       |
    | ``feature_name``| string    | feature column name                  |
    | ``psi``         | float32   | population stability index           |
    | ``js_divergence`` | float32 | Jensen-Shannon divergence ``[0, 1]`` |
    | ``mean_shift``  | float32   | ``(μ_curr - μ_base) / max(σ_base, ε)`` (z-score units) |
    | ``std_ratio``   | float32   | ``σ_curr / max(σ_base, ε)`` — vol regime indicator |
    | ``severity``    | string    | ``info``/``warn``/``critical``/``no_data`` |

    Parameters
    ----------
    baseline_df
        ``load_baseline``-decoded DataFrame.  Must have ``feature_name``,
        ``mean``, ``std``, ``hist_edges`` (ndarray) and ``hist_counts``
        (ndarray) columns.
    current_features
        Today's features in the **same coordinate system** the baseline
        was built in (i.e. already pushed through the frozen training
        scaler).  Rows = observations, cols = feature names.
    cluster_id
        Stamped on every output row so multi-cluster tables can be
        ``pd.concat``-ed freely.
    epsilon
        Numerical floor used by PSI / std-ratio to avoid div-by-zero
        on degenerate near-constant baselines.

    Returns
    -------
    pd.DataFrame
        Long-format drift table (one row per baseline feature).
    """
    if baseline_df is None or baseline_df.empty:
        return _empty_drift_table()

    required = {"feature_name", "mean", "std", "hist_edges", "hist_counts"}
    missing  = required - set(baseline_df.columns)
    if missing:
        raise ValueError(
            f"baseline_df is missing required columns: {sorted(missing)}.  "
            f"Did you forget to call monitoring.loaders.load_baseline?"
        )

    rows: List[Dict[str, Any]] = []
    for _, brow in baseline_df.iterrows():
        feature_name    = str(brow["feature_name"])
        baseline_edges  = np.asarray(brow["hist_edges"],  dtype=np.float64)
        baseline_counts = np.asarray(brow["hist_counts"], dtype=np.float64)
        baseline_mean   = float(brow["mean"]) if pd.notna(brow["mean"]) else np.nan
        baseline_std    = float(brow["std"])  if pd.notna(brow["std"])  else np.nan

        if feature_name not in current_features.columns:
            rows.append(_no_data_row(cluster_id, feature_name))
            continue

        current_raw   = current_features[feature_name].to_numpy(dtype=np.float64)
        current_valid = current_raw[np.isfinite(current_raw)]
        if current_valid.size == 0 or baseline_counts.size == 0:
            rows.append(_no_data_row(cluster_id, feature_name))
            continue

        psi = population_stability_index(
            baseline_counts, baseline_edges, current_valid, epsilon=epsilon,
        )

        # JSD needs paired histograms on the same edges — bin today's
        # values with the persisted baseline edges first.  We reuse
        # the same outlier-clipping convention as PSI so the two
        # numbers tell a consistent story.
        clipped = np.clip(current_valid, baseline_edges[0], baseline_edges[-1])
        current_counts, _ = np.histogram(clipped, bins=baseline_edges)
        jsd = js_divergence(baseline_counts, current_counts, epsilon=epsilon)

        # Mean shift in baseline-std units; std ratio in raw scale.
        # ``epsilon`` guards against degenerate near-constant baselines
        # (std ≈ 0) — the resulting numbers are noisy by definition in
        # that regime, but better than divide-by-zero.
        std_floor = (
            max(baseline_std, epsilon)
            if np.isfinite(baseline_std) and baseline_std > 0
            else epsilon
        )
        current_mean = float(np.mean(current_valid))
        current_std  = float(np.std(current_valid, ddof=0))
        mean_shift   = (
            (current_mean - baseline_mean) / std_floor
            if np.isfinite(baseline_mean) else np.nan
        )
        std_ratio = current_std / std_floor

        rows.append({
            "cluster_id":    cluster_id,
            "feature_name":  feature_name,
            "psi":           psi,
            "js_divergence": jsd,
            "mean_shift":    mean_shift,
            "std_ratio":     std_ratio,
            "severity":      classify_severity(psi),
        })

    return _to_drift_dataframe(rows)


def _empty_drift_table() -> pd.DataFrame:
    """Empty long-format drift table with the canonical dtype schema."""
    return _to_drift_dataframe([])


def _no_data_row(cluster_id: str, feature_name: str) -> Dict[str, Any]:
    """Drift-table row for a feature we cannot score (missing / all-NaN)."""
    return {
        "cluster_id":    cluster_id,
        "feature_name":  feature_name,
        "psi":           np.nan,
        "js_divergence": np.nan,
        "mean_shift":    np.nan,
        "std_ratio":     np.nan,
        "severity":      SEVERITY_NO_DATA,
    }


def _to_drift_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    """Coerce a row list to the canonical drift-table dtype schema.

    Centralised so empty + populated tables share the same dtype
    fingerprint — important for ``pd.concat`` to work without dtype
    promotion warnings when callers stitch per-cluster tables together.
    """
    df = pd.DataFrame(rows, columns=list(_DRIFT_TABLE_COLUMNS))
    return df.astype(_DRIFT_TABLE_DTYPES)


# ═════════════════════════════════════════════════════════════════════
# Portfolio-level summary
# ═════════════════════════════════════════════════════════════════════

def build_portfolio_drift_summary(
    drift_tables: Sequence[pd.DataFrame],
) -> Dict[str, Any]:
    """Aggregate per-cluster drift tables into portfolio-level KPIs.

    Consumed by the monitoring run manifest (``drift_summary.json``) and
    the Monitoring tab health-strip KPIs.  Severity is classified off
    ``mean_psi`` using the same thresholds applied to individual
    features, so the portfolio's "warn" line matches the per-feature
    severity boundary the user already sees in the heatmap.

    Returns a dict with:

    ===================  =================================================
    key                  meaning
    ===================  =================================================
    ``n_clusters``        clusters represented across the tables
    ``n_features_total``  scoreable features (excluding ``no_data`` rows)
    ``n_features_info``   features with severity ``info``
    ``n_features_warn``   features with severity ``warn``
    ``n_features_crit``   features with severity ``critical``
    ``n_features_nodata`` features that could not be scored
    ``mean_psi``          mean PSI across all scoreable features
    ``max_psi``           max PSI across all scoreable features
    ``worst_cluster``     ``cluster_id`` holding the max-PSI feature
    ``worst_feature``     ``feature_name`` holding the max PSI
    ``severity``          portfolio-level severity, classified off ``mean_psi``
    ===================  =================================================

    Tolerates empty / ``None`` inputs by returning an empty-state dict
    with zero counts and ``severity == "no_data"`` so callers don't have
    to gate on row counts.
    """
    if drift_tables is None or len(drift_tables) == 0:
        return _empty_summary()

    nonempty = [t for t in drift_tables if t is not None and not t.empty]
    if not nonempty:
        return _empty_summary()

    combined   = pd.concat(nonempty, ignore_index=True)
    n_clusters = int(combined["cluster_id"].nunique())

    sev_counts = combined["severity"].value_counts().to_dict()
    n_info   = int(sev_counts.get(SEVERITY_INFO,     0))
    n_warn   = int(sev_counts.get(SEVERITY_WARN,     0))
    n_crit   = int(sev_counts.get(SEVERITY_CRITICAL, 0))
    n_nodata = int(sev_counts.get(SEVERITY_NO_DATA,  0))
    n_total  = n_info + n_warn + n_crit

    scoreable = combined[combined["severity"] != SEVERITY_NO_DATA]
    if scoreable.empty:
        mean_psi:      float          = float("nan")
        max_psi:       float          = float("nan")
        worst_cluster: Optional[str]  = None
        worst_feature: Optional[str]  = None
    else:
        mean_psi      = float(scoreable["psi"].mean())
        idx_max       = scoreable["psi"].idxmax()
        max_psi       = float(scoreable.loc[idx_max, "psi"])
        worst_cluster = str(scoreable.loc[idx_max, "cluster_id"])
        worst_feature = str(scoreable.loc[idx_max, "feature_name"])

    return {
        "n_clusters":         n_clusters,
        "n_features_total":   n_total,
        "n_features_info":    n_info,
        "n_features_warn":    n_warn,
        "n_features_crit":    n_crit,
        "n_features_nodata":  n_nodata,
        "mean_psi":           mean_psi,
        "max_psi":            max_psi,
        "worst_cluster":      worst_cluster,
        "worst_feature":      worst_feature,
        "severity":           classify_severity(mean_psi),
    }


def _empty_summary() -> Dict[str, Any]:
    """Empty-state portfolio summary; keeps every key the manifest writer expects."""
    return {
        "n_clusters":         0,
        "n_features_total":   0,
        "n_features_info":    0,
        "n_features_warn":    0,
        "n_features_crit":    0,
        "n_features_nodata":  0,
        "mean_psi":           float("nan"),
        "max_psi":            float("nan"),
        "worst_cluster":      None,
        "worst_feature":      None,
        "severity":           SEVERITY_NO_DATA,
    }


__all__ = [
    # constants
    "SCHEMA_VERSION",
    "PSI_WARN_THRESHOLD", "PSI_CRITICAL_THRESHOLD",
    "SEVERITY_INFO", "SEVERITY_WARN", "SEVERITY_CRITICAL", "SEVERITY_NO_DATA",
    # primitives
    "population_stability_index",
    "js_divergence",
    "classify_severity",
    # builders
    "build_drift_table",
    "build_portfolio_drift_summary",
]
