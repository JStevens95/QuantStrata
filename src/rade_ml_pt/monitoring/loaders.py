"""Loaders for monitoring baseline parquets.

Decode the JSON-encoded ``hist_edges_json`` / ``hist_counts_json``
columns produced by :func:`monitoring.baselines.save_feature_baseline`
into ready-to-use NumPy arrays so callers (drift computation, UI
distribution overlays) don't have to repeat the ``json.loads`` dance.

Kept deliberately small — this module is just the reader half of the
training/inference symmetry; the writer half lives in ``baselines.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Columns that MUST be present in any baseline parquet we're prepared
# to consume.  Hoisted to module scope so the assertion + the error
# message stay in sync (one source of truth).
_REQUIRED_COLUMNS = (
    "feature_name",
    "mean",
    "std",
    "hist_edges_json",
    "hist_counts_json",
)


def load_baseline(parquet_path: Union[str, Path]) -> pd.DataFrame:
    """Read a ``baseline_feature_stats.parquet`` and decode its JSON columns.

    Parameters
    ----------
    parquet_path
        Path to the parquet file produced by
        :func:`monitoring.baselines.save_feature_baseline`.

    Returns
    -------
    pd.DataFrame
        The original parquet schema (``feature_name``, ``mean``, ``std``,
        ``min``, ``max``, ``p05``…``p95``, ``hist_edges_json``,
        ``hist_counts_json``) plus **two new in-memory columns**:

        * ``hist_edges``  — ``np.ndarray[float64]`` of length 51 (or
          empty when the feature was all-NaN at training time).
        * ``hist_counts`` — ``np.ndarray[int64]``  of length 50 (or
          empty in the same case).

        The two JSON-string columns are kept on the frame for
        traceability / debugging — drop them yourself if not needed
        downstream.

    Raises
    ------
    FileNotFoundError
        If ``parquet_path`` does not exist.
    ValueError
        If the parquet is missing the expected ``hist_edges_json`` /
        ``hist_counts_json`` (or any other required column).
    """
    parquet_path = Path(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"baseline parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    missing = set(_REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"baseline parquet at {parquet_path} is missing required columns "
            f"{sorted(missing)}.  Was it written by an older version of "
            f"monitoring.baselines.save_feature_baseline (or by something else "
            f"entirely)?"
        )

    df = df.copy()
    df["hist_edges"]  = df["hist_edges_json"].map(_decode_json_array_f64)
    df["hist_counts"] = df["hist_counts_json"].map(_decode_json_array_i64)
    return df


def _decode_json_array_f64(s: object) -> np.ndarray:
    """JSON-string-of-floats → ``np.ndarray[float64]`` (empty on error/empty).

    Defensive: tolerates ``None`` / ``NaN`` / malformed JSON by
    returning an empty array, so a single corrupt row in the baseline
    parquet can't poison an entire run.  Pair with the ``hist_edges``
    length checks downstream in drift code which already treat
    ``size == 0`` as "no_data".
    """
    if s is None:
        return np.empty(0, dtype=np.float64)
    if isinstance(s, float) and np.isnan(s):
        return np.empty(0, dtype=np.float64)
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return np.empty(0, dtype=np.float64)
    return np.asarray(parsed, dtype=np.float64)


def _decode_json_array_i64(s: object) -> np.ndarray:
    """JSON-string-of-ints → ``np.ndarray[int64]`` (empty on error/empty)."""
    if s is None:
        return np.empty(0, dtype=np.int64)
    if isinstance(s, float) and np.isnan(s):
        return np.empty(0, dtype=np.int64)
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return np.empty(0, dtype=np.int64)
    return np.asarray(parsed, dtype=np.int64)


__all__ = ["load_baseline"]
