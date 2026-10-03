"""
Per-feature summary statistics for the PRISM Data Quality tab.

Writes a per-feature parquet of count / mean / std / percentiles /
min / max (F4 schema).  Per the PRISM pipeline output contract
(``docs/platform_designs/prism_retool_migration.md`` §11.15.1) the
training-baseline flavour of this artifact is produced unconditionally
at the end of every training run; the per-split flavour is produced by
the eval pipeline in Phase 5a-eval.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


def build_feature_summary_df(
    features: pd.DataFrame,
    *,
    cluster_id: str,
) -> pd.DataFrame:
    """
    Compute the per-feature summary DataFrame (F4 schema) without
    writing to disk.

    Percentiles and min / max / mean / std are computed on **finite**
    values only (NaN and ±inf are excluded).  Columns with zero finite
    values emit a row with stats set to ``NaN``; we prefer this over
    dropping the column so that downstream consumers see a deterministic
    row count equal to ``features.shape[1]``.

    Standard deviation is population (``ddof=0``), matching the baseline
    feature stats writer in ``monitoring/baselines.py`` — any later
    drift metric that forms a ratio or shift against the baseline stays
    on the same estimator.

    Separated from :func:`save_feature_summary` so the eval pipeline
    can concatenate per-cluster DataFrames into a single ensemble-level
    parquet per split (PRISM contract §11.15.2 Phase 5e).

    Schema:

    - ``cluster_id``    (string)
    - ``feature_name``  (string)
    - ``count``         (int32 — number of finite values)
    - ``mean``, ``std`` (float32)
    - ``p01``, ``p50``, ``p99`` (float32)
    - ``min``, ``max``  (float32)

    Raises
    ------
    ValueError
        If ``features`` has zero rows or zero columns.
    """
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            f"Cannot build feature summary for '{cluster_id}': features is empty "
            f"(shape={features.shape})"
        )

    rows = []
    for col in features.columns:
        arr = features[col].to_numpy(dtype=np.float64, copy=False)
        valid = arr[np.isfinite(arr)]

        if valid.size == 0:
            rows.append({
                "cluster_id": cluster_id,
                "feature_name": str(col),
                "count": 0,
                "mean": np.nan, "std": np.nan,
                "p01": np.nan, "p50": np.nan, "p99": np.nan,
                "min": np.nan, "max": np.nan,
            })
            continue

        p01, p50, p99 = np.percentile(valid, [1, 50, 99])
        rows.append({
            "cluster_id": cluster_id,
            "feature_name": str(col),
            "count": int(valid.size),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "p01": float(p01),
            "p50": float(p50),
            "p99": float(p99),
            "min": float(valid.min()),
            "max": float(valid.max()),
        })

    return pd.DataFrame(rows).astype({
        "cluster_id": "string",
        "feature_name": "string",
        "count": np.int32,
        "mean": np.float32, "std": np.float32,
        "p01": np.float32, "p50": np.float32, "p99": np.float32,
        "min": np.float32, "max": np.float32,
    })


def save_feature_summary(
    out_path: Union[str, Path],
    features: pd.DataFrame,
    *,
    cluster_id: str,
) -> None:
    """
    Write per-feature summary statistics to a parquet (F4 schema).

    Thin wrapper around :func:`build_feature_summary_df` that adds
    file-level Arrow metadata (``_schema_version``, ``cluster_id``) and
    writes the result.  Used by the training pipeline to emit
    per-member ``feature_summary_train_baseline.parquet``.

    Parameters
    ----------
    out_path : str | Path
        Destination parquet file.  Parent dirs are created.
    features : pd.DataFrame
        Feature frame to summarise.  One row emitted per column.
    cluster_id : str
        Member's cluster id.

    Raises
    ------
    ValueError
        If ``features`` has zero rows or zero columns.
    """
    df = build_feature_summary_df(features, cluster_id=cluster_id)

    # ``pa.table`` (module-level factory) avoids the spurious
    # "Parameter 'cls_1' unfilled" warning PyCharm's pyarrow stub
    # produces against the ``Table.from_pandas`` classmethod.  For a
    # default-RangeIndex DataFrame the behaviour is equivalent.
    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(SCHEMA_VERSION).encode(),
        b"cluster_id": cluster_id.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    logger.info("Saved %s (%d features)", out_path.name, len(df))
