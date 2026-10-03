"""
Completeness analysis for the PRISM Data Quality tab.

Writes a per-feature parquet of null / nan / inf / zero counts and
dtype.  Per the PRISM pipeline output contract
(``docs/platform_designs/prism_retool_migration.md`` §11.15.1) the
training-baseline flavour of this artifact is produced unconditionally
at the end of every training run; the per-split flavour is produced by
the eval pipeline in Phase 5a-eval.

The writer is agnostic to *which* DataFrame it receives — it just
describes whatever columns it is handed.  The caller (training or eval)
is responsible for choosing which feature matrix to snapshot.
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


def build_completeness_df(
    features: pd.DataFrame,
    *,
    cluster_id: str,
) -> pd.DataFrame:
    """
    Compute the per-feature completeness DataFrame (F2 schema) without
    writing to disk.

    One row per column of ``features``.  Separated from
    :func:`save_completeness` so the eval pipeline can concatenate
    per-cluster DataFrames into a single ensemble-level parquet per
    split (PRISM contract §11.15.2 Phase 5e).

    Schema:

    - ``cluster_id``    (string)
    - ``feature_name``  (string)
    - ``dtype``         (string — original pandas dtype)
    - ``n_rows``        (int32)
    - ``n_null``        (int32 — ``pd.isna`` count; subsumes NaN and pd.NA)
    - ``null_rate``     (float32 — ``n_null / n_rows``)
    - ``n_distinct``    (int32 — over non-null values)
    - ``n_zero``        (int32 — exact zeros, excluding nulls and inf)
    - ``n_inf``         (int32 — positive + negative infinity)
    - ``n_nan``         (int32 — IEEE-754 NaN specifically)

    ``n_null`` is the broad count; ``n_nan`` is the narrow subset that
    is specifically IEEE-754 NaN.  Both are emitted so consumers can
    distinguish "missing by design" (e.g. pd.NA) from "missing by
    numeric failure" (e.g. a div-by-zero upstream).

    Raises
    ------
    ValueError
        If ``features`` has zero rows or zero columns.
    """
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            f"Cannot build completeness for '{cluster_id}': features is empty "
            f"(shape={features.shape})"
        )

    n_rows = int(features.shape[0])
    rows = []
    for col in features.columns:
        s = features[col]
        arr = s.to_numpy(copy=False)

        is_null = pd.isna(arr)
        if np.issubdtype(arr.dtype, np.floating):
            is_inf = np.isinf(arr)
            is_nan = np.isnan(arr)
        else:
            is_inf = np.zeros(n_rows, dtype=bool)
            is_nan = np.zeros(n_rows, dtype=bool)

        n_null = int(is_null.sum())
        n_inf = int(is_inf.sum())
        n_nan = int(is_nan.sum())

        finite_mask = ~is_null & ~is_inf
        n_zero = int((arr[finite_mask] == 0).sum()) if finite_mask.any() else 0
        n_distinct = int(s.dropna().nunique())

        rows.append({
            "cluster_id": cluster_id,
            "feature_name": str(col),
            "dtype": str(s.dtype),
            "n_rows": n_rows,
            "n_null": n_null,
            "null_rate": n_null / n_rows if n_rows > 0 else 0.0,
            "n_distinct": n_distinct,
            "n_zero": n_zero,
            "n_inf": n_inf,
            "n_nan": n_nan,
        })

    return pd.DataFrame(rows).astype({
        "cluster_id": "string",
        "feature_name": "string",
        "dtype": "string",
        "n_rows": np.int32,
        "n_null": np.int32,
        "null_rate": np.float32,
        "n_distinct": np.int32,
        "n_zero": np.int32,
        "n_inf": np.int32,
        "n_nan": np.int32,
    })


def save_completeness(
    out_path: Union[str, Path],
    features: pd.DataFrame,
    *,
    cluster_id: str,
) -> None:
    """
    Write per-feature completeness statistics to a parquet (F2 schema).

    Thin wrapper around :func:`build_completeness_df` that adds
    file-level Arrow metadata (``_schema_version``, ``cluster_id``) and
    writes the result.  Used by the training pipeline to emit
    per-member ``completeness_train_baseline.parquet``.

    Parameters
    ----------
    out_path : str | Path
        Destination parquet file.  Parent dirs are created.
    features : pd.DataFrame
        Feature frame to describe.  One row emitted per column.
    cluster_id : str
        Member's cluster id.  Populates every row's ``cluster_id`` so
        concatenations across members by the eval pipeline work without
        extra joins.

    Raises
    ------
    ValueError
        If ``features`` has zero rows or zero columns.
    """
    df = build_completeness_df(features, cluster_id=cluster_id)

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
