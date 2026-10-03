"""
Training-time baselines for drift monitoring.

Produces per-member baseline parquets under ``{REG}/{MEM_VER}/monitoring/``
that the eval pipeline later compares against to compute PSI, JS divergence,
and related drift metrics (Phase 4b / E-series).

Per the PRISM pipeline output contract
(``docs/platform_designs/prism_retool_migration.md`` §11.15.1) the feature
baseline file is produced unconditionally at the end of every training run;
no config flag gates emission.

Residual baselines are produced by the eval pipeline instead — see §11.15.2.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
N_HIST_BINS = 50


def save_feature_baseline(
    out_path: Union[str, Path],
    features: pd.DataFrame,
    *,
    cluster_id: str,
) -> None:
    """
    Write per-feature baseline statistics for drift monitoring.

    One row per column of ``features``.  See §4.1 of the PRISM migration
    doc for the authoritative schema.  Histograms use 50 fixed
    equal-width bins spanning the ``[min, max]`` range of each training
    feature; these bin edges are persisted in ``hist_edges_json`` so the
    eval pipeline can re-bin current-split data into the same buckets
    for PSI / JS computation without having to infer edges.

    Columns that are entirely non-finite (all NaN / inf) are emitted
    with stats set to ``NaN`` and empty histogram arrays.  Constant
    columns get a tiny epsilon-widened histogram range so
    ``np.histogram`` does not error; downstream drift code is expected
    to detect near-zero variance from ``std`` and suppress PSI.

    Parameters
    ----------
    out_path : str | Path
        Destination parquet file.  Parent dirs are created as needed.
    features : pd.DataFrame
        Training-slice of the feature matrix (rows = timesteps /
        scenarios, cols = feature names).  Expected to already be in
        the scaled space the model consumes.
    cluster_id : str
        Owning member's cluster id.  Not stored as a column (the file
        lives under ``{MEM_VER}/``, so cluster identity is already
        unambiguous) but is embedded in pyarrow schema metadata for
        join debugging.

    Raises
    ------
    ValueError
        If ``features`` has zero rows or zero columns.
    """
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError(
            f"Cannot build feature baseline for '{cluster_id}': features is empty "
            f"(shape={features.shape})"
        )

    rows = []
    for col in features.columns:
        arr = features[col].to_numpy(dtype=np.float64, copy=False)
        valid = arr[np.isfinite(arr)]

        if valid.size == 0:
            rows.append({
                "feature_name": str(col),
                "mean": np.nan, "std": np.nan,
                "min": np.nan, "max": np.nan,
                "p05": np.nan, "p25": np.nan, "p50": np.nan,
                "p75": np.nan, "p95": np.nan,
                "n_bins": N_HIST_BINS,
                "hist_edges_json": json.dumps([]),
                "hist_counts_json": json.dumps([]),
            })
            continue

        vmin = float(valid.min())
        vmax = float(valid.max())
        p05, p25, p50, p75, p95 = np.percentile(valid, [5, 25, 50, 75, 95])

        if vmin == vmax:
            # Constant column — widen the range minimally so np.histogram
            # produces N_HIST_BINS well-defined (if degenerate) bins.
            eps = max(abs(vmin) * 1e-6, 1e-12)
            edges = np.linspace(vmin - eps, vmax + eps, N_HIST_BINS + 1)
        else:
            edges = np.linspace(vmin, vmax, N_HIST_BINS + 1)
        counts, _ = np.histogram(valid, bins=edges)

        rows.append({
            "feature_name": str(col),
            "mean": float(valid.mean()),
            "std": float(valid.std(ddof=0)),
            "min": vmin,
            "max": vmax,
            "p05": float(p05),
            "p25": float(p25),
            "p50": float(p50),
            "p75": float(p75),
            "p95": float(p95),
            "n_bins": N_HIST_BINS,
            "hist_edges_json": json.dumps(edges.tolist()),
            "hist_counts_json": json.dumps(counts.tolist()),
        })

    df = pd.DataFrame(rows).astype({
        "feature_name": "string",
        "mean": np.float32, "std": np.float32,
        "min": np.float32, "max": np.float32,
        "p05": np.float32, "p25": np.float32, "p50": np.float32,
        "p75": np.float32, "p95": np.float32,
        "n_bins": np.int32,
        "hist_edges_json": "string",
        "hist_counts_json": "string",
    })

    # Use the module-level ``pa.table`` factory rather than
    # ``Table.from_pandas`` — the latter is a classmethod and PyCharm's
    # pyarrow stub mishandles ``cls`` detection, producing a spurious
    # "Parameter 'cls_1' unfilled" warning.  ``pa.table`` with a pandas
    # DataFrame that has a default RangeIndex is equivalent to
    # ``from_pandas(preserve_index=None)``: no index column is emitted.
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
