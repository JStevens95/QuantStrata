"""Writers + readers for monitoring run artifacts.

Symmetric with inference's ``_post_infer_cluster`` / ``post_infer`` —
two layers:

* **Per-cluster**: :func:`write_drift_table_parquet` (the per-cluster
  output of :func:`monitoring.drift.build_drift_table`).
* **Run-level**:   :func:`write_drift_summary_json` (portfolio KPIs)
  and :func:`write_monitoring_manifest_json` (entry-point pointer that
  the API result reader / UI use to discover every artifact this run
  produced).

JSON writers sanitise ``NaN`` → ``null`` because ``json.dump`` with
``allow_nan=True`` produces ``NaN`` literals that are NOT valid JSON
and break every standards-compliant reader (incl. JavaScript /
``json.loads`` in non-Python tooling).
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Bumped whenever the manifest / drift_table schemas change in a
# back-incompatible way.  Mirrors ``monitoring.baselines.SCHEMA_VERSION``
# and ``monitoring.drift.SCHEMA_VERSION``; all three are independent
# but conventionally move together.
SCHEMA_VERSION: int = 1


# ═════════════════════════════════════════════════════════════════════
# Per-cluster drift table (parquet)
# ═════════════════════════════════════════════════════════════════════

def write_drift_table_parquet(
    drift_table: pd.DataFrame,
    out_path:    Union[str, Path],
    *,
    cluster_id:       str,
    ensemble_version: str,
    run_id:           str,
) -> None:
    """Persist one cluster's drift table.

    Schema lives in :data:`monitoring.drift._DRIFT_TABLE_COLUMNS`; this
    writer enforces those columns and embeds ``cluster_id`` /
    ``ensemble_version`` / ``run_id`` / ``_schema_version`` as pyarrow
    schema metadata so a stray parquet file can be reconciled with its
    parent run without consulting the manifest.

    Parameters
    ----------
    drift_table
        Output of :func:`monitoring.drift.build_drift_table`.  Must
        contain the canonical column set (validated lightly here —
        a missing column raises rather than silently dropping data).
    out_path
        Destination file.  Parent dirs are created on demand.
    cluster_id, ensemble_version, run_id
        Provenance triplet stamped on the schema metadata.

    Raises
    ------
    ValueError
        If ``drift_table`` is missing required drift-table columns.
    """
    required = {
        "cluster_id", "feature_name", "psi", "js_divergence",
        "mean_shift", "std_ratio", "severity",
    }
    missing = required - set(drift_table.columns)
    if missing:
        raise ValueError(
            f"drift_table for cluster '{cluster_id}' is missing columns "
            f"{sorted(missing)}.  Did you build it via "
            f"monitoring.drift.build_drift_table?"
        )

    # See ``monitoring.baselines.save_feature_baseline`` for the
    # rationale on ``pa.table`` over ``Table.from_pandas`` (the latter
    # trips a pyarrow stub bug in PyCharm).
    table = pa.table(drift_table)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version":   str(SCHEMA_VERSION).encode(),
        b"cluster_id":        cluster_id.encode(),
        b"ensemble_version":  ensemble_version.encode(),
        b"run_id":            run_id.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(out_path))
    logger.info(
        "Wrote drift table for cluster '%s' (%d features, severity counts %s)",
        cluster_id,
        len(drift_table),
        dict(drift_table["severity"].value_counts()),
    )


def read_drift_table_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """Read a drift table parquet written by :func:`write_drift_table_parquet`.

    Pure pass-through to ``pd.read_parquet`` plus a friendlier
    ``FileNotFoundError`` message.  The result has the exact same
    column / dtype schema produced by
    :func:`monitoring.drift.build_drift_table`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"drift table parquet not found: {path}")
    return pd.read_parquet(path)


# ═════════════════════════════════════════════════════════════════════
# Run-level JSON artifacts (drift_summary, manifest)
# ═════════════════════════════════════════════════════════════════════

def write_drift_summary_json(
    summary:  Mapping[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Write the portfolio-level drift summary as JSON.

    ``summary`` is typically the output of
    :func:`monitoring.drift.build_portfolio_drift_summary`.  NaN
    values are coerced to ``None`` (rendered as JSON ``null``) so the
    file is strictly valid JSON — see module docstring for why.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitised = _sanitise_for_json(summary)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sanitised, f, indent=2, sort_keys=True)
    logger.info("Wrote drift summary → %s", out_path)


def read_drift_summary_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a drift_summary.json written by :func:`write_drift_summary_json`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"drift summary json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_monitoring_manifest_json(
    manifest: Mapping[str, Any],
    out_path: Union[str, Path],
) -> None:
    """Write the run manifest as JSON.

    The manifest is the canonical entry point that the API result
    reader and the UI use to discover every artifact this run
    produced.  See :class:`pipelines.ensemble.monitor.MonitoringResult`
    for the producer side and ``RADE_UI_DESIGN.md`` for the schema
    expected by the M.4 API.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sanitised = _sanitise_for_json(manifest)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(sanitised, f, indent=2, sort_keys=True)
    logger.info("Wrote monitoring manifest → %s", out_path)


def read_monitoring_manifest_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Read a manifest.json written by :func:`write_monitoring_manifest_json`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"monitoring manifest json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════
# Internal — JSON sanitisation
# ═════════════════════════════════════════════════════════════════════

def _sanitise_for_json(obj: Any) -> Any:
    """Recursive NaN/Inf → None + ndarray → list normalisation.

    ``json.dump(..., allow_nan=True)`` emits ``NaN`` / ``Infinity``
    literals that are NOT in the JSON spec and silently break every
    non-Python consumer.  We normalise here so call-sites don't have
    to remember.

    Also coerces:
    * ``np.ndarray`` → ``list`` (so cluster-id arrays / hist arrays
      embedded in metadata don't crash the encoder)
    * ``np.integer`` / ``np.floating`` → native Python ``int`` /
      ``float`` (round-trip parity across pandas / numpy versions)
    * ``Path`` → ``str``  (manifests embed paths frequently)
    """
    if obj is None:
        return None
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitise_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitise_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitise_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if not math.isfinite(v) else v
    if isinstance(obj, float):
        return None if not math.isfinite(obj) else obj
    return obj


__all__ = [
    "SCHEMA_VERSION",
    # parquet
    "write_drift_table_parquet",
    "read_drift_table_parquet",
    # json
    "write_drift_summary_json",
    "read_drift_summary_json",
    "write_monitoring_manifest_json",
    "read_monitoring_manifest_json",
]
