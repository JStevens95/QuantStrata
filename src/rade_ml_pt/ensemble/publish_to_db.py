"""
Publish pre-computed evaluation artifacts to a SQLite database.

Reads the JSON and NPZ files produced by
``EnsembleEvalPipeline._save_all_artifacts()`` from the
``evaluation/`` directory and inserts them into SQLite using
batch inserts for performance.

Usage
-----
::

    from src.rade_ml_pt.ensemble.publish_to_db import publish_to_sqlite

    publish_to_sqlite(
        eval_dir="/path/to/evaluation",
        db_path="/path/to/ensemble.db",  # created if missing
    )
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SCHEMA_FILE = Path(__file__).parent / "db_schema.sql"


def publish_to_sqlite(
    eval_dir: str,
    db_path: Optional[str] = None,
) -> str:
    """Read evaluation artifacts and insert into a SQLite database.

    Parameters
    ----------
    eval_dir : str
        Path to the ``evaluation/`` directory containing both core
        artifacts and pre-computed summaries.
    db_path : str or None
        Output database path.  Defaults to ``evaluation/ensemble.db``.

    Returns
    -------
    str
        Path to the created/updated database file.
    """
    t0 = time.perf_counter()
    root = Path(eval_dir)

    if db_path is None:
        db_path = str(root / "ensemble.db")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    schema_sql = _SCHEMA_FILE.read_text()
    conn.executescript(schema_sql)

    version_meta = _read_json(root / "ensemble_version.json")
    if not version_meta:
        raise FileNotFoundError(f"ensemble_version.json not found in {root}")

    version = version_meta["version"]

    for table in [
        "ensemble_versions", "cluster_predictions", "portfolio_summary",
        "portfolio_percentiles", "worst_scenarios", "trade_metrics",
        "trade_cluster_map", "graph_stats", "cluster_attributes",
        "group_summaries", "group_correlations", "cluster_metrics",
        "ensemble_metrics",
    ]:
        conn.execute(f"DELETE FROM {table} WHERE version=?", (version,))

    # ── ensemble_versions ─────────────────────────────────────────
    conn.execute(
        "INSERT INTO ensemble_versions "
        "(version, n_clusters, n_trades, aggregation, strategy) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            version,
            version_meta.get("n_clusters", 0),
            version_meta.get("n_trades", 0),
            version_meta.get("aggregation"),
            version_meta.get("execution_strategy"),
        ),
    )

    # ── cluster_attributes ────────────────────────────────────────
    attrs = _read_json(root / "cluster_attributes.json") or {}
    rows = []
    for cid, ca in attrs.items():
        for attr_name, attr_val in ca.items():
            rows.append((
                version, cid, attr_name,
                str(attr_val) if attr_val is not None else None,
            ))
    conn.executemany(
        "INSERT INTO cluster_attributes "
        "(version, cluster_id, attribute_name, attribute_value) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )

    # ── trade_cluster_map ─────────────────────────────────────────
    tcm = _read_json(root / "trade_cluster_map.json") or {}
    conn.executemany(
        "INSERT INTO trade_cluster_map "
        "(version, trade_id, cluster_id) VALUES (?, ?, ?)",
        [(version, tid, cid) for tid, cid in tcm.items()],
    )

    # ── graph_stats ───────────────────────────────────────────────
    gs = _read_json(root / "graph_stats.json") or {}
    conn.executemany(
        "INSERT INTO graph_stats "
        "(version, cluster_id, n_nodes, n_edges, density, mean_weight) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (version, cid, s.get("n_nodes", 0), s.get("n_edges", 0),
             s.get("density", 0), s.get("mean_weight", 0))
            for cid, s in gs.items()
        ],
    )

    # ── Per-split data ────────────────────────────────────────────
    splits = version_meta.get("splits", ["test", "val", "train"])
    for split in splits:
        _publish_portfolio(conn, root, version, split)
        _publish_cluster_summary(conn, root, version, split)
        _publish_trade_metrics(conn, root, version, split)
        _publish_group_data(conn, root, version, split)

    conn.commit()
    conn.close()

    elapsed = time.perf_counter() - t0
    logger.info("Published to SQLite: %s (%.2fs)", db_path, elapsed)
    return db_path


# ======================================================================
# Per-split publishers
# ======================================================================

def _publish_portfolio(
    conn: sqlite3.Connection, root: Path, version: str, split: str,
) -> None:
    """Insert portfolio summary, percentiles, and worst scenarios."""
    npz_path = root / "portfolio_summary" / f"{split}.npz"
    if npz_path.exists():
        data = np.load(npz_path, allow_pickle=False)
        preds = data["predictions"]
        targets = data["targets"]
        conn.executemany(
            "INSERT INTO portfolio_summary "
            "(version, split, scenario_idx, prediction, target) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (version, split, int(i), float(preds[i]), float(targets[i]))
                for i in range(len(preds))
            ],
        )

    pct = _read_json(root / "portfolio_summary" / f"{split}_percentiles.json")
    if pct:
        conn.executemany(
            "INSERT INTO portfolio_percentiles "
            "(version, split, percentile, pred_value, target_value, abs_error) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (version, split, p,
                 v.get("prediction"), v.get("target"), v.get("abs_error"))
                for p, v in pct.items()
            ],
        )

    worst = _read_json(root / "portfolio_summary" / f"{split}_worst.json")
    if worst:
        conn.executemany(
            "INSERT INTO worst_scenarios "
            "(version, split, rank, scenario_idx, prediction, target, abs_error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (version, split, w["rank"], w["scenario_idx"],
                 w.get("prediction"), w.get("target"), w.get("abs_error"))
                for w in worst
            ],
        )


def _publish_cluster_summary(
    conn: sqlite3.Connection, root: Path, version: str, split: str,
) -> None:
    """Insert cluster-level prediction summaries."""
    npz_path = root / "cluster_summary" / f"{split}.npz"
    if not npz_path.exists():
        return

    data = np.load(npz_path, allow_pickle=False)
    cluster_ids = {key.rsplit("_", 1)[0] for key in data.files}

    rows = []
    for cid in sorted(cluster_ids):
        preds = data.get(f"{cid}_pred")
        targets = data.get(f"{cid}_target")
        if preds is None or targets is None:
            continue
        for i in range(len(preds)):
            rows.append(
                (version, cid, split, int(i), float(preds[i]), float(targets[i]))
            )

    conn.executemany(
        "INSERT INTO cluster_predictions "
        "(version, cluster_id, split, scenario_idx, prediction, target) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )


def _publish_trade_metrics(
    conn: sqlite3.Connection, root: Path, version: str, split: str,
) -> None:
    """Insert per-trade summary metrics."""
    metrics = _read_json(root / "trade_metrics" / f"{split}.json")
    if not metrics:
        return
    conn.executemany(
        "INSERT INTO trade_metrics "
        "(version, cluster_id, split, trade_id, mae, rmse, max_ae, p95_ae, "
        "mean_residual, std_residual) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (version, m["cluster_id"], split, m["trade_id"],
             m.get("mae"), m.get("rmse"), m.get("max_ae"), m.get("p95_ae"),
             m.get("mean_residual"), m.get("std_residual"))
            for m in metrics
        ],
    )


def _publish_group_data(
    conn: sqlite3.Connection, root: Path, version: str, split: str,
) -> None:
    """Insert group summaries and correlations as JSON blobs."""
    gs = _read_json(root / "group_summaries" / f"{split}.json")
    if gs:
        conn.execute(
            "INSERT INTO group_summaries (version, split, data) VALUES (?, ?, ?)",
            (version, split, json.dumps(gs)),
        )

    gc = _read_json(root / "group_correlations" / f"{split}.json")
    if gc:
        conn.execute(
            "INSERT INTO group_correlations (version, split, data) VALUES (?, ?, ?)",
            (version, split, json.dumps(gc)),
        )


# ======================================================================
# Helpers
# ======================================================================

def _read_json(path: Path):
    """Read a JSON file, returning ``None`` if it doesn't exist."""
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
