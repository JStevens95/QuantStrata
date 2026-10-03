"""Read-side counterpart to the monitoring pipeline's writers.

Serves the manifest + drift artifacts written by
:class:`EnsembleMonitoringPipeline` for any monitoring run on disk.
The dashboard's data plane (``GET /monitoring/runs/...``) reads
exclusively through this class; the control plane (``/load``,
``/scenarios``, ``/validate``, ``/run``, ``/promote``, ``/status``,
``/events``) goes through :class:`MonitoringStateManager` for the
*active* run.

On-disk layout
--------------
Mirrors the monitoring pipeline's
:class:`MonitoringRunPaths`.  Per run::

    <base_artifacts_dir>/monitoring_runs/<run_id>/monitoring/
    ├── manifest.json                             ← entry point
    ├── drift_summary.json
    ├── clusters/<cid>/drift_table.parquet
    └── inference/                                ← optional (after promote)
        ├── manifest.json
        ├── cluster_summary/...
        ├── portfolio_summary/...
        └── trade_predictions/...

The ``inference/`` subdir is M.3's promote-to-predictions output and
is NOT served by this reader in M.4 — predictions data plane is
deferred to M.4.5 (will compose :class:`InferenceResultReader` over
``<run_dir>/monitoring/`` when shipped).

State model — Option A (process-wide singleton)
-----------------------------------------------
Mirrors :class:`InferenceResultReader`: one reader per process,
constructed at lifespan startup with the base artifacts dir from
:class:`Settings`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

from src.rade_ml_pt.monitoring.run_paths import (
    DRIFT_SUMMARY_FILENAME,
    DRIFT_TABLE_FILENAME,
    MANIFEST_FILENAME,
    MONITORING_RUNS_DIRNAME,
    MONITORING_SUBDIRNAME,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Per-run filesystem layout
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MonitoringRunReaderPaths:
    """Resolved on-disk paths for one monitoring run (reader view).

    Symmetric with inference's :class:`RunPaths` and with the writer-
    side :class:`MonitoringRunPaths`.  Pure derived data — does NOT
    verify any path exists.  The router checks existence and raises
    the appropriate HTTPException itself.
    """

    run_id:              str
    run_root:            Path   # <base>/monitoring_runs/<run_id>
    monitoring_dir:      Path   # <base>/monitoring_runs/<run_id>/monitoring
    manifest_path:       Path
    drift_summary_path:  Path
    clusters_dir:        Path   # holds <cid>/drift_table.parquet

    def cluster_drift_table_path(self, cluster_id: str) -> Path:
        """``{clusters_dir}/<cluster_id>/drift_table.parquet``."""
        return self.clusters_dir / cluster_id / DRIFT_TABLE_FILENAME


# ──────────────────────────────────────────────────────────────────────
# Module-level loaders
# ──────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


# ──────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────

class MonitoringResultReader:
    """Typed accessor over the monitoring-runs directory.

    Stateless — every method resolves the on-disk layout from
    ``run_id`` and reads the file.  No cache (runs are append-only).

    Attributes
    ----------
    base_dir
        Resolved ``settings.artifacts_dir`` — root under which the
        ``monitoring_runs/<run_id>/monitoring/`` tree lives.
    """

    def __init__(self, base_artifacts_dir: str):
        self.base_dir = Path(base_artifacts_dir)

    # ── Path resolution ─────────────────────────────────────────────

    def paths_for(self, run_id: str) -> MonitoringRunReaderPaths:
        """Resolve the per-run filesystem layout.  Pure path arithmetic."""
        run_root       = self.base_dir / MONITORING_RUNS_DIRNAME / run_id
        monitoring_dir = run_root / MONITORING_SUBDIRNAME
        return MonitoringRunReaderPaths(
            run_id              = run_id,
            run_root            = run_root,
            monitoring_dir      = monitoring_dir,
            manifest_path       = monitoring_dir / MANIFEST_FILENAME,
            drift_summary_path  = monitoring_dir / DRIFT_SUMMARY_FILENAME,
            clusters_dir        = monitoring_dir / "clusters",
        )

    # ── Discovery ───────────────────────────────────────────────────

    def list_run_ids(self) -> List[str]:
        """Return all monitoring run IDs on disk, most recent first.

        The run-id format embeds an ISO-8601 timestamp so descending
        alphabetical order is also descending temporal order for any
        given ensemble version (matches inference's run-id ordering
        contract).

        Returns
        -------
        list of str
            Empty list when ``monitoring_runs/`` doesn't exist yet.
        """
        root = self.base_dir / MONITORING_RUNS_DIRNAME
        if not root.is_dir():
            return []
        return sorted(
            (p.name for p in root.iterdir() if p.is_dir()),
            reverse=True,
        )

    # ── Manifest ────────────────────────────────────────────────────

    def load_manifest(self, run_id: str) -> Dict[str, Any]:
        """Read ``manifest.json`` for ``run_id``.

        Raises
        ------
        FileNotFoundError
            If the manifest doesn't exist (run still in flight or
            never completed).
        """
        path = self.paths_for(run_id).manifest_path
        if not path.exists():
            raise FileNotFoundError(
                f"Monitoring manifest not found for run '{run_id}': {path}"
            )
        return _load_json(path)

    # ── Drift summary ───────────────────────────────────────────────

    def load_drift_summary(self, run_id: str) -> Dict[str, Any]:
        """Read ``drift_summary.json`` (portfolio aggregate)."""
        path = self.paths_for(run_id).drift_summary_path
        if not path.exists():
            raise FileNotFoundError(
                f"Drift summary not found for run '{run_id}': {path}"
            )
        return _load_json(path)

    # ── Per-cluster drift table ─────────────────────────────────────

    def load_cluster_drift_table(
        self,
        run_id:     str,
        cluster_id: str,
    ) -> pd.DataFrame:
        """Read a single cluster's drift table parquet."""
        path = self.paths_for(run_id).cluster_drift_table_path(cluster_id)
        if not path.exists():
            raise FileNotFoundError(
                f"Drift table not found for run '{run_id}' cluster "
                f"'{cluster_id}': {path}"
            )
        return _load_parquet(path)

    # ── Cluster severity index (cheap aggregate for the UI heatmap) ─

    def load_cluster_severity_index(self, run_id: str) -> List[Dict[str, Any]]:
        """Per-cluster severity rollup derived from the manifest's drift_summary.

        The portfolio drift_summary already aggregates severity at the
        cluster level, so we read it once and project to a flat list
        of ``{cluster_id, severity, max_psi, mean_psi, n_features}``
        rows for the UI heatmap.  Avoids loading every cluster
        parquet just to render the cluster picker.

        Returns
        -------
        list of dict
            One entry per cluster.  Empty list when the summary has
            no per-cluster breakdown (``no_data`` severity at the
            portfolio level).
        """
        summary = self.load_drift_summary(run_id)
        clusters = summary.get("clusters") or []
        # The summary writer normalises clusters to a list of dicts;
        # round-trip defensively here in case the schema evolves.
        return [
            {
                "cluster_id":  str(c.get("cluster_id", "")),
                "severity":    str(c.get("severity", "no_data")),
                "max_psi":     c.get("max_psi"),
                "mean_psi":    c.get("mean_psi"),
                "n_features":  c.get("n_features"),
            }
            for c in clusters
        ]

    # ── Run summary ─────────────────────────────────────────────────

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        """Lightweight summary suitable for the run-history table.

        Reads only the manifest (small JSON), no parquets.  Falls
        back to a well-formed ``status='in_progress'`` summary when
        the manifest doesn't exist yet so the UI can list in-flight
        runs.
        """
        try:
            manifest = self.load_manifest(run_id)
            status   = (
                "promoted"
                if manifest.get("predictions") is not None
                else "complete"
            )
        except FileNotFoundError:
            manifest = {}
            status   = "in_progress"

        drift_summary: Dict[str, Any] = manifest.get("drift_summary") or {}
        predictions: Optional[Dict[str, Any]] = manifest.get("predictions")

        return {
            "run_id":                run_id,
            "ensemble_version":      manifest.get("ensemble_version"),
            "status":                status,
            "created_at":            manifest.get("created_at"),
            "n_scenarios":           manifest.get("n_scenarios"),
            "n_clusters":            manifest.get("n_clusters"),
            "n_clusters_affected":   manifest.get("n_clusters_affected"),
            "n_clusters_unaffected": manifest.get("n_clusters_unaffected"),
            "severity":              drift_summary.get("severity"),
            "mean_psi":              drift_summary.get("mean_psi"),
            "max_psi":               drift_summary.get("max_psi"),
            "has_predictions":       predictions is not None,
        }


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton
# ──────────────────────────────────────────────────────────────────────

_monitoring_reader: Optional[MonitoringResultReader] = None


def set_monitoring_result_reader(reader: MonitoringResultReader) -> None:
    """Inject the reader at app-lifespan startup."""
    global _monitoring_reader
    _monitoring_reader = reader


def get_monitoring_result_reader() -> MonitoringResultReader:
    """FastAPI ``Depends`` — returns the singleton reader."""
    if _monitoring_reader is None:
        raise RuntimeError(
            "MonitoringResultReader not initialised. Server not ready."
        )
    return _monitoring_reader


__all__ = [
    "MonitoringRunReaderPaths",
    "MonitoringResultReader",
    "set_monitoring_result_reader",
    "get_monitoring_result_reader",
]
