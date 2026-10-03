"""Path conventions + run-id helpers for monitoring runs.

Mirrors ``ensemble.api.services.inference_state.per_run_artifacts_dir``
for inference, kept in monitoring/ so the writer / pipeline / API layers
agree on the layout without anyone importing the inference state
manager.  Single source of truth for "where does monitoring write?".

Layout (mirrors ``inference_runs/<run_id>/inference/...``)::

    {artifacts_dir}/monitoring_runs/<run_id>/monitoring/
      ├── manifest.json
      ├── drift_summary.json
      └── clusters/<cid>/drift_table.parquet
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

# Top-level directory under ``artifacts_dir`` for monitoring runs.
# Symmetric with ``inference_runs`` so callers can list both with a
# single ``Path(artifacts_dir).iterdir()`` if they want a unified runs
# index later (M.4 / M.5).
MONITORING_RUNS_DIRNAME: str = "monitoring_runs"

# Nested sub-directory holding the manifest + per-cluster artifacts.
# Mirrors inference's ``inference/`` sub-dir.  Keep it stable across
# M.2 / M.3 / M.4 — the API result reader and the UI both join from
# this constant.
MONITORING_SUBDIRNAME:   str = "monitoring"

# Default file names — also constants so writers + readers stay in
# sync without string typos.
MANIFEST_FILENAME:        str = "manifest.json"
DRIFT_SUMMARY_FILENAME:   str = "drift_summary.json"
DRIFT_TABLE_FILENAME:     str = "drift_table.parquet"

# Run-id segment that distinguishes monitoring runs from inference runs
# when both share an ``ensemble_version`` prefix.  Lets future tooling
# (or grep) tell them apart at a glance.
RUN_ID_SEGMENT:           str = "monitor"


def monitoring_run_id(
    ensemble_version: str,
    *,
    timestamp: Union[datetime, None] = None,
) -> str:
    """Build a deterministic monitoring run-id.

    Format::

        <ensemble_version>__monitor__<UTC_ISO_TS>

    Where ``UTC_ISO_TS`` is ``YYYY-MM-DDTHH-MM-SSZ`` (colon → dash so
    the id is safe as a directory name on every filesystem we care
    about).  Mirrors inference's ``<ensemble_version>__<TS>`` pattern,
    inserting ``monitor`` so monitoring + inference runs are easy to
    distinguish in a flat listing.

    Parameters
    ----------
    ensemble_version
        Ensemble version or tag this run is monitoring.
    timestamp
        Optional override (UTC).  Defaults to ``datetime.now(timezone.utc)``.
        Tests inject a fixed value for determinism.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        # Defensive: callers passing naive datetimes get UTC assumed,
        # not local time — the artefact path must be reproducible.
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    ts_str = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ensemble_version}__{RUN_ID_SEGMENT}__{ts_str}"


# Run-id parser regex.  Lazy import-time compile so callers that never
# parse run-ids don't pay the cost.  Matches the format produced by
# :func:`monitoring_run_id` strictly — anything looser would invite
# silent collisions with future run-id variants.
_RUN_ID_REGEX = re.compile(
    r"^(?P<ensemble_version>.+)__"
    rf"{re.escape(RUN_ID_SEGMENT)}__"
    r"(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)$"
)


def parse_monitoring_run_id(run_id: str) -> dict:
    """Reverse :func:`monitoring_run_id` — return ``{ensemble_version, ts}``.

    Returns an empty dict on parse failure rather than raising; callers
    that need strict validation should check ``bool(...)`` on the
    result.  Used by the M.4 API to populate ``MonitoringRunMeta``
    from a directory listing without re-reading every manifest.
    """
    m = _RUN_ID_REGEX.match(run_id)
    if not m:
        return {}
    return {"ensemble_version": m.group("ensemble_version"), "ts": m.group("ts")}


def per_run_artifacts_dir(base_artifacts_dir: Union[str, Path], run_id: str) -> str:
    """Where this run's artifacts live on disk.

    ``{base_artifacts_dir}/monitoring_runs/<run_id>``.  The pipeline's
    writers further nest a ``monitoring/`` directory underneath for the
    manifest + parquets, so the final manifest path is
    ``{base}/monitoring_runs/<run_id>/monitoring/manifest.json``.

    Centralised here so the pipeline, the writer, and any future
    listing endpoint agree on the layout — same convention used by
    inference (see ``ensemble.api.services.inference_state``).
    """
    return str(Path(base_artifacts_dir) / MONITORING_RUNS_DIRNAME / run_id)


@dataclass(frozen=True)
class MonitoringRunPaths:
    """Resolved on-disk paths for a single monitoring run.

    Constructing this dataclass does NOT touch the filesystem; calling
    :meth:`ensure_dirs` does.  Keep the two concerns separate so unit
    tests can exercise path resolution without ``tmp_path``.

    Attributes
    ----------
    artifacts_dir
        Base artifacts directory (the same value passed to the
        :class:`EnsembleMonitoringPipeline`).  All other paths are
        rooted here.
    run_id
        Monitoring run identifier (see :func:`monitoring_run_id`).
    ensemble_version
        Ensemble version this run is monitoring.  Kept on the dataclass
        so callers don't have to re-derive it from ``run_id`` by
        regex.
    """

    artifacts_dir:    Path
    run_id:           str
    ensemble_version: str

    # ─── Run-level paths ─────────────────────────────────────────────

    @property
    def run_dir(self) -> Path:
        """``{artifacts_dir}/monitoring_runs/<run_id>``."""
        return self.artifacts_dir / MONITORING_RUNS_DIRNAME / self.run_id

    @property
    def monitoring_dir(self) -> Path:
        """``{run_dir}/monitoring`` — holds manifest + summary + clusters/."""
        return self.run_dir / MONITORING_SUBDIRNAME

    @property
    def manifest_path(self) -> Path:
        """``{monitoring_dir}/manifest.json``."""
        return self.monitoring_dir / MANIFEST_FILENAME

    @property
    def drift_summary_path(self) -> Path:
        """``{monitoring_dir}/drift_summary.json`` — portfolio KPIs."""
        return self.monitoring_dir / DRIFT_SUMMARY_FILENAME

    @property
    def clusters_dir(self) -> Path:
        """``{monitoring_dir}/clusters`` — parent of per-cluster sub-dirs."""
        return self.monitoring_dir / "clusters"

    # ─── Per-cluster paths ───────────────────────────────────────────

    def cluster_dir(self, cluster_id: str) -> Path:
        """``{clusters_dir}/<cluster_id>``."""
        return self.clusters_dir / cluster_id

    def cluster_drift_table_path(self, cluster_id: str) -> Path:
        """``{cluster_dir}/drift_table.parquet`` — output of build_drift_table."""
        return self.cluster_dir(cluster_id) / DRIFT_TABLE_FILENAME

    # ─── Filesystem side-effects ─────────────────────────────────────

    def ensure_dirs(self) -> None:
        """Create the run + monitoring + clusters directories.

        Per-cluster sub-directories are created lazily by the writer
        each time it writes a cluster table — saves us from having to
        know the cluster list at path-construction time.
        """
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)
        self.clusters_dir.mkdir(parents=True, exist_ok=True)


__all__ = [
    # constants
    "MONITORING_RUNS_DIRNAME",
    "MONITORING_SUBDIRNAME",
    "MANIFEST_FILENAME",
    "DRIFT_SUMMARY_FILENAME",
    "DRIFT_TABLE_FILENAME",
    "RUN_ID_SEGMENT",
    # functions
    "monitoring_run_id",
    "parse_monitoring_run_id",
    "per_run_artifacts_dir",
    # dataclass
    "MonitoringRunPaths",
]
