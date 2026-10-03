"""Read-side counterpart to Stage 6's writers.

Serves the manifest + parquets written by
:meth:`EnsembleInferencePipeline.post_infer` for any inference run
on disk.  The dashboard's data plane reads exclusively through this
class; the control plane (``/load``, ``/scenarios``, ``/validate``,
``/run``, ``/status``, ``/events``) goes through
:class:`InferenceStateManager` for the *active* run.

On-disk layout
--------------
Mirrors the eval pipeline's "one subdir per artifact family"
convention.  The directory and filename constants are imported
from :mod:`pipelines.ensemble.infer` (the writer side) — single
source of truth, so writer and reader can never drift.  Per run::

    <base_artifacts_dir>/inference_runs/<run_id>/inference/
    ├── manifest.json                                    ← entry point
    ├── cluster_summary/
    │   └── cluster_predictions.parquet                  ← long-format
    ├── portfolio_summary/
    │   └── portfolio_predictions.parquet                ← long-format
    └── trade_predictions/
        ├── <cluster_id>_scaled.parquet                  ← wide, model space
        └── <cluster_id>_original.parquet                ← wide, original space

Companion to
------------
:class:`ArtifactReader` (eval artifacts).  Two readers rather than
one because:

* The layouts differ — eval is keyed by ensemble version, inference
  is keyed by run_id.
* The caching story differs — eval artifacts are updated in place
  for the active version (mtime cache); inference artifacts are
  append-only by run_id (no cache needed).

State model — Option A (process-wide singleton)
-----------------------------------------------
Mirrors :class:`InferenceStateManager`: one reader per process,
constructed at lifespan startup with the base artifacts dir from
:class:`Settings`.  Migrating to per-tenant readers (Option B for
multi-tenant deployments) is mechanical — wrap the singleton slot
in a dict keyed by tenant ID.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

# Single source of truth for directory / filename constants.  Defined
# alongside the writer in pipelines.ensemble.infer so both sides
# of the contract stay in lockstep.
from src.rade_ml_pt.pipelines.ensemble.infer import (
    CLUSTER_SUMMARY_DIRNAME,
    CLUSTER_SUMMARY_FILENAME,
    INFERENCE_DIRNAME,
    MANIFEST_FILENAME,
    PORTFOLIO_SUMMARY_DIRNAME,
    PORTFOLIO_SUMMARY_FILENAME,
    TRADE_PREDICTIONS_DIRNAME,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# On-disk layout constants (reader-only)
# ──────────────────────────────────────────────────────────────────────

# Subdirectory under ``settings.artifacts_dir`` where all inference
# runs land.  Matches :func:`per_run_artifacts_dir` in
# :mod:`services.inference_state` so the writer (which builds
# per-run artifacts dirs) and the reader (which discovers them)
# agree without cross-coupling.
INFERENCE_RUNS_DIR: str = "inference_runs"

# Re-export the inference subdirectory constant under its
# historical name so any external caller that imports
# ``INFERENCE_SUBDIR`` from this module keeps working.
INFERENCE_SUBDIR:   str = INFERENCE_DIRNAME

# Allowed values for the ``space=`` query parameter on
# ``GET /inference/runs/{id}/clusters/{cid}/trades``.  Defined as a
# module constant so the router can use it in OpenAPI docs and the
# reader can use it for validation.
VALID_SPACES = ("scaled", "original")


# ──────────────────────────────────────────────────────────────────────
# Per-run filesystem layout
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RunPaths:
    """Resolved on-disk paths for one inference run.

    Frozen so the dataclass is hashable / cache-keyable.  Pure
    derived data — does NOT verify any path exists.  Callers check
    existence to raise the appropriate ``HTTPException`` themselves.

    Layout (single source of truth: ``pipelines.ensemble.infer``)::

        <inference_dir>/
        ├── manifest.json
        ├── cluster_summary/<CLUSTER_SUMMARY_FILENAME>
        ├── portfolio_summary/<PORTFOLIO_SUMMARY_FILENAME>
        └── trade_predictions/<cid>_{scaled,original}.parquet
    """

    run_id:                 str
    run_root:               Path   # <base>/inference_runs/<run_id>
    inference_dir:          Path   # <base>/inference_runs/<run_id>/inference
    manifest_path:          Path
    cluster_summary_path:   Path   # cluster_summary/cluster_predictions.parquet
    portfolio_summary_path: Path   # portfolio_summary/portfolio_predictions.parquet
    trade_predictions_dir:  Path   # holds <cid>_{scaled,original}.parquet

    def cluster_parquet(self, cluster_id: str, space: str) -> Path:
        """Resolve the per-cluster wide parquet path.

        ``space`` must be one of :data:`VALID_SPACES`; the API
        validates this upstream but we re-check defensively to keep
        the file-path layer self-contained.
        """
        if space not in VALID_SPACES:
            raise ValueError(
                f"space must be one of {VALID_SPACES}; got {space!r}"
            )
        return self.trade_predictions_dir / f"{cluster_id}_{space}.parquet"


# ──────────────────────────────────────────────────────────────────────
# Module-level loaders
# ──────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> Dict[str, Any]:
    """Read a JSON file into a plain dict."""
    return json.loads(path.read_text())


def _load_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file into a pandas DataFrame.

    Routed through ``pyarrow.parquet.read_table`` (rather than
    ``pd.read_parquet``) for consistency with :class:`ArtifactReader`
    and to keep pyarrow as the only parquet engine dependency.
    """
    return pq.read_table(path).to_pandas()


# ──────────────────────────────────────────────────────────────────────
# Reader
# ──────────────────────────────────────────────────────────────────────

class InferenceResultReader:
    """Typed accessor over the inference-runs directory.

    Instances are essentially stateless — every method resolves the
    on-disk layout from ``run_id`` and reads the file.  No mtime
    cache (runs are append-only) but typical access patterns are
    cold (one read per UI navigation event) so caching wouldn't pay
    for itself.

    Attributes
    ----------
    base_dir
        Resolved ``settings.artifacts_dir`` — root under which the
        ``inference_runs/<run_id>/inference/`` tree lives.
    """

    def __init__(self, base_artifacts_dir: str):
        self.base_dir = Path(base_artifacts_dir)

    # ── Path resolution ─────────────────────────────────────────────

    def paths_for(self, run_id: str) -> RunPaths:
        """Resolve the per-run filesystem layout.

        Pure path arithmetic — does NOT check that any file exists.
        The router does an existence check and raises a 404 / 409
        as appropriate.
        """
        run_root      = self.base_dir / INFERENCE_RUNS_DIR / run_id
        inference_dir = run_root / INFERENCE_SUBDIR
        return RunPaths(
            run_id                 = run_id,
            run_root               = run_root,
            inference_dir          = inference_dir,
            manifest_path          = inference_dir / MANIFEST_FILENAME,
            cluster_summary_path   = (
                inference_dir / CLUSTER_SUMMARY_DIRNAME / CLUSTER_SUMMARY_FILENAME
            ),
            portfolio_summary_path = (
                inference_dir / PORTFOLIO_SUMMARY_DIRNAME / PORTFOLIO_SUMMARY_FILENAME
            ),
            trade_predictions_dir  = inference_dir / TRADE_PREDICTIONS_DIRNAME,
        )

    # ── Discovery ───────────────────────────────────────────────────

    def list_run_ids(self) -> List[str]:
        """Return all run IDs on disk, most recent first.

        Sorts by directory name descending.  The pipeline's run_id
        format is ``<ensemble_version>__<UTC_yyyymmdd_HHMMSS>`` so
        descending alphabetical order is also descending temporal
        order within the same ensemble version.

        Returns
        -------
        list of str
            Empty list when ``inference_runs/`` doesn't exist yet
            (fresh deployment with no runs).
        """
        root = self.base_dir / INFERENCE_RUNS_DIR
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
            If the manifest doesn't exist — typically because the
            run is still in flight (the worker thread hasn't reached
            ``post_infer`` yet) or never completed successfully.
        """
        path = self.paths_for(run_id).manifest_path
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest not found for run '{run_id}': {path}"
            )
        return _load_json(path)

    # ── Run summary ─────────────────────────────────────────────────

    def run_summary(self, run_id: str) -> Dict[str, Any]:
        """Lightweight summary suitable for the run-history table.

        Reads only the manifest (small JSON), not the parquets.
        When the manifest doesn't exist we still return a
        well-formed summary with ``status='in_progress'`` so the
        UI can list in-flight runs.

        Returns
        -------
        dict
            Keys match :class:`RunSummary` (pydantic model).
        """
        paths = self.paths_for(run_id)
        try:
            manifest = self.load_manifest(run_id)
            status   = "complete"
        except FileNotFoundError:
            manifest = {}
            status   = "in_progress"

        return {
            "run_id":           run_id,
            "ensemble_version": manifest.get("ensemble_version"),
            "generated_at":     manifest.get("generated_at"),
            "n_scenarios":      int(manifest.get("n_scenarios") or 0),
            "n_clusters":       len(manifest.get("clusters") or []),
            "status":           status,
            "manifest_path":    str(paths.manifest_path),
            "latency_seconds":  manifest.get("latency_seconds"),
        }

    # ── Long-format parquets (run-level summaries) ──────────────────

    def load_portfolio(self, run_id: str) -> pd.DataFrame:
        """Read ``portfolio_predictions.parquet`` for ``run_id``.

        Long-format: one row per scenario, columns are
        ``scenario_label``, ``sum_pnl_scaled``, ``sum_pnl_original``,
        ``n_clusters``.
        """
        path = self.paths_for(run_id).portfolio_summary_path
        if not path.exists():
            raise FileNotFoundError(
                f"Portfolio summary not found for run '{run_id}': {path}"
            )
        return _load_parquet(path)

    def load_clusters_summary(self, run_id: str) -> pd.DataFrame:
        """Read ``cluster_predictions.parquet`` for ``run_id``.

        Long-format: one row per cluster × scenario.  Schema is
        whatever :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
        emits in its ``summary_df`` — currently ``scenario_label``,
        ``cluster_id``, ``sum_pnl_scaled``, ``sum_pnl_original``,
        ``mean_pnl_original``, ``std_pnl_original``,
        ``min_pnl_original``, ``max_pnl_original``.
        """
        path = self.paths_for(run_id).cluster_summary_path
        if not path.exists():
            raise FileNotFoundError(
                f"Cluster summary not found for run '{run_id}': {path}"
            )
        return _load_parquet(path)

    # ── Wide-format parquets (per-cluster trade-level) ──────────────

    def load_cluster_trades(
        self,
        run_id:     str,
        cluster_id: str,
        space:      str = "original",
    ) -> pd.DataFrame:
        """Read one per-cluster wide parquet.

        Parameters
        ----------
        space
            Either ``'scaled'`` (model output space) or
            ``'original'`` (inverse-transformed, notional-restored).
            See :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
            for the difference.  Default is ``'original'`` since
            that's what the user-facing dashboard reads by default.

        Returns
        -------
        pd.DataFrame
            Wide format — index = ``scenario_label``, columns =
            trade IDs, values = predicted PnL in the requested
            space.
        """
        path = self.paths_for(run_id).cluster_parquet(cluster_id, space)
        if not path.exists():
            raise FileNotFoundError(
                f"Cluster trades not found: run='{run_id}', "
                f"cluster='{cluster_id}', space='{space}': {path}"
            )
        return _load_parquet(path)

    # ── Convenience: snapshots embedded in the manifest ─────────────

    def load_validation_snapshot(
        self, run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Pull the ``ValidationReport`` snapshot out of the manifest.

        Cheaper than the active run's ``GET /validate`` because no
        on-disk parquet is involved — the snapshot is captured in
        ``manifest.json`` at write time.

        Returns ``None`` if the manifest doesn't carry one (older
        runs written before the snapshot was added).
        """
        return self.load_manifest(run_id).get("validation")

    def load_scenarios_snapshot(
        self, run_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Pull the ``LoadedScenariosReport`` snapshot out of the manifest."""
        return self.load_manifest(run_id).get("scenarios")


# ──────────────────────────────────────────────────────────────────────
# Process-wide singleton (mirrors InferenceStateManager pattern)
# ──────────────────────────────────────────────────────────────────────

_reader: Optional[InferenceResultReader] = None


def set_result_reader(reader: InferenceResultReader) -> None:
    """Inject the reader at app-lifespan startup."""
    global _reader
    _reader = reader


def get_result_reader() -> InferenceResultReader:
    """FastAPI ``Depends`` — returns the singleton reader.

    Raises
    ------
    RuntimeError
        If called before the lifespan startup has run — indicates
        a misconfigured server.
    """
    if _reader is None:
        raise RuntimeError(
            "InferenceResultReader not initialised. Server not ready."
        )
    return _reader
