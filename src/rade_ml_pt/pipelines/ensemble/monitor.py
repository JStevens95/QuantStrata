"""Ensemble monitoring pipeline.

Composes :class:`EnsembleInferencePipeline` to reuse the already-stable
load → load_scenarios → validate_scenarios stages, then computes per-
cluster drift against the training-time baselines emitted by
:mod:`monitoring.baselines`.

Pipeline waterfall::

    EnsembleMonitoringPipeline.run(new_scenario_dir)
    │
    ├─ 1. inference.load()                       (model + per-cluster contexts)
    ├─ 2. inference.load_scenarios(...)          (parse shock CSVs)
    ├─ 3. inference.validate_scenarios()         (routing + cheap-path check)
    ├─ 4. for cid in members:
    │       today_features = _today_features(ctx, decision, shocks, labels)
    │       baseline_df    = load_baseline(version_dir(ctx) / "monitoring" / "...")
    │       drift_table    = build_drift_table(baseline_df, today_features, ...)
    │       write_drift_table_parquet(...)
    ├─ 5. portfolio_summary = build_portfolio_drift_summary([drift_tables])
    │     write_drift_summary_json(...)
    │     write_monitoring_manifest_json(...)
    │
    └─ return MonitoringResult(run_id, summary, drift_tables, manifest_path)

NOT implemented in M.2 (deferred to M.3):
* "Promote-to-predictions" step that re-uses the inference pipeline's
  ``run_inference()`` to write ``predictions_*.parquet`` into the SAME
  monitoring run directory.  The hook point is :meth:`run` itself —
  M.3 will add a ``promote_to_predictions(run_id)`` method that
  re-loads the run's manifest and chains into inference.

Threading + parallelism: M.2 runs clusters sequentially.  Drift compute
is sub-second per cluster; the wall-clock bottleneck is the lazy load
of ``cluster_assets`` for affected clusters (shared with inference).
M.4 / M.5 can add ``ThreadPoolExecutor`` if portfolios grow large.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from src.rade_ml_pt.core.types import InferenceResult
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.monitoring.drift import (
    SEVERITY_NO_DATA,
    build_drift_table,
    build_portfolio_drift_summary,
)
from src.rade_ml_pt.monitoring.loaders import load_baseline
from src.rade_ml_pt.monitoring.run_paths import (
    MonitoringRunPaths,
    monitoring_run_id,
)
from src.rade_ml_pt.monitoring.writers import (
    write_drift_summary_json,
    write_drift_table_parquet,
    write_monitoring_manifest_json,
)
from src.rade_ml_pt.pipelines.ensemble.infer import (
    ClusterRoutingDecision,
    EnsembleInferencePipeline,
)
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    EmitFn,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_RUNNING,
    event,
    noop_emit,
)

logger = logging.getLogger(__name__)


# Monitoring schema version stamped onto the manifest.  Bump only on
# back-incompatible manifest shape changes — adding new optional keys
# does NOT require a bump.
MANIFEST_SCHEMA_VERSION: int = 1

# Filename of the baseline parquet under each member's ``version_dir``.
# Must match what ``monitoring.baselines.save_feature_baseline`` writes
# from the training pipeline.
_BASELINE_RELPATH = Path("monitoring") / "baseline_feature_stats.parquet"

# Pipeline stage tag used in activity log emissions.  Kept as a plain
# string (cast at the call-site) because the inference event vocab is
# a Literal[...]; widening it would require touching ``infer_events.py``
# which we explicitly avoid in M.2.  The UI's monitoring activity log
# (M.6) will filter on this tag.
_STAGE_MONITORING: str = "monitoring"

# Sub-directory name (RELATIVE to ``monitoring_dir``) where
# ``promote_to_predictions`` lands its prediction artifacts.  Mirrors
# ``infer.INFERENCE_DIRNAME`` ("inference") — we duplicate the literal
# rather than import it so this module stays insulated from any future
# rename inside infer.py.  Final layout:
#     monitoring_runs/<run_id>/monitoring/inference/manifest.json
#     monitoring_runs/<run_id>/monitoring/inference/cluster_summary/...
#     ...
# achieved by stash-swapping ``config.artifacts_dir`` to point at
# ``monitoring_dir`` before calling ``run_inference()`` — see
# :meth:`EnsembleMonitoringPipeline.promote_to_predictions` for the
# safety wrapper.
_PROMOTE_PREDICTIONS_SUBDIR: str = "inference"


# ═════════════════════════════════════════════════════════════════════
# Result dataclass
# ═════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MonitoringResult:
    """Frozen return value of :meth:`EnsembleMonitoringPipeline.run`.

    Programmatic callers consume this directly; the API layer (M.4)
    will translate it into a Pydantic response model.  Heavy
    per-cluster ``drift_tables`` are also persisted to disk under
    :attr:`MonitoringRunPaths.cluster_drift_table_path` — the in-memory
    copy is convenient for tests and short-lived UI sessions but the
    disk copy is the source of truth for cross-process consumers.
    """

    run_id:            str
    ensemble_version:  str
    artifacts_dir:     Path
    manifest_path:     Path
    portfolio_summary: Dict[str, Any]
    drift_tables:      Dict[str, pd.DataFrame] = field(default_factory=dict)
    n_scenarios:       int                     = 0
    n_clusters:        int                     = 0
    n_affected:        int                     = 0
    n_unaffected:      int                     = 0
    wall_seconds:      float                   = 0.0


@dataclass(frozen=True)
class PromoteResult:
    """Frozen return value of :meth:`EnsembleMonitoringPipeline.promote_to_predictions`.

    Captures the location of the new prediction artifacts plus a
    handle on the underlying :class:`InferenceResult` for programmatic
    callers that want to inspect predictions in memory.  The on-disk
    layout (``predictions_dir``) is the authoritative source for
    cross-process consumers (API result reader, UI).

    Attributes
    ----------
    run_id
        Monitoring run-id this promote was attached to.  Same value
        as :attr:`MonitoringResult.run_id` — kept here so the
        promote result is self-describing without joining back to
        the originating MonitoringResult.
    predictions_dir
        Absolute path to the directory ``run_inference()`` wrote
        into.  Always equal to ``monitoring_dir / "inference"``.
    manifest_path
        Absolute path to the (now-rewritten) monitoring ``manifest.json``.
        The manifest's ``predictions`` field points at
        ``predictions_dir`` via a relative-to-``run_dir`` path.
    inference_result
        The :class:`InferenceResult` returned by the underlying
        ``run_inference()`` call.  Useful for in-memory inspection
        of predictions without re-reading parquets.
    promoted_at
        ISO-8601 UTC timestamp (seconds precision) of when the
        promote completed.  Mirrors the manifest's ``promoted_at``.
    wall_seconds
        End-to-end wall-clock time for the promote (forward pass +
        post-infer writes + manifest rewrite).
    n_clusters_predicted
        Number of clusters that produced predictions.  Same as
        ``inference_result.metadata.get("n_clusters")`` when
        available; otherwise inferred from the cluster artifacts
        on disk.
    """

    run_id:                str
    predictions_dir:       Path
    manifest_path:         Path
    inference_result:      InferenceResult
    promoted_at:           str
    wall_seconds:          float
    n_clusters_predicted:  int


# ═════════════════════════════════════════════════════════════════════
# Pipeline
# ═════════════════════════════════════════════════════════════════════

class EnsembleMonitoringPipeline:
    """Run drift monitoring on an ensemble against a new scenario set.

    Composes :class:`EnsembleInferencePipeline` for the load /
    load_scenarios / validate stages — see module docstring for the
    full waterfall.

    Parameters
    ----------
    ensemble_config
        Must have ``registry_dir`` and ``artifacts_dir`` set.  The
        same config a non-UI inference run would use; monitoring uses
        a ``monitoring_runs/<run_id>/`` sibling layout to inference
        artifacts.
    ensemble_version
        Ensemble version or tag to monitor.
    session
        Optional pre-loaded :class:`EnsembleSession` (UI / backend
        warm path).  Must match ``ensemble_version`` — the inference
        pipeline enforces this in :meth:`load`.
    on_event
        Optional lifecycle-event callback.  Forwarded to the
        composed inference pipeline AND used directly by this class
        for monitoring-specific events.  Defaults to a no-op.
    """

    def __init__(
        self,
        ensemble_config:  EnsembleConfig,
        ensemble_version: str = "latest",
        session:          Optional[Any] = None,
        *,
        on_event:         Optional[EmitFn] = None,
    ) -> None:
        self.config           = ensemble_config
        self.ensemble_version = ensemble_version
        self._session         = session
        self._emit: EmitFn    = on_event if on_event is not None else noop_emit

        # Composed inference pipeline — owns the model + contexts +
        # validation report.  Created on every instance so monitoring
        # runs are independent (no cross-run state leak).
        self._inference_pipeline = EnsembleInferencePipeline(
            ensemble_config  = ensemble_config,
            ensemble_version = ensemble_version,
            session          = session,
            on_event         = on_event,
        )

        # Populated by run().
        self._run_paths:    Optional[MonitoringRunPaths] = None
        self._drift_tables: Dict[str, pd.DataFrame]      = {}

        # Cached after validate_scenarios() so compute_drift() can
        # consume it without reaching into the composed inference
        # pipeline's private state.  ``run()`` populates this via
        # the staged ``validate_scenarios()`` proxy below.
        self._validation_report = None

        # Cached after run() so promote_to_predictions can re-build
        # the manifest without re-aggregating the per-cluster drift
        # tables.  Reset to None after a successful promote so the
        # manifest write is a one-shot transition (M.3 promotes are
        # NOT re-runnable on the same instance — caller constructs a
        # new pipeline + new run_id if they want another promote).
        self._cached_portfolio_summary: Optional[Dict[str, Any]] = None

        # Populated by promote_to_predictions() — exposes the inference
        # result for callers that want in-memory access without
        # re-reading parquets.  Mirrors the on-disk state captured in
        # the rewritten manifest's "predictions" block.
        self._promotion: Optional[PromoteResult] = None

    # ─────────────────────────────────────────────────────────────────
    # Public API — staged (UI / API driver)
    # ─────────────────────────────────────────────────────────────────
    #
    # These four methods (``load`` / ``load_scenarios`` /
    # ``validate_scenarios`` / ``run``) mirror the inference pipeline's
    # public surface 1:1, so the M.4 API router can drive monitoring
    # through the same four-stage state machine the inference router
    # uses — without reaching into ``self._inference_pipeline`` (which
    # would couple the router to monitor.py's internals).
    #
    # Stage 1-3 are pure forwarders to the composed inference pipeline:
    # monitoring needs the same model + same scenarios + same validation
    # before it can compute drift, so re-implementing them would just
    # duplicate the inference contract.
    # ─────────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Stage 1 — cold-load the ensemble + per-cluster contexts.

        Pure forwarder to :meth:`EnsembleInferencePipeline.load` —
        the underlying load is identical to the inference path.
        """
        self._inference_pipeline.load()

    def load_scenarios(
        self,
        new_scenario_dir: Optional[Union[str, Path]] = None,
    ):
        """Stage 2 — parse the new-scenario shock folder.

        Pure forwarder to
        :meth:`EnsembleInferencePipeline.load_scenarios`.  Returns
        the :class:`LoadedScenariosReport` so the API router can wrap
        it in :class:`LoadScenariosResponse`.
        """
        return self._inference_pipeline.load_scenarios(new_scenario_dir)

    def validate_scenarios(self):
        """Stage 3 — compute per-cluster routing decisions + run-level errors.

        Forwards to :meth:`EnsembleInferencePipeline.validate_scenarios`
        and caches the report on this instance so
        :meth:`compute_drift` can consume it without reaching into the
        composed inference pipeline's private ``_validation_report``.

        Returns
        -------
        ValidationReport
            Wrapped by the API router into a :class:`ValidateResponse`.
        """
        report = self._inference_pipeline.validate_scenarios()
        self._validation_report = report
        return report

    def compute_drift(self) -> MonitoringResult:
        """Stage 4 — compute drift artefacts from the validated scenarios.

        Assumes :meth:`load`, :meth:`load_scenarios`, and
        :meth:`validate_scenarios` have already run on this instance.
        Symmetric with the four-stage state machine the inference
        pipeline exposes, so the M.4 API router can drive monitoring
        stage-by-stage without ever calling :meth:`run` itself.

        On disk: writes the manifest + drift_summary + per-cluster
        drift tables under ``monitoring_runs/<run_id>/monitoring/``.

        Returns
        -------
        MonitoringResult
            Frozen handle on the run's KPIs + on-disk paths.  Same
            shape :meth:`run` returns.

        Raises
        ------
        RuntimeError
            If :meth:`validate_scenarios` hasn't been called on this
            instance.
        ValueError
            If the cached validation report contains user-input errors
            (mirrors :meth:`run`'s behaviour).
        """
        if self._validation_report is None:
            raise RuntimeError(
                "compute_drift() requires a prior validate_scenarios() "
                "call on this instance."
            )
        report = self._validation_report

        if not report.is_valid:
            raise ValueError(
                f"Monitoring validation failed with "
                f"{len(report.errors)} error(s): {report.errors}"
            )

        t0 = time.perf_counter()
        try:
            self._run_paths    = self._init_run_paths()
            self._drift_tables = self._compute_drift_for_all_clusters()
            portfolio_summary  = build_portfolio_drift_summary(
                list(self._drift_tables.values())
            )
            # Cache for promote_to_predictions() — same rationale as
            # the prior :meth:`run` body.
            self._cached_portfolio_summary = portfolio_summary
            self._write_run_artifacts(portfolio_summary)
        except Exception as exc:
            self._emit(_mon_event(
                "Pipeline failed", status=STATUS_FAIL,
                target=type(exc).__name__, detail=str(exc),
            ))
            raise

        wall         = time.perf_counter() - t0
        n_affected   = report.affected_count
        n_unaffected = report.unaffected_count

        assert self._run_paths is not None  # set in the try-block

        result = MonitoringResult(
            run_id            = self._run_paths.run_id,
            ensemble_version  = self.ensemble_version,
            artifacts_dir     = self._run_paths.run_dir,
            manifest_path     = self._run_paths.manifest_path,
            portfolio_summary = portfolio_summary,
            drift_tables      = self._drift_tables,
            n_scenarios       = report.n_scenarios,
            n_clusters        = n_affected + n_unaffected,
            n_affected        = n_affected,
            n_unaffected      = n_unaffected,
            wall_seconds      = wall,
        )

        logger.info(
            "EnsembleMonitoringPipeline.compute_drift: done "
            "(%.3fs, %d clusters, severity=%s)",
            wall, result.n_clusters, portfolio_summary.get("severity"),
        )
        self._emit(_mon_event(
            "Pipeline complete", status=STATUS_OK,
            target=f"{wall * 1000:.0f} ms · {result.n_clusters} clusters",
            detail=f"severity={portfolio_summary.get('severity')}",
        ))
        return result

    def run(
        self,
        new_scenario_dir: Optional[Union[str, Path]] = None,
    ) -> MonitoringResult:
        """Execute the full monitoring pipeline.

        Convenience wrapper that calls the four staged methods in
        order — symmetric with
        :meth:`EnsembleInferencePipeline.run` so callers can swap one
        for the other when they want a drift-only view of today's
        scenarios.

        Parameters
        ----------
        new_scenario_dir
            Directory containing the new-scenario shock CSVs.  Falls
            back to ``config.metadata['inference']['new_scenario_dir']``
            if ``None`` (matching the inference pipeline's lookup).

        Raises
        ------
        ValueError
            If the inference validation reports user-input errors
            (e.g. unaffected cluster with missing scenario labels).
        """
        logger.info("EnsembleMonitoringPipeline: starting")
        self._emit(_mon_event("Pipeline started", status=STATUS_RUNNING,
                              target=self.ensemble_version))

        # Delegate to the staged methods so the orchestrated path and
        # the API-driven (stage-by-stage) path share one implementation
        # — eliminates a drift surface between ``run()`` and the M.4
        # router.  ``compute_drift`` emits the terminal ``Pipeline
        # complete`` event.
        self.load()
        self.load_scenarios(new_scenario_dir)
        self.validate_scenarios()
        return self.compute_drift()

    def promote_to_predictions(self) -> PromoteResult:
        """Forward-pass the same scenarios this run() just drifted on.

        Optional follow-up to :meth:`run` — re-uses the already-loaded
        ensemble + per-cluster contexts + validated scenarios to
        produce per-trade predicted PnL artifacts inside the SAME
        monitoring run directory.

        On-disk layout after a successful promote::

            monitoring_runs/<run_id>/monitoring/
              ├── manifest.json              ← rewritten with predictions block
              ├── drift_summary.json
              ├── clusters/<cid>/drift_table.parquet
              └── inference/                 ← NEW
                  ├── manifest.json
                  ├── cluster_summary/cluster_predictions.parquet
                  ├── portfolio_summary/portfolio_predictions.parquet
                  └── trade_predictions/<cid>_<space>.parquet

        Mechanism: temporarily set
        ``inference_pipeline.config.artifacts_dir`` to ``monitoring_dir``
        so the existing ``run_inference()`` writes its standard
        ``<root>/inference/...`` tree INSIDE the monitoring run rather
        than to the global inference dir.  The swap is wrapped in
        ``try/finally`` so partial failures restore the config
        cleanly.

        Returns
        -------
        PromoteResult
            Self-describing handle on the new predictions
            (``predictions_dir``, ``inference_result``, timing).

        Raises
        ------
        RuntimeError
            If :meth:`run` has not been called on this instance, OR
            if a promote has already succeeded on this instance
            (promotes are NOT re-runnable — construct a new pipeline
            for a fresh run_id).
        """
        if self._run_paths is None or self._cached_portfolio_summary is None:
            raise RuntimeError(
                "promote_to_predictions() requires a successful prior run() "
                "on this instance.  Call pipeline.run(new_scenario_dir) first."
            )
        if self._promotion is not None:
            raise RuntimeError(
                "promote_to_predictions() has already succeeded on this "
                "instance.  Construct a new EnsembleMonitoringPipeline if "
                "you need to monitor + promote a fresh scenario set."
            )

        logger.info(
            "EnsembleMonitoringPipeline.promote_to_predictions: starting "
            "(run_id=%s)", self._run_paths.run_id,
        )
        self._emit(_mon_event(
            "Promote started", status=STATUS_RUNNING,
            target=self.ensemble_version,
        ))
        t0 = time.perf_counter()

        config            = self._inference_pipeline.config
        original_root     = config.artifacts_dir
        # Stash-swap target: ``run_inference()`` joins ``INFERENCE_DIRNAME``
        # onto this, so predictions land at ``monitoring_dir/inference/``.
        # We mkdir explicitly first because run_inference's own mkdir
        # is conditional on ``config.artifacts_dir`` being truthy and
        # we want to fail loudly if the swap target is unwritable.
        swap_target = self._run_paths.monitoring_dir
        swap_target.mkdir(parents=True, exist_ok=True)

        try:
            config.artifacts_dir = str(swap_target)
            inference_result = self._inference_pipeline.run_inference()
        except Exception as exc:
            self._emit(_mon_event(
                "Promote failed", status=STATUS_FAIL,
                target=type(exc).__name__, detail=str(exc),
            ))
            raise
        finally:
            # Restore even on failure so the inference pipeline (and
            # any callers sharing this config object) sees the
            # original artifacts_dir afterwards.
            config.artifacts_dir = original_root

        wall         = time.perf_counter() - t0
        promoted_at  = datetime.now(timezone.utc).isoformat(timespec="seconds")
        predictions_dir = swap_target / _PROMOTE_PREDICTIONS_SUBDIR

        # Derive ``n_clusters_predicted`` from the inference result
        # metadata when present; fall back to counting cluster
        # parquets on disk so the UI's "N clusters scored" KPI is
        # never silently wrong even if ``infer.py`` evolves.
        n_clusters_predicted = self._derive_n_clusters_predicted(
            inference_result, predictions_dir,
        )

        predictions_block = self._build_predictions_block(
            predictions_dir      = predictions_dir,
            promoted_at          = promoted_at,
            wall_seconds         = wall,
            n_clusters_predicted = n_clusters_predicted,
        )

        # Re-write the monitoring manifest with the new predictions
        # pointer.  All other manifest fields stay byte-identical
        # because we rebuild from the cached portfolio_summary.
        manifest = self._build_manifest(
            self._cached_portfolio_summary,
            predictions_block=predictions_block,
        )
        write_monitoring_manifest_json(
            manifest = manifest,
            out_path = self._run_paths.manifest_path,
        )

        promote_result = PromoteResult(
            run_id               = self._run_paths.run_id,
            predictions_dir      = predictions_dir,
            manifest_path        = self._run_paths.manifest_path,
            inference_result     = inference_result,
            promoted_at          = promoted_at,
            wall_seconds         = wall,
            n_clusters_predicted = n_clusters_predicted,
        )
        self._promotion = promote_result

        logger.info(
            "EnsembleMonitoringPipeline.promote_to_predictions: done "
            "(%.3fs, %d clusters predicted, predictions_dir=%s)",
            wall, n_clusters_predicted, predictions_dir,
        )
        self._emit(_mon_event(
            "Promote complete", status=STATUS_OK,
            target=f"{wall * 1000:.0f} ms · {n_clusters_predicted} clusters",
            detail=f"predictions_dir={predictions_dir.name}",
        ))
        return promote_result

    # ─────────────────────────────────────────────────────────────────
    # Internal — drift compute
    # ─────────────────────────────────────────────────────────────────

    def _compute_drift_for_all_clusters(self) -> Dict[str, pd.DataFrame]:
        """Walk the validation report, compute a drift table per cluster.

        Two paths:
        * **Affected** cluster → today's features = output of
          :meth:`HybridGnnRnnInferencePipeline.build_new_scenario_inputs`
          (same call inference makes; the heavy ``cluster_assets``
          load happens once and is shared by a later promote step).
        * **Unaffected** cluster → today's features =
          ``ctx.elementary_pnl.loc[scenario_labels]`` (the historical
          PnL slice for the requested scenario labels).  This still
          tells us "did today's scenarios pick a slice of history
          that looks different from training?".

        Missing baselines or shape mismatches degrade gracefully — the
        affected feature column emits a ``severity = "no_data"`` row
        rather than crashing the run.
        """
        assert self._inference_pipeline._validation_report is not None, \
            "validate_scenarios() must run first"
        assert self._inference_pipeline._new_scenario_shocks is not None, \
            "load_scenarios() must run first"

        report          = self._inference_pipeline._validation_report
        shocks          = self._inference_pipeline._new_scenario_shocks
        scenario_labels = report.scenario_labels

        drift_tables: Dict[str, pd.DataFrame] = {}

        for decision in report.cluster_decisions:
            cid = decision.cluster_id
            ctx = self._inference_pipeline._inference_contexts[cid]

            self._emit(_mon_event(
                "Computing drift", status=STATUS_RUNNING, target=cid,
                detail="affected" if decision.is_affected else "unaffected",
            ))

            try:
                today_features = self._today_features(
                    ctx              = ctx,
                    decision         = decision,
                    shocks           = shocks,
                    scenario_labels  = scenario_labels,
                )
                baseline_df = self._load_cluster_baseline(ctx, cluster_id=cid)

                drift_table = build_drift_table(
                    baseline_df      = baseline_df,
                    current_features = today_features,
                    cluster_id       = cid,
                )
            except FileNotFoundError as exc:
                # No baseline for this cluster — emit a no_data row per
                # feature in the current frame so the portfolio
                # summary still reflects the cluster's coverage.
                logger.warning(
                    "No baseline found for cluster '%s' (%s) — drift row will "
                    "be emitted as no_data", cid, exc,
                )
                drift_table = _no_data_table_for_cluster(cid)

            drift_tables[cid] = drift_table

            sev_counts = dict(drift_table["severity"].value_counts())
            self._emit(_mon_event(
                "Cluster drift ready", status=STATUS_OK, target=cid,
                detail=f"{len(drift_table)} features · {sev_counts}",
            ))

        return drift_tables

    @staticmethod
    def _today_features(
        ctx:             Any,
        decision:        ClusterRoutingDecision,
        shocks:          Dict[str, Any],
        scenario_labels: List[str],
    ) -> pd.DataFrame:
        """Build today's feature matrix for one cluster.

        Returns a ``DataFrame`` shaped ``(n_scenarios × n_features)``
        in the same coordinate system the training baseline was built
        in (scaled space).  Compatible with
        :func:`monitoring.drift.build_drift_table`.

        Affected path delegates to the same builder inference uses, so
        the two pipelines see byte-identical "today" features for any
        cluster they both touch.  Unaffected path slices the
        cluster's historical ``elementary_pnl`` at the requested
        scenario labels — same data the cheap-path inference loads.

        The heavy ``HybridGnnRnnInferencePipeline`` import is deferred
        to the affected branch so a pure unaffected-only run (or a
        unit test) doesn't pull in the static-replication dependency
        chain.
        """
        if decision.is_affected:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                HybridGnnRnnInferencePipeline,
            )
            result = HybridGnnRnnInferencePipeline.build_new_scenario_inputs(
                ctx                 = ctx,
                new_scenario_shocks = shocks,
                scenario_labels     = scenario_labels,
                is_affected         = True,
            )
            inputs = result["inputs"]
            elementary_pnl = inputs.elementary_pnl
            if elementary_pnl is None:
                raise RuntimeError(
                    f"build_new_scenario_inputs returned no elementary_pnl "
                    f"for cluster '{decision.cluster_id}'."
                )
            return elementary_pnl

        # Unaffected path — historical PnL at the requested labels.
        # Labels are guaranteed present by the validate_scenarios()
        # cheap-path eligibility check (any missing label would have
        # been surfaced as an error and we'd never reach this branch).
        hist = ctx.elementary_pnl
        if hist is None:
            raise RuntimeError(
                f"Cluster '{decision.cluster_id}' is unaffected but its "
                f"InferenceContext has no elementary_pnl loaded."
            )
        return hist.loc[scenario_labels]

    @staticmethod
    def _load_cluster_baseline(ctx: Any, *, cluster_id: str) -> pd.DataFrame:
        """Locate + load the training-time baseline parquet.

        Resolution order (first hit wins):
        1. ``ctx._cluster_assets_path.parent`` — set by
           ``load_inference_context_from_dir``; works for both
           registry cold-loads and session warm-loads.
        2. Future: a ``ctx.version_dir`` field if we add one.

        Raises
        ------
        FileNotFoundError
            If no baseline parquet can be resolved / read.  The caller
            converts this into a no_data drift row.
        """
        if ctx._cluster_assets_path is None:
            raise FileNotFoundError(
                f"Cluster '{cluster_id}': InferenceContext has no "
                f"_cluster_assets_path; cannot derive baseline location."
            )
        version_dir = ctx._cluster_assets_path.parent
        baseline_path = version_dir / _BASELINE_RELPATH
        return load_baseline(baseline_path)

    # ─────────────────────────────────────────────────────────────────
    # Internal — artifact writing
    # ─────────────────────────────────────────────────────────────────

    def _init_run_paths(self) -> MonitoringRunPaths:
        """Mint a fresh run-id + create the on-disk layout."""
        run_id = monitoring_run_id(self.ensemble_version)
        artifacts_dir = Path(self.config.artifacts_dir)
        paths = MonitoringRunPaths(
            artifacts_dir    = artifacts_dir,
            run_id           = run_id,
            ensemble_version = self.ensemble_version,
        )
        paths.ensure_dirs()
        logger.info(
            "Monitoring run dir: %s (run_id=%s)", paths.run_dir, paths.run_id,
        )
        return paths

    def _write_run_artifacts(self, portfolio_summary: Dict[str, Any]) -> None:
        """Persist per-cluster parquets + run-level JSON files."""
        assert self._run_paths is not None

        # Per-cluster drift tables.
        for cid, drift_table in self._drift_tables.items():
            write_drift_table_parquet(
                drift_table      = drift_table,
                out_path         = self._run_paths.cluster_drift_table_path(cid),
                cluster_id       = cid,
                ensemble_version = self.ensemble_version,
                run_id           = self._run_paths.run_id,
            )

        # Run-level summary + manifest.
        write_drift_summary_json(
            summary  = portfolio_summary,
            out_path = self._run_paths.drift_summary_path,
        )
        write_monitoring_manifest_json(
            manifest = self._build_manifest(portfolio_summary),
            out_path = self._run_paths.manifest_path,
        )

    def _build_manifest(
        self,
        portfolio_summary: Dict[str, Any],
        *,
        predictions_block: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Construct the canonical run manifest.

        Mirrors the inference manifest schema where it makes sense
        (``schema_version``, ``run_id``, ``ensemble_version``,
        ``created_at``) and adds monitoring-specific keys
        (``drift_summary``, ``cluster_drift_tables``, ``predictions``).

        ``predictions_block`` is ``None`` on the initial post-run
        manifest write (drift-only) and populated by
        :meth:`promote_to_predictions` on the rewrite.  Same field
        either way so manifest consumers (UI / API result reader) can
        check ``if manifest["predictions"] is not None`` to decide
        whether to render the prediction views.
        """
        assert self._run_paths is not None
        report   = self._inference_pipeline._validation_report
        loaded   = self._inference_pipeline._loaded_scenarios

        # Per-cluster parquet paths stored RELATIVE to the run dir so
        # the manifest moves with the directory tree.  The API reader
        # joins them back against the manifest's own location.
        cluster_drift_tables: Dict[str, str] = {}
        for cid in self._drift_tables:
            abs_path = self._run_paths.cluster_drift_table_path(cid)
            rel_path = abs_path.relative_to(self._run_paths.run_dir)
            cluster_drift_tables[cid] = str(rel_path)

        return {
            "schema_version":       MANIFEST_SCHEMA_VERSION,
            "run_id":                self._run_paths.run_id,
            "ensemble_version":      self.ensemble_version,
            "created_at":            datetime.now(timezone.utc).isoformat(
                                         timespec="seconds"),
            "input_mode":            "new_scenarios",
            "new_scenario_dir":      loaded.new_scenario_dir if loaded else None,
            "n_scenarios":           report.n_scenarios if report else 0,
            "n_clusters":            report.affected_count + report.unaffected_count
                                     if report else 0,
            "n_clusters_affected":   report.affected_count   if report else 0,
            "n_clusters_unaffected": report.unaffected_count if report else 0,
            "drift_summary":         portfolio_summary,
            "cluster_drift_tables":  cluster_drift_tables,
            # Null on first write, populated by promote_to_predictions.
            "predictions":           predictions_block,
        }

    def _build_predictions_block(
        self,
        *,
        predictions_dir:       Path,
        promoted_at:           str,
        wall_seconds:          float,
        n_clusters_predicted:  int,
    ) -> Dict[str, Any]:
        """Build the manifest's ``predictions`` block.

        Uses paths RELATIVE to the monitoring run dir so the manifest
        is portable if the run dir is ever moved (e.g. archived to
        long-term storage).  Consumers reconstruct the absolute path
        by joining against ``manifest_path.parent.parent`` (the
        ``run_dir``).
        """
        assert self._run_paths is not None
        rel_predictions_dir = predictions_dir.relative_to(self._run_paths.run_dir)
        return {
            "subdir":             str(rel_predictions_dir),
            "manifest_path":      str(rel_predictions_dir / "manifest.json"),
            "promoted_at":        promoted_at,
            "wall_seconds":       round(wall_seconds, 3),
            "n_clusters":         n_clusters_predicted,
        }

    @staticmethod
    def _derive_n_clusters_predicted(
        inference_result: InferenceResult,
        predictions_dir:  Path,
    ) -> int:
        """Best-effort count of clusters that produced predictions.

        Resolution order:
        1. ``inference_result.metadata['n_clusters']`` if set by
           ``run_inference()``.
        2. Count of ``trade_predictions/<cid>_<space>.parquet`` files
           on disk (one per cluster per space).  Divided by two
           because each cluster typically writes both scaled +
           original space files.
        3. Zero if neither path resolves a positive count — the
           manifest is still valid, the KPI just renders as ``"—"``.
        """
        meta_count = inference_result.metadata.get("n_clusters") \
            if inference_result.metadata else None
        if isinstance(meta_count, int) and meta_count > 0:
            return meta_count

        trade_preds_dir = predictions_dir / "trade_predictions"
        if not trade_preds_dir.is_dir():
            return 0
        parquets = list(trade_preds_dir.glob("*.parquet"))
        if not parquets:
            return 0
        # Trade prediction files are named ``<cid>_<space>.parquet``;
        # unique cluster ids derived by stripping the trailing
        # ``_scaled``/``_original`` suffix.
        cids = {p.stem.rsplit("_", 1)[0] for p in parquets}
        return len(cids)


# ═════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════

def _mon_event(
    phase: str,
    *,
    status: str = STATUS_OK,
    target: Optional[str] = None,
    detail: Optional[str] = None,
) -> Dict[str, Any]:
    """Thin wrapper around :func:`event` that stamps the monitoring stage.

    Suppresses the type-checker warning for passing a non-Literal
    string into ``Stage`` — see ``_STAGE_MONITORING`` docstring.
    """
    return event(
        _STAGE_MONITORING,  # type: ignore[arg-type]
        phase,
        status=status,  # type: ignore[arg-type]
        target=target,
        detail=detail,
    )


def _no_data_table_for_cluster(cluster_id: str) -> pd.DataFrame:
    """Single-row no_data drift table when a cluster has no baseline.

    Lets the portfolio summary's ``n_clusters`` count this cluster
    even though we couldn't score any features for it.  The single
    row carries ``feature_name="__no_baseline__"`` so the UI heatmap
    has something concrete to render in the cluster's column.
    """
    return build_drift_table(
        baseline_df = pd.DataFrame([{
            "feature_name": "__no_baseline__",
            "mean":         float("nan"),
            "std":          float("nan"),
            "hist_edges":   [],
            "hist_counts":  [],
        }]),
        current_features = pd.DataFrame(),
        cluster_id       = cluster_id,
    )


__all__ = [
    "EnsembleMonitoringPipeline",
    "MonitoringResult",
    "PromoteResult",
    "MANIFEST_SCHEMA_VERSION",
]
