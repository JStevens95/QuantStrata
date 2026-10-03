"""
Ensemble inference pipeline.

Two ways to drive a run:

1. **One-shot (programmatic / CLI):**
   Construct with an ``EnsembleConfig`` (and optionally an
   :class:`EnsembleSession`) and call ``run()``.  The pipeline
   executes the staged flow internally —
   ``load → load_scenarios → validate_scenarios → run_inference`` —
   and returns an :class:`InferenceResult`.

2. **Staged (UI):**
   Construct with an ``EnsembleConfig`` *plus* an
   :class:`EnsembleSession` so the model + contexts are warm-loaded
   from RAM instead of disk, then invoke each stage method
   individually.  Each button on the *Inference Console* triggers
   one stage and surfaces its report (or errors) to the user.

Per-cluster work (input building, routing classification) lives
in :class:`HybridGnnRnnInferencePipeline` as static methods; this
pipeline is the orchestrator + routing-decision layer.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.core.types import InferenceResult
from src.rade_ml_pt.pipelines.ensemble.infer_events import (
    EmitFn,
    STAGE_INFERENCE,
    STATUS_FAIL,
    STATUS_OK,
    STATUS_RUNNING,
    event,
    noop_emit,
)

if TYPE_CHECKING:
    from src.rade_ml_pt.ensemble.session import EnsembleSession
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dict_to_inference_context(
    raw: Dict[str, Any],
    data_config_override: Any = None,
) -> "InferenceContext":
    """
    Convert a raw context dict (as returned by ``load_inference_context_from_dir``)
    into an ``InferenceContext`` dataclass.

    Parameters
    ----------
    raw : dict
        Keys match those produced by ``load_inference_context_from_dir``.
    data_config_override : optional
        If provided, replaces the ``data_config`` from *raw*.
    """
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

    dc = data_config_override if data_config_override is not None else raw.get("data_config")
    return InferenceContext(
        data_config=dc,
        encoder=raw.get("encoder"),
        encoder_results=raw.get("encoder_results"),
        graph_builder=raw.get("graph_builder"),
        graph_results=raw.get("graph_results"),
        elementary_pnl=raw.get("elementary_pnl"),
        elementary_scaler=raw.get("elementary_scaler"),
        elementary_attributes=raw.get("elementary_attribs"),
        target_scaler=raw.get("target_scaler"),
        target_attributes=raw.get("target_attribs"),
        trade_universe=raw.get("trade_universe"),
        cluster_info=raw.get("cluster_info"),
        cluster_assets=raw.get("cluster_assets"),
        cluster_rf_keys=raw.get("cluster_rf_keys"),
        _cluster_assets_path=raw.get("_cluster_assets_path"),
        cluster_elem_trades=raw.get("cluster_elem_trades"),
    )


# ------------------------------------------------------------------
# Stage reports
# ------------------------------------------------------------------
#
# The pipeline returns one frozen dataclass per stage, used by both
# programmatic callers and the UI activity log.  Heavy artefacts
# (shock DataFrames, ensemble model, predictions array) stay inside
# the pipeline; the reports below are intentionally small and JSON-
# serialisable so they can live in a ``dcc.Store`` or HTTP response.
#

@dataclass(frozen=True)
class LoadedScenariosReport:
    """Summary of what :meth:`EnsembleInferencePipeline.load_scenarios`
    parsed from the new-scenario directory.

    Attributes
    ----------
    new_scenario_dir
        Absolute or relative path to the folder that was parsed.
    risk_factor_names
        Risk-factor stems (filename minus ``.csv``).  Order is the
        order ``os.listdir`` returned them — not significant.
    n_risk_factors
        Convenience: ``len(risk_factor_names)``.  Stored for cheap
        UI consumption.
    n_scenarios
        Number of rows in every shock CSV.  Validated equal across
        files at parse time.
    scenario_labels
        The canonical scenario index, shared across all shock files
        in this run.  Used by :meth:`load_scenarios` /
        :meth:`validate_scenarios` to look up rows in each cluster's
        ``elementary_pnl`` history.

    Notes
    -----
    The actual shock dictionaries — i.e. the heavy
    ``Dict[rf_name, Dict[scenario_label, Dict[knot, value]]]``
    parsed from disk — live on the pipeline as
    ``self._new_scenario_shocks`` and are *not* exposed via this
    report.  UI code consumes only the lightweight summary.
    """
    new_scenario_dir:    str
    risk_factor_names:   List[str]
    n_risk_factors:      int
    n_scenarios:         int
    scenario_labels:     List[str]

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view (defensive copies of list fields)."""
        return {
            "new_scenario_dir":  self.new_scenario_dir,
            "risk_factor_names": list(self.risk_factor_names),
            "n_risk_factors":    self.n_risk_factors,
            "n_scenarios":       self.n_scenarios,
            "scenario_labels":   list(self.scenario_labels),
        }


@dataclass(frozen=True)
class LoadedNewTradesReport:
    """Stub — populated when ``new_trades`` mode is implemented.

    Counterpart to :class:`LoadedScenariosReport` for the
    ``new_trades`` input mode.  Currently carries only the path
    that was loaded; additional summary fields land alongside the
    real :meth:`EnsembleInferencePipeline.load_new_trades` body.
    """
    new_trades_path: str

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view."""
        return {"new_trades_path": self.new_trades_path}


@dataclass(frozen=True)
class ClusterRoutingDecision:
    """Per-cluster decision made by :meth:`EnsembleInferencePipeline.validate`.

    Drives the per-cluster branch inside
    :meth:`EnsembleInferencePipeline.run_inference`: affected clusters
    take the full price-and-predict path
    (``_build_cluster_inputs``); unaffected clusters take the cheap
    historical-lookup path (``_load_cluster_inputs``).

    Attributes
    ----------
    cluster_id
        The cluster this decision is about.
    is_affected
        True iff the cluster shares at least one risk factor with the
        loaded scenario shocks.
    intersecting_risk_factors
        Sorted list of RF names that are in both the cluster's
        instrument universe and ``new_scenario_shocks``.  Empty when
        ``is_affected`` is False.
    n_elementary_trades
        Cluster's reduced elementary-trade count (post dimensionality
        reduction).  Surfaced here so the UI can render trade counts
        in the routing table without re-loading contexts.
    n_target_trades
        Cluster's target-trade count.  Same UI-convenience rationale.
    missing_scenario_labels
        Populated by ``validate_scenarios()`` *only* for unaffected
        clusters whose history doesn't cover all requested scenario
        labels.
        Affected clusters always have ``[]`` here (they take the full
        re-pricing path, so historical-label lookup never happens).
        Non-empty on an unaffected cluster ⇒ cheap path blocked ⇒
        :class:`ValidationReport` gets a corresponding entry in
        ``errors``, flipping ``is_valid`` to False.
    """
    cluster_id:                  str
    is_affected:                 bool
    intersecting_risk_factors:   List[str]
    n_elementary_trades:         int
    n_target_trades:             int
    missing_scenario_labels:     List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view (defensive copies of list fields)."""
        return {
            "cluster_id":                self.cluster_id,
            "is_affected":               self.is_affected,
            "intersecting_risk_factors": list(self.intersecting_risk_factors),
            "n_elementary_trades":       self.n_elementary_trades,
            "n_target_trades":           self.n_target_trades,
            "missing_scenario_labels":   list(self.missing_scenario_labels),
        }


@dataclass(frozen=True)
class ValidationReport:
    """Output of :meth:`EnsembleInferencePipeline.validate_scenarios`.

    Two kinds of issues are surfaced separately:

    * **errors** — *block the run*.  ``is_valid`` is False whenever
      this list is non-empty.  Typical errors: scenario labels not
      found in an unaffected cluster's history; no cluster intersects
      any shocked RF (nothing to run).
    * **warnings** — surfaced to the user but **do not** block the
      run.  e.g. a single cluster intersects shocks, shock magnitudes
      look small.

    The pipeline itself never raises on a validation issue.  Those
    are user-input problems that should be fixable by editing scenario
    files or selecting a different ensemble version; the caller
    (programmatic, backend, or UI) decides what to do with the report.

    Attributes
    ----------
    ensemble_version
        The resolved ensemble version this report was built against.
    n_scenarios
        Number of scenarios in the loaded shock files.
    scenario_labels
        Canonical scenario index used during validation.
    cluster_decisions
        One :class:`ClusterRoutingDecision` per cluster in the
        ensemble, in the order the ensemble's ``cluster_ids`` lists
        them.
    errors, warnings
        Validation issues — see notes above.

    Properties
    ----------
    is_valid
        ``len(errors) == 0``.
    affected_cluster_ids, unaffected_cluster_ids
        Derived from ``cluster_decisions``.  Single source of truth
        (these cannot drift from ``cluster_decisions``).
    affected_count, unaffected_count, cheap_path_used
        Convenience counters / boolean for UI display.
    """
    ensemble_version:    str
    n_scenarios:         int
    scenario_labels:     List[str]
    cluster_decisions:   List[ClusterRoutingDecision] = field(default_factory=list)
    errors:              List[str]                    = field(default_factory=list)
    warnings:            List[str]                    = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """True iff no blocking errors were recorded."""
        return len(self.errors) == 0

    @property
    def affected_cluster_ids(self) -> List[str]:
        """Cluster IDs that intersect with at least one shocked RF."""
        return [d.cluster_id for d in self.cluster_decisions if d.is_affected]

    @property
    def unaffected_cluster_ids(self) -> List[str]:
        """Cluster IDs with no shocked RF — eligible for the cheap path."""
        return [d.cluster_id for d in self.cluster_decisions if not d.is_affected]

    @property
    def affected_count(self) -> int:
        return len(self.affected_cluster_ids)

    @property
    def unaffected_count(self) -> int:
        return len(self.unaffected_cluster_ids)

    @property
    def cheap_path_used(self) -> bool:
        """True iff any cluster will take the cheap (historical-lookup) path."""
        return self.unaffected_count > 0

    def decision_for(
        self, cluster_id: str,
    ) -> Optional[ClusterRoutingDecision]:
        """Look up the routing decision for one cluster (linear scan).

        Returns ``None`` if the cluster ID isn't in the report — which
        should never happen for an ensemble member the pipeline knows
        about, but the method is defensive.
        """
        for d in self.cluster_decisions:
            if d.cluster_id == cluster_id:
                return d
        return None

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view, including derived properties.

        The derived fields (``is_valid``, ``affected_cluster_ids``
        etc.) are included so UI code can render the report without
        having to re-derive them.
        """
        return {
            "ensemble_version":       self.ensemble_version,
            "n_scenarios":            self.n_scenarios,
            "scenario_labels":        list(self.scenario_labels),
            "cluster_decisions":      [d.to_dict() for d in self.cluster_decisions],
            "errors":                 list(self.errors),
            "warnings":               list(self.warnings),
            "is_valid":               self.is_valid,
            "affected_cluster_ids":   self.affected_cluster_ids,
            "unaffected_cluster_ids": self.unaffected_cluster_ids,
            "affected_count":         self.affected_count,
            "unaffected_count":       self.unaffected_count,
            "cheap_path_used":        self.cheap_path_used,
        }


# ------------------------------------------------------------------
# Inference artifact on-disk layout (Stage 6)
#
# Mirrors the eval pipeline's convention of one subdirectory per
# artifact family.  Single source of truth for the names —
# `services.result_reader` re-imports these so writer and reader
# can never drift again.
# ------------------------------------------------------------------

INFERENCE_DIRNAME:        str = "inference"
CLUSTER_SUMMARY_DIRNAME:  str = "cluster_summary"
PORTFOLIO_SUMMARY_DIRNAME:str = "portfolio_summary"
TRADE_PREDICTIONS_DIRNAME:str = "trade_predictions"

CLUSTER_SUMMARY_FILENAME:   str = "cluster_predictions.parquet"
PORTFOLIO_SUMMARY_FILENAME: str = "portfolio_predictions.parquet"
MANIFEST_FILENAME:          str = "manifest.json"


# ------------------------------------------------------------------
# Per-cluster artifact (Stage 6)
# ------------------------------------------------------------------

@dataclass
class ClusterArtifact:
    """Lightweight handle on one cluster's persisted predictions.

    Built inside :meth:`EnsembleInferencePipeline._post_infer_cluster`
    right after the two per-cluster parquets are written, and
    accumulated through :meth:`run_inference` for downstream
    aggregation in :meth:`post_infer`.

    ``summary_df`` is the per-scenario aggregate frame returned by
    :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
    (columns: ``scenario_label``, ``cluster_id``, ``sum_pnl_scaled``,
    ``sum_pnl_original``, ``mean_pnl_original``, ``std_pnl_original``,
    ``min_pnl_original``, ``max_pnl_original``).  Kept in-memory only
    long enough for ``post_infer`` to stack across clusters and write
    the run-level cluster + portfolio summaries; the heavy wide
    frames (``scaled_wide`` / ``original_wide``) have already been
    flushed to disk by the time this dataclass exists.

    ``eq=False`` so the dataclass doesn't try to compare its
    embedded DataFrame (pandas refuses ``==`` on heterogeneous
    frames); this dataclass is identity-compared in practice.

    Attributes
    ----------
    cluster_id
        Cluster this artifact belongs to.
    n_trades
        Number of target trades in the cluster (== width of both
        wide parquets).
    n_scenarios
        Number of scenario rows scored for this cluster.
    scaled_path, original_path
        Absolute filesystem paths to the two wide parquets the API
        result-reader will serve.
    trade_ids
        Canonical column ordering of both wide parquets (taken from
        ``target_scaler.feature_names_in_``).  Surfaced in the
        manifest so the UI can list cluster contents without
        opening the parquet.
    summary_df
        Per-scenario summary aggregates for this cluster.
        ``repr=False`` because pandas DataFrames make logs unreadable.
    """

    cluster_id:    str
    n_trades:      int
    n_scenarios:   int
    scaled_path:   str
    original_path: str
    trade_ids:     List[str]                  = field(default_factory=list)
    summary_df:    Optional[pd.DataFrame]     = field(default=None, repr=False, compare=False)

    def to_manifest_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view written into ``manifest.json``.

        Deliberately excludes ``summary_df`` (heavy + redundant — the
        same data is rolled up into the run-level cluster summary
        parquet).
        """
        return {
            "cluster_id":    self.cluster_id,
            "n_trades":      self.n_trades,
            "n_scenarios":   self.n_scenarios,
            "scaled_path":   self.scaled_path,
            "original_path": self.original_path,
            "trade_ids":     list(self.trade_ids),
        }


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class EnsembleInferencePipeline:
    """
    Run inference through an ensemble of models.

    Supports two input modes (set in config.metadata["inference"]["input_mode"]):
    - ``new_scenarios``: Same trades, new risk-factor scenario data.
    - ``new_trades``: New trade attributes to route and predict (not yet implemented).

    Parameters
    ----------
    ensemble_config : EnsembleConfig
        Must have ``registry_dir`` set (ignored when *session* is provided).
    ensemble_version : str
        Ensemble version or tag to load.
    session : EnsembleSession or None
        If provided, skip registry loading and use the session's cached
        models + inference contexts.  The session must have Phase 3 loaded.
    """

    def __init__(
        self,
        ensemble_config: EnsembleConfig,
        ensemble_version: str = "latest",
        session: Optional["EnsembleSession"] = None,
        *,
        on_event: Optional[EmitFn] = None,
    ) -> None:
        """Initialise the pipeline.

        Parameters
        ----------
        ensemble_config : EnsembleConfig
            Pipeline configuration.  Must have ``registry_dir`` set
            when ``session`` is ``None``.
            ``metadata['inference']['input_mode']`` selects
            ``'new_scenarios'`` (default) or ``'new_trades'``.
        ensemble_version : str
            Ensemble version or tag to load.  When a ``session`` is
            provided this must match ``session.ensemble_version`` —
            see :meth:`load` for the safety check.
        session : EnsembleSession, optional
            Pre-loaded session (UI / backend path).  When provided,
            :meth:`load` reuses the session's models + contexts
            instead of reading the registry.  When ``None``,
            :meth:`load` cold-loads from ``ensemble_config.registry_dir``.
        on_event : EmitFn, optional
            Lifecycle-event callback.  ``None`` (default) silences
            events — the pipeline runs exactly as it did before the
            hook existed.  Pass an :class:`EventCollector` to stream
            a narration into the UI activity log.
        """
        # Constructor inputs — never reassigned after this point.
        self.config           = ensemble_config
        self.ensemble_version = ensemble_version
        self._session         = session
        self._emit: EmitFn    = on_event if on_event is not None else noop_emit

        # Model layer — populated by load().
        self._ensemble                                       = None
        self._ens_config:         Optional[EnsembleConfig]   = None
        self._member_versions:    Optional[Dict[str, str]]   = None
        self._inference_contexts: Dict[str, Any]             = {}

        # new_scenarios input — populated by load_scenarios().
        self._new_scenario_shocks: Optional[Dict[str, Any]]        = None
        self._loaded_scenarios:    Optional[LoadedScenariosReport] = None

        # new_trades input — populated by load_new_trades() (stub for now;
        # LoadedNewTradesReport dataclass lands with the real body).
        self._new_trades_payload:  Optional[Dict[str, Any]]            = None
        self._loaded_new_trades:   Optional["LoadedNewTradesReport"]   = None

        # Validation — written by validate_scenarios() / validate_new_trades(),
        # read by run_inference() and _build_result().
        self._validation_report:   Optional[ValidationReport]          = None

    # ==================================================================
    # Orchestration
    # ==================================================================

    def run(self) -> InferenceResult:
        """Execute the full staged inference pipeline.

        Convenience wrapper that calls the four staged methods in
        order.  The middle pair is mode-specific — the input mode is
        read from ``config.metadata['inference']['input_mode']``::

            load()
            ├─ new_scenarios → load_scenarios()  → validate_scenarios()
            └─ new_trades    → load_new_trades() → validate_new_trades()
            run_inference()

        Validation failures (user-input problems surfaced in
        ``report.errors``) raise :class:`ValueError`.  System
        failures propagate the underlying exception.

        Lifecycle events
        ----------------
        Emits the outer ``Pipeline started`` / ``Pipeline complete`` /
        ``Pipeline failed`` envelope here; each inner stage emits its
        own start/OK/fail events.

        Returns
        -------
        InferenceResult
        """
        logger.info("EnsembleInferencePipeline: starting")
        self._emit(event(
            STAGE_INFERENCE, "Pipeline started",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))
        t0 = time.perf_counter()

        try:
            input_mode = (
                self.config.metadata
                    .get("inference", {})
                    .get("input_mode", "new_scenarios")
            )
            if input_mode not in {"new_scenarios", "new_trades"}:
                raise ValueError(
                    f"Unknown input_mode '{input_mode}'. "
                    f"Supported: 'new_scenarios', 'new_trades'."
                )

            self.load()

            if input_mode == "new_scenarios":
                self.load_scenarios()
                report = self.validate_scenarios()
            else:  # "new_trades" — validated above
                self.load_new_trades()
                report = self.validate_new_trades()

            if not report.is_valid:
                raise ValueError(
                    f"Inference validation failed with "
                    f"{len(report.errors)} error(s): {report.errors}"
                )

            result = self.run_inference()
        except Exception as exc:
            self._emit(event(
                STAGE_INFERENCE, "Pipeline failed",
                status=STATUS_FAIL,
                target=type(exc).__name__, detail=str(exc),
            ))
            raise

        wall = time.perf_counter() - t0
        logger.info(
            "EnsembleInferencePipeline: done (%.3fs, %d samples)",
            wall, result.n_samples,
        )
        self._emit(event(
            STAGE_INFERENCE, "Pipeline complete",
            status=STATUS_OK,
            target=f"{wall * 1000:.0f} ms · {result.n_samples} samples",
        ))
        return result

    def load(self) -> None:
        """Load the ensemble model + per-cluster inference contexts.

        Idempotent — returns immediately if already loaded.  Source
        selection is decided at ``__init__`` time, not here:

        * **Session given** → reuse the pre-loaded models + contexts
          (warm path).  The session must match the requested
          ``ensemble_version`` and have finished loading.
        * **No session** → cold-load from the registry at
          ``config.registry_dir``.

        There is no silent fallback: a mismatched or not-ready
        session raises rather than falling back to the registry.

        Raises
        ------
        RuntimeError
            If a session was provided whose version mismatches, or
            whose inference contexts have not finished loading.

        UI usage
        --------
        Bound to the "Load model" button on the *Inference Console*.
        Failures emit a ``status="fail"`` event from the underlying
        loader before re-raising.
        """
        if self._ensemble is not None:
            return  # idempotent — already loaded

        if self._session is None:
            self._load_from_registry()
            return

        if self._session.ensemble_version != self.ensemble_version:
            raise RuntimeError(
                f"Session is loaded for ensemble version "
                f"'{self._session.ensemble_version}' but the pipeline was "
                f"constructed for '{self.ensemble_version}'. Construct "
                f"the pipeline with the matching session, or pass "
                f"session=None."
            )
        if not self._session.all_inference_ready:
            raise RuntimeError(
                f"Session for '{self._session.ensemble_version}' has "
                f"not finished loading per-cluster inference contexts "
                f"(all_inference_ready=False). Wait for the session to "
                f"finish loading before constructing the pipeline."
            )

        self._load_from_session()

    def load_scenarios(
        self,
        new_scenario_dir: Optional[Union[str, Path]] = None,
    ) -> LoadedScenariosReport:
        """Parse new-scenario shock CSVs into the pipeline state.

        Three steps:

          1. Resolve the scenario directory (argument or config fallback).
          2. Parse every ``<risk_factor>.csv`` under that directory.
          3. Cross-check that all shock files share the same scenario
             label index, and extract the canonical list.

        Returns a small JSON-friendly :class:`LoadedScenariosReport`
        summarising what was parsed; the heavy shock dict itself
        stays in ``self._new_scenario_shocks`` for
        :meth:`validate_scenarios` and the input builder to consume.

        Parameters
        ----------
        new_scenario_dir
            Folder containing one CSV per shocked risk factor
            (filename minus ``.csv`` is the RF key).  Falls back to
            ``self.config.metadata['inference']['new_scenario_dir']``
            if ``None``.

        Raises
        ------
        RuntimeError
            If :meth:`load` hasn't been called yet.
        ValueError
            If no scenario directory is provided, or shock files
            disagree on scenario labels / counts.

        Side effects
        ------------
        Invalidates any prior :meth:`validate_scenarios` result by
        resetting ``self._validation_report`` to ``None``.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before load_scenarios()."
            )

        # --- Step 1: resolve the directory ---
        if new_scenario_dir is None:
            new_scenario_dir = (
                self.config.metadata
                    .get("inference", {})
                    .get("new_scenario_dir")
            )
        if not new_scenario_dir:
            raise ValueError(
                "new_scenario_dir is required (pass as argument or set "
                "config.metadata['inference']['new_scenario_dir'])."
            )
        new_scenario_dir = str(new_scenario_dir)

        # --- Step 2 + 3: parse + cross-check ---
        self._emit(event(
            STAGE_INFERENCE, "Loading new-scenario shocks",
            status=STATUS_RUNNING, target=new_scenario_dir,
        ))

        try:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                HybridGnnRnnInferencePipeline,
            )
            shocks    = HybridGnnRnnInferencePipeline.load_new_scenarios(new_scenario_dir)
            canonical = self._extract_canonical_index(shocks)
        except Exception as exc:
            self._emit(event(
                STAGE_INFERENCE, "Scenario load failed",
                status=STATUS_FAIL, target=new_scenario_dir, detail=str(exc),
            ))
            raise

        # --- Commit to instance state + build the report ---
        self._new_scenario_shocks = shocks
        self._validation_report   = None  # any prior validation is now stale

        rf_names = sorted(shocks.keys())
        report = LoadedScenariosReport(
            new_scenario_dir  = new_scenario_dir,
            risk_factor_names = rf_names,
            n_risk_factors    = len(rf_names),
            n_scenarios       = len(canonical),
            scenario_labels   = canonical,
        )
        self._loaded_scenarios = report

        self._emit(event(
            STAGE_INFERENCE, "Shocks loaded",
            status=STATUS_OK, target=new_scenario_dir,
            detail=f"{report.n_risk_factors} RFs · {report.n_scenarios} scenarios",
        ))
        return report

    def load_new_trades(
        self,
        new_trades_path: Optional[Union[str, Path]] = None,
    ) -> "LoadedNewTradesReport":
        """Parse a new-trades payload into the pipeline state.

        Counterpart to :meth:`load_scenarios` for the ``new_trades``
        input mode.  Will populate ``self._new_trades_payload`` and
        ``self._loaded_new_trades`` (analogous to the new-scenarios
        slots) so :meth:`validate_new_trades` and :meth:`run_inference`
        can pick them up.

        Raises
        ------
        RuntimeError
            If :meth:`load` hasn't been called yet.
        NotImplementedError
            Always — the per-cluster trade routing helper on
            :class:`HybridGnnRnnInferencePipeline` does not exist
            yet.  Extension point: drop the parsing + routing body
            in here once that helper is added.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before load_new_trades()."
            )
        raise NotImplementedError(
            "new_trades input mode is not yet implemented. The "
            "staged pipeline scaffold exists; the missing piece is "
            "per-cluster trade routing — see "
            "HybridGnnRnnInferencePipeline."
        )

    def validate_scenarios(self) -> ValidationReport:
        """Classify each cluster and check cheap-path eligibility.

        For each ensemble member, decides whether the cluster is
        *affected* by the loaded shocks (cluster RFs ∩ shock RFs ≠ ∅).
        For unaffected clusters, additionally checks that the cluster's
        historical ``elementary_pnl`` index contains every requested
        scenario label (cheap-path eligibility).

        Returns a :class:`ValidationReport` with one
        :class:`ClusterRoutingDecision` per ensemble member.  Errors
        and warnings live in ``report.errors`` / ``report.warnings``;
        ``report.is_valid`` is True iff no errors were found.

        Validation issues are treated as user-input problems and
        **never raise** here — the caller (``run()`` or the UI)
        decides what to do with the report.

        Raises
        ------
        RuntimeError
            If :meth:`load` or :meth:`load_scenarios` haven't run.

        UI usage
        --------
        Bound to the "Validate" button on the *Inference Console*.
        The routing-table card renders ``report.cluster_decisions``;
        the alert banner renders ``report.errors`` / ``report.warnings``.
        The Run button is enabled only when ``report.is_valid`` is True.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before validate_scenarios()."
            )
        if self._loaded_scenarios is None:
            raise RuntimeError(
                "load_scenarios() must be called before validate_scenarios()."
            )

        self._emit(event(
            STAGE_INFERENCE, "Validating scenarios",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))

        scenario_labels = self._loaded_scenarios.scenario_labels
        shock_rfs       = sorted(self._new_scenario_shocks.keys())

        decisions: List[ClusterRoutingDecision] = []
        errors:    List[str]                    = []
        warnings:  List[str]                    = []

        # --- Per-cluster classification + eligibility check ---
        for cid in self._ens_config.cluster_ids:
            ctx = self._inference_contexts.get(cid)
            if ctx is None:
                errors.append(f"Cluster '{cid}': no inference context loaded.")
                continue

            decision = self._build_routing_decision(
                cluster_id         = cid,
                ctx                = ctx,
                shock_risk_factors = shock_rfs,
                scenario_labels    = scenario_labels,
            )

            # Surface cheap-path eligibility failures into the report
            # so the UI can render them in the alert banner.
            if not decision.is_affected and decision.missing_scenario_labels:
                missing = decision.missing_scenario_labels
                errors.append(
                    f"Cluster '{cid}' is unaffected but its history "
                    f"is missing {len(missing)}/{len(scenario_labels)} "
                    f"requested scenario labels (first 3: {missing[:3]})."
                )

            decisions.append(decision)

        # --- Cross-cluster sanity check ---
        # Treated as a WARNING (not an error) — when no cluster
        # intersects any shocked risk factor, every cluster takes the
        # cheap historical-lookup path.  The run is still meaningful
        # (the dashboard is comparing the new scenarios against the
        # cluster's historical PnL distribution), it just won't show
        # any *new* re-pricing.  Surfacing this as a hard error would
        # block the cheap-path use case entirely.
        if decisions and not any(d.is_affected for d in decisions):
            warnings.append(
                "No clusters intersect any shocked risk factor — every "
                "cluster will take the cheap historical-lookup path."
            )

        # --- Build and cache the report ---
        report = ValidationReport(
            ensemble_version  = self.ensemble_version,
            n_scenarios       = self._loaded_scenarios.n_scenarios,
            scenario_labels   = scenario_labels,
            cluster_decisions = decisions,
            errors            = errors,
            warnings          = warnings,
        )
        self._validation_report = report

        self._emit(event(
            STAGE_INFERENCE, "Validation complete",
            status=STATUS_OK if report.is_valid else STATUS_FAIL,
            target=f"{report.affected_count} affected · {report.unaffected_count} unaffected",
            detail=f"{len(report.errors)} error(s)" if report.errors else None,
        ))
        return report

    def validate_new_trades(self) -> ValidationReport:
        """Validate the loaded new-trades payload.

        Counterpart to :meth:`validate_scenarios` for the
        ``new_trades`` input mode.  Will populate
        ``self._validation_report`` with one routing decision per
        cluster — likely a different dataclass shape (a future
        ``TradeRoutingDecision``) sitting alongside
        :class:`ClusterRoutingDecision` in the union typing of
        ``ValidationReport.cluster_decisions``.

        Raises
        ------
        RuntimeError
            If :meth:`load` or :meth:`load_new_trades` haven't run.
        NotImplementedError
            Always — pending the matching trade-routing logic.
        """
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before validate_new_trades()."
            )
        if self._loaded_new_trades is None:
            raise RuntimeError(
                "load_new_trades() must be called before validate_new_trades()."
            )
        raise NotImplementedError(
            "new_trades validation is not yet implemented. The "
            "plug-in point is here — populate self._validation_report "
            "with one ClusterRoutingDecision (or TradeRoutingDecision) "
            "per cluster and return it."
        )

    def run_inference(self) -> InferenceResult:
        """Build inputs (per-cluster routed), predict, post-process.

        Final stage of the staged pipeline.  Mode-agnostic: dispatches
        on ``config.metadata['inference']['input_mode']`` via
        :meth:`_build_member_inputs`.

        Five steps:

          1. Prerequisite checks  (load / load_* / validate_* completed).
          2. Build member inputs  (per-mode, per-cluster routing inside).
          3. Forward pass         (``EnsembleModel.predict``).
          4. Build the result     (:meth:`_build_result`).
          5. Post-inference       (logging + optional artifact writes).

        Raises
        ------
        RuntimeError
            If any prior stage is missing, or the cached validation
            report contains errors (``report.is_valid`` is False).

        Returns
        -------
        InferenceResult
        """
        # --- Step 1: prerequisite checks ---
        if self._ensemble is None:
            raise RuntimeError(
                "load() must be called before run_inference()."
            )
        if self._loaded_scenarios is None and self._loaded_new_trades is None:
            raise RuntimeError(
                "load_scenarios() or load_new_trades() must be called "
                "before run_inference()."
            )
        if self._validation_report is None:
            raise RuntimeError(
                "validate_scenarios() (or validate_new_trades()) "
                "must be called before run_inference()."
            )
        if not self._validation_report.is_valid:
            first = (
                self._validation_report.errors[0]
                if self._validation_report.errors else "?"
            )
            raise RuntimeError(
                f"Cannot run inference — validation reported "
                f"{len(self._validation_report.errors)} error(s). "
                f"First: {first}"
            )

        # --- Step 2: build member inputs (per-mode dispatch) ---
        t0         = time.perf_counter()
        infer_meta = self.config.metadata.get("inference", {})
        input_mode = infer_meta.get("input_mode", "new_scenarios")
        batch_size = int(infer_meta.get("batch_size", 128))
        member_inputs, extra_meta = self._build_member_inputs(input_mode)

        # --- Step 3: forward pass — predict per cluster, write parquets ---
        #
        # We iterate cluster-by-cluster (rather than the legacy
        # ``self._ensemble.predict(member_inputs)`` one-shot call) so we
        # can:
        #
        #   * use ``predict_member_chunked`` to bound activation peak
        #     memory per cluster,
        #   * call ``_post_infer_cluster`` immediately after each
        #     cluster's predictions land — flushing wide parquets to
        #     disk so the in-memory footprint stays bounded across
        #     long runs, and
        #   * free ``member_inputs[cid]`` as soon as its predictions
        #     are computed (deep-copied risk_factor_shocks are the
        #     biggest contributor and never need to outlive the
        #     forward pass).
        #
        # ``_ensemble._combine`` is called once at the end with the
        # full per-cluster prediction dict — same aggregation contract
        # as ``predict``, just sourced one cluster at a time.
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        artifacts_root: Optional[Path] = None
        if self.config.artifacts_dir:
            artifacts_root = Path(self.config.artifacts_dir) / INFERENCE_DIRNAME
            artifacts_root.mkdir(parents=True, exist_ok=True)

        scenario_labels   = list(self._validation_report.scenario_labels)
        cluster_artifacts: List[ClusterArtifact]    = []
        member_preds:      Dict[str, np.ndarray]    = {}

        self._emit(event(
            STAGE_INFERENCE, "Forward pass started",
            status=STATUS_RUNNING,
            target=(
                f"{self._validation_report.affected_count} affected · "
                f"{self._validation_report.unaffected_count} unaffected"
            ),
        ))

        for cid in self._ensemble.router.cluster_ids:
            if cid not in self._ensemble.members:
                logger.warning(
                    "run_inference: no member model for cluster %s; "
                    "skipping (router/ensemble drift).", cid,
                )
                continue
            cluster_inputs = member_inputs.get(cid)
            if cluster_inputs is None:
                continue

            # Per-cluster memory-bounded forward pass.  Returns
            # predictions in the model's SCALED output space —
            # transform back to original units inside
            # _post_infer_cluster via transform_predictions.
            cluster_preds = HybridGnnRnnInferencePipeline.predict_member_chunked(
                ensemble       = self._ensemble,
                cluster_id     = cid,
                cluster_inputs = cluster_inputs,
                batch_size     = batch_size,
            )
            member_preds[cid] = cluster_preds

            # Stage 6: persist scaled + original parquets and capture
            # a lightweight artifact handle for downstream aggregation.
            # Only written when artifacts_dir is configured — pure
            # in-memory runs still produce an InferenceResult, just
            # without per-cluster parquets.
            if artifacts_root is not None:
                ctx = self._inference_contexts.get(cid)
                if ctx is None:
                    logger.warning(
                        "run_inference: no inference context for cluster %s; "
                        "skipping per-cluster artifact write.", cid,
                    )
                else:
                    cluster_artifacts.append(
                        self._post_infer_cluster(
                            cluster_id      = cid,
                            cluster_preds   = cluster_preds,
                            scenario_labels = scenario_labels,
                            ctx             = ctx,
                            out_dir         = artifacts_root,
                        )
                    )

            # Release the per-cluster input dict as soon as we no
            # longer need it.  Affected clusters carry deep-copied
            # risk_factor_shocks which are by far the largest object
            # in the member input dict.
            member_inputs[cid] = None

        if not member_preds:
            raise RuntimeError(
                "run_inference: no member produced predictions — "
                "ensemble may be empty or all clusters were skipped."
            )

        # Single aggregation call across all per-cluster predictions —
        # identical contract to EnsembleModel.predict, just with the
        # iteration lifted into this pipeline so we can interleave
        # per-cluster post-processing above.
        combined = self._ensemble._combine(member_preds)   # noqa: SLF001 — internal-but-stable

        self._emit(event(
            STAGE_INFERENCE, "Forward pass complete",
            status=STATUS_OK, target=f"{combined.shape[0]} scenarios",
        ))

        # --- Step 4: build result ---
        result = self._build_result(combined, infer_meta, extra_meta)
        result.latency_seconds = time.perf_counter() - t0

        # --- Step 5: post-inference (manifest + run-level summaries) ---
        self.post_infer(result, cluster_artifacts=cluster_artifacts)
        return result

    # ==================================================================
    # Loading
    # ==================================================================

    def _load_from_registry(self) -> None:
        """Cold-load: read ensemble + per-cluster contexts from registry.

        Three layers, one event per layer:

          1. Ensemble config + member versions (``EnsembleRegistry``).
          2. Member models, assembled into one ``EnsembleModel``
             (``EnsembleBuilder`` + ``ModelRegistry``).
          3. Per-cluster inference contexts
             (``load_inference_context_from_dir``).

        Step 3 reads the same on-disk artifacts that
        ``hybrid_gnn_rnn/infer.py`` consumes for single-cluster
        inference; the encoder / graph-builder dispatch happens
        inside ``_dict_to_inference_context``.
        """
        # --- Step 1: ensemble config + member versions ---
        self._emit(event(
            STAGE_INFERENCE, "Loading ensemble from registry",
            status=STATUS_RUNNING, target=self.ensemble_version,
        ))
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, version = ens_registry.load(self.ensemble_version)
        self._ens_config      = config
        self._member_versions = member_versions

        # --- Step 2: assemble the ensemble model ---
        from src.rade_ml_pt.registry.store import ModelRegistry

        model_registry = ModelRegistry(self.config.registry_dir)
        builder        = EnsembleBuilder(model_registry)
        self._ensemble = builder.build(config, member_versions)
        self._emit(event(
            STAGE_INFERENCE, "Ensemble assembled",
            status=STATUS_OK, target=f"{config.n_members} members",
        ))

        # --- Step 3: per-cluster inference contexts (parallel) ---
        #
        # The per-cluster ``InferenceContext`` is by far the slowest
        # part of cold-load — each context reads encoder / graph-
        # builder pickles plus, lazily, a 10-30 MB cluster_assets
        # joblib.  These loads are IO-bound (and pickle.load releases
        # the GIL during the underlying C read) so a ThreadPoolExecutor
        # is the right primitive: modest concurrency gives a ~2-3×
        # wall-clock reduction without the import / serialisation
        # overhead of multiprocessing.
        #
        # Worker count resolution order:
        #   1. ``config.metadata['inference']['loader_max_workers']``
        #      — explicit, per-run override (UI / config knob).
        #   2. ``config.max_workers`` — existing EnsembleConfig field,
        #      reused here for backward-compat with callers that set
        #      it for the execution_strategy knob.
        #   3. ``min(8, n_clusters)`` — sane default for IO-bound work
        #      that never overshoots when the ensemble is small.
        #
        # Failure model: we collect every failure inside the pool and
        # raise *once* after the executor shuts down.  Avoids the
        # "first exception cancels remaining futures" footgun and
        # gives the UI a complete list of broken clusters in one go.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            load_inference_context_from_dir,
        )

        loader_workers = (
            self.config.metadata.get("inference", {}).get("loader_max_workers")
            or self.config.max_workers
            or min(8, max(1, len(config.cluster_ids)))
        )
        loader_workers = max(1, int(loader_workers))

        logger.info(
            "Loading %d inference contexts with %d worker(s)",
            len(config.cluster_ids), loader_workers,
        )

        contexts: Dict[str, Any]           = {}
        failures: List[Tuple[str, str, Exception]] = []

        with ThreadPoolExecutor(
            max_workers = loader_workers,
            thread_name_prefix = "infer-ctx-load",
        ) as executor:
            # Submit every cluster up front so the pool can start
            # draining immediately.  The future ↦ (cid, ver, dir)
            # map gives ``as_completed`` enough info to attribute
            # errors and emit per-cluster events without inspecting
            # the closure.
            futures: Dict[Any, Tuple[str, str, Path]] = {}
            for cid in config.cluster_ids:
                ver         = member_versions[cid]
                version_dir = Path(self.config.registry_dir) / ver
                self._emit(event(
                    STAGE_INFERENCE, "Cluster context loading",
                    status=STATUS_RUNNING, target=cid,
                ))
                future = executor.submit(
                    load_inference_context_from_dir, version_dir,
                )
                futures[future] = (cid, ver, version_dir)

            # Drain in completion order — fastest clusters surface
            # their "loaded" events first, which the UI's activity
            # log streams immediately.  Conversion through
            # ``_dict_to_inference_context`` happens on the main
            # thread (cheap; keeps the raw-dict shape thread-local).
            for future in as_completed(futures):
                cid, ver, version_dir = futures[future]
                try:
                    raw = future.result()
                except Exception as exc:
                    self._emit(event(
                        STAGE_INFERENCE, "Cluster context failed",
                        status=STATUS_FAIL, target=cid, detail=str(exc),
                    ))
                    failures.append((cid, str(version_dir), exc))
                    continue

                contexts[cid] = _dict_to_inference_context(raw)
                self._emit(event(
                    STAGE_INFERENCE, "Cluster context loaded",
                    status=STATUS_OK, target=cid,
                ))
                logger.info(
                    "Loaded inference context for cluster '%s' (version '%s')",
                    cid, ver,
                )

        # Re-raise as a single aggregated ValueError so the UI can
        # render every broken cluster in one alert rather than
        # cascading through retries.
        if failures:
            first_cid, first_dir, first_exc = failures[0]
            broken = ", ".join(f"'{c}'" for c, _, _ in failures)
            raise ValueError(
                f"Could not load inference context for {len(failures)} "
                f"cluster(s): {broken}. First failure — cluster "
                f"'{first_cid}' (dir={first_dir}): {first_exc}"
            ) from first_exc

        self._inference_contexts = contexts

        logger.info(
            "Loaded ensemble '%s' (%d members) from registry",
            version, config.n_members,
        )

    def _load_from_session(self) -> None:
        """Warm-load: reuse the session's pre-loaded models + contexts.

        Called by :meth:`load` only after the session has been
        validated (matching version, fully loaded).  Just copies
        references — no IO, no model construction.
        """
        self._emit(event(
            STAGE_INFERENCE, "Reusing pre-loaded session",
            status=STATUS_RUNNING, target=self._session.ensemble_version,
        ))

        # Copy refs from the session (no IO).
        self._ens_config      = self._session.config
        self._member_versions = self._session.member_versions
        self._ensemble        = self._session.ensemble_model

        # Per-cluster inference contexts can live on the session as
        # either already-typed InferenceContext objects or the raw
        # dicts returned by load_inference_context_from_dir (legacy
        # sessions).  Normalise to InferenceContext here so downstream
        # code only deals with one shape.
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

        for cid in self._ens_config.cluster_ids:
            state = self._session._inference[cid]
            ctx   = state.inference_context
            if isinstance(ctx, InferenceContext):
                self._inference_contexts[cid] = ctx
            else:
                self._inference_contexts[cid] = _dict_to_inference_context(
                    ctx, data_config_override=state.data_config,
                )

        logger.info(
            "Using pre-loaded session (ensemble '%s', %d members)",
            self._session.ensemble_version, self._ens_config.n_members,
        )
        self._emit(event(
            STAGE_INFERENCE, "Session ready",
            status=STATUS_OK,
            target=f"{self._ens_config.n_members} members",
        ))

    # ==================================================================
    # Validation helpers
    # ==================================================================
    #
    # All private to the pipeline.  Operate on a single cluster's
    # ``InferenceContext`` (or — for ``_extract_canonical_index`` — the
    # already-parsed shock dict) and return primitive types or the
    # frozen ``ClusterRoutingDecision`` dataclass.  No IO; no model
    # forward pass; cheap to call.
    #

    def _extract_canonical_index(
        self,
        new_scenario_shocks: Dict[str, Dict[Any, Any]],
    ) -> List[str]:
        """Cross-shock-file: extract + validate the canonical scenario index.

        Every shock CSV must share the same scenario labels in the
        same order.  Returns the canonical list.

        Raises
        ------
        ValueError
            If no shocks were provided, or files disagree on labels
            or counts.  Errors include up to 3 mismatch examples so
            the user can fix their input files.
        """
        if not new_scenario_shocks:
            raise ValueError(
                "No shock files were loaded — empty new_scenario_dir?"
            )

        canonical:    Optional[List[str]] = None
        canonical_rf: Optional[str]       = None

        for rf, scenarios in new_scenario_shocks.items():
            labels = [str(k) for k in scenarios.keys()]
            if canonical is None:
                canonical    = labels
                canonical_rf = rf
                continue
            if len(labels) != len(canonical):
                raise ValueError(
                    f"Shock files disagree on scenario count: "
                    f"'{canonical_rf}' has {len(canonical)} scenarios, "
                    f"'{rf}' has {len(labels)}."
                )
            if labels != canonical:
                mismatches = [
                    (i, a, b)
                    for i, (a, b) in enumerate(zip(canonical, labels))
                    if a != b
                ]
                raise ValueError(
                    f"Shock files disagree on scenario labels "
                    f"('{canonical_rf}' vs '{rf}'). First mismatches "
                    f"(index, canonical, observed): {mismatches[:3]}"
                )

        assert canonical is not None  # for type checker
        return canonical

    def _build_routing_decision(
        self,
        cluster_id: str,
        ctx: "InferenceContext",
        shock_risk_factors: Iterable[str],
        scenario_labels: List[str],
    ) -> ClusterRoutingDecision:
        """Assemble one cluster's routing decision from primitives.

        Calls into :class:`HybridGnnRnnInferencePipeline` for the
        cluster-level classification work (RF intersection, cheap-path
        eligibility), then wraps the result + trade counts in the
        ensemble-level :class:`ClusterRoutingDecision` dataclass.

        Trade counts are pulled from ``ctx.trade_universe`` so the
        returned decision is rich enough to drive the UI's routing-
        table card without any further context lookups.
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        intersecting = HybridGnnRnnInferencePipeline.intersecting_risk_factors(
            ctx, shock_risk_factors,
        )
        is_affected  = bool(intersecting)

        # Eligibility check only matters for unaffected clusters
        # (affected clusters take the re-pricing path, which doesn't
        # depend on the historical label index).
        missing: List[str] = []
        if not is_affected:
            missing = HybridGnnRnnInferencePipeline.missing_scenario_labels(
                ctx, scenario_labels,
            )

        n_elementary = 0
        n_target     = 0
        if ctx.trade_universe is not None:
            n_elementary = len(ctx.trade_universe.get("elementary_ids") or [])
            n_target     = len(ctx.trade_universe.get("target_ids")     or [])

        return ClusterRoutingDecision(
            cluster_id                = cluster_id,
            is_affected               = is_affected,
            intersecting_risk_factors = intersecting,
            n_elementary_trades       = n_elementary,
            n_target_trades           = n_target,
            missing_scenario_labels   = missing,
        )

    # ==================================================================
    # Input building
    # ==================================================================

    def _build_member_inputs(
        self,
        input_mode: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Mode dispatcher for input building.

        Called from :meth:`run_inference` after the matching
        ``validate_*`` method has populated ``self._validation_report``.

        Returns
        -------
        tuple of (member_inputs, extra_meta)
            *member_inputs*: ``{cluster_id: 7-key model input dict}``
            ready for ``EnsembleModel.predict``.
            *extra_meta*: auxiliary info (sample_ids per member).
        """
        if input_mode == "new_scenarios":
            return self._build_new_scenarios_inputs()

        if input_mode == "new_trades":
            raise NotImplementedError(
                "new_trades inference is not yet supported in the ensemble "
                "pipeline. The underlying HybridGnnRnnInferencePipeline does "
                "not implement _prepare_new_trade_inputs yet."
            )

        raise ValueError(f"Unknown input_mode: {input_mode}")

    def _build_new_scenarios_inputs(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Per-cluster routing loop for new-scenarios mode.

        Walks ``self._validation_report.cluster_decisions`` in order
        and delegates each cluster's input building to
        :meth:`HybridGnnRnnInferencePipeline.build_new_scenario_inputs`,
        which internally dispatches on ``decision.is_affected``:

        * **Affected** clusters take the full re-pricing path.
        * **Unaffected** clusters take the cheap historical-lookup path.

        Returns
        -------
        tuple of (member_inputs, extra_meta)
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        assert self._validation_report is not None, \
            "validate_scenarios() must run first"
        assert self._new_scenario_shocks is not None, \
            "load_scenarios() must run first"

        member_inputs: Dict[str, Any] = {}
        extra_meta:    Dict[str, Any] = {"sample_ids": {}}

        scenario_labels = self._validation_report.scenario_labels
        shocks          = self._new_scenario_shocks

        for decision in self._validation_report.cluster_decisions:
            cid  = decision.cluster_id
            ctx  = self._inference_contexts[cid]
            path = "affected" if decision.is_affected else "unaffected"

            self._emit(event(
                STAGE_INFERENCE, "Building cluster inputs",
                status=STATUS_RUNNING, target=cid,
                detail=f"path={path}",
            ))
            try:
                result = HybridGnnRnnInferencePipeline.build_new_scenario_inputs(
                    ctx                 = ctx,
                    new_scenario_shocks = shocks,
                    scenario_labels     = scenario_labels,
                    is_affected         = decision.is_affected,
                )

                member_inputs[cid] = result["inputs"]
                extra_meta["sample_ids"][cid] = result.get("sample_ids")

                n_windows = result["metadata"]["n_scenarios"]
                logger.info(
                    "Built inputs for cluster '%s' (path=%s, %d windows)",
                    cid, path, n_windows,
                )
                self._emit(event(
                    STAGE_INFERENCE, "Cluster inputs ready",
                    status=STATUS_OK, target=cid,
                    detail=f"path={path} · {n_windows} scenarios",
                ))
            except Exception as exc:
                self._emit(event(
                    STAGE_INFERENCE, "Cluster input build failed",
                    status=STATUS_FAIL, target=cid, detail=str(exc),
                ))
                raise

        return member_inputs, extra_meta

    # ==================================================================
    # Post-inference (Stage 6)
    #
    # Three responsibilities:
    #
    #   1. ``_post_infer_cluster`` — per-cluster: inverse-scale +
    #      restore notional sign via ``transform_predictions``, write
    #      ``_scaled.parquet`` / ``_original.parquet``, return a
    #      ``ClusterArtifact`` handle.
    #   2. ``post_infer`` — run-level: aggregate the per-cluster
    #      summaries into ``cluster_predictions.parquet`` and
    #      ``portfolio_predictions.parquet``, then write the
    #      manifest.
    #   3. ``_write_run_manifest`` — emit the canonical
    #      ``manifest.json`` the API result-reader and the dashboard
    #      use as the entry point to every artifact this run
    #      produced.
    # ==================================================================

    def _post_infer_cluster(
        self,
        cluster_id:      str,
        cluster_preds:   np.ndarray,
        scenario_labels: List[str],
        ctx:             "InferenceContext",
        out_dir:         Path,
    ) -> ClusterArtifact:
        """Transform one cluster's scaled predictions and persist them.

        Steps
        -----
        1. Call :meth:`HybridGnnRnnInferencePipeline.transform_predictions`
           to get ``(scaled_wide, original_wide, summary_df)``.
        2. Write both wide DataFrames to
           ``<out_dir>/clusters/<cluster_id>_{scaled,original}.parquet``.
           Wide-format is intentional — the dashboard's per-cluster
           trade drill-down reads scenarios as rows and trades as
           columns, matching the parquet layout directly.
        3. Emit an activity event so the UI can render progress
           per cluster.
        4. Return a :class:`ClusterArtifact` carrying the summary
           frame (for post_infer aggregation) and the parquet
           paths (for the manifest).

        Returns
        -------
        ClusterArtifact
            Lightweight handle suitable for accumulation in a list
            without retaining the heavy wide frames (those have
            already been flushed to disk).
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        scaled_wide, original_wide, summary_df = (
            HybridGnnRnnInferencePipeline.transform_predictions(
                cluster_id      = cluster_id,
                cluster_preds   = cluster_preds,
                scenario_labels = scenario_labels,
                ctx             = ctx,
            )
        )

        # Per-cluster trade-level outputs live under
        # ``trade_predictions/<cid>_<space>.parquet`` — symmetric with
        # the eval pipeline's ``members/<cid>/predictions/<split>.npz``
        # convention (one subdirectory per artifact family).
        trade_predictions_dir = out_dir / TRADE_PREDICTIONS_DIRNAME
        trade_predictions_dir.mkdir(parents=True, exist_ok=True)
        scaled_path   = trade_predictions_dir / f"{cluster_id}_scaled.parquet"
        original_path = trade_predictions_dir / f"{cluster_id}_original.parquet"

        scaled_wide.to_parquet(scaled_path)
        original_wide.to_parquet(original_path)

        n_scenarios = int(len(scaled_wide))
        n_trades    = int(len(scaled_wide.columns))
        trade_ids   = list(scaled_wide.columns)

        self._emit(event(
            STAGE_INFERENCE,
            f"Wrote cluster artifacts",
            status = STATUS_OK,
            target = cluster_id,
            detail = f"{n_trades} trades × {n_scenarios} scenarios",
        ))

        return ClusterArtifact(
            cluster_id    = cluster_id,
            n_trades      = n_trades,
            n_scenarios   = n_scenarios,
            scaled_path   = str(scaled_path),
            original_path = str(original_path),
            trade_ids     = trade_ids,
            summary_df    = summary_df,
        )

    def post_infer(
        self,
        result:            InferenceResult,
        *,
        cluster_artifacts: Optional[List[ClusterArtifact]] = None,
    ) -> None:
        """Aggregate per-cluster artifacts and write run-level summaries.

        Replaces the legacy flat-CSV writer.  When invoked with no
        ``cluster_artifacts`` (e.g. by a programmatic caller running
        the pipeline against an in-memory configuration) the method
        is a no-op aside from the summary log line — preserving
        backwards compatibility for callers that don't care about
        on-disk artifacts.

        When called from :meth:`run_inference` with the accumulated
        list of :class:`ClusterArtifact` objects, three outputs land
        under ``<artifacts_dir>/inference/``:

          * ``cluster_predictions.parquet`` — long-format per-cluster
            × per-scenario aggregate stats.  Stacked directly from
            ``ClusterArtifact.summary_df`` so the column schema
            matches :meth:`transform_predictions`'s contract.
          * ``portfolio_predictions.parquet`` — per-scenario
            portfolio totals (sum across clusters per scenario).
            ``sum_pnl_*`` is sum-additive across clusters; mean / std
            / min / max are not, so we re-derive them from
            ``sum_pnl_original`` at the portfolio level (one value
            per scenario).
          * ``manifest.json`` — canonical entry-point pointing at
            every artifact this run produced (see
            :meth:`_write_run_manifest`).

        Parameters
        ----------
        result
            The :class:`InferenceResult` :meth:`_build_result` just
            produced — used for the summary log line and the
            ``n_scenarios`` / ``latency_seconds`` fields in the
            manifest.
        cluster_artifacts
            The list :meth:`run_inference` accumulated from
            per-cluster ``_post_infer_cluster`` calls.  None ⇒ this
            invocation came from outside :meth:`run_inference`;
            parquets + manifest are not written.
        """
        if result.predictions is not None:
            preds = result.predictions
            logger.info(
                "Ensemble inference summary: n_samples=%d, mean=%.4f, "
                "std=%.4f, min=%.4f, max=%.4f",
                result.n_samples, np.mean(preds), np.std(preds),
                np.min(preds), np.max(preds),
            )

        if not self.config.artifacts_dir or cluster_artifacts is None:
            return

        out_dir = Path(self.config.artifacts_dir) / INFERENCE_DIRNAME
        out_dir.mkdir(parents=True, exist_ok=True)

        cluster_summary_path:   Optional[Path] = None
        portfolio_summary_path: Optional[Path] = None

        if cluster_artifacts:
            # Stack per-cluster summary frames into one long DataFrame.
            # transform_predictions guarantees the column schema is
            # uniform across clusters, so a vanilla concat is safe.
            cluster_summary_df  = pd.concat(
                [a.summary_df for a in cluster_artifacts if a.summary_df is not None],
                ignore_index=True,
            )
            cluster_summary_dir = out_dir / CLUSTER_SUMMARY_DIRNAME
            cluster_summary_dir.mkdir(parents=True, exist_ok=True)
            cluster_summary_path = cluster_summary_dir / CLUSTER_SUMMARY_FILENAME
            cluster_summary_df.to_parquet(cluster_summary_path)

            # Portfolio-level: sum sum_pnl_* across clusters per
            # scenario (these are linear-additive).  Mean / std /
            # min / max are NOT additive across clusters so we
            # surface them as the portfolio-level aggregates derived
            # from per-cluster sums — one row per scenario.
            portfolio_summary_df  = (
                cluster_summary_df
                .groupby("scenario_label", as_index=False)
                .agg(
                    sum_pnl_scaled   = ("sum_pnl_scaled",   "sum"),
                    sum_pnl_original = ("sum_pnl_original", "sum"),
                    n_clusters       = ("cluster_id",       "nunique"),
                )
            )
            portfolio_summary_dir = out_dir / PORTFOLIO_SUMMARY_DIRNAME
            portfolio_summary_dir.mkdir(parents=True, exist_ok=True)
            portfolio_summary_path = portfolio_summary_dir / PORTFOLIO_SUMMARY_FILENAME
            portfolio_summary_df.to_parquet(portfolio_summary_path)

            self._emit(event(
                STAGE_INFERENCE, "Run summaries written",
                status = STATUS_OK,
                target = f"{len(cluster_artifacts)} clusters",
            ))

        manifest_path = self._write_run_manifest(
            out_dir                = out_dir,
            result                 = result,
            cluster_artifacts      = cluster_artifacts,
            cluster_summary_path   = cluster_summary_path,
            portfolio_summary_path = portfolio_summary_path,
        )
        logger.info("Inference manifest written to %s", manifest_path)
        self._emit(event(
            STAGE_INFERENCE, "Manifest written",
            status = STATUS_OK,
            target = str(manifest_path.name),
        ))

    def _write_run_manifest(
        self,
        out_dir:                Path,
        result:                 InferenceResult,
        cluster_artifacts:      List[ClusterArtifact],
        cluster_summary_path:   Optional[Path],
        portfolio_summary_path: Optional[Path],
    ) -> Path:
        """Write the canonical ``manifest.json`` describing the run.

        The manifest is the result-reader's entry point — the
        dashboard fetches it first, then deep-links to whichever
        per-cluster / run-level parquets it needs.  Schema is
        intentionally permissive (values are not validated by a
        pydantic model on the server side) so future fields can be
        added without breaking older clients.

        Returns
        -------
        Path
            Path of the manifest that was written (always
            ``<out_dir>/manifest.json``).
        """
        manifest: Dict[str, Any] = {
            "schema_version":          "1",
            "generated_at":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ensemble_version":        self.ensemble_version,
            "input_mode":              (result.metadata or {}).get(
                "input_mode", "new_scenarios",
            ),
            "n_scenarios":             int(result.n_samples) if result.n_samples else 0,
            "scenario_labels": (
                list(self._validation_report.scenario_labels)
                if self._validation_report is not None else []
            ),
            "clusters": [a.to_manifest_dict() for a in (cluster_artifacts or [])],
            "cluster_summary_path":    str(cluster_summary_path) if cluster_summary_path   else None,
            "portfolio_summary_path":  str(portfolio_summary_path) if portfolio_summary_path else None,
            "validation": (
                self._validation_report.to_dict()
                if self._validation_report is not None else None
            ),
            "scenarios": (
                self._loaded_scenarios.to_dict()
                if self._loaded_scenarios is not None else None
            ),
            "latency_seconds":         (
                float(result.latency_seconds)
                if getattr(result, "latency_seconds", None) is not None else None
            ),
        }

        path = out_dir / MANIFEST_FILENAME
        path.write_text(json.dumps(manifest, indent=2))
        return path

    # ==================================================================
    # Result building
    # ==================================================================

    def _build_result(
        self,
        combined: np.ndarray,
        infer_meta: Dict[str, Any],
        extra_meta: Dict[str, Any],
    ) -> InferenceResult:
        """Wrap aggregated predictions into an :class:`InferenceResult`.

        Always called from :meth:`run_inference`, whose prerequisite
        checks guarantee that the ensemble model + validation report
        are populated by the time we land here.

        The result's ``metadata`` carries everything the UI needs to
        render the run summary card: input mode, cluster identifiers,
        routing breakdown (affected/unaffected lists, cheap-path
        flag), scenario labels, and per-member sample IDs.
        """
        assert self._validation_report is not None, \
            "_build_result called before validation (run_inference should guard this)"

        meta: Dict[str, Any] = {
            "input_mode":          infer_meta.get("input_mode", "new_scenarios"),
            "cluster_ids":         self._ensemble.router.cluster_ids if self._ensemble else [],
            "n_members":           self._ens_config.n_members if self._ens_config else 0,
            "affected_clusters":   self._validation_report.affected_cluster_ids,
            "unaffected_clusters": self._validation_report.unaffected_cluster_ids,
            "cheap_path_used":     self._validation_report.cheap_path_used,
            "scenario_labels":     self._validation_report.scenario_labels,
        }
        if extra_meta.get("sample_ids"):
            meta["per_member_sample_ids"] = extra_meta["sample_ids"]

        # Flatten per-member sample IDs into a single ordered list,
        # falling back to whatever the caller stashed in infer_meta.
        all_sample_ids = None
        per_member_ids = extra_meta.get("sample_ids", {})
        if per_member_ids:
            all_sample_ids = []
            for cid in sorted(per_member_ids.keys()):
                ids = per_member_ids[cid]
                if ids:
                    all_sample_ids.extend(ids)
            all_sample_ids = all_sample_ids or None

        return InferenceResult(
            predictions   = combined,
            n_samples     = combined.shape[0],
            sample_ids    = all_sample_ids or infer_meta.get("sample_ids"),
            model_version = self.ensemble_version,
            metadata      = meta,
        )
