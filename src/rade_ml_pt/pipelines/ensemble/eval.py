"""
Ensemble evaluation pipeline.

Loads an ensemble from the registry, routes test data to each member,
collects per-member and ensemble-level metrics, and saves evaluation
artifacts for the UI dashboard.

The per-member evaluation loop respects ``EnsembleConfig.execution_strategy``
via the same dispatch pattern used by ``EnsembleTrainPipeline``.  The
module-level worker ``evaluate_single_member`` is picklable for future
multi-process / distributed backends.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.ensemble.metrics import (
    compute_ensemble_metrics,
    compute_per_member_metrics,
    aggregate_member_metrics,
)

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import EvaluationResult

logger = logging.getLogger(__name__)

_SUPPORTED_STRATEGIES = {"sequential", "process_pool", "gpu_parallel"}
_PLANNED_STRATEGIES = {"distributed"}
_EVAL_SPLITS = ("train", "val", "test")
_PRIMARY_SPLIT = "test"

_EVAL_MANIFEST_SCHEMA_VERSION = 2  # v2: +spaces_available (Phase 3.1)
_PORTFOLIO_TIMESERIES_SCHEMA_VERSION = 3  # v3: +*_original columns (Phase 3.1)
_CLUSTER_TIMESERIES_SCHEMA_VERSION   = 3  # v3: +*_original columns (Phase 3.1)
_TRADE_METRICS_SCHEMA_VERSION = 1
_ENSEMBLE_METRICS_SCHEMA_VERSION = 1
_PER_MEMBER_METRICS_SCHEMA_VERSION = 1
_GROUP_CORRELATIONS_SCHEMA_VERSION = 1
_CLUSTER_ATTRIBUTES_SCHEMA_VERSION = 1
_GRAPH_STATS_SCHEMA_VERSION = 1
_COMPLETENESS_SCHEMA_VERSION = 1
_FEATURE_SUMMARY_SCHEMA_VERSION = 1


# ======================================================================
# PerSplitData — single source of truth for the per-split loop
# ======================================================================

@dataclass
class PerSplitData:
    """All per-split data needed by every eval-pipeline writer.

    Built once at the top of the per-split loop by
    :func:`_build_per_split_data` and consumed by every ``_save_*``
    helper.  Carries both PnL spaces (scaled + original) so writers
    stay pure side-effects — no transform or aggregation logic in
    them.

    Per-cluster wide DataFrames are the canonical form:
      - ``index``   = ``scenario_label`` (or positional ``"0", "1", …``)
      - ``columns`` = trade IDs in the scaler's canonical order

    ``preds_scaled`` / ``targets_scaled`` are keyed for every cluster
    with predictions for this split.  ``preds_original`` /
    ``targets_original`` are keyed only for the subset of clusters
    whose scaler artefacts loaded successfully — the rest fall through
    to NaN in the timeseries writers' ``*_original`` columns and to
    no ``{split}_original.npz`` shard.

    All aggregations (cluster sums, portfolio sums) are derived once
    in ``__post_init__`` so writers never recompute them.
    """
    split:           str
    n_scenarios:     int
    n_trades:        int
    scenario_labels: List[str]                          # may be empty
    preds_scaled:    Dict[str, pd.DataFrame]
    targets_scaled:  Dict[str, pd.DataFrame]
    preds_original:  Dict[str, pd.DataFrame]
    targets_original: Dict[str, pd.DataFrame]

    # Derived — cluster-level sums (axis=1 over wide frames).
    cluster_pred_sums:            Dict[str, np.ndarray] = field(init=False)
    cluster_target_sums:          Dict[str, np.ndarray] = field(init=False)
    cluster_pred_sums_original:   Dict[str, np.ndarray] = field(init=False)
    cluster_target_sums_original: Dict[str, np.ndarray] = field(init=False)

    # Derived — portfolio-level sums (reduce over clusters).
    # ``*_original`` is ``None`` when no cluster contributed (scaler
    # missing for all members) — the timeseries writer reads that as
    # NaN-fill.
    preds_1d:            np.ndarray           = field(init=False)
    targets_1d:          np.ndarray           = field(init=False)
    preds_1d_original:   Optional[np.ndarray] = field(init=False)
    targets_1d_original: Optional[np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        self.cluster_pred_sums = {
            cid: df.to_numpy().sum(axis=1)
            for cid, df in self.preds_scaled.items()
        }
        self.cluster_target_sums = {
            cid: df.to_numpy().sum(axis=1)
            for cid, df in self.targets_scaled.items()
        }
        self.cluster_pred_sums_original = {
            cid: df.to_numpy().sum(axis=1)
            for cid, df in self.preds_original.items()
        }
        self.cluster_target_sums_original = {
            cid: df.to_numpy().sum(axis=1)
            for cid, df in self.targets_original.items()
        }

        if self.cluster_pred_sums:
            self.preds_1d = np.add.reduce(
                list(self.cluster_pred_sums.values()),
            )
            self.targets_1d = np.add.reduce(
                list(self.cluster_target_sums.values()),
            )
        else:
            self.preds_1d   = np.zeros(self.n_scenarios, dtype=np.float32)
            self.targets_1d = np.zeros(self.n_scenarios, dtype=np.float32)

        if self.cluster_pred_sums_original:
            self.preds_1d_original = np.add.reduce(
                list(self.cluster_pred_sums_original.values()),
            )
            self.targets_1d_original = np.add.reduce(
                list(self.cluster_target_sums_original.values()),
            )
        else:
            self.preds_1d_original   = None
            self.targets_1d_original = None

    # ── Convenience accessors for writers that prefer ndarray dicts ──
    def preds_scaled_arrays(self) -> Dict[str, np.ndarray]:
        """ndarray view of every scaled prediction wide frame."""
        return {cid: df.to_numpy() for cid, df in self.preds_scaled.items()}

    def targets_scaled_arrays(self) -> Dict[str, np.ndarray]:
        """ndarray view of every scaled target wide frame."""
        return {cid: df.to_numpy() for cid, df in self.targets_scaled.items()}


# ======================================================================
# Module-level helpers
# ======================================================================

_DEFAULT_BATCH_SIZE = 32


def _resolve_batch_size(version_dir: Path) -> int:
    """Read batch_size from the member's saved data config, or fall back to default."""
    dc_path = version_dir / "data_config.json"
    if dc_path.exists():
        try:
            with open(dc_path, "r") as f:
                cfg = json.load(f)
            bs = cfg.get("batch_size")
            if bs is not None:
                return int(bs)
        except Exception:
            pass
    return _DEFAULT_BATCH_SIZE


# ======================================================================
# Module-level worker — picklable, portable across processes / nodes
# ======================================================================

def evaluate_single_member(
    cluster_id: str,
    model: Any,
    registry_root_dir: str,
    member_version: str,
    split: str = "test",
) -> Optional[Dict[str, Any]]:
    """
    Evaluate one member model on a single cached dataset split.

    Module-level so it can be dispatched to ``ProcessPoolExecutor`` or
    ``ray.remote`` in future parallel strategies.

    Parameters
    ----------
    cluster_id : str
        Cluster identifier.
    model : nn.Module
        Member model in eval mode.
    registry_root_dir : str
        Root registry directory (to locate cached datasets).
    member_version : str
        Registry version string for this member.
    split : str
        Dataset split to evaluate (``"train"``, ``"val"``, or ``"test"``).

    Returns
    -------
    dict or None
        ``{"predictions": ndarray, "targets": ndarray, "metrics": dict}``
        or ``None`` if the split is not available.
    """
    from src.rade_ml_pt.evaluation.evaluator import Evaluator

    version_dir = Path(registry_root_dir) / member_version
    ds_dir = version_dir / "datasets"

    ds_path = ds_dir / f"{split}.pt"
    if not ds_path.exists():
        logger.debug(
            "No %s data for member '%s' (version '%s'). Skipping.",
            split, cluster_id, member_version,
        )
        return None

    batch_size = _resolve_batch_size(version_dir)

    loader = None
    import torch
    from torch.utils.data import DataLoader
    try:
        from src.rade_ml_pt.data.dataset import _collate_dict_batch
        dataset = torch.load(str(ds_path), weights_only=False)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=_collate_dict_batch,
        )
    except Exception as exc:
        logger.warning(
            "Could not load cached %s data for '%s': %s", split, cluster_id, exc,
        )

    if loader is None:
        return None

    evaluator = Evaluator(model=model)
    eval_result = evaluator.run(loader)

    return {
        "predictions": eval_result.predictions,
        "targets": eval_result.targets,
        "metrics": eval_result.metrics,
    }


# ======================================================================
# Pipeline
# ======================================================================

class EnsembleEvalPipeline:
    """
    Evaluate an ensemble model by running each member on its cluster's
    test data and aggregating the results.

    Parameters
    ----------
    ensemble_config : EnsembleConfig
        Must have ``registry_dir`` set.
    ensemble_version : str
        Ensemble version or tag to load (default ``"latest"``).
    run_member_eval : bool
        If ``True``, run the full ``HybridGnnRnnEvalPipeline`` for each
        member (cold-start from registry).  Produces per-member evaluation
        plots, portfolio PnL analytics, and inverse-transformed metrics.
        Default ``False`` (lightweight metrics only).
    member_eval_pipeline : str or None
        Dotpath to the member eval pipeline class.  Defaults to
        ``HybridGnnRnnEvalPipeline`` when ``None``.
    """

    _DEFAULT_MEMBER_EVAL = (
        "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.eval.HybridGnnRnnEvalPipeline"
    )

    def __init__(
        self,
        ensemble_config: EnsembleConfig,
        ensemble_version: str = "latest",
        run_member_eval: bool = False,
        member_eval_pipeline: Optional[str] = None,
    ) -> None:
        self.config = ensemble_config
        self.ensemble_version = ensemble_version
        self.run_member_eval = run_member_eval
        self._member_eval_dotpath = member_eval_pipeline or self._DEFAULT_MEMBER_EVAL

    def run(self) -> Dict[str, Any]:
        """
        Execute the full ensemble evaluation.

        Steps
        -----
        1. Load ensemble (all members + router) from registry.
        2. Dispatch per-member evaluation across all available splits.
        3. Compute per-member and ensemble-level metrics for each split.
        4. Save evaluation artifacts.

        Returns
        -------
        dict
            Top-level keys (``ensemble_metrics``, ``per_member_metrics``,
            ``member_summary``) correspond to the primary split (test).
            ``additional_splits`` holds the same structure for train/val
            when those datasets are available.
        """
        strategy = self.config.execution_strategy
        logger.info("EnsembleEvalPipeline: starting (strategy='%s')", strategy)
        t0 = time.perf_counter()

        # Load ensemble.
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, resolved_version = ens_registry.load(self.ensemble_version)

        from src.rade_ml_pt.registry.store import ModelRegistry
        registry = ModelRegistry(self.config.registry_dir)
        builder = EnsembleBuilder(registry)
        ensemble = builder.build(config, member_versions)

        # Full per-member evaluation (cold-start pipeline with plots).
        if self.run_member_eval:
            self._run_member_eval_pipelines(config, member_versions)

        # Dispatch per-member evaluation (returns per-split structures).
        t_eval = time.perf_counter()
        split_preds, split_targets, split_metrics = (
            self._dispatch_evaluation(config, ensemble, registry, member_versions)
        )
        logger.info(
            "Dispatch evaluation: %.1fs (%d clusters × %d splits)",
            time.perf_counter() - t_eval,
            len(config.cluster_ids),
            len(split_preds),
        )

        # Compute metrics for every available split.
        all_split_results: Dict[str, Dict[str, Any]] = {}

        for split in split_preds:
            preds = split_preds[split]
            targets = split_targets[split]

            pm_metrics = compute_per_member_metrics(preds, targets)
            rollup = aggregate_member_metrics(pm_metrics)

            ens_metrics: Dict[str, float] = {}
            if preds:
                try:
                    c_preds = ensemble._combine(preds)
                    c_targets = ensemble._combine(targets)
                    ens_metrics = compute_ensemble_metrics(c_preds, c_targets)
                except Exception as exc:
                    logger.warning("Could not compute ensemble metrics for '%s': %s", split, exc)

            all_split_results[split] = {
                "ensemble_metrics": ens_metrics,
                "per_member_metrics": pm_metrics,
                "member_summary": rollup,
            }

        primary = all_split_results.get(_PRIMARY_SPLIT, {})
        additional = {s: v for s, v in all_split_results.items() if s != _PRIMARY_SPLIT}

        # Save artifacts.
        if self.config.artifacts_dir:
            self._save_all_artifacts(
                resolved_version, all_split_results, config,
                split_preds, split_targets,
                member_versions=member_versions,
            )

        elapsed = time.perf_counter() - t0
        logger.info("EnsembleEvalPipeline: done (%.1fs)", elapsed)

        return {
            "ensemble_version": resolved_version,
            "ensemble_metrics": primary.get("ensemble_metrics", {}),
            "per_member_metrics": primary.get("per_member_metrics", {}),
            "member_summary": primary.get("member_summary", {}),
            "additional_splits": additional,
        }

    # ------------------------------------------------------------------
    # Full per-member evaluation (cold-start pipeline)
    # ------------------------------------------------------------------

    def _run_member_eval_pipelines(
        self,
        config: EnsembleConfig,
        member_versions: Dict[str, str],
    ) -> None:
        """Run the full member eval pipeline for each cluster.

        Uses the configured ``member_eval_pipeline`` class (cold-start from
        registry).  Each member's artifacts are saved under
        ``<artifacts_dir>/members/<cluster_id>/``.
        """
        import importlib

        module_path, cls_name = self._member_eval_dotpath.rsplit(".", 1)
        try:
            eval_cls = getattr(importlib.import_module(module_path), cls_name)
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Could not import member eval pipeline (%s): %s. "
                "Skipping per-member evaluation.",
                self._member_eval_dotpath, exc,
            )
            return

        for cid in config.cluster_ids:
            version = member_versions.get(cid)
            if not version:
                logger.warning("No version for '%s'; skipping member eval.", cid)
                continue

            logger.info("--- Running member eval pipeline: '%s' (version='%s') ---", cid, version)
            try:
                from src.rade_ml_pt.pipelines.config import PipelineConfig

                member_raw = config.member_configs.get(cid, {})
                member_config = PipelineConfig(
                    data_config=member_raw.get("data_config"),
                    model_config=member_raw.get("model_config"),
                    registry_dir=self.config.registry_dir,
                    artifacts_dir=str(
                        Path(self.config.artifacts_dir) / "members" / cid
                    ) if self.config.artifacts_dir else None,
                    version_or_tag=version,
                    metadata={
                        **member_raw.get("metadata", {}),
                        "cluster_id": cid,
                    },
                )

                eval_pipeline = eval_cls(member_config)
                eval_pipeline.run()
                logger.info("Member '%s' evaluation complete.", cid)
            except Exception as exc:
                logger.warning(
                    "Member '%s' eval pipeline failed (non-fatal): %s", cid, exc,
                )

    # ------------------------------------------------------------------
    # Execution strategy dispatch
    # ------------------------------------------------------------------

    def _dispatch_evaluation(
        self,
        config: EnsembleConfig,
        ensemble: Any,
        registry: Any,
        member_versions: Dict[str, str],
    ) -> tuple:
        """
        Route to the configured execution strategy for member evaluation.

        Adding a new strategy:
            1. Implement ``_run_<strategy>_eval`` returning the same 3-tuple.
            2. Add the strategy name to ``_SUPPORTED_STRATEGIES``.
            3. Add an ``elif`` branch here.
        """
        strategy = self.config.execution_strategy

        if strategy == "sequential":
            return self._run_sequential_eval(config, ensemble, registry, member_versions)
        elif strategy == "process_pool":
            return self._run_threaded_eval(config, ensemble, registry, member_versions)
        elif strategy == "gpu_parallel":
            return self._run_gpu_parallel_eval(config, ensemble, registry, member_versions)
        elif strategy in _PLANNED_STRATEGIES:
            raise NotImplementedError(
                f"Execution strategy '{strategy}' is planned but not yet "
                f"implemented for evaluation. Available now: "
                f"{sorted(_SUPPORTED_STRATEGIES)}."
            )
        else:
            raise ValueError(
                f"Unknown execution_strategy '{strategy}'. "
                f"Supported: {sorted(_SUPPORTED_STRATEGIES)}. "
                f"Planned: {sorted(_PLANNED_STRATEGIES)}."
            )

    # ------------------------------------------------------------------
    # Strategy: sequential
    # ------------------------------------------------------------------

    def _run_sequential_eval(
        self,
        config: EnsembleConfig,
        ensemble: Any,
        registry: Any,
        member_versions: Dict[str, str],
    ) -> tuple:
        """Evaluate all members sequentially across all available splits."""
        split_preds: Dict[str, Dict[str, np.ndarray]] = {}
        split_targets: Dict[str, Dict[str, np.ndarray]] = {}
        split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        for cid in config.cluster_ids:
            logger.info("--- Evaluating member '%s' ---", cid)
            for split in _EVAL_SPLITS:
                member_result = evaluate_single_member(
                    cluster_id=cid,
                    model=ensemble.members[cid],
                    registry_root_dir=str(registry.root_dir),
                    member_version=member_versions[cid],
                    split=split,
                )
                if member_result is not None:
                    split_preds.setdefault(split, {})[cid] = member_result["predictions"]
                    split_targets.setdefault(split, {})[cid] = member_result["targets"]
                    split_metrics.setdefault(split, {})[cid] = member_result["metrics"]

        return split_preds, split_targets, split_metrics

    # ------------------------------------------------------------------
    # Strategy: process_pool (threaded, models already in memory)
    # ------------------------------------------------------------------

    def _run_threaded_eval(
        self,
        config: EnsembleConfig,
        ensemble: Any,
        registry: Any,
        member_versions: Dict[str, str],
    ) -> tuple:
        """Evaluate members in parallel using ThreadPoolExecutor.

        ThreadPoolExecutor is preferred over ProcessPoolExecutor for eval
        because the models are already loaded in the main process — no
        expensive pickling required.  PyTorch releases the GIL during
        forward passes, so threads achieve real parallelism for both CPU
        and CUDA inference.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        split_preds: Dict[str, Dict[str, np.ndarray]] = {}
        split_targets: Dict[str, Dict[str, np.ndarray]] = {}
        split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        max_w = self.config.max_workers or len(config.cluster_ids)
        logger.info("process_pool eval: %d members, %d threads", len(config.cluster_ids), max_w)

        tasks = [
            (cid, split)
            for cid in config.cluster_ids
            for split in _EVAL_SPLITS
        ]

        def _eval_task(args):
            cid, split = args
            return cid, split, evaluate_single_member(
                cluster_id=cid,
                model=ensemble.members[cid],
                registry_root_dir=str(registry.root_dir),
                member_version=member_versions[cid],
                split=split,
            )

        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = {pool.submit(_eval_task, t): t for t in tasks}
            for future in as_completed(futures):
                cid, split, member_result = future.result()
                if member_result is not None:
                    split_preds.setdefault(split, {})[cid] = member_result["predictions"]
                    split_targets.setdefault(split, {})[cid] = member_result["targets"]
                    split_metrics.setdefault(split, {})[cid] = member_result["metrics"]

        return split_preds, split_targets, split_metrics

    # ------------------------------------------------------------------
    # Strategy: gpu_parallel (models moved to per-GPU devices)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_gpu_ids(config: EnsembleConfig) -> "List[int]":
        """Return the list of CUDA device IDs to use."""
        if config.gpu_device_ids:
            return list(config.gpu_device_ids)
        import torch
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError(
                "gpu_parallel strategy requires at least one CUDA GPU, "
                "but torch.cuda.device_count() == 0."
            )
        return list(range(n))

    def _run_gpu_parallel_eval(
        self,
        config: EnsembleConfig,
        ensemble: Any,
        registry: Any,
        member_versions: Dict[str, str],
    ) -> tuple:
        """Evaluate members in parallel with each model pinned to a CUDA device.

        Models are moved to their assigned GPU, then evaluation runs in a
        ThreadPoolExecutor.  PyTorch releases the GIL during CUDA kernels,
        so threads give true parallelism across GPUs.
        """
        import torch
        from concurrent.futures import ThreadPoolExecutor, as_completed

        gpu_ids = self._resolve_gpu_ids(config)
        n_gpus = len(gpu_ids)

        for i, cid in enumerate(config.cluster_ids):
            device = torch.device(f"cuda:{gpu_ids[i % n_gpus]}")
            ensemble.members[cid].to(device)
            logger.info("Moved member '%s' to %s", cid, device)

        max_w = min(self.config.max_workers or n_gpus, n_gpus)
        logger.info(
            "gpu_parallel eval: %d members across %d GPUs, %d threads",
            len(config.cluster_ids), n_gpus, max_w,
        )

        split_preds: Dict[str, Dict[str, np.ndarray]] = {}
        split_targets: Dict[str, Dict[str, np.ndarray]] = {}
        split_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

        tasks = [
            (cid, split)
            for cid in config.cluster_ids
            for split in _EVAL_SPLITS
        ]

        def _eval_task(args):
            cid, split = args
            return cid, split, evaluate_single_member(
                cluster_id=cid,
                model=ensemble.members[cid],
                registry_root_dir=str(registry.root_dir),
                member_version=member_versions[cid],
                split=split,
            )

        with ThreadPoolExecutor(max_workers=max_w) as pool:
            futures = {pool.submit(_eval_task, t): t for t in tasks}
            for future in as_completed(futures):
                cid, split, member_result = future.result()
                if member_result is not None:
                    split_preds.setdefault(split, {})[cid] = member_result["predictions"]
                    split_targets.setdefault(split, {})[cid] = member_result["targets"]
                    split_metrics.setdefault(split, {})[cid] = member_result["metrics"]

        return split_preds, split_targets, split_metrics

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def _save_all_artifacts(
        self,
        version: str,
        all_split_results: Dict[str, Dict[str, Any]],
        config: EnsembleConfig,
        split_preds: Dict[str, Dict[str, np.ndarray]],
        split_targets: Dict[str, Dict[str, np.ndarray]],
        member_versions: Optional[Dict[str, str]] = None,
    ) -> None:
        """Persist the PRISM evaluation contract artifacts.

        Output layout under ``{ART}/ensemble/{version}/evaluation/``:

        Run-scoped (written once per run):
          - ``manifest.json``                    — index of every artifact
          - ``trade_cluster_map.json``           — trade_id → cluster_id
          - ``ensemble_metrics.parquet``         (B4)
          - ``per_member_metrics.parquet``       (B5)
          - ``cluster_attributes.parquet``       (B7)
          - ``graph_stats.parquet``              (B8)
          - ``members/{cid}/{graph_results.joblib,trade_universe.json,
                            training_curves.parquet,elementary_pnl.parquet}``
            (staged from the training registry by the graph copier)

        Per-split (train / val / test):
          - ``members/{cid}/predictions/{split}.npz``                  (raw shards)
          - ``portfolio_summary/portfolio_timeseries_{split}.parquet`` (B1)
          - ``cluster_summary/cluster_timeseries_{split}.parquet``     (B2)
          - ``trade_metrics/trade_metrics_{split}.parquet``            (B3)
          - ``group_correlations/group_correlations_{split}.parquet``  (B6)
          - ``quality/completeness_{split}.parquet``                   (5e)
          - ``quality/feature_summary_{split}.parquet``                (5e)

        Orchestration is intentionally three-phase:

          1. **Run metadata** — manifest + trade-cluster map written
             upfront so a partial / failing run still leaves a
             structurally-valid manifest behind for the API to consume.
          2. **Shared inputs** — scenario labels, cluster attributes,
             scaler artefacts, and canonical trade IDs loaded once and
             reused across every split-scoped writer.
          3. **Run-scoped parquets** then a **per-split loop** that
             builds a :class:`PerSplitData` bundle once (compute) and
             dispatches one pure writer per artifact family (write).

        Phase 3.1 (scaled ↔ original PnL) is folded into the per-split
        compute step: :func:`_build_per_split_data` runs the
        scaled → original transform for every cluster with scaler
        artefacts and stores both spaces' wide DataFrames on the
        bundle.  Writers consume the bundle and emit the parallel
        ``{split}_original.npz`` shards (sibling to the legacy scaled
        NPZ) and the ``*_original`` columns on the cluster / portfolio
        timeseries — all without any extra branching at the
        orchestrator level.  Clusters / runs without scaler coverage
        transparently degrade to NaN-filled original columns and emit
        no original-space NPZ; the v3 schemas stay canonical
        regardless of coverage.
        """
        t0 = time.perf_counter()
        eval_dir = (
            Path(self.config.artifacts_dir) / "ensemble" / version / "evaluation"
        )
        eval_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Run metadata ──────────────────────────────────────────
        cluster_trade_indices = EnsembleBuilder._build_cluster_trade_indices(config)
        _write_json(
            eval_dir / "manifest.json",
            _build_eval_manifest(
                version, config, cluster_trade_indices, all_split_results,
            ),
        )
        _write_json(
            eval_dir / "trade_cluster_map.json",
            {
                str(tid): cid
                for cid in config.cluster_ids
                for tid in config.cluster_mapping.get(cid, [])
            },
        )

        # ── 2. Shared inputs (computed once, reused across splits) ──
        scenario_labels_by_split = _load_scenario_labels_by_split(
            config, member_versions,
        )
        cluster_attrs = _build_cluster_attributes(config)
        # Phase 3.1 — per-cluster (target_scaler, target_attributes)
        # tuples loaded once and reused across every split.  Empty when
        # no member exposes scaler artefacts (legacy runs) — the
        # per-split transform then becomes a no-op and the timeseries
        # writers NaN-fill the new ``*_original`` columns.
        cluster_artefacts = _load_member_target_artifacts(
            config, member_versions,
        )
        if not cluster_artefacts:
            logger.warning(
                "No per-cluster scaler artefacts found — eval will emit "
                "scaled-space only (Phase 3.1 *_original columns will be NaN)."
            )
        # Canonical trade-ID order per cluster — resolved once so every
        # per-split write sees consistent labels across splits / spaces.
        trade_ids_by_cluster = _build_trade_ids_by_cluster(
            clusters          = list(config.cluster_ids),
            cluster_artefacts = cluster_artefacts,
            config            = config,
        )

        # ── 3a. Run-scoped parquets ─────────────────────────────────
        _save_ensemble_metrics_parquet(eval_dir, all_split_results)
        _save_per_member_metrics_parquet(eval_dir, all_split_results)
        _save_cluster_attributes_parquet(eval_dir, cluster_attrs)
        _save_graph_stats_parquet(eval_dir, config, member_versions)
        _copy_member_graph_artifacts(eval_dir, config, member_versions)

        # ── 3b. Per-split loop — compute once, write many ───────────
        for split in all_split_results:
            t_split = time.perf_counter()

            if split not in split_preds:
                logger.info(
                    "  [%s] no predictions — skipping per-split writes", split,
                )
                continue

            # Step (a) — pure compute: wrap raw ndarrays as labelled
            # wide DataFrames, run the scaled→original transform for
            # clusters with scaler artefacts, and derive all sums.
            ps = _build_per_split_data(
                split                = split,
                preds_by_cluster     = split_preds[split],
                targets_by_cluster   = split_targets[split],
                scenario_labels      = scenario_labels_by_split.get(split, []),
                trade_ids_by_cluster = trade_ids_by_cluster,
                cluster_artefacts    = cluster_artefacts,
            )
            logger.info(
                "  [%s] %d clusters, %d scenarios, %d trades",
                ps.split, len(ps.preds_scaled), ps.n_scenarios, ps.n_trades,
            )

            # Step (b) — pure writers: each takes ``ps`` (or a slice
            # of it) and produces one artifact family.  No transform
            # or aggregation logic in any of these calls.
            _save_member_predictions_npz_split(eval_dir, ps)
            _save_portfolio_timeseries_parquet(eval_dir, ps)
            _save_cluster_timeseries_parquet(eval_dir, ps)
            _save_trade_metrics_parquet(
                eval_dir, ps.split,
                ps.preds_scaled_arrays(),
                ps.targets_scaled_arrays(),
                config,
            )
            _save_group_correlations_parquet(
                eval_dir, ps.split,
                ps.cluster_pred_sums,
                ps.cluster_target_sums,
                cluster_attrs,
            )
            _save_quality_parquets_for_split(
                eval_dir, ps.split, config, member_versions,
            )

            logger.info(
                "  [%s] total: %.1fs",
                ps.split, time.perf_counter() - t_split,
            )

        logger.info(
            "Evaluation artifacts saved to %s (%.2fs)",
            eval_dir, time.perf_counter() - t0,
        )


# ------------------------------------------------------------------
# Module-level helpers (stateless, no self)
# ------------------------------------------------------------------

def _write_json(path: Path, data: Any, compact: bool = False) -> None:
    """Write *data* as JSON, converting numpy types.

    Parameters
    ----------
    compact : bool
        If True, write without indentation (smaller files for large
        lists like trade metrics).
    """
    from src.rade_ml_pt.core import json_safe
    kwargs = {"default": json_safe}
    if not compact:
        kwargs["indent"] = 2
    with open(path, "w") as f:
        json.dump(data, f, **kwargs)


def _load_scenario_labels_by_split(
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]],
) -> Dict[str, List[str]]:
    """Resolve scenario labels for every split from a member's ``trade_universe.json``.

    Period-based splits mean every ensemble member shares the same
    ``train_scenarios`` / ``val_scenarios`` / ``test_scenarios`` lists, so we
    can read from whichever member we encounter first.  Returns an empty
    dict if no member carries a ``trade_universe.json`` (e.g. legacy runs);
    downstream writers must tolerate that and fall back to positional-only
    labels.

    Mirrors the member-resolution fallback in ``_save_graph_stats_parquet``.
    """
    if not member_versions:
        member_versions = config.metadata.get("job", {}).get("member_versions", {})

    if not isinstance(member_versions, dict):
        return {}

    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue
        tu_path = Path(config.registry_dir) / member_version / "trade_universe.json"
        if not tu_path.exists():
            continue
        try:
            with open(tu_path) as f:
                tu = json.load(f)
        except Exception as exc:
            logger.debug("Could not read trade_universe for '%s': %s", cid, exc)
            continue

        return {
            "train": [str(s) for s in tu.get("train_scenarios", [])],
            "val": [str(s) for s in tu.get("val_scenarios", [])],
            "test": [str(s) for s in tu.get("test_scenarios", [])],
        }

    return {}


def _build_eval_manifest(
    version:               str,
    config:                EnsembleConfig,
    cluster_trade_indices: Dict[str, Any],
    all_split_results:     Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Build the eval-run manifest dict.

    The ``artifacts`` block is the single source of truth that maps an
    artifact family to its on-disk path template and its column contract.
    Each entry carries:

    - ``schema_version``   — bump this in the writer when columns change.
    - ``path_template``    — relative to ``eval_dir``; ``{split}`` is
                             substituted for per-split artifacts, no
                             substitution for run-scoped ones.
    - ``identity_columns`` — composite primary key (and any non-measure
                             columns that identify each row).
    - ``measure_columns``  — value columns the consumer reads.

    For families whose columns are dynamic (one column per cluster-key
    in ``cluster_attributes``, per-cluster feature panels in
    ``completeness`` / ``feature_summary``), only the well-known columns
    are declared and a ``note`` records that extras may appear.

    Phase 3.1 — the manifest declares ``spaces_available`` at the run
    level; per-artifact entries that carry both spaces split their
    ``measure_columns`` into ``{scaled: [...], original: [...]}`` and
    advertise a ``spaces`` key.  Per-cluster trade-level predictions
    are persisted as NPZ shards (``members/{cid}/predictions/...``)
    not as a parquet family — see :func:`_save_member_predictions_npz`.
    Artifacts that are space-agnostic (metrics, attributes, graph
    stats, DQ) stay on the flat ``measure_columns`` shape — clients
    keying off a per-entry ``spaces`` field handle both cases
    uniformly.
    """
    return {
        "_schema_version":  _EVAL_MANIFEST_SCHEMA_VERSION,
        "version":          version,
        "evaluated_at":     datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "trade_ids":        config.all_trade_ids,
        "cluster_ids":      config.cluster_ids,
        "cluster_trade_indices": {
            k: v.tolist() if isinstance(v, np.ndarray) else list(v)
            for k, v in cluster_trade_indices.items()
        },
        "splits_available": sorted(all_split_results.keys()),
        "spaces_available": ["scaled", "original"],
        "artifacts": {
            # ── Run-scoped ─────────────────────────────────────────
            "ensemble_metrics": {
                "schema_version":   _ENSEMBLE_METRICS_SCHEMA_VERSION,
                "path_template":    "ensemble_metrics.parquet",
                "identity_columns": ["split"],
                "measure_columns":  [
                    "mae", "mse", "rmse", "max_ae", "p95_ae", "p99_ae",
                ],
            },
            "per_member_metrics": {
                "schema_version":   _PER_MEMBER_METRICS_SCHEMA_VERSION,
                "path_template":    "per_member_metrics.parquet",
                "identity_columns": ["cluster_id", "split"],
                "measure_columns":  [
                    "mae", "mse", "rmse", "max_ae", "p95_ae", "p99_ae",
                    "n_targets", "n_scenarios",
                ],
            },
            "cluster_attributes": {
                "schema_version":   _CLUSTER_ATTRIBUTES_SCHEMA_VERSION,
                "path_template":    "cluster_attributes.parquet",
                "identity_columns": ["cluster_id"],
                "measure_columns":  ["n_trades"],
                "note":             (
                    "Additional dynamic string columns appear per "
                    "cluster-key declared in the ensemble config "
                    "(e.g. currency_code, product, desk)."
                ),
            },
            "graph_stats": {
                "schema_version":   _GRAPH_STATS_SCHEMA_VERSION,
                "path_template":    "graph_stats.parquet",
                "identity_columns": ["cluster_id"],
                "measure_columns":  [
                    "n_nodes", "n_edges", "density", "mean_weight",
                ],
            },

            # ── Per-split long-format ──────────────────────────────
            "portfolio_timeseries": {
                "schema_version":   _PORTFOLIO_TIMESERIES_SCHEMA_VERSION,
                "path_template":    "portfolio_summary/portfolio_timeseries_{split}.parquet",
                "identity_columns": ["scenario_idx", "scenario_label", "split"],
                "spaces":           ["scaled", "original"],
                "measure_columns":  {
                    "scaled": [
                        "predictions", "targets",
                        "error", "abs_error", "squared_error",
                    ],
                    "original": [
                        "predictions_original", "targets_original",
                        "error_original", "abs_error_original",
                        "squared_error_original",
                    ],
                },
            },
            "cluster_timeseries": {
                "schema_version":   _CLUSTER_TIMESERIES_SCHEMA_VERSION,
                "path_template":    "cluster_summary/cluster_timeseries_{split}.parquet",
                "identity_columns": ["cluster_id", "scenario_idx", "scenario_label", "split"],
                "spaces":           ["scaled", "original"],
                "measure_columns":  {
                    "scaled": [
                        "predictions", "targets",
                        "error", "abs_error", "squared_error",
                    ],
                    "original": [
                        "predictions_original", "targets_original",
                        "error_original", "abs_error_original",
                        "squared_error_original",
                    ],
                },
            },
            "trade_metrics": {
                "schema_version":   _TRADE_METRICS_SCHEMA_VERSION,
                "path_template":    "trade_metrics/trade_metrics_{split}.parquet",
                "identity_columns": ["cluster_id", "trade_id", "split"],
                "measure_columns":  [
                    "mae", "mse", "rmse", "max_ae", "p95_ae", "p99_ae",
                    "mean_residual", "std_residual", "n_scenarios",
                ],
            },
            "group_correlations": {
                "schema_version":   _GROUP_CORRELATIONS_SCHEMA_VERSION,
                "path_template":    "group_correlations/group_correlations_{split}.parquet",
                "identity_columns": ["attribute", "group_a", "group_b", "split"],
                "measure_columns":  ["rho", "n_scenarios"],
            },
            "completeness": {
                "schema_version":   _COMPLETENESS_SCHEMA_VERSION,
                "path_template":    "quality/completeness_{split}.parquet",
                "identity_columns": ["cluster_id", "feature_name"],
                "measure_columns":  [
                    "dtype", "n_rows", "n_null", "null_rate",
                    "n_distinct", "n_zero", "n_inf", "n_nan",
                ],
            },
            "feature_summary": {
                "schema_version":   _FEATURE_SUMMARY_SCHEMA_VERSION,
                "path_template":    "quality/feature_summary_{split}.parquet",
                "identity_columns": ["cluster_id", "feature_name"],
                "measure_columns":  [
                    "count", "mean", "std",
                    "p01", "p50", "p99",
                    "min", "max",
                ],
            },
        },
    }


# ----------------------------------------------------------------------
# Phase 3.1 — original-space helpers
# ----------------------------------------------------------------------
#
# Three independent, single-responsibility helpers that the per-split
# orchestrator (:func:`_compute_original_space_per_split`) composes.
# Kept module-level (no ``self``) so they can be unit-tested in
# isolation against a fixture cluster directory.

def _load_member_target_artifacts(
    config:          EnsembleConfig,
    member_versions: Optional[Dict[str, str]],
) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """Read each cluster's ``(target_scaler, target_attributes)`` from the model store.

    Returns a mapping of ``cluster_id → (scaler, attributes)`` — plain
    tuples, no ``InferenceContext`` wrapper, no inference-pipeline
    import.  The eval pipeline owns its own reads from each member's
    version_dir, mirroring how
    :meth:`HybridGnnRnnEvalPipeline._load_cached_data` loads the same
    files in the per-member flow.

    Resolution of ``member_versions`` prefers the explicit argument,
    then falls back to ``config.metadata["job"]["member_versions"]`` —
    same chain as :func:`_load_scenario_labels_by_split` so legacy runs
    with implicit member versioning still work.

    Clusters whose artefacts are missing or unreadable are dropped from
    the returned mapping (a warning is logged); the per-split
    orchestrator falls back to scaled-only output for those clusters
    rather than failing the whole eval run.
    """
    if not member_versions:
        member_versions = (
            config.metadata.get("job", {}).get("member_versions", {}) or {}
        )
    if not isinstance(member_versions, dict):
        return {}

    artefacts: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue
        version_dir = Path(config.registry_dir) / str(member_version)
        scaler_path = version_dir / "target_scaler.pkl"
        attrs_path  = version_dir / "target_attributes.json"

        if not scaler_path.exists() or not attrs_path.exists():
            logger.warning(
                "Original-space artefacts skipped for cluster '%s' — missing "
                "target_scaler.pkl or target_attributes.json under %s",
                cid, version_dir,
            )
            continue

        try:
            scaler = joblib.load(scaler_path)
        except Exception as exc:
            logger.warning(
                "Could not load target_scaler.pkl for cluster '%s': %s — "
                "original-space skipped.", cid, exc,
            )
            continue
        try:
            with open(attrs_path) as f:
                attrs = json.load(f)
        except Exception as exc:
            logger.warning(
                "Could not load target_attributes.json for cluster '%s': %s — "
                "original-space skipped.", cid, exc,
            )
            continue

        artefacts[cid] = (scaler, attrs)

    return artefacts


def _build_trade_ids_by_cluster(
    clusters:          List[str],
    cluster_artefacts: Dict[str, Tuple[Any, Dict[str, Any]]],
    config:            EnsembleConfig,
) -> Dict[str, List[str]]:
    """Resolve the canonical trade-ID order for every cluster.

    Source preference (highest → lowest):

      1. ``scaler.feature_names_in_`` — the canonical training-time
         order the model was scaled in.  This is what every other
         artifact (NPZ ``trade_ids``, parquet columns) commits to.
      2. ``config.cluster_mapping[cid]`` — positional from the
         ensemble config; matches the scaler when both are derived
         from the same upstream universe definition.
      3. Positional last-resort ``["{cid}_trade_0", …]`` — guards
         against legacy runs where neither source is populated.

    Called once at the top of :meth:`_save_all_artifacts` so every
    per-split write sees the same labels (no risk of drift between
    splits / spaces).
    """
    out: Dict[str, List[str]] = {}
    for cid in clusters:
        pair = cluster_artefacts.get(cid)
        if pair is not None:
            ids = [
                str(t) for t in getattr(pair[0], "feature_names_in_", [])
            ]
            if ids:
                out[cid] = ids
                continue
        ids = [str(t) for t in config.cluster_mapping.get(cid, [])]
        if ids:
            out[cid] = ids
            continue
        out[cid] = []  # caller fills positional fallback once it knows n_trades
    return out


def _transform_cluster_to_original(
    cluster_id:      str,
    preds_2d:        np.ndarray,
    targets_2d:      np.ndarray,
    scenario_labels: Optional[List[str]],
    scaler:          Any,
    attributes:      Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inverse-scale + notional-sign-restore one cluster's preds + targets.

    Composes two calls to the neutral
    :func:`src.rade_ml_pt.utilities.predictions_transform.transform_to_original`
    utility — same math definition as the inference pipeline, no
    cross-pipeline imports.  The eval pipeline reads its scaler /
    attributes directly from the model store (see
    :func:`_load_member_target_artifacts`) so this helper is purely a
    composer over the shared utility.

    Returns
    -------
    preds_original_wide   : pd.DataFrame  # [n_scenarios, n_trades]
        Inverse-scaled + notional-sign-restored predictions.
    targets_original_wide : pd.DataFrame  # [n_scenarios, n_trades]
        Inverse-scaled + notional-sign-restored targets.

    Both DataFrames share the same row index (``scenario_label``) and
    column labels (trade IDs in the scaler's canonical order).  The
    scaled-space frames are *not* returned — the caller already has
    the raw ndarrays in z-space and wraps them as DataFrames with the
    same labels in :func:`_build_per_split_data` (no need to round-trip
    through the transform utility just to re-attach metadata).
    """
    from src.rade_ml_pt.utilities.predictions_transform import (
        transform_to_original,
    )

    _, preds_original_wide, _ = transform_to_original(
        cluster_id        = cluster_id,
        cluster_preds     = preds_2d,
        scenario_labels   = scenario_labels,
        scaler            = scaler,
        target_attributes = attributes,
    )
    _, targets_original_wide, _ = transform_to_original(
        cluster_id        = cluster_id,
        cluster_preds     = targets_2d,
        scenario_labels   = scenario_labels,
        scaler            = scaler,
        target_attributes = attributes,
    )
    return preds_original_wide, targets_original_wide


def _save_member_predictions_npz(
    eval_dir:        Path,
    cluster_id:      str,
    split:           str,
    space:           str,
    *,
    predictions:     np.ndarray,
    targets:         np.ndarray,
    trade_ids:       List[str],
    scenario_labels: List[str],
) -> Path:
    """Write one cluster's per-split predictions + targets NPZ shard.

    Filename convention ::

        members/{cluster_id}/predictions/
            {split}.npz             # space == "scaled" (legacy name)
            {split}_original.npz    # space == "original"

    The scaled filename intentionally has *no* ``_scaled`` suffix to
    preserve backwards compatibility with the many existing readers
    (ensemble session, prediction-stream API router, legacy UI
    backends).  Consumers locate the file via
    :meth:`ArtifactPaths.member_predictions(cluster_id, split, space)`
    so they never hard-code the naming asymmetry.

    Schema — every NPZ shard carries:

      - ``predictions``      float32 ``[n_scenarios, n_trades]``
      - ``targets``          float32 ``[n_scenarios, n_trades]``
      - ``trade_ids``        unicode ``[n_trades]``       (Phase 3.1)
      - ``scenario_labels``  unicode ``[n_scenarios]``    (Phase 3.1)

    All four arrays are pickle-free (``<U`` dtype for the string
    arrays) so existing consumers reading with
    ``np.load(path, allow_pickle=False)`` keep working unchanged —
    ``trade_ids`` and ``scenario_labels`` are additive keys that pre-3.1
    readers ignore.
    """
    member_dir = eval_dir / "members" / cluster_id / "predictions"
    member_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{split}.npz" if space == "scaled" else f"{split}_{space}.npz"
    out_path = member_dir / filename

    np.savez_compressed(
        out_path,
        predictions     = predictions.astype(np.float32),
        targets         = targets.astype(np.float32),
        trade_ids       = np.asarray(trade_ids).astype(str),
        scenario_labels = np.asarray(scenario_labels).astype(str),
    )
    return out_path


def _transform_split_to_original(
    split:                str,
    preds_by_cluster:     Dict[str, np.ndarray],
    targets_by_cluster:   Dict[str, np.ndarray],
    scenario_labels:      List[str],
    cluster_artefacts:    Dict[str, Tuple[Any, Dict[str, Any]]],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Pure compute: run ``_transform_cluster_to_original`` for every
    cluster with scaler artefacts, return the original-space wide
    DataFrames for ``preds`` and ``targets``.

    No I/O.  Clusters without a ``(scaler, attributes)`` entry in
    ``cluster_artefacts`` are silently absent from the returned dicts;
    clusters whose transform raises are logged and skipped.  Both fall
    through to NaN in the timeseries writers' ``*_original`` columns
    and to no original-space NPZ shard for that cluster.
    """
    preds_original:   Dict[str, pd.DataFrame] = {}
    targets_original: Dict[str, pd.DataFrame] = {}
    if not cluster_artefacts:
        return preds_original, targets_original

    t0 = time.perf_counter()
    for cid in sorted(preds_by_cluster):
        pair = cluster_artefacts.get(cid)
        if pair is None:
            continue
        scaler, attrs = pair
        try:
            preds_original_wide, targets_original_wide = (
                _transform_cluster_to_original(
                    cluster_id      = cid,
                    preds_2d        = preds_by_cluster[cid],
                    targets_2d      = targets_by_cluster[cid],
                    scenario_labels = scenario_labels or None,
                    scaler          = scaler,
                    attributes      = attrs,
                )
            )
        except Exception as exc:
            logger.warning(
                "  [%s] cluster '%s' original-space transform failed: "
                "%s — falling back to scaled-only for this cluster.",
                split, cid, exc,
            )
            continue
        preds_original[cid]   = preds_original_wide
        targets_original[cid] = targets_original_wide

    if preds_original:
        logger.info(
            "  [%s] original-space transforms: %.1fs (%d clusters)",
            split, time.perf_counter() - t0, len(preds_original),
        )
    return preds_original, targets_original


def _build_per_split_data(
    split:                str,
    preds_by_cluster:     Dict[str, np.ndarray],
    targets_by_cluster:   Dict[str, np.ndarray],
    scenario_labels:      List[str],
    trade_ids_by_cluster: Dict[str, List[str]],
    cluster_artefacts:    Dict[str, Tuple[Any, Dict[str, Any]]],
) -> PerSplitData:
    """Build the canonical ``PerSplitData`` bundle for one split.

    Pure compute — wraps the raw scaled ndarrays as labelled wide
    DataFrames and runs the scaled → original transform for every
    cluster with scaler artefacts.  No I/O, no side effects.

    Trade-ID resolution is precomputed by
    :func:`_build_trade_ids_by_cluster` so every per-split wrap uses
    consistent labels across splits (and across scaled / original
    spaces within a split).
    """
    n_scenarios = next(
        (v.shape[0] for v in preds_by_cluster.values()), 0,
    )
    n_trades = sum(
        v.shape[1] if v.ndim > 1 else 1
        for v in preds_by_cluster.values()
    )

    scenario_index = (
        list(scenario_labels)
        if scenario_labels
        else [str(i) for i in range(n_scenarios)]
    )

    def _wrap(arr: np.ndarray, cid: str) -> pd.DataFrame:
        # Honour the resolved trade IDs; fall back to a positional
        # last-resort if the resolver returned an empty list (legacy
        # config with no cluster_mapping).
        cols = trade_ids_by_cluster.get(cid) or []
        if not cols:
            n_t = arr.shape[1] if arr.ndim > 1 else 1
            cols = [f"{cid}_trade_{j}" for j in range(n_t)]
        return pd.DataFrame(arr, index=scenario_index, columns=cols)

    preds_scaled   = {cid: _wrap(a, cid) for cid, a in preds_by_cluster.items()}
    targets_scaled = {cid: _wrap(a, cid) for cid, a in targets_by_cluster.items()}

    preds_original, targets_original = _transform_split_to_original(
        split             = split,
        preds_by_cluster  = preds_by_cluster,
        targets_by_cluster= targets_by_cluster,
        scenario_labels   = scenario_labels,
        cluster_artefacts = cluster_artefacts,
    )

    return PerSplitData(
        split            = split,
        n_scenarios      = n_scenarios,
        n_trades         = n_trades,
        scenario_labels  = list(scenario_labels),
        preds_scaled     = preds_scaled,
        targets_scaled   = targets_scaled,
        preds_original   = preds_original,
        targets_original = targets_original,
    )


def _save_member_predictions_npz_split(
    eval_dir: Path, ps: PerSplitData,
) -> None:
    """Write the scaled NPZ for every cluster + the original NPZ for
    the subset of clusters that have scaler artefacts.

    Pure dispatch — no business logic.  Both NPZ flavours go through
    :func:`_save_member_predictions_npz` with labels pulled straight
    off the wide DataFrames so scaled / original stay in lock-step.
    """
    for cid, df in ps.preds_scaled.items():
        _save_member_predictions_npz(
            eval_dir, cid, ps.split, "scaled",
            predictions     = df.to_numpy(),
            targets         = ps.targets_scaled[cid].to_numpy(),
            trade_ids       = list(df.columns),
            scenario_labels = list(df.index),
        )
    for cid, df in ps.preds_original.items():
        _save_member_predictions_npz(
            eval_dir, cid, ps.split, "original",
            predictions     = df.to_numpy(),
            targets         = ps.targets_original[cid].to_numpy(),
            trade_ids       = list(df.columns),
            scenario_labels = list(df.index),
        )


def _save_portfolio_timeseries_parquet(
    eval_dir: Path, ps: PerSplitData,
) -> None:
    """Write ``portfolio_timeseries_{split}.parquet`` (PRISM contract §11.15.2, B1).

    Long-format columnar time series for portfolio-level predictions /
    actuals, with precomputed error fields.  Reads everything it needs
    off ``ps`` — no transform or aggregation logic in this writer.

    Schema (v3):

    - ``scenario_idx``           int32   — 0-based within this split
    - ``scenario_label``         string  — date-like label from elementary PnL index
    - ``split``                  string  — "train" / "val" / "test"
    - ``predictions``            float32 — portfolio predicted PnL (scaled space)
    - ``targets``                float32 — portfolio actual PnL (scaled space)
    - ``error``                  float32 — predictions − targets (scaled)
    - ``abs_error``              float32 — |error|
    - ``squared_error``          float32 — error²
    - ``predictions_original``   float32 — portfolio predicted PnL in original (notional) units (Phase 3.1; NaN if no scaler)
    - ``targets_original``       float32 — portfolio actual PnL in original units (Phase 3.1; NaN if no scaler)
    - ``error_original``         float32 — predictions_original − targets_original
    - ``abs_error_original``     float32 — |error_original|
    - ``squared_error_original`` float32 — error_original²

    File-level metadata: ``_schema_version`` = ``3``, ``split``.

    Schema-change history
    ---------------------
    v3 (Phase 3.1): five ``*_original`` columns appended.  The scaled
    columns are preserved unchanged so v2 consumers keep reading the
    same numbers.  When ``ps.preds_1d_original`` is ``None`` (no
    cluster contributed) the original columns are NaN-filled — the v3
    schema stays canonical regardless of coverage.

    Raises
    ------
    RuntimeError
        If ``ps.scenario_labels`` is non-empty but its length does
        not match ``ps.preds_1d``.
    """
    split = ps.split
    n = int(ps.preds_1d.shape[0])
    if n == 0:
        logger.warning(
            "  [%s] portfolio_timeseries_%s.parquet skipped (no scenarios)",
            split, split,
        )
        return

    if ps.scenario_labels and len(ps.scenario_labels) != n:
        raise RuntimeError(
            f"portfolio_timeseries_{split}.parquet: "
            f"scenario_labels length {len(ps.scenario_labels)} != "
            f"preds_1d length {n}. "
            f"Check trade_universe.json alignment with eval inputs."
        )

    if not ps.scenario_labels:
        logger.warning(
            "  [%s] no scenario_labels available; falling back to positional",
            split,
        )
        labels = [str(i) for i in range(n)]
    else:
        labels = list(ps.scenario_labels)

    preds_f32   = ps.preds_1d.astype(np.float32, copy=False)
    targets_f32 = ps.targets_1d.astype(np.float32, copy=False)
    error       = preds_f32 - targets_f32

    # Original-space columns — NaN-filled when no cluster contributed
    # original-space PnL so the v3 schema stays canonical.
    if ps.preds_1d_original is not None and ps.targets_1d_original is not None:
        preds_orig_f32   = ps.preds_1d_original.astype(np.float32, copy=False)
        targets_orig_f32 = ps.targets_1d_original.astype(np.float32, copy=False)
    else:
        preds_orig_f32   = np.full(n, np.nan, dtype=np.float32)
        targets_orig_f32 = np.full(n, np.nan, dtype=np.float32)
    error_orig = preds_orig_f32 - targets_orig_f32

    df = pd.DataFrame({
        "scenario_idx":           np.arange(n, dtype=np.int32),
        "scenario_label":         labels,
        "split":                  [split] * n,
        "predictions":            preds_f32,
        "targets":                targets_f32,
        "error":                  error,
        "abs_error":              np.abs(error),
        "squared_error":          error * error,
        "predictions_original":   preds_orig_f32,
        "targets_original":       targets_orig_f32,
        "error_original":         error_orig,
        "abs_error_original":     np.abs(error_orig),
        "squared_error_original": error_orig * error_orig,
    }).astype({
        "scenario_idx":           np.int32,
        "scenario_label":         "string",
        "split":                  "string",
        "predictions":            np.float32,
        "targets":                np.float32,
        "error":                  np.float32,
        "abs_error":              np.float32,
        "squared_error":          np.float32,
        "predictions_original":   np.float32,
        "targets_original":       np.float32,
        "error_original":         np.float32,
        "abs_error_original":     np.float32,
        "squared_error_original": np.float32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_PORTFOLIO_TIMESERIES_SCHEMA_VERSION).encode(),
        b"split": split.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_dir = eval_dir / "portfolio_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"portfolio_timeseries_{split}.parquet"
    pq.write_table(table, str(out_path))
    logger.info("  [%s] saved %s (%d rows)", split, out_path.name, n)


def _save_cluster_timeseries_parquet(
    eval_dir: Path, ps: PerSplitData,
) -> None:
    """Write ``cluster_timeseries_{split}.parquet`` (PRISM contract §11.15.2, B2).

    Long-format per-cluster × per-scenario portfolio sums.  One row
    per ``(cluster_id, scenario_idx)``; reads everything it needs off
    ``ps`` — no aggregation logic in this writer.

    Schema (v3):

    - ``cluster_id``             string  — e.g. "cluster_0"
    - ``scenario_idx``           int32   — 0-based within this split
    - ``scenario_label``         string  — date-like label from elementary PnL index
    - ``split``                  string  — "train" / "val" / "test"
    - ``predictions``            float32 — cluster portfolio predicted PnL (scaled)
    - ``targets``                float32 — cluster portfolio actual PnL (scaled)
    - ``error``                  float32 — predictions − targets (scaled)
    - ``abs_error``              float32 — |error|
    - ``squared_error``          float32 — error²
    - ``predictions_original``   float32 — cluster predicted PnL in original (notional) units (Phase 3.1)
    - ``targets_original``       float32 — cluster actual PnL in original units (Phase 3.1)
    - ``error_original``         float32 — predictions_original − targets_original
    - ``abs_error_original``     float32 — |error_original|
    - ``squared_error_original`` float32 — error_original²

    Rows are sorted by ``(cluster_id, scenario_idx)`` for read locality
    under the dominant filter-by-cluster access pattern.  File metadata:
    ``_schema_version`` = ``3``, ``split``.

    Schema-change history
    ---------------------
    v3 (Phase 3.1): five ``*_original`` columns appended.  Per-cluster
    NaN-fill is supported — a cluster missing from
    ``ps.cluster_pred_sums_original`` / ``ps.cluster_target_sums_original``
    still emits its rows but with NaN in the original columns.

    Raises
    ------
    RuntimeError
        If any cluster's series length disagrees with the others, or
        with ``ps.scenario_labels`` (when non-empty).
    """
    split = ps.split
    if not ps.cluster_pred_sums:
        logger.warning(
            "  [%s] cluster_timeseries_%s.parquet skipped (no clusters)",
            split, split,
        )
        return

    cluster_ids = sorted(ps.cluster_pred_sums.keys())
    n_scenarios = int(ps.cluster_pred_sums[cluster_ids[0]].shape[0])
    for cid in cluster_ids:
        n_p = int(ps.cluster_pred_sums[cid].shape[0])
        n_t = int(ps.cluster_target_sums[cid].shape[0])
        if n_p != n_scenarios or n_t != n_scenarios:
            raise RuntimeError(
                f"cluster_timeseries_{split}.parquet: cluster '{cid}' "
                f"length mismatch (preds={n_p}, targets={n_t}, expected={n_scenarios})"
            )

    if ps.scenario_labels and len(ps.scenario_labels) != n_scenarios:
        raise RuntimeError(
            f"cluster_timeseries_{split}.parquet: "
            f"scenario_labels length {len(ps.scenario_labels)} != "
            f"cluster series length {n_scenarios}."
        )

    if not ps.scenario_labels:
        logger.warning(
            "  [%s] no scenario_labels available; falling back to positional",
            split,
        )
        labels_per_scenario = [str(i) for i in range(n_scenarios)]
    else:
        labels_per_scenario = list(ps.scenario_labels)

    n_rows = len(cluster_ids) * n_scenarios
    cluster_col = np.repeat(np.asarray(cluster_ids, dtype=object), n_scenarios)
    scenario_idx_col = np.tile(
        np.arange(n_scenarios, dtype=np.int32), len(cluster_ids),
    )
    scenario_label_col = np.tile(
        np.asarray(labels_per_scenario, dtype=object), len(cluster_ids),
    )

    preds = np.concatenate([
        ps.cluster_pred_sums[cid].astype(np.float32, copy=False)
        for cid in cluster_ids
    ])
    actuals = np.concatenate([
        ps.cluster_target_sums[cid].astype(np.float32, copy=False)
        for cid in cluster_ids
    ])
    error = preds - actuals

    # Original-space columns — per-cluster NaN-fill where the cluster
    # had no scaler coverage; full-NaN column when no cluster did.
    def _stack_or_nan(
        sums_by_cid: Optional[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        if not sums_by_cid:
            return np.full(n_rows, np.nan, dtype=np.float32)
        out = np.empty(n_rows, dtype=np.float32)
        for i, cid in enumerate(cluster_ids):
            block = sums_by_cid.get(cid)
            start = i * n_scenarios
            end   = start + n_scenarios
            if block is None:
                out[start:end] = np.nan
            else:
                out[start:end] = np.asarray(block, dtype=np.float32)
        return out

    preds_orig   = _stack_or_nan(ps.cluster_pred_sums_original or None)
    targets_orig = _stack_or_nan(ps.cluster_target_sums_original or None)
    error_orig   = preds_orig - targets_orig

    df = pd.DataFrame({
        "cluster_id":             cluster_col,
        "scenario_idx":           scenario_idx_col,
        "scenario_label":         scenario_label_col,
        "split":                  np.full(n_rows, split, dtype=object),
        "predictions":            preds,
        "targets":                actuals,
        "error":                  error,
        "abs_error":              np.abs(error),
        "squared_error":          error * error,
        "predictions_original":   preds_orig,
        "targets_original":       targets_orig,
        "error_original":         error_orig,
        "abs_error_original":     np.abs(error_orig),
        "squared_error_original": error_orig * error_orig,
    }).astype({
        "cluster_id":             "string",
        "scenario_idx":           np.int32,
        "scenario_label":         "string",
        "split":                  "string",
        "predictions":            np.float32,
        "targets":                np.float32,
        "error":                  np.float32,
        "abs_error":              np.float32,
        "squared_error":          np.float32,
        "predictions_original":   np.float32,
        "targets_original":       np.float32,
        "error_original":         np.float32,
        "abs_error_original":     np.float32,
        "squared_error_original": np.float32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_CLUSTER_TIMESERIES_SCHEMA_VERSION).encode(),
        b"split": split.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_dir = eval_dir / "cluster_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"cluster_timeseries_{split}.parquet"
    pq.write_table(table, str(out_path))
    logger.info(
        "  [%s] saved %s (%d rows, %d clusters × %d scenarios)",
        split, out_path.name, n_rows, len(cluster_ids), n_scenarios,
    )


def _build_cluster_attributes(config: EnsembleConfig) -> Dict[str, Any]:
    """Build ``{cluster_id: {attr: value, n_trades: int}}``.

    Sources cluster attributes via :meth:`EnsembleConfig.get_cluster_keys_for_router`
    so all three resolution paths are honoured uniformly:

    1. ``cluster_keys`` — pre-built dict-of-dicts (``{cid: {attr: val}}``).
    2. ``cluster_key`` + ``cluster_key_values`` — top-level lists zipped per cid.
    3. ``metadata['job']['cluster_key']`` + ``metadata['job']['cluster_key_values']``
       — fallback for configs serialised by the ensemble training pipeline.

    This matches the train-side ``_build_member_summary`` logic, so the
    Governance / Cluster-Attributes / Group-Correlations parquets always
    see the same attribute view that the TradeRouter uses at inference.
    Without this, a config that only populates ``cluster_keys`` (no
    ``cluster_key``/``cluster_key_values``) would silently produce
    attribute-less clusters here and skip ``group_correlations.parquet``
    with no visible diagnostic.
    """
    cluster_keys = config.get_cluster_keys_for_router() or {}
    attrs: Dict[str, Any] = {}
    for cid in config.cluster_ids:
        entry: Dict[str, Any] = dict(cluster_keys.get(cid) or {})
        entry["n_trades"] = len(config.cluster_mapping.get(cid, []))
        attrs[cid] = entry
    return attrs


def _save_trade_metrics_parquet(
    eval_dir: Path,
    split: str,
    preds: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    config: EnsembleConfig,
) -> None:
    """Write ``trade_metrics_{split}.parquet`` (PRISM contract §11.15.2, B3).

    One row per trade, aggregated across all scenarios in this split.
    Vectorised across trades within each cluster to avoid per-trade
    Python loops.  Replaces the legacy ``trade_metrics/{split}.json``.

    Schema (v1):

    - ``cluster_id``     string
    - ``trade_id``       string
    - ``split``          string
    - ``mae``            float32
    - ``mse``            float32
    - ``rmse``           float32
    - ``max_ae``         float32
    - ``p95_ae``         float32
    - ``p99_ae``         float32
    - ``mean_residual``  float32
    - ``std_residual``   float32
    - ``n_scenarios``    int32
    """
    if not preds:
        logger.warning(
            "  [%s] trade_metrics_%s.parquet skipped (no clusters)",
            split, split,
        )
        return

    rows = []
    for cid in sorted(preds):
        p, t = preds[cid], targets[cid]
        if p.ndim == 1:
            p = p.reshape(-1, 1)
            t = t.reshape(-1, 1)

        residuals = p - t
        abs_err = np.abs(residuals)

        mae = np.mean(abs_err, axis=0)
        mse = np.mean(residuals ** 2, axis=0)
        rmse = np.sqrt(mse)
        max_ae = np.max(abs_err, axis=0)
        p95_ae = np.percentile(abs_err, 95, axis=0)
        p99_ae = np.percentile(abs_err, 99, axis=0)
        mean_res = np.mean(residuals, axis=0)
        std_res = np.std(residuals, axis=0)
        n_scenarios = int(p.shape[0])

        trade_ids = config.cluster_mapping.get(cid, [])
        for j in range(p.shape[1]):
            tid = str(trade_ids[j]) if j < len(trade_ids) else f"{cid}_trade_{j}"
            rows.append({
                "cluster_id": str(cid),
                "trade_id": tid,
                "split": split,
                "mae": float(mae[j]),
                "mse": float(mse[j]),
                "rmse": float(rmse[j]),
                "max_ae": float(max_ae[j]),
                "p95_ae": float(p95_ae[j]),
                "p99_ae": float(p99_ae[j]),
                "mean_residual": float(mean_res[j]),
                "std_residual": float(std_res[j]),
                "n_scenarios": n_scenarios,
            })

    df = pd.DataFrame(rows).astype({
        "cluster_id": "string",
        "trade_id": "string",
        "split": "string",
        "mae": np.float32,
        "mse": np.float32,
        "rmse": np.float32,
        "max_ae": np.float32,
        "p95_ae": np.float32,
        "p99_ae": np.float32,
        "mean_residual": np.float32,
        "std_residual": np.float32,
        "n_scenarios": np.int32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_TRADE_METRICS_SCHEMA_VERSION).encode(),
        b"split": split.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_dir = eval_dir / "trade_metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"trade_metrics_{split}.parquet"
    pq.write_table(table, str(out_path))
    logger.info(
        "  [%s] saved %s (%d rows, %d clusters)",
        split, out_path.name, len(df), len(preds),
    )


def _save_ensemble_metrics_parquet(
    eval_dir: Path,
    all_split_results: Dict[str, Dict[str, Any]],
) -> None:
    """Write ``ensemble_metrics.parquet`` (PRISM contract §11.15.2, B4).

    One row per split with portfolio-level aggregate metrics.  Replaces
    the per-split ``ensemble_metrics{_split}.json`` legacy files.

    Schema (v1):

    - ``split``   string
    - ``mae``, ``mse``, ``rmse``, ``max_ae``, ``p95_ae``, ``p99_ae``  float32
    """
    rows = []
    for split, results in all_split_results.items():
        em = results.get("ensemble_metrics") or {}
        if not em:
            continue
        rows.append({
            "split": split,
            "mae": float(em.get("mae", np.nan)),
            "mse": float(em.get("mse", np.nan)),
            "rmse": float(em.get("rmse", np.nan)),
            "max_ae": float(em.get("max_ae", np.nan)),
            "p95_ae": float(em.get("p95_ae", np.nan)),
            "p99_ae": float(em.get("p99_ae", np.nan)),
        })

    if not rows:
        logger.warning("ensemble_metrics.parquet skipped (no metrics)")
        return

    df = pd.DataFrame(rows).astype({
        "split": "string",
        "mae": np.float32,
        "mse": np.float32,
        "rmse": np.float32,
        "max_ae": np.float32,
        "p95_ae": np.float32,
        "p99_ae": np.float32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_ENSEMBLE_METRICS_SCHEMA_VERSION).encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = eval_dir / "ensemble_metrics.parquet"
    pq.write_table(table, str(out_path))
    logger.info("  saved %s (%d splits)", out_path.name, len(df))


def _save_per_member_metrics_parquet(
    eval_dir: Path,
    all_split_results: Dict[str, Dict[str, Any]],
) -> None:
    """Write ``per_member_metrics.parquet`` (PRISM contract §11.15.2, B5).

    One row per ``(cluster_id, split)`` with per-member aggregate
    metrics.  Replaces the per-split ``per_member_metrics{_split}.json``
    legacy files.

    Schema (v1):

    - ``cluster_id``   string
    - ``split``        string
    - ``mae``, ``mse``, ``rmse``, ``max_ae``, ``p95_ae``, ``p99_ae``  float32
    - ``n_targets``    int32
    - ``n_scenarios``  int32
    """
    rows = []
    for split, results in all_split_results.items():
        pm = results.get("per_member_metrics") or {}
        for cid, m in sorted(pm.items()):
            rows.append({
                "cluster_id": str(cid),
                "split": split,
                "mae": float(m.get("mae", np.nan)),
                "mse": float(m.get("mse", np.nan)),
                "rmse": float(m.get("rmse", np.nan)),
                "max_ae": float(m.get("max_ae", np.nan)),
                "p95_ae": float(m.get("p95_ae", np.nan)),
                "p99_ae": float(m.get("p99_ae", np.nan)),
                "n_targets": int(m.get("n_targets", 0)),
                "n_scenarios": int(m.get("n_scenarios", 0)),
            })

    if not rows:
        logger.warning("per_member_metrics.parquet skipped (no metrics)")
        return

    df = pd.DataFrame(rows).astype({
        "cluster_id": "string",
        "split": "string",
        "mae": np.float32,
        "mse": np.float32,
        "rmse": np.float32,
        "max_ae": np.float32,
        "p95_ae": np.float32,
        "p99_ae": np.float32,
        "n_targets": np.int32,
        "n_scenarios": np.int32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_PER_MEMBER_METRICS_SCHEMA_VERSION).encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = eval_dir / "per_member_metrics.parquet"
    pq.write_table(table, str(out_path))
    logger.info("  saved %s (%d rows)", out_path.name, len(df))


def _save_group_correlations_parquet(
    eval_dir: Path,
    split: str,
    cluster_pred_sums: Dict[str, np.ndarray],
    cluster_target_sums: Dict[str, np.ndarray],
    cluster_attrs: Dict[str, Any],
) -> None:
    """Write ``group_correlations_{split}.parquet`` (PRISM contract §11.15.2, B6).

    Long-format upper-triangular pairwise residual correlations between
    groups defined by each cluster attribute (ccy, product, desk, etc.).
    Replaces ``group_correlations/{split}.json``.

    Schema (v1):

    - ``attribute``     string  — e.g. "currency_code"
    - ``group_a``       string  — e.g. "GBP"
    - ``group_b``       string  — e.g. "USD"
    - ``split``         string
    - ``rho``           float32 — Pearson correlation of residual series
    - ``n_scenarios``   int32

    Attributes with fewer than two distinct groups are skipped (no
    cross-group correlation possible).
    """
    if not cluster_attrs or not cluster_pred_sums:
        logger.warning(
            "  [%s] group_correlations_%s.parquet skipped: "
            "cluster_attrs empty=%s, cluster_pred_sums empty=%s",
            split, split, not cluster_attrs, not cluster_pred_sums,
        )
        return

    attr_keys = {
        k for ca in cluster_attrs.values()
        for k in ca if k != "n_trades"
    }
    if not attr_keys:
        logger.warning(
            "  [%s] group_correlations_%s.parquet skipped: "
            "no cluster attributes beyond n_trades. Check that "
            "EnsembleConfig.cluster_key / cluster_key_values are populated.",
            split, split,
        )
        return

    rows = []
    for attr in sorted(attr_keys):
        groups: Dict[str, list] = {}
        for cid, ca in cluster_attrs.items():
            val = ca.get(attr)
            if val is not None and cid in cluster_pred_sums:
                groups.setdefault(str(val), []).append(cid)

        if len(groups) < 2:
            continue

        group_residuals: Dict[str, np.ndarray] = {}
        for grp, cids in groups.items():
            if not cids:
                continue
            grp_pred = np.add.reduce([cluster_pred_sums[c] for c in cids])
            grp_tgt = np.add.reduce([cluster_target_sums[c] for c in cids])
            group_residuals[grp] = grp_pred - grp_tgt

        if len(group_residuals) < 2:
            continue

        df_res = pd.DataFrame(group_residuals)
        corr = df_res.corr()
        n_scen = int(df_res.shape[0])

        cols = list(corr.columns)
        for i, g_a in enumerate(cols):
            for j in range(i + 1, len(cols)):
                g_b = cols[j]
                rho = corr.iloc[i, j]
                if pd.isna(rho):
                    continue
                rows.append({
                    "attribute": attr,
                    "group_a": str(g_a),
                    "group_b": str(g_b),
                    "split": split,
                    "rho": float(rho),
                    "n_scenarios": n_scen,
                })

    if not rows:
        logger.info(
            "  [%s] group_correlations_%s.parquet skipped (no pairs)",
            split, split,
        )
        return

    df = pd.DataFrame(rows).astype({
        "attribute": "string",
        "group_a": "string",
        "group_b": "string",
        "split": "string",
        "rho": np.float32,
        "n_scenarios": np.int32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_GROUP_CORRELATIONS_SCHEMA_VERSION).encode(),
        b"split": split.encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_dir = eval_dir / "group_correlations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"group_correlations_{split}.parquet"
    pq.write_table(table, str(out_path))
    logger.info(
        "  [%s] saved %s (%d pairs across %d attrs)",
        split, out_path.name, len(df), len(attr_keys),
    )


def _save_cluster_attributes_parquet(
    eval_dir: Path,
    cluster_attrs: Dict[str, Any],
) -> None:
    """Write ``cluster_attributes.parquet`` (PRISM contract §11.15.2, B7).

    One row per cluster in wide format.  Columns are ``cluster_id``,
    ``n_trades``, plus one string column per cluster-key present in the
    ensemble config (asset class, product, currency, desk, etc.).

    Replaces the legacy ``cluster_attributes.json``.
    """
    if not cluster_attrs:
        logger.warning("cluster_attributes.parquet skipped (empty attrs)")
        return

    attr_cols = sorted({
        k for ca in cluster_attrs.values() for k in ca if k != "n_trades"
    })

    rows = []
    for cid in sorted(cluster_attrs):
        ca = cluster_attrs[cid]
        row: Dict[str, Any] = {
            "cluster_id": str(cid),
            "n_trades": int(ca.get("n_trades", 0)),
        }
        for k in attr_cols:
            val = ca.get(k)
            row[k] = None if val is None else str(val)
        rows.append(row)

    df = pd.DataFrame(rows)
    astype_map: Dict[str, Any] = {"cluster_id": "string", "n_trades": np.int32}
    for k in attr_cols:
        astype_map[k] = "string"
    df = df.astype(astype_map)

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_CLUSTER_ATTRIBUTES_SCHEMA_VERSION).encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = eval_dir / "cluster_attributes.parquet"
    pq.write_table(table, str(out_path))
    logger.info(
        "  saved %s (%d clusters, %d attrs)",
        out_path.name, len(df), len(attr_cols),
    )


def _save_graph_stats_parquet(
    eval_dir: Path,
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]] = None,
) -> None:
    """Write ``graph_stats.parquet`` (PRISM contract §11.15.2, B8).

    One row per cluster.  Loads each member's ``graph_results.joblib``
    from the registry and extracts node / edge counts, density, and
    mean edge weight.  Replaces ``graph_stats.json``.

    Schema (v1):

    - ``cluster_id``    string
    - ``n_nodes``       int32
    - ``n_edges``       int32
    - ``density``       float32
    - ``mean_weight``   float32

    Missing / unreadable member joblibs default to zeros so the contract
    always emits one row per cluster; consumers can filter on
    ``n_nodes > 0`` to drop placeholders.
    """
    if not member_versions:
        member_versions = config.metadata.get("job", {}).get("member_versions", {})

    rows = []
    for cid in config.cluster_ids:
        n_nodes = 0
        n_edges = 0
        density = 0.0
        mean_w = 0.0

        member_version = (
            member_versions.get(cid) if isinstance(member_versions, dict) else None
        )
        if member_version:
            graph_path = (
                Path(config.registry_dir) / member_version / "graph_results.joblib"
            )
            if graph_path.exists():
                try:
                    import joblib
                    gr = joblib.load(graph_path)
                    values = gr.get("sparse_values")
                    shape = gr.get("sparse_shape", [0, 0])
                    n_nodes = int(shape[0]) if shape[0] > 0 else 0
                    n_edges = (
                        int(len(np.asarray(values))) if values is not None else 0
                    )
                    if n_nodes > 0:
                        density = n_edges / (n_nodes * n_nodes)
                    if values is not None and n_edges > 0:
                        mean_w = float(np.mean(values))
                except Exception as exc:
                    logger.debug(
                        "Could not read graph stats for '%s': %s", cid, exc,
                    )

        rows.append({
            "cluster_id": str(cid),
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "density": density,
            "mean_weight": mean_w,
        })

    df = pd.DataFrame(rows).astype({
        "cluster_id": "string",
        "n_nodes": np.int32,
        "n_edges": np.int32,
        "density": np.float32,
        "mean_weight": np.float32,
    })

    table = pa.table(df)
    schema_metadata = {
        **(table.schema.metadata or {}),
        b"_schema_version": str(_GRAPH_STATS_SCHEMA_VERSION).encode(),
    }
    table = table.replace_schema_metadata(schema_metadata)

    out_path = eval_dir / "graph_stats.parquet"
    pq.write_table(table, str(out_path))
    logger.info("  saved %s (%d clusters)", out_path.name, len(df))


def _copy_member_graph_artifacts(
    eval_dir: Path,
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]] = None,
) -> None:
    """Stage per-member graph & training artefacts into the eval directory.

    For each cluster, copies the following files from
    ``{registry_dir}/{member_version}/`` into
    ``{eval_dir}/members/{cluster_id}/``:

    * ``graph_results.joblib`` — sparse adjacency used by the Trade-Graph
      UI tab (``sparse_indices`` / ``sparse_values`` / ``sparse_shape``).
    * ``trade_universe.json`` — target / elementary trade split used to
      colour nodes and populate the Selected-Trade card.
    * ``training_curves.parquet`` — per-epoch ``train_loss`` (and any
      additional per-epoch series the trainer emitted, e.g. ``val_loss``,
      ``mae``, ``val_mae``, …).  Consumed by the Cluster Deep-Dive
      training-curves chart.

    All three files are already referenced by the member artefacts in
    the registry; copying them into the eval bundle makes the evaluation
    directory self-contained (shippable to a read-only environment
    without the training registry) and keeps the API a pure reader over
    ``artifacts_dir``.  This matches the pattern set by
    ``members/{cluster_id}/predictions/{split}.npz``.

    Missing source files are logged at debug and skipped — the UI
    already tolerates the absence of trade-graph / training-curve data
    gracefully, so legacy runs without these files continue to work.
    """
    if not member_versions:
        member_versions = config.metadata.get("job", {}).get("member_versions", {})

    if not isinstance(member_versions, dict) or not member_versions:
        logger.info("  skipping graph-artefact staging (no member versions)")
        return

    files_to_copy = (
        "graph_results.joblib",
        "trade_universe.json",
        "training_curves.parquet",
        # Staged for the Cluster Deep-Dive "Elementary PnL Explorer"
        # row.  Elementary trades are *model inputs* (raw PnL of the
        # hedge instruments / atomic legs), so the chart plots the
        # parquet values directly — there is no prediction / target
        # distinction.  Same staging pattern as ``training_curves``:
        # copy under ``members/{cid}/`` so the evaluation directory
        # stays self-contained without the training registry.
        "elementary_pnl.parquet",
    )
    n_clusters_staged = 0
    n_files_staged = 0

    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue

        src_dir = Path(config.registry_dir) / member_version
        dst_dir = eval_dir / "members" / cid
        dst_dir.mkdir(parents=True, exist_ok=True)

        cluster_files_staged = 0
        for filename in files_to_copy:
            src = src_dir / filename
            if not src.exists():
                logger.debug(
                    "  member graph artefact missing: %s (cluster=%s)",
                    src, cid,
                )
                continue
            try:
                shutil.copy2(src, dst_dir / filename)
                cluster_files_staged += 1
            except Exception as exc:
                logger.warning(
                    "  could not stage %s for cluster '%s': %s",
                    filename, cid, exc,
                )

        if cluster_files_staged > 0:
            n_clusters_staged += 1
            n_files_staged += cluster_files_staged

    logger.info(
        "  staged member graph + training artefacts: %d files across "
        "%d clusters", n_files_staged, n_clusters_staged,
    )


def _save_quality_parquets_for_split(
    eval_dir: Path,
    split: str,
    config: EnsembleConfig,
    member_versions: Optional[Dict[str, str]],
) -> None:
    """Write ``quality/completeness_{split}.parquet`` and
    ``quality/feature_summary_{split}.parquet`` combined across clusters
    (PRISM contract §11.15.2, Phase 5e).

    For each cluster, loads ``elementary_pnl.parquet`` and the split
    indices recorded in ``trade_universe.json`` under ``{split}_indices``,
    builds per-feature F2 / F4 DataFrames via the quality helpers, and
    concatenates them into one ensemble-level parquet per flavour.
    Clusters with no features for this split (empty indices, missing
    parquet, unreadable) are skipped with a debug/warning log; the
    parquets are still written if at least one member contributes.
    """
    from src.rade_ml_pt.quality.completeness import build_completeness_df
    from src.rade_ml_pt.quality.summary import build_feature_summary_df

    if not member_versions:
        member_versions = config.metadata.get("job", {}).get("member_versions", {})

    if not isinstance(member_versions, dict):
        logger.warning(
            "  [%s] DQ skipped: member_versions is %s, expected dict",
            split, type(member_versions).__name__,
        )
        return

    completeness_frames: List[pd.DataFrame] = []
    feature_summary_frames: List[pd.DataFrame] = []

    for cid in config.cluster_ids:
        member_version = member_versions.get(cid)
        if not member_version:
            continue

        version_dir = Path(config.registry_dir) / member_version
        elem_path = version_dir / "elementary_pnl.parquet"
        tu_path = version_dir / "trade_universe.json"
        if not elem_path.exists() or not tu_path.exists():
            logger.debug(
                "Skipping DQ for '%s' (%s): missing elementary_pnl or trade_universe",
                cid, split,
            )
            continue

        try:
            with open(tu_path) as f:
                tu = json.load(f)

            # ``trade_universe.json`` written by the hybrid_gnn_rnn training
            # pipeline is always a JSON object (dict).  Older / hand-edited
            # artefacts can drift to a list shape — guard explicitly so we
            # surface a clear log line instead of an opaque
            # ``'list' object has no attribute 'get'`` AttributeError.
            if not isinstance(tu, dict):
                logger.warning(
                    "Skipping DQ for '%s' (%s): trade_universe.json is %s, "
                    "expected dict (path=%s)",
                    cid, split, type(tu).__name__, tu_path,
                )
                continue

            split_indices = tu.get(f"{split}_indices") or []
            if len(split_indices) == 0:
                continue

            elem_df = pd.read_parquet(elem_path)
            split_features = elem_df.iloc[np.asarray(split_indices)]

            completeness_frames.append(
                build_completeness_df(split_features, cluster_id=cid)
            )
            feature_summary_frames.append(
                build_feature_summary_df(split_features, cluster_id=cid)
            )
        except Exception as exc:
            logger.warning(
                "Could not build DQ for '%s' (%s): %s: %s",
                cid, split, type(exc).__name__, exc,
            )

    quality_dir = eval_dir / "quality"
    quality_dir.mkdir(parents=True, exist_ok=True)

    if completeness_frames:
        c_df = pd.concat(completeness_frames, ignore_index=True)
        c_table = pa.table(c_df).replace_schema_metadata({
            b"_schema_version": str(_COMPLETENESS_SCHEMA_VERSION).encode(),
            b"split": split.encode(),
        })
        pq.write_table(
            c_table, str(quality_dir / f"completeness_{split}.parquet"),
        )
        logger.info(
            "  [%s] saved completeness_%s.parquet (%d clusters, %d rows)",
            split, split, len(completeness_frames), len(c_df),
        )
    else:
        logger.warning("  [%s] no DQ completeness rows", split)

    if feature_summary_frames:
        f_df = pd.concat(feature_summary_frames, ignore_index=True)
        f_table = pa.table(f_df).replace_schema_metadata({
            b"_schema_version": str(_FEATURE_SUMMARY_SCHEMA_VERSION).encode(),
            b"split": split.encode(),
        })
        pq.write_table(
            f_table, str(quality_dir / f"feature_summary_{split}.parquet"),
        )
        logger.info(
            "  [%s] saved feature_summary_%s.parquet (%d clusters, %d rows)",
            split, split, len(feature_summary_frames), len(f_df),
        )
    else:
        logger.warning("  [%s] no DQ feature-summary rows", split)
