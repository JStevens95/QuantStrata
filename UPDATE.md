# DB-Ready Ensemble Analytics — Source Code Update

This file contains the **complete source code** for every file created or modified as part of the DB-ready ensemble analytics implementation. Copy each file to the corresponding path in your project.

## How to use

1. For **MODIFIED** files — replace the existing file entirely with the contents below.
2. For **NEW** files — create the file at the specified path and paste the contents.
3. After copying, ensure the `db_ready/` artifacts exist by running the ensemble evaluation pipeline with `save_db_artifacts=True` (the default).
4. Optionally publish to SQLite: `from src.rade_ml_pt.ensemble.publish_to_db import publish_to_sqlite; publish_to_sqlite("/path/to/db_ready")`

---

## File: src/rade_ml_pt/ensemble/config.py
**Action:** MODIFIED — added `save_db_artifacts: bool = True` field

```python
"""
Ensemble configuration dataclasses.

``EnsembleConfig`` aggregates per-cluster member configs, trade-to-cluster
mapping, aggregation strategy, and infrastructure paths.  It is the single
config object consumed by all ensemble pipelines and the EnsembleBuilder.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.rade_ml_pt.core import json_safe
from src.rade_ml_pt.pipelines.config import PipelineConfig


@dataclass
class EnsembleConfig:
    """
    Top-level configuration for an ensemble of model members.

    Attributes
    ----------
    member_configs : dict
        ``{cluster_id: PipelineConfig_dict}`` — per-cluster pipeline config.
    pipeline_class : dict
        ``{cluster_id: dotpath_str}`` — pipeline class for each cluster.
        Omit or set to ``None`` for a cluster to use the default
        (``HybridGnnRnnTrainPipeline``).
    cluster_mapping : dict
        ``{cluster_id: [trade_id, ...]}`` — assigns every target trade to
        exactly one cluster.
    cluster_keys : dict or None
        ``{cluster_id: {attr_name: value, ...}}`` — optional attribute-based
        routing for new trades. Can be set directly or derived from
        ``cluster_key`` + ``cluster_key_values`` via ``get_cluster_keys_for_router()``.
    cluster_key : list or None
        Shared list of attribute names that define routing, e.g. ``["ccy", "desk", "product"]``.
        Used with ``cluster_key_values`` to build the key dict per cluster.
    cluster_key_values : dict or None
        ``{cluster_id: [value_ccy, value_desk, value_product, ...]}`` — per-cluster values
        in the same order as ``cluster_key``. E.g. cluster_0 = ``["GBP", "FLOW_RATES", "EUROPEAN"]``
        means cluster_0 has ccy=GBP, desk=FLOW_RATES, product=EUROPEAN.
    aggregation : str
        Aggregation strategy: ``"concat"`` (disjoint clusters) or
        ``"weighted_mean"`` (overlapping clusters).
    weights : dict or None
        ``{cluster_id: float}`` — member weights for weighted-mean
        aggregation.  Ignored when *aggregation* is ``"concat"``.
    execution_strategy : str
        How to execute per-member pipelines.  ``"sequential"`` runs one at
        a time (default).  Future values: ``"process_pool"`` (multi-CPU),
        ``"gpu_parallel"`` (multi-GPU), ``"distributed"`` (Ray / cloud).
    max_workers : int or None
        Maximum number of parallel workers for ``process_pool`` and
        ``gpu_parallel``.  ``None`` means one worker per cluster.
    gpu_device_ids : list of int or None
        Explicit GPU device IDs for ``gpu_parallel`` strategy.  ``None``
        means use all available GPUs via ``torch.cuda.device_count()``.
        Ignored by other strategies.
    registry_dir : str or None
        Root directory for ensemble and member registries.
    artifacts_dir : str or None
        Root directory for ensemble artifacts (plots, metrics, predictions).
    metadata : dict
        Arbitrary key-value pairs forwarded into run records.
    """

    member_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pipeline_class: Dict[str, Optional[str]] = field(default_factory=dict)
    cluster_mapping: Dict[str, List[str]] = field(default_factory=dict)
    cluster_keys: Optional[Dict[str, Dict[str, Any]]] = None
    cluster_key: Optional[List[str]] = None
    cluster_key_values: Optional[Dict[str, List[Any]]] = None
    aggregation: str = "concat"
    weights: Optional[Dict[str, float]] = None
    execution_strategy: str = "sequential"
    max_workers: Optional[int] = None
    gpu_device_ids: Optional[List[int]] = None
    registry_dir: Optional[str] = None
    artifacts_dir: Optional[str] = None
    save_db_artifacts: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def cluster_ids(self) -> List[str]:
        """Ordered list of cluster identifiers."""
        return sorted(self.cluster_mapping.keys())

    @property
    def n_members(self) -> int:
        return len(self.cluster_mapping)

    @property
    def all_trade_ids(self) -> List[str]:
        """Flat list of every trade ID across all clusters."""
        ids: List[str] = []
        for cid in self.cluster_ids:
            ids.extend(self.cluster_mapping[cid])
        return ids

    def get_cluster_keys_for_router(self) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Return ``{cluster_id: {attr_name: value, ...}}`` for TradeRouter.

        Resolution order:

        1. ``cluster_keys`` (top-level, pre-built dict).
        2. ``cluster_key`` + ``cluster_key_values`` (top-level lists).
        3. ``metadata['job']['cluster_key']`` + ``metadata['job']['cluster_key_values']``
           (fallback for configs built by ensemble training pipelines).
        """
        if isinstance(self.cluster_keys, dict):
            return self.cluster_keys

        job = self.metadata.get("job", {})
        ck = self.cluster_key or self.cluster_keys or job.get("cluster_key")
        ckv = self.cluster_key_values or job.get("cluster_key_values")

        if ck is not None and ckv is not None:
            return {
                cid: dict(zip(ck, values))
                for cid, values in ckv.items()
            }
        return None

    def get_member_pipeline_config(self, cluster_id: str) -> PipelineConfig:
        """Build a ``PipelineConfig`` for one member from its dict representation."""
        raw = self.member_configs.get(cluster_id, {})
        return PipelineConfig(
            training_config=raw.get("training_config"),
            data_config=raw.get("data_config"),
            model_config=raw.get("model_config"),
            registry_dir=self.registry_dir,
            tracking_dir=raw.get("tracking_dir"),
            artifacts_dir=self.artifacts_dir,
            version_or_tag=raw.get("version_or_tag", "latest"),
            metadata={
                **raw.get("metadata", {}),
                "cluster_id": cluster_id,
                "trade_ids": self.cluster_mapping.get(cluster_id, []),
            },
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EnsembleConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=json_safe)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "EnsembleConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "EnsembleConfig":
        """Load from a YAML file (requires ``pyyaml``)."""
        import yaml
        from src.rade_ml_pt.core.config import sanitize_yaml_values
        with open(path, "r") as f:
            return cls.from_dict(sanitize_yaml_values(yaml.safe_load(f)))
```

---

## File: src/rade_ml_pt/pipelines/ensemble/eval.py
**Action:** MODIFIED — added `_save_artifacts_db()` method and call in `run()`

```python
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

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
        split_preds, split_targets, split_metrics = (
            self._dispatch_evaluation(config, ensemble, registry, member_versions)
        )

        # Compute metrics for every available split.
        all_split_results: Dict[str, Dict[str, Any]] = {}
        combined_arrays: Dict[str, Dict[str, np.ndarray]] = {}

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
                    combined_arrays[split] = {
                        "predictions": c_preds,
                        "targets": c_targets,
                    }
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
            self._save_artifacts(
                resolved_version, all_split_results, config,
                split_preds, split_targets, combined_arrays,
            )
            if config.save_db_artifacts:
                self._save_artifacts_db(
                    resolved_version, all_split_results, config,
                    split_preds, split_targets, combined_arrays,
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

    def _save_artifacts(
        self,
        version: str,
        all_split_results: Dict[str, Dict[str, Any]],
        config: EnsembleConfig,
        split_preds: Dict[str, Dict[str, np.ndarray]],
        split_targets: Dict[str, Dict[str, np.ndarray]],
        combined_arrays: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """Save evaluation artifacts for all splits.

        Primary split (test) files are written without a suffix for backward
        compatibility.  Additional splits get a ``_<split>`` suffix.

        Also persists per-member and combined prediction/target arrays as
        ``.npz`` files, plus a ``manifest.json`` that maps trade IDs to
        cluster positions — enough for the dashboard to reconstruct
        portfolio-level views without re-running evaluation.
        """
        eval_dir = Path(self.config.artifacts_dir) / "ensemble" / version / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        cluster_trade_indices = EnsembleBuilder._build_cluster_trade_indices(config)

        # --- Manifest (written once, shared across splits) ---------------
        manifest = {
            "trade_ids": config.all_trade_ids,
            "cluster_ids": config.cluster_ids,
            "cluster_trade_indices": {
                k: v.tolist() if isinstance(v, np.ndarray) else list(v)
                for k, v in cluster_trade_indices.items()
            },
            "splits_available": sorted(all_split_results.keys()),
        }
        with open(eval_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        # --- Per-split artifacts -----------------------------------------
        for split, results in all_split_results.items():
            suffix = "" if split == _PRIMARY_SPLIT else f"_{split}"

            with open(eval_dir / f"ensemble_metrics{suffix}.json", "w") as f:
                json.dump(results["ensemble_metrics"], f, indent=2)

            with open(eval_dir / f"per_member_metrics{suffix}.json", "w") as f:
                json.dump(results["per_member_metrics"], f, indent=2)

            with open(eval_dir / f"member_rollup{suffix}.json", "w") as f:
                json.dump(results["member_summary"], f, indent=2)

            # Per-member prediction/target arrays
            if split in split_preds:
                for cid in split_preds[split]:
                    member_dir = eval_dir / "members" / cid / "predictions"
                    member_dir.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(
                        member_dir / f"{split}.npz",
                        predictions=split_preds[split][cid],
                        targets=split_targets[split][cid],
                    )

            # Combined (portfolio-level) arrays
            if split in combined_arrays:
                combined_dir = eval_dir / "combined"
                combined_dir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    combined_dir / f"{split}.npz",
                    predictions=combined_arrays[split]["predictions"],
                    targets=combined_arrays[split]["targets"],
                )

            # Plots
            try:
                from src.rade_ml_pt.ensemble.plots import save_ensemble_plots
                plots_dir = eval_dir / "plots" / split
                save_ensemble_plots(
                    results["per_member_metrics"], plots_dir,
                    results["ensemble_metrics"],
                )
                logger.info("Saved %s evaluation plots to %s", split, plots_dir)
            except Exception as exc:
                logger.warning("Could not generate %s plots: %s", split, exc)

        logger.info("Ensemble evaluation artifacts saved to %s", eval_dir)

    # ------------------------------------------------------------------
    # DB-ready artifact persistence
    # ------------------------------------------------------------------

    def _save_artifacts_db(
        self,
        version: str,
        all_split_results: Dict[str, Dict[str, Any]],
        config: EnsembleConfig,
        split_preds: Dict[str, Dict[str, np.ndarray]],
        split_targets: Dict[str, Dict[str, np.ndarray]],
        combined_arrays: Dict[str, Dict[str, np.ndarray]],
    ) -> None:
        """Save pre-computed, dashboard-optimised artifacts to ``db_ready/``.

        These files provide instant loading for the dashboard without
        constructing the expensive ``GlobalPredictionStore``.  All data
        is stored in JSON or compressed NPZ — portable across languages
        and ready for ingestion into a relational database.

        Parameters
        ----------
        version : str
            Resolved ensemble version string.
        all_split_results : dict
            ``{split: {"ensemble_metrics", "per_member_metrics", ...}}``.
        config : EnsembleConfig
            Ensemble configuration used for this evaluation.
        split_preds : dict
            ``{split: {cluster_id: ndarray[n_scenarios, n_trades]}}``.
        split_targets : dict
            Same structure as *split_preds*.
        combined_arrays : dict
            ``{split: {"predictions": ndarray, "targets": ndarray}}``.
        """
        t0 = time.perf_counter()
        db_dir = Path(self.config.artifacts_dir) / "ensemble" / version / "db_ready"
        db_dir.mkdir(parents=True, exist_ok=True)

        # ── Ensemble version metadata ────────────────────────────────
        version_meta = {
            "version": version,
            "n_clusters": config.n_members,
            "n_trades": len(config.all_trade_ids),
            "aggregation": config.aggregation,
            "execution_strategy": config.execution_strategy,
            "splits": sorted(all_split_results.keys()),
        }
        self._write_json(db_dir / "ensemble_version.json", version_meta)

        # ── Cluster attributes ───────────────────────────────────────
        cluster_attrs = self._build_cluster_attributes(config)
        self._write_json(db_dir / "cluster_attributes.json", cluster_attrs)

        # ── Trade-cluster mapping (inverted for fast lookup) ─────────
        trade_cluster_map = {}
        for cid in config.cluster_ids:
            for tid in config.cluster_mapping.get(cid, []):
                trade_cluster_map[str(tid)] = cid
        self._write_json(db_dir / "trade_cluster_map.json", trade_cluster_map)

        # ── Per-split pre-computed artifacts ──────────────────────────
        portfolio_dir = db_dir / "portfolio_summary"
        portfolio_dir.mkdir(exist_ok=True)
        cluster_dir = db_dir / "cluster_summary"
        cluster_dir.mkdir(exist_ok=True)
        trade_dir = db_dir / "trade_metrics"
        trade_dir.mkdir(exist_ok=True)
        group_dir = db_dir / "group_summaries"
        group_dir.mkdir(exist_ok=True)
        corr_dir = db_dir / "group_correlations"
        corr_dir.mkdir(exist_ok=True)

        for split in all_split_results:
            self._save_portfolio_summary(
                portfolio_dir, split, combined_arrays.get(split),
            )
            self._save_cluster_summaries(
                cluster_dir, split, split_preds.get(split, {}),
                split_targets.get(split, {}),
            )
            self._save_trade_metrics(
                trade_dir, split, split_preds.get(split, {}),
                split_targets.get(split, {}), config,
            )
            self._save_group_summaries(
                group_dir, corr_dir, split, split_preds.get(split, {}),
                split_targets.get(split, {}), config, cluster_attrs,
            )

        # ── Graph stats (reads joblib from registry) ─────────────────
        self._save_graph_stats(db_dir, config)

        elapsed = time.perf_counter() - t0
        logger.info(
            "DB-ready artifacts saved to %s (%.2fs)", db_dir, elapsed,
        )

    # ------------------------------------------------------------------
    # _save_artifacts_db sub-routines
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        """Write *data* as indented JSON, converting numpy types."""
        from src.rade_ml_pt.core import json_safe
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=json_safe)

    @staticmethod
    def _build_cluster_attributes(config: EnsembleConfig) -> Dict[str, Any]:
        """Build a flat ``{cluster_id: {attr: value, n_trades: int}}`` dict."""
        attrs: Dict[str, Any] = {}
        key_names = config.cluster_key or []
        key_values = config.cluster_key_values or {}
        for cid in config.cluster_ids:
            entry: Dict[str, Any] = {}
            if key_names and cid in key_values:
                for name, val in zip(key_names, key_values[cid]):
                    entry[name] = val
            entry["n_trades"] = len(config.cluster_mapping.get(cid, []))
            attrs[cid] = entry
        return attrs

    def _save_portfolio_summary(
        self,
        out_dir: Path,
        split: str,
        combined: Optional[Dict[str, np.ndarray]],
    ) -> None:
        """Save portfolio-level 1-D predictions, targets, percentiles, worst scenarios."""
        if combined is None:
            return
        preds_1d = combined["predictions"].sum(axis=1)
        targets_1d = combined["targets"].sum(axis=1)
        abs_errors = np.abs(preds_1d - targets_1d)

        percentiles = {}
        for p in (1, 5, 25, 50, 75, 95, 99):
            percentiles[f"P{p}"] = {
                "prediction": float(np.percentile(preds_1d, p)),
                "target": float(np.percentile(targets_1d, p)),
                "abs_error": float(np.percentile(abs_errors, p)),
            }
        percentiles["mean"] = {
            "prediction": float(np.mean(preds_1d)),
            "target": float(np.mean(targets_1d)),
            "abs_error": float(np.mean(abs_errors)),
        }

        worst_idx = np.argsort(abs_errors)[::-1][:20]
        worst_scenarios = [
            {
                "rank": int(i + 1),
                "scenario_idx": int(idx),
                "prediction": float(preds_1d[idx]),
                "target": float(targets_1d[idx]),
                "abs_error": float(abs_errors[idx]),
            }
            for i, idx in enumerate(worst_idx)
        ]

        np.savez_compressed(
            out_dir / f"{split}.npz",
            predictions=preds_1d.astype(np.float32),
            targets=targets_1d.astype(np.float32),
        )
        self._write_json(out_dir / f"{split}_percentiles.json", percentiles)
        self._write_json(out_dir / f"{split}_worst.json", worst_scenarios)

    def _save_cluster_summaries(
        self,
        out_dir: Path,
        split: str,
        preds: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
    ) -> None:
        """Save per-cluster 1-D summed predictions and targets."""
        cluster_preds = {}
        cluster_targets = {}
        for cid in sorted(preds.keys()):
            cluster_preds[cid] = preds[cid].sum(axis=1).astype(np.float32)
            cluster_targets[cid] = targets[cid].sum(axis=1).astype(np.float32)

        np.savez_compressed(
            out_dir / f"{split}.npz",
            **{f"{cid}_pred": cluster_preds[cid] for cid in cluster_preds},
            **{f"{cid}_target": cluster_targets[cid] for cid in cluster_targets},
        )

    def _save_trade_metrics(
        self,
        out_dir: Path,
        split: str,
        preds: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        config: EnsembleConfig,
    ) -> None:
        """Save per-trade summary metrics (MAE, RMSE, MaxAE, P95AE, etc.)."""
        rows = []
        for cid in sorted(preds.keys()):
            p = preds[cid]
            t = targets[cid]
            if p.ndim == 1:
                p = p.reshape(-1, 1)
                t = t.reshape(-1, 1)

            trade_ids = config.cluster_mapping.get(cid, [])
            for j in range(p.shape[1]):
                residuals = p[:, j] - t[:, j]
                abs_err = np.abs(residuals)
                tid = str(trade_ids[j]) if j < len(trade_ids) else f"trade_{j}"
                rows.append({
                    "cluster_id": cid,
                    "trade_id": tid,
                    "mae": round(float(np.mean(abs_err)), 6),
                    "rmse": round(float(np.sqrt(np.mean(residuals ** 2))), 6),
                    "max_ae": round(float(np.max(abs_err)), 6),
                    "p95_ae": round(float(np.percentile(abs_err, 95)), 6),
                    "mean_residual": round(float(np.mean(residuals)), 6),
                    "std_residual": round(float(np.std(residuals)), 6),
                })
        self._write_json(out_dir / f"{split}.json", rows)

    def _save_group_summaries(
        self,
        group_dir: Path,
        corr_dir: Path,
        split: str,
        preds: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray],
        config: EnsembleConfig,
        cluster_attrs: Dict[str, Any],
    ) -> None:
        """Save group-level (desk/ccy/product) summaries and correlation matrices."""
        if not cluster_attrs:
            return

        attr_keys = set()
        for ca in cluster_attrs.values():
            attr_keys.update(k for k in ca if k != "n_trades")
        if not attr_keys:
            return

        cluster_sums: Dict[str, np.ndarray] = {}
        cluster_target_sums: Dict[str, np.ndarray] = {}
        for cid in sorted(preds.keys()):
            cluster_sums[cid] = preds[cid].sum(axis=1)
            cluster_target_sums[cid] = targets[cid].sum(axis=1)

        group_results: Dict[str, Dict[str, Any]] = {}
        correlations: Dict[str, Any] = {}

        for attr_key in sorted(attr_keys):
            groups: Dict[str, list] = {}
            for cid, ca in cluster_attrs.items():
                val = ca.get(attr_key)
                if val is not None and cid in cluster_sums:
                    groups.setdefault(str(val), []).append(cid)

            group_data = {}
            group_residuals: Dict[str, np.ndarray] = {}
            for grp, cids in sorted(groups.items()):
                grp_pred = sum(cluster_sums[c] for c in cids if c in cluster_sums)
                grp_tgt = sum(cluster_target_sums[c] for c in cids if c in cluster_target_sums)
                if isinstance(grp_pred, (int, float)):
                    continue
                residual = grp_pred - grp_tgt
                n_trades = sum(cluster_attrs.get(c, {}).get("n_trades", 0) for c in cids)
                group_data[grp] = {
                    "n_trades": n_trades,
                    "n_clusters": len(cids),
                    "mae": round(float(np.mean(np.abs(residual))), 6),
                    "rmse": round(float(np.sqrt(np.mean(residual ** 2))), 6),
                }
                group_residuals[grp] = residual

            group_results[attr_key] = group_data

            if len(group_residuals) > 1:
                import pandas as pd
                df = pd.DataFrame(group_residuals)
                corr_matrix = df.corr()
                correlations[attr_key] = {
                    "columns": corr_matrix.columns.tolist(),
                    "values": corr_matrix.values.tolist(),
                }

        self._write_json(group_dir / f"{split}.json", group_results)
        if correlations:
            self._write_json(corr_dir / f"{split}.json", correlations)

    def _save_graph_stats(
        self, db_dir: Path, config: EnsembleConfig,
    ) -> None:
        """Compute and save per-cluster graph statistics from registry joblobs."""
        stats: Dict[str, Any] = {}
        for cid in config.cluster_ids:
            member_version = None
            job_meta = config.metadata.get("job", {})
            member_versions = job_meta.get("member_versions", {})
            if isinstance(member_versions, dict):
                member_version = member_versions.get(cid)
            if not member_version:
                stats[cid] = {"n_nodes": 0, "n_edges": 0, "density": 0, "mean_weight": 0}
                continue
            graph_path = Path(config.registry_dir) / member_version / "graph_results.joblib"
            if not graph_path.exists():
                stats[cid] = {"n_nodes": 0, "n_edges": 0, "density": 0, "mean_weight": 0}
                continue
            try:
                import joblib
                gr = joblib.load(graph_path)
                indices = gr.get("sparse_indices")
                values = gr.get("sparse_values")
                shape = gr.get("sparse_shape", [0, 0])
                n_nodes = shape[0] if shape[0] > 0 else 0
                nnz = len(np.array(values)) if values is not None else 0
                density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0
                mean_w = float(np.mean(values)) if values is not None and nnz > 0 else 0
                stats[cid] = {
                    "n_nodes": int(n_nodes),
                    "n_edges": int(nnz),
                    "density": round(density, 6),
                    "mean_weight": round(mean_w, 4),
                }
            except Exception as exc:
                logger.debug("Could not read graph stats for '%s': %s", cid, exc)
                stats[cid] = {"n_nodes": 0, "n_edges": 0, "density": 0, "mean_weight": 0}

        self._write_json(db_dir / "graph_stats.json", stats)
```

---

## File: src/rade_ml_pt/ensemble/db_schema.sql
**Action:** NEW

```sql
-- Ensemble Analytics DB-ready schema.
-- Compatible with both SQLite and Postgres.

CREATE TABLE IF NOT EXISTS ensemble_versions (
    version         TEXT PRIMARY KEY,
    n_clusters      INTEGER NOT NULL,
    n_trades        INTEGER NOT NULL,
    aggregation     TEXT,
    strategy        TEXT,
    evaluated_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS ensemble_metrics (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    REAL,
    PRIMARY KEY (version, split, metric_name)
);

CREATE TABLE IF NOT EXISTS cluster_metrics (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    split           TEXT NOT NULL,
    metric_name     TEXT NOT NULL,
    metric_value    REAL,
    PRIMARY KEY (version, cluster_id, split, metric_name)
);

CREATE TABLE IF NOT EXISTS cluster_attributes (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    attribute_name  TEXT NOT NULL,
    attribute_value TEXT,
    PRIMARY KEY (version, cluster_id, attribute_name)
);

CREATE TABLE IF NOT EXISTS cluster_predictions (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    split           TEXT NOT NULL,
    scenario_idx    INTEGER NOT NULL,
    prediction      REAL NOT NULL,
    target          REAL NOT NULL,
    PRIMARY KEY (version, cluster_id, split, scenario_idx)
);

CREATE INDEX IF NOT EXISTS idx_cp_split
    ON cluster_predictions(version, split);

CREATE TABLE IF NOT EXISTS portfolio_summary (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    scenario_idx    INTEGER NOT NULL,
    prediction      REAL NOT NULL,
    target          REAL NOT NULL,
    PRIMARY KEY (version, split, scenario_idx)
);

CREATE TABLE IF NOT EXISTS portfolio_percentiles (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    percentile      TEXT NOT NULL,
    pred_value      REAL,
    target_value    REAL,
    abs_error       REAL,
    PRIMARY KEY (version, split, percentile)
);

CREATE TABLE IF NOT EXISTS worst_scenarios (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    scenario_idx    INTEGER NOT NULL,
    prediction      REAL,
    target          REAL,
    abs_error       REAL,
    PRIMARY KEY (version, split, rank)
);

CREATE TABLE IF NOT EXISTS trade_metrics (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    split           TEXT NOT NULL,
    trade_id        TEXT NOT NULL,
    mae             REAL,
    rmse            REAL,
    max_ae          REAL,
    p95_ae          REAL,
    mean_residual   REAL,
    std_residual    REAL,
    PRIMARY KEY (version, cluster_id, split, trade_id)
);

CREATE TABLE IF NOT EXISTS trade_cluster_map (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    trade_id        TEXT NOT NULL,
    cluster_id      TEXT NOT NULL,
    PRIMARY KEY (version, trade_id)
);

CREATE TABLE IF NOT EXISTS graph_stats (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    cluster_id      TEXT NOT NULL,
    n_nodes         INTEGER,
    n_edges         INTEGER,
    density         REAL,
    mean_weight     REAL,
    PRIMARY KEY (version, cluster_id)
);

-- JSON-blob tables for group summaries (simple storage).
CREATE TABLE IF NOT EXISTS group_summaries (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    data            TEXT NOT NULL,
    PRIMARY KEY (version, split)
);

CREATE TABLE IF NOT EXISTS group_correlations (
    version         TEXT NOT NULL REFERENCES ensemble_versions(version),
    split           TEXT NOT NULL,
    data            TEXT NOT NULL,
    PRIMARY KEY (version, split)
);
```

---

## File: src/rade_ml_pt/ensemble/publish_to_db.py
**Action:** NEW

```python
"""
Publish ``db_ready/`` artifacts to a SQLite database.

Reads the JSON and NPZ files produced by
``EnsembleEvalPipeline._save_artifacts_db()`` and inserts them into
a SQLite database using batch inserts for performance.

Usage
-----
::

    from src.rade_ml_pt.ensemble.publish_to_db import publish_to_sqlite

    publish_to_sqlite(
        db_ready_dir="/path/to/evaluation/db_ready",
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
    db_ready_dir: str,
    db_path: Optional[str] = None,
) -> str:
    """
    Read ``db_ready/`` files and insert into a SQLite database.

    Parameters
    ----------
    db_ready_dir : str
        Path to the ``db_ready/`` directory.
    db_path : str or None
        Output database path.  Defaults to ``db_ready/ensemble.db``.

    Returns
    -------
    str
        Path to the created/updated database file.
    """
    t0 = time.perf_counter()
    root = Path(db_ready_dir)

    if db_path is None:
        db_path = str(root / "ensemble.db")

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    # Create schema
    schema_sql = _SCHEMA_FILE.read_text()
    conn.executescript(schema_sql)

    version_meta = _read_json(root / "ensemble_version.json")
    if not version_meta:
        raise FileNotFoundError(f"ensemble_version.json not found in {root}")

    version = version_meta["version"]

    # Remove old data for this version (idempotent re-publish)
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
        "INSERT INTO ensemble_versions (version, n_clusters, n_trades, aggregation, strategy) "
        "VALUES (?, ?, ?, ?, ?)",
        (version, version_meta.get("n_clusters", 0), version_meta.get("n_trades", 0),
         version_meta.get("aggregation"), version_meta.get("execution_strategy")),
    )

    # ── cluster_attributes ────────────────────────────────────────
    attrs = _read_json(root / "cluster_attributes.json") or {}
    rows = []
    for cid, ca in attrs.items():
        for attr_name, attr_val in ca.items():
            rows.append((version, cid, attr_name, str(attr_val) if attr_val is not None else None))
    conn.executemany(
        "INSERT INTO cluster_attributes (version, cluster_id, attribute_name, attribute_value) "
        "VALUES (?, ?, ?, ?)", rows,
    )

    # ── trade_cluster_map ─────────────────────────────────────────
    tcm = _read_json(root / "trade_cluster_map.json") or {}
    conn.executemany(
        "INSERT INTO trade_cluster_map (version, trade_id, cluster_id) VALUES (?, ?, ?)",
        [(version, tid, cid) for tid, cid in tcm.items()],
    )

    # ── graph_stats ───────────────────────────────────────────────
    gs = _read_json(root / "graph_stats.json") or {}
    conn.executemany(
        "INSERT INTO graph_stats (version, cluster_id, n_nodes, n_edges, density, mean_weight) "
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
            "INSERT INTO portfolio_summary (version, split, scenario_idx, prediction, target) "
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
    cluster_ids = set()
    for key in data.files:
        cid = key.rsplit("_", 1)[0]
        cluster_ids.add(cid)

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
```

---

## File: src/ui/apps/ensemble_analytics_db/\_\_init\_\_.py
**Action:** NEW

```python
"""
DB-ready Ensemble Analytics dashboard.

Uses pre-computed artifacts from ``db_ready/`` instead of building
a ``GlobalPredictionStore``, resulting in sub-second load times
even at 75K+ targets.
"""
from src.ui.apps.ensemble_analytics_db.app import create_app

__all__ = ["create_app"]
```

---

## File: src/ui/apps/ensemble_analytics_db/app.py
**Action:** NEW

```python
"""
Dash application factory for the DB-ready Ensemble Analytics dashboard.

Shares all UI code (tabs, figures, components, theme) with the original
``ensemble_analytics`` app but uses a ``DataBackend`` (file or SQLite)
instead of the ``GlobalPredictionStore``.
"""
from __future__ import annotations

import logging
from typing import Optional

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from src.ui.apps.ensemble_analytics.config import (
    APP_TITLE,
    TAB_ORDER,
    TAB_OVERVIEW,
)
from src.ui.apps.ensemble_analytics.theme.colors import BG_PRIMARY, TEXT_PRIMARY, ACCENT_BLUE
from src.ui.apps.ensemble_analytics.theme.styles import NAVBAR_STYLE, CONTAINER_STYLE
from src.ui.apps.ensemble_analytics.theme.plotly_template import PLOTLY_TEMPLATE

logger = logging.getLogger(__name__)

_DB_APP_TITLE = "Ensemble Analytics (DB) — Hybrid GNN-RNN"


def create_app(
    registry_dir: str,
    artifacts_dir: str,
    version: str = "latest",
    backend: str = "file",
    db_path: Optional[str] = None,
    debug: bool = False,
) -> dash.Dash:
    """
    Build and return the DB-ready Dash application.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load on startup.
    backend : str
        Data backend: ``"file"`` or ``"sqlite"``.
    db_path : str or None
        Path to SQLite database (required when *backend* is ``"sqlite"``).
    debug : bool
        Enable Dash debug mode.

    Returns
    -------
    dash.Dash
        Fully configured application ready for ``app.run()``.
    """
    import plotly.io as pio
    pio.templates["ensemble_dark"] = PLOTLY_TEMPLATE
    pio.templates.default = "ensemble_dark"

    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
        title=_DB_APP_TITLE,
    )

    from src.ui.apps.ensemble_analytics_db.data.session_manager import initialise
    initialise(registry_dir, artifacts_dir, version, backend=backend, db_path=db_path)

    app.layout = _build_layout(version)

    from src.ui.apps.ensemble_analytics_db.callbacks import register_all_callbacks
    register_all_callbacks(app)

    return app


def _build_layout(version: str) -> dbc.Container:
    """Assemble the top-level page layout (shared with original app)."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session

    session = get_session()
    config = session.config

    ens_registry = session._ens_registry
    available_versions = []
    try:
        available_versions = [
            {
                "label": f"{v['version']} ({v['n_members']} clusters, {v['n_trades']} trades)",
                "value": v["version"],
            }
            for v in ens_registry.list_versions()
        ]
    except Exception:
        available_versions = [{"label": version, "value": version}]

    navbar = dbc.Navbar(
        dbc.Container(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            html.H4(
                                _DB_APP_TITLE,
                                className="mb-0",
                                style={"color": TEXT_PRIMARY, "fontWeight": "600"},
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="ensemble-version-selector",
                                options=available_versions,
                                value=session.ensemble_version,
                                clearable=False,
                                style={
                                    "width": "340px",
                                    "backgroundColor": BG_PRIMARY,
                                    "color": TEXT_PRIMARY,
                                },
                            ),
                            width="auto",
                        ),
                        dbc.Col(
                            html.Span(
                                f"{config.n_members} clusters · "
                                f"{len(config.all_trade_ids)} trades",
                                style={"color": "#8b949e", "fontSize": "13px"},
                            ),
                            width="auto",
                            className="ms-3 d-flex align-items-center",
                        ),
                    ],
                    align="center",
                    className="g-3",
                ),
            ],
            fluid=True,
        ),
        style=NAVBAR_STYLE,
        dark=True,
    )

    tab_bar = dcc.Tabs(
        id="main-tabs",
        value=TAB_OVERVIEW,
        children=[
            dcc.Tab(label=label, value=tab_id)
            for tab_id, label in TAB_ORDER
        ],
        style={"borderBottom": "1px solid #30363d"},
    )

    return dbc.Container(
        [
            navbar,
            dcc.Store(id="active-split", data="test"),
            dcc.Store(id="active-cluster", data=None),
            html.Div(style={"height": "8px"}),
            tab_bar,
            html.Div(id="tab-content", style={"marginTop": "16px"}),
        ],
        fluid=True,
        style=CONTAINER_STYLE,
    )
```

---

## File: src/ui/apps/ensemble_analytics_db/data/\_\_init\_\_.py
**Action:** NEW

```python
"""
Data layer for the DB-ready ensemble analytics dashboard.

Provides ``FileBackend`` and ``SqliteBackend`` implementations of a
common ``DataBackend`` interface.  All dashboard callbacks read data
through this layer instead of the ``GlobalPredictionStore``.
"""
```

---

## File: src/ui/apps/ensemble_analytics_db/data/backend.py
**Action:** NEW

```python
"""
Data backend abstraction for the DB-ready dashboard.

Defines the ``DataBackend`` protocol and two concrete implementations:

* ``FileBackend``  — reads from ``db_ready/`` files (JSON + NPZ).
* ``SqliteBackend`` — reads from a local SQLite database.

Both return identical data shapes so callbacks are backend-agnostic.
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ======================================================================
# Abstract backend
# ======================================================================

class DataBackend(ABC):
    """Common read-only interface consumed by all dashboard callbacks."""

    # ── Portfolio-level ───────────────────────────────────────────

    @abstractmethod
    def get_portfolio_summary(self, split: str) -> Optional[Dict[str, Any]]:
        """Return ``{"predictions": 1D, "targets": 1D}`` for the whole portfolio."""

    @abstractmethod
    def get_portfolio_percentiles(self, split: str) -> Dict[str, Any]:
        """Return pre-computed percentile table data."""

    @abstractmethod
    def get_worst_scenarios(self, split: str) -> List[Dict[str, Any]]:
        """Return top-20 worst scenarios by absolute error."""

    # ── Cluster-level ─────────────────────────────────────────────

    @abstractmethod
    def get_cluster_summary(self, split: str) -> Dict[str, Dict[str, np.ndarray]]:
        """Return ``{cluster_id: {"predictions": 1D, "targets": 1D}}``."""

    @abstractmethod
    def get_cluster_predictions(
        self, cluster_id: str, split: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Return per-trade ``{"predictions": 2D, "targets": 2D, "trade_ids": list}``."""

    # ── Trade-level ───────────────────────────────────────────────

    @abstractmethod
    def get_trade_metrics(self, split: str) -> List[Dict[str, Any]]:
        """Return per-trade summary metrics (MAE, RMSE, etc.)."""

    # ── Group-level ───────────────────────────────────────────────

    @abstractmethod
    def get_group_summaries(self, split: str) -> Dict[str, Any]:
        """Return ``{group_col: {group_val: {mae, rmse, ...}}}``."""

    @abstractmethod
    def get_group_correlations(self, split: str) -> Dict[str, Any]:
        """Return ``{group_col: {"columns": [...], "values": [[...]]}}``."""

    # ── Metadata ──────────────────────────────────────────────────

    @abstractmethod
    def get_cluster_attributes(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{cluster_id: {attr: value, n_trades: int}}``."""

    @abstractmethod
    def get_trade_cluster_map(self) -> Dict[str, str]:
        """Return ``{trade_id: cluster_id}``."""

    @abstractmethod
    def get_graph_stats(self) -> Dict[str, Dict[str, Any]]:
        """Return ``{cluster_id: {n_nodes, n_edges, density, mean_weight}}``."""

    @abstractmethod
    def get_ensemble_version_meta(self) -> Dict[str, Any]:
        """Return version, n_clusters, n_trades, etc."""


# ======================================================================
# File backend — reads from db_ready/ directory
# ======================================================================

class FileBackend(DataBackend):
    """Reads pre-computed artifacts directly from the ``db_ready/`` directory.

    Parameters
    ----------
    db_ready_dir : str or Path
        Path to ``{artifacts_dir}/ensemble/{version}/db_ready/``.
    eval_dir : str or Path
        Path to ``{artifacts_dir}/ensemble/{version}/evaluation/``.
        Used for on-demand per-trade predictions (the existing NPZ files).
    """

    def __init__(self, db_ready_dir: str, eval_dir: str) -> None:
        self._root = Path(db_ready_dir)
        self._eval_dir = Path(eval_dir)
        self._cache: Dict[str, Any] = {}

    # ── Helpers ───────────────────────────────────────────────────

    def _read_json(self, path: Path) -> Any:
        key = str(path)
        if key not in self._cache:
            if not path.exists():
                self._cache[key] = None
            else:
                with open(path) as f:
                    self._cache[key] = json.load(f)
        return self._cache[key]

    def _read_npz(self, path: Path) -> Optional[Dict[str, np.ndarray]]:
        if not path.exists():
            return None
        return dict(np.load(path, allow_pickle=False))

    # ── Portfolio ─────────────────────────────────────────────────

    def get_portfolio_summary(self, split: str) -> Optional[Dict[str, Any]]:
        npz = self._read_npz(self._root / "portfolio_summary" / f"{split}.npz")
        if npz is None:
            return None
        return {
            "predictions": npz["predictions"],
            "targets": npz["targets"],
        }

    def get_portfolio_percentiles(self, split: str) -> Dict[str, Any]:
        return self._read_json(
            self._root / "portfolio_summary" / f"{split}_percentiles.json",
        ) or {}

    def get_worst_scenarios(self, split: str) -> List[Dict[str, Any]]:
        return self._read_json(
            self._root / "portfolio_summary" / f"{split}_worst.json",
        ) or []

    # ── Cluster ───────────────────────────────────────────────────

    def get_cluster_summary(self, split: str) -> Dict[str, Dict[str, np.ndarray]]:
        npz = self._read_npz(self._root / "cluster_summary" / f"{split}.npz")
        if npz is None:
            return {}
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for key, arr in npz.items():
            cid, suffix = key.rsplit("_", 1)
            result.setdefault(cid, {})[
                "predictions" if suffix == "pred" else "targets"
            ] = arr
        return result

    def get_cluster_predictions(
        self, cluster_id: str, split: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        """Load the full ``[n_scenarios x n_trades]`` arrays for one cluster.

        Reads from the original per-member NPZ in ``evaluation/members/``.
        """
        npz_path = (
            self._eval_dir / "members" / cluster_id / "predictions" / f"{split}.npz"
        )
        if not npz_path.exists():
            return None
        data = np.load(npz_path, allow_pickle=False)
        return {
            "predictions": data["predictions"],
            "targets": data["targets"],
        }

    # ── Trade metrics ─────────────────────────────────────────────

    def get_trade_metrics(self, split: str) -> List[Dict[str, Any]]:
        return self._read_json(
            self._root / "trade_metrics" / f"{split}.json",
        ) or []

    # ── Group ─────────────────────────────────────────────────────

    def get_group_summaries(self, split: str) -> Dict[str, Any]:
        return self._read_json(
            self._root / "group_summaries" / f"{split}.json",
        ) or {}

    def get_group_correlations(self, split: str) -> Dict[str, Any]:
        return self._read_json(
            self._root / "group_correlations" / f"{split}.json",
        ) or {}

    # ── Metadata ──────────────────────────────────────────────────

    def get_cluster_attributes(self) -> Dict[str, Dict[str, Any]]:
        return self._read_json(self._root / "cluster_attributes.json") or {}

    def get_trade_cluster_map(self) -> Dict[str, str]:
        return self._read_json(self._root / "trade_cluster_map.json") or {}

    def get_graph_stats(self) -> Dict[str, Dict[str, Any]]:
        return self._read_json(self._root / "graph_stats.json") or {}

    def get_ensemble_version_meta(self) -> Dict[str, Any]:
        return self._read_json(self._root / "ensemble_version.json") or {}


# ======================================================================
# SQLite backend
# ======================================================================

class SqliteBackend(DataBackend):
    """Reads pre-computed artifacts from a local SQLite database.

    Parameters
    ----------
    db_path : str or Path
        Path to the ``.db`` file.
    eval_dir : str or Path
        Path to ``evaluation/`` for on-demand per-trade NPZ reads.
    version : str
        Ensemble version to query.
    """

    def __init__(self, db_path: str, eval_dir: str, version: str) -> None:
        import sqlite3
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._eval_dir = Path(eval_dir)
        self._version = version

    def _query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        cur = self._conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def _query_one(self, sql: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        rows = self._query(sql, params)
        return rows[0] if rows else None

    # ── Portfolio ─────────────────────────────────────────────────

    def get_portfolio_summary(self, split: str) -> Optional[Dict[str, Any]]:
        rows = self._query(
            "SELECT scenario_idx, prediction, target "
            "FROM portfolio_summary WHERE version=? AND split=? "
            "ORDER BY scenario_idx",
            (self._version, split),
        )
        if not rows:
            return None
        return {
            "predictions": np.array([r["prediction"] for r in rows], dtype=np.float32),
            "targets": np.array([r["target"] for r in rows], dtype=np.float32),
        }

    def get_portfolio_percentiles(self, split: str) -> Dict[str, Any]:
        rows = self._query(
            "SELECT percentile, pred_value, target_value, abs_error "
            "FROM portfolio_percentiles WHERE version=? AND split=?",
            (self._version, split),
        )
        return {
            r["percentile"]: {
                "prediction": r["pred_value"],
                "target": r["target_value"],
                "abs_error": r["abs_error"],
            }
            for r in rows
        }

    def get_worst_scenarios(self, split: str) -> List[Dict[str, Any]]:
        return self._query(
            "SELECT rank, scenario_idx, prediction, target, abs_error "
            "FROM worst_scenarios WHERE version=? AND split=? ORDER BY rank",
            (self._version, split),
        )

    # ── Cluster ───────────────────────────────────────────────────

    def get_cluster_summary(self, split: str) -> Dict[str, Dict[str, np.ndarray]]:
        rows = self._query(
            "SELECT cluster_id, scenario_idx, prediction, target "
            "FROM cluster_predictions WHERE version=? AND split=? "
            "ORDER BY cluster_id, scenario_idx",
            (self._version, split),
        )
        from itertools import groupby
        result: Dict[str, Dict[str, np.ndarray]] = {}
        for cid, group in groupby(rows, key=lambda r: r["cluster_id"]):
            records = list(group)
            result[cid] = {
                "predictions": np.array([r["prediction"] for r in records], dtype=np.float32),
                "targets": np.array([r["target"] for r in records], dtype=np.float32),
            }
        return result

    def get_cluster_predictions(
        self, cluster_id: str, split: str,
    ) -> Optional[Dict[str, np.ndarray]]:
        npz_path = (
            self._eval_dir / "members" / cluster_id / "predictions" / f"{split}.npz"
        )
        if not npz_path.exists():
            return None
        data = np.load(npz_path, allow_pickle=False)
        return {
            "predictions": data["predictions"],
            "targets": data["targets"],
        }

    # ── Trade metrics ─────────────────────────────────────────────

    def get_trade_metrics(self, split: str) -> List[Dict[str, Any]]:
        return self._query(
            "SELECT cluster_id, trade_id, mae, rmse, max_ae, p95_ae, "
            "mean_residual, std_residual "
            "FROM trade_metrics WHERE version=? AND split=?",
            (self._version, split),
        )

    # ── Group ─────────────────────────────────────────────────────

    def get_group_summaries(self, split: str) -> Dict[str, Any]:
        row = self._query_one(
            "SELECT data FROM group_summaries WHERE version=? AND split=?",
            (self._version, split),
        )
        return json.loads(row["data"]) if row else {}

    def get_group_correlations(self, split: str) -> Dict[str, Any]:
        row = self._query_one(
            "SELECT data FROM group_correlations WHERE version=? AND split=?",
            (self._version, split),
        )
        return json.loads(row["data"]) if row else {}

    # ── Metadata ──────────────────────────────────────────────────

    def get_cluster_attributes(self) -> Dict[str, Dict[str, Any]]:
        rows = self._query(
            "SELECT cluster_id, attribute_name, attribute_value "
            "FROM cluster_attributes WHERE version=?",
            (self._version,),
        )
        result: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            result.setdefault(r["cluster_id"], {})[r["attribute_name"]] = r["attribute_value"]
        return result

    def get_trade_cluster_map(self) -> Dict[str, str]:
        rows = self._query(
            "SELECT trade_id, cluster_id FROM trade_cluster_map WHERE version=?",
            (self._version,),
        )
        return {r["trade_id"]: r["cluster_id"] for r in rows}

    def get_graph_stats(self) -> Dict[str, Dict[str, Any]]:
        rows = self._query(
            "SELECT cluster_id, n_nodes, n_edges, density, mean_weight "
            "FROM graph_stats WHERE version=?",
            (self._version,),
        )
        return {
            r["cluster_id"]: {
                "n_nodes": r["n_nodes"], "n_edges": r["n_edges"],
                "density": r["density"], "mean_weight": r["mean_weight"],
            }
            for r in rows
        }

    def get_ensemble_version_meta(self) -> Dict[str, Any]:
        return self._query_one(
            "SELECT * FROM ensemble_versions WHERE version=?",
            (self._version,),
        ) or {}
```

---

## File: src/ui/apps/ensemble_analytics_db/data/session_manager.py
**Action:** NEW

```python
"""
Singleton wrapper around ``DbEnsembleSession``.

Mirrors the interface of ``ensemble_analytics.data.session_manager``
but initialises from a ``DataBackend`` (file or SQLite) instead of
the heavyweight ``EnsembleSession`` + ``GlobalPredictionStore``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from src.rade_ml_pt.ensemble.session import EnsembleSession
from src.ui.apps.ensemble_analytics_db.data.backend import DataBackend

logger = logging.getLogger(__name__)

_session: Optional[EnsembleSession] = None
_backend: Optional[DataBackend] = None
_registry_dir: Optional[str] = None
_artifacts_dir: Optional[str] = None


def initialise(
    registry_dir: str,
    artifacts_dir: str,
    version: str = "latest",
    backend: str = "file",
    db_path: Optional[str] = None,
) -> None:
    """
    Create the singleton session and data backend.

    Parameters
    ----------
    registry_dir : str
        Root directory for model and ensemble registries.
    artifacts_dir : str
        Root directory for evaluation artifacts.
    version : str
        Ensemble version or tag to load.
    backend : str
        ``"file"`` for FileBackend, ``"sqlite"`` for SqliteBackend.
    db_path : str or None
        Path to SQLite database (required when *backend* is ``"sqlite"``).
    """
    global _session, _backend, _registry_dir, _artifacts_dir

    _registry_dir = registry_dir
    _artifacts_dir = artifacts_dir

    _session = EnsembleSession(
        registry_dir=registry_dir,
        artifacts_dir=artifacts_dir,
    )
    _session.load_metadata(version)
    _session.load_display_artifacts()

    resolved_version = _session.ensemble_version
    eval_dir = str(
        Path(artifacts_dir) / "ensemble" / resolved_version / "evaluation"
    )
    db_ready_dir = str(
        Path(artifacts_dir) / "ensemble" / resolved_version / "db_ready"
    )

    if backend == "sqlite":
        if db_path is None:
            db_path = str(Path(db_ready_dir) / "ensemble.db")
        from src.ui.apps.ensemble_analytics_db.data.backend import SqliteBackend
        _backend = SqliteBackend(db_path, eval_dir, resolved_version)
        logger.info("Initialised SqliteBackend: %s", db_path)
    else:
        from src.ui.apps.ensemble_analytics_db.data.backend import FileBackend
        _backend = FileBackend(db_ready_dir, eval_dir)
        logger.info("Initialised FileBackend: %s", db_ready_dir)

    logger.info(
        "DB session initialised: version=%s, clusters=%d, backend=%s",
        resolved_version, _session.config.n_members, backend,
    )


def get_session() -> EnsembleSession:
    """Return the singleton ``EnsembleSession`` (for metadata access)."""
    if _session is None:
        raise RuntimeError(
            "Session not initialised. Call initialise() at app startup."
        )
    return _session


def get_backend() -> DataBackend:
    """Return the active ``DataBackend`` (for pre-computed data access)."""
    if _backend is None:
        raise RuntimeError(
            "Backend not initialised. Call initialise() at app startup."
        )
    return _backend


def reload(version: str = "latest") -> None:
    """Reload the session for a different ensemble version."""
    if _registry_dir is None or _artifacts_dir is None:
        raise RuntimeError("Cannot reload — session was never initialised.")
    initialise(_registry_dir, _artifacts_dir, version)
```

---

## File: src/ui/apps/ensemble_analytics_db/data/prediction_store.py
**Action:** NEW

```python
"""
Pre-computed prediction access for the DB-ready dashboard.

Replaces the ``GlobalPredictionStore`` pattern with lightweight reads
from the ``DataBackend``.  Provides portfolio-level summaries and
on-demand single-cluster predictions.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def get_portfolio_summary(split: str = "test") -> Optional[Dict[str, Any]]:
    """
    Return pre-summed portfolio predictions and targets.

    Returns
    -------
    dict or None
        ``{"predictions": 1D ndarray, "targets": 1D ndarray}``
    """
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_portfolio_summary(split)


def get_portfolio_percentiles(split: str = "test") -> Dict[str, Any]:
    """Return pre-computed percentile statistics."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_portfolio_percentiles(split)


def get_worst_scenarios(split: str = "test") -> List[Dict[str, Any]]:
    """Return top-20 worst scenarios by absolute error."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_worst_scenarios(split)


def get_cluster_summary(split: str = "test") -> Dict[str, Dict[str, np.ndarray]]:
    """Return per-cluster summed predictions and targets."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_cluster_summary(split)


def get_cluster_predictions(
    cluster_id: str, split: str = "test",
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load full per-trade predictions for one cluster (on-demand).

    Returns
    -------
    dict or None
        ``{"predictions": 2D, "targets": 2D}``
    """
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_cluster_predictions(cluster_id, split)


def get_trade_metrics(split: str = "test") -> List[Dict[str, Any]]:
    """Return per-trade summary metrics (MAE, RMSE, etc.)."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_trade_metrics(split)


def get_group_summaries(split: str = "test") -> Dict[str, Any]:
    """Return group-level summaries (desk/ccy/product)."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_group_summaries(split)


def get_group_correlations(split: str = "test") -> Dict[str, Any]:
    """Return cross-group correlation matrices."""
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
    return get_backend().get_group_correlations(split)
```

---

## File: src/ui/apps/ensemble_analytics_db/data/trade_catalogue.py
**Action:** NEW

```python
"""
Trade catalogue built from pre-computed ``db_ready/`` artifacts.

Constructs a pandas DataFrame from ``trade_cluster_map.json`` and
``cluster_attributes.json`` without loading the full cluster displays.
Falls back to the original session method if the pre-computed files
are missing attributes.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

_cache: Optional[pd.DataFrame] = None


def get_trade_catalogue() -> pd.DataFrame:
    """
    Return the global trade catalogue DataFrame.

    Falls back to the original ``session.build_global_trade_catalogue()``
    if the ``DataBackend`` does not have enough attribute data.
    """
    global _cache
    if _cache is not None:
        return _cache

    from src.ui.apps.ensemble_analytics_db.data.session_manager import (
        get_session,
        get_backend,
    )

    backend = get_backend()
    tcm = backend.get_trade_cluster_map()
    attrs = backend.get_cluster_attributes()

    if tcm:
        rows = []
        for trade_id, cluster_id in tcm.items():
            row = {"trade_id": trade_id, "cluster_id": cluster_id}
            ca = attrs.get(cluster_id, {})
            row.update(ca)
            rows.append(row)
        _cache = pd.DataFrame(rows)
    else:
        _cache = get_session().build_global_trade_catalogue()

    return _cache


def invalidate() -> None:
    """Clear the cache (called on session reload)."""
    global _cache
    _cache = None
```

---

## File: src/ui/apps/ensemble_analytics_db/data/graph_data_loader.py
**Action:** NEW

```python
"""
Cluster graph-data loader — delegates to the original session.

Graph adjacency data lives in ``graph_results.joblib`` in the registry
and is not pre-computed into ``db_ready/``.  This module proxies to the
original ``EnsembleSession.load_cluster_graph_data``.
"""
from __future__ import annotations

from typing import Any, Dict


def get_graph_data(cluster_id: str) -> Dict[str, Any]:
    """
    Return graph adjacency and encoder feature data for one cluster.

    Returns
    -------
    dict
        Keys: ``graph_results``, ``encoder_results``, ``trade_universe``.
    """
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
    return get_session().load_cluster_graph_data(cluster_id)
```

---

## File: src/ui/apps/ensemble_analytics_db/data/market_data_loader.py
**Action:** NEW

```python
"""
Cluster market-data loader — delegates to the original session.

Market data lives in ``cluster_assets.joblib`` in the registry and is
not pre-computed into ``db_ready/``.  This module proxies to the
original ``EnsembleSession.load_cluster_market_data``.
"""
from __future__ import annotations

from typing import Any, Dict


def get_market_data(cluster_id: str) -> Dict[str, Any]:
    """
    Return market / risk-factor shock data for one cluster.

    Returns
    -------
    dict
        ``{asset_name: {rf_name: np.ndarray}}``.
    """
    from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
    return get_session().load_cluster_market_data(cluster_id)
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/\_\_init\_\_.py
**Action:** NEW

```python
"""
Callback registration hub for the DB-ready dashboard.

Mirrors ``ensemble_analytics.callbacks`` but uses the DB-backed data
layer.  Tab layouts are imported from the original app (shared UI).
"""
from __future__ import annotations

import dash
from dash import Input, Output, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    TAB_OVERVIEW,
    TAB_EVALUATION,
    TAB_CLUSTER_DEEP_DIVE,
    TAB_MARKET_DATA,
    TAB_TRADE_GRAPH,
    TAB_INFERENCE,
    TAB_GOVERNANCE,
)


def register_all_callbacks(app: dash.Dash) -> None:
    """Register every callback module and the top-level tab router."""

    @app.callback(
        Output("tab-content", "children"),
        Input("main-tabs", "value"),
    )
    def render_tab(tab_id: str):
        """Swap the main content area based on the active tab."""
        if tab_id == TAB_OVERVIEW:
            from src.ui.apps.ensemble_analytics.tabs.overview import layout
            return layout()
        elif tab_id == TAB_EVALUATION:
            from src.ui.apps.ensemble_analytics.tabs.evaluation import layout
            return layout()
        elif tab_id == TAB_CLUSTER_DEEP_DIVE:
            from src.ui.apps.ensemble_analytics.tabs.cluster_deep_dive import layout
            return layout()
        elif tab_id == TAB_MARKET_DATA:
            from src.ui.apps.ensemble_analytics.tabs.market_data import layout
            return layout()
        elif tab_id == TAB_TRADE_GRAPH:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph import layout
            return layout()
        elif tab_id == TAB_INFERENCE:
            from src.ui.apps.ensemble_analytics.tabs.inference import layout
            return layout()
        elif tab_id == TAB_GOVERNANCE:
            from src.ui.apps.ensemble_analytics.tabs.governance import layout
            return layout()
        return html.Div("Tab not found.")

    @app.callback(
        Output("tab-content", "children", allow_duplicate=True),
        Input("ensemble-version-selector", "value"),
        prevent_initial_call=True,
    )
    def reload_version(version):
        """Reload session when the version dropdown changes."""
        if not version:
            return no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import reload
        from src.ui.apps.ensemble_analytics_db.data.trade_catalogue import invalidate
        reload(version)
        invalidate()
        return render_tab(TAB_OVERVIEW)

    from src.ui.apps.ensemble_analytics_db.callbacks.overview_cb import register as reg_overview
    from src.ui.apps.ensemble_analytics_db.callbacks.evaluation_cb import register as reg_evaluation
    from src.ui.apps.ensemble_analytics_db.callbacks.cluster_deep_dive_cb import register as reg_deep_dive
    from src.ui.apps.ensemble_analytics_db.callbacks.market_data_cb import register as reg_market_data
    from src.ui.apps.ensemble_analytics_db.callbacks.trade_graph_cb import register as reg_trade_graph
    from src.ui.apps.ensemble_analytics_db.callbacks.inference_cb import register as reg_inference
    from src.ui.apps.ensemble_analytics_db.callbacks.governance_cb import register as reg_governance

    reg_overview(app)
    reg_evaluation(app)
    reg_deep_dive(app)
    reg_market_data(app)
    reg_trade_graph(app)
    reg_inference(app)
    reg_governance(app)
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/overview_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 1 — Overview (DB-ready version).

Reads portfolio scatter data from pre-computed ``portfolio_summary``
instead of the ``GlobalPredictionStore``.
"""
from __future__ import annotations

import numpy as np
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.kpi_card import kpi_card
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.heatmaps import multi_metric_cluster_heatmap


def register(app):
    """Register Overview tab callbacks on *app*."""

    @app.callback(
        Output("overview-kpi-row", "children"),
        Output("overview-scatter-container", "children"),
        Output("overview-bar-container", "children"),
        Output("overview-heatmap-container", "children"),
        Output("overview-table-container", "children"),
        Input("overview-split-toggle", "value"),
    )
    def update_overview(split: str):
        """Rebuild all Overview visuals when the split toggle changes."""
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_portfolio_summary

        session = get_session()
        ens_display = session.ensemble_display
        if ens_display is None:
            return no_update, no_update, no_update, no_update, no_update

        # ── KPI cards ─────────────────────────────────────────────
        ens_metrics = ens_display.ensemble_metrics.get(split, {})
        kpi_keys = ["mae", "rmse", "max_ae", "p95_ae", "p99_ae"]
        kpi_cards = []
        for key in kpi_keys:
            val = ens_metrics.get(key)
            display_val = f"{val:.4f}" if val is not None else "N/A"
            kpi_cards.append(
                dbc.Col(
                    kpi_card(
                        title=METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        value=display_val,
                    ),
                    md=True,
                )
            )

        # ── Portfolio scatter (from pre-computed summary) ─────────
        scatter_fig = html.Div("No prediction data available.")
        portfolio = get_portfolio_summary(split)
        if portfolio is not None:
            scatter_fig = dcc.Graph(
                figure=pred_vs_target_scatter(
                    portfolio["predictions"], portfolio["targets"],
                    title=f"Portfolio PnL — {split.capitalize()}",
                ),
                config={"displayModeBar": False},
            )

        # ── Member comparison bar ─────────────────────────────────
        pm_metrics = ens_display.per_member_metrics.get(split, {})
        cluster_ids = session.config.cluster_ids
        cluster_attrs = session.cluster_attributes
        mae_by_cluster = {
            cid: pm_metrics.get(cid, {}).get("mae", 0.0)
            for cid in cluster_ids
        }
        hover_text = {}
        for cid in cluster_ids:
            ca = cluster_attrs.get(cid, {}) if cluster_attrs else {}
            if ca:
                hover_text[cid] = ", ".join(f"{k}={v}" for k, v in ca.items() if v is not None)
            else:
                hover_text[cid] = ""

        bar_fig = html.Div(
            dcc.Graph(
                figure=member_comparison_bar(
                    cluster_ids, mae_by_cluster,
                    metric_name="MAE",
                    title=f"MAE by Cluster — {split.capitalize()}",
                    hover_text=hover_text,
                ),
                config={"displayModeBar": False},
            ),
            style={"maxHeight": "450px", "overflowY": "auto"},
        )

        # ── Multi-metric cluster heatmap ──────────────────────────
        heatmap_fig = dcc.Graph(
            figure=multi_metric_cluster_heatmap(
                cluster_ids, pm_metrics,
                title=f"Cluster Metrics Heatmap — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )

        # ── Member table ──────────────────────────────────────────
        cluster_attrs = session.cluster_attributes
        column_defs = [
            {"field": "cluster_id", "headerName": "Cluster", "pinned": "left"},
        ]
        first_attrs = next(iter(cluster_attrs.values()), {}) if cluster_attrs else {}
        attr_cols = list(first_attrs.keys())
        for ac in attr_cols:
            column_defs.append({"field": ac, "headerName": ac.replace("_", " ").title()})

        metric_keys = list(next(iter(pm_metrics.values()), {}).keys()) if pm_metrics else []
        all_metric_vals = {mk: [] for mk in metric_keys}
        for cid in cluster_ids:
            for mk in metric_keys:
                v = pm_metrics.get(cid, {}).get(mk)
                if v is not None:
                    all_metric_vals[mk].append(v)

        p25 = {mk: float(np.percentile(vs, 25)) if vs else 0 for mk, vs in all_metric_vals.items()}
        p75 = {mk: float(np.percentile(vs, 75)) if vs else 0 for mk, vs in all_metric_vals.items()}

        for mk in metric_keys:
            column_defs.append({
                "field": mk,
                "headerName": METRIC_DISPLAY_NAMES.get(mk, mk.upper()),
                "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                "cellStyle": {
                    "styleConditions": [
                        {"condition": f"params.value < {p25[mk]}", "style": {"color": "#3fb950"}},
                        {"condition": f"params.value >= {p25[mk]} && params.value <= {p75[mk]}", "style": {"color": "#d29922"}},
                        {"condition": f"params.value > {p75[mk]}", "style": {"color": "#f85149"}},
                    ]
                },
            })

        column_defs.append({"field": "n_trades", "headerName": "# Trades"})

        row_data = []
        for cid in cluster_ids:
            row = {"cluster_id": cid}
            ca = cluster_attrs.get(cid, {})
            for ac in attr_cols:
                row[ac] = ca.get(ac, "")
            row.update(pm_metrics.get(cid, {}))
            row["n_trades"] = len(session.config.cluster_mapping.get(cid, []))
            row_data.append(row)

        table = metric_table(
            column_defs=column_defs,
            row_data=row_data,
            table_id="overview-member-table",
            sort_model=[{"colId": "mae", "sort": "asc"}],
        )

        return kpi_cards, scatter_fig, bar_fig, heatmap_fig, table
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/evaluation_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 2 — Evaluation (DB-ready version).

Portfolio and group views use pre-computed summaries.  Cluster
drill-down loads a single cluster's predictions on demand.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    EVAL_SUB_PORTFOLIO,
    EVAL_SUB_DESK,
    EVAL_SUB_PRODUCT,
    EVAL_SUB_CCY,
    EVAL_SUB_CLUSTER,
    EVAL_GROUP_COLUMNS,
    METRIC_DISPLAY_NAMES,
)
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.timeseries import pnl_timeseries, overlaid_group_timeseries
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram, violin_overlay
from src.ui.apps.ensemble_analytics.figures.bar_charts import member_comparison_bar
from src.ui.apps.ensemble_analytics.figures.tables import percentile_table_data, worst_scenarios_data


def register(app):
    """Register Evaluation tab callbacks on *app*."""

    # ── Sub-tab routing ───────────────────────────────────────────
    @app.callback(
        Output("eval-sub-tab-content", "children"),
        Input("eval-sub-tabs", "value"),
    )
    def render_eval_sub_tab(sub_tab: str):
        if sub_tab == EVAL_SUB_PORTFOLIO:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.portfolio import layout
            return layout()
        elif sub_tab == EVAL_SUB_DESK:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_desk import layout
            return layout()
        elif sub_tab == EVAL_SUB_PRODUCT:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_product import layout
            return layout()
        elif sub_tab == EVAL_SUB_CCY:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_ccy import layout
            return layout()
        elif sub_tab == EVAL_SUB_CLUSTER:
            from src.ui.apps.ensemble_analytics.tabs.evaluation.by_cluster import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── Filter visibility & options ───────────────────────────────
    _FILTER_ROW = {"display": "flex", "alignItems": "center", "marginRight": "20px"}
    _HIDDEN = {"display": "none"}

    @app.callback(
        Output("eval-filter-desk", "style"),
        Output("eval-filter-product", "style"),
        Output("eval-filter-ccy", "style"),
        Output("eval-filter-cluster", "style"),
        Input("eval-sub-tabs", "value"),
    )
    def toggle_filter_visibility(sub_tab):
        return (
            _FILTER_ROW if sub_tab == EVAL_SUB_DESK else _HIDDEN,
            _FILTER_ROW if sub_tab == EVAL_SUB_PRODUCT else _HIDDEN,
            _FILTER_ROW if sub_tab == EVAL_SUB_CCY else _HIDDEN,
            _FILTER_ROW if sub_tab == EVAL_SUB_CLUSTER else _HIDDEN,
        )

    @app.callback(
        Output("eval-desk-filter-desk", "options"),
        Output("eval-product-filter-product_type", "options"),
        Output("eval-ccy-filter-ccy", "options"),
        Output("eval-cluster-cluster-dropdown", "options"),
        Output("eval-cluster-cluster-dropdown", "value"),
        Input("eval-sub-tabs", "value"),
    )
    def populate_filter_options(_sub_tab):
        from src.ui.apps.ensemble_analytics_db.data.trade_catalogue import get_trade_catalogue
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session

        catalogue = get_trade_catalogue()
        session = get_session()

        def _opts(logical_col):
            actual = EVAL_GROUP_COLUMNS.get(logical_col, logical_col)
            if catalogue is not None and actual in catalogue.columns:
                vals = sorted(catalogue[actual].dropna().unique().tolist())
                return [{"label": v, "value": v} for v in vals]
            return []

        cluster_opts, default_cluster = [], None
        if session and session.config:
            attrs = session.cluster_attributes
            for cid in session.config.cluster_ids:
                if attrs and cid in attrs:
                    parts = [f"{k}={v}" for k, v in attrs[cid].items() if v is not None]
                    label = f"{cid}  ({', '.join(parts)})" if parts else cid
                else:
                    label = cid
                cluster_opts.append({"label": label, "value": cid})
            default_cluster = session.config.cluster_ids[0] if session.config.cluster_ids else None

        return _opts("desk"), _opts("product_type"), _opts("ccy"), cluster_opts, default_cluster

    # ── Portfolio sub-tab (pre-computed) ──────────────────────────
    @app.callback(
        Output("eval-portfolio-ts", "children"),
        Output("eval-portfolio-scatter", "children"),
        Output("eval-portfolio-residual", "children"),
        Output("eval-portfolio-percentile", "children"),
        Output("eval-portfolio-worst", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
    )
    def update_portfolio(split: str, sub_tab: str):
        if sub_tab != EVAL_SUB_PORTFOLIO:
            return no_update, no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.prediction_store import (
            get_portfolio_summary,
        )

        portfolio = get_portfolio_summary(split)
        if portfolio is None:
            msg = html.Div("No prediction data available for this split.")
            return msg, msg, msg, msg, msg

        portfolio_preds = portfolio["predictions"]
        portfolio_targets = portfolio["targets"]

        ts_fig = dcc.Graph(
            figure=pnl_timeseries(portfolio_preds, portfolio_targets,
                                  title=f"Portfolio PnL — {split.capitalize()}"),
            config={"displayModeBar": False},
        )
        scatter_fig = dcc.Graph(
            figure=pred_vs_target_scatter(portfolio_preds, portfolio_targets,
                                          title=f"Pred vs Target — {split.capitalize()}"),
            config={"displayModeBar": False},
        )
        residual_fig = dcc.Graph(
            figure=residual_histogram(portfolio_preds, portfolio_targets,
                                      title=f"Residual Distribution — {split.capitalize()}"),
            config={"displayModeBar": False},
        )

        pct_cols, pct_rows = percentile_table_data(portfolio_preds, portfolio_targets)
        pct_table = metric_table(pct_cols, pct_rows, "eval-portfolio-pct-table", height="220px")

        worst_cols, worst_rows = worst_scenarios_data(portfolio_preds, portfolio_targets)
        worst_table = metric_table(worst_cols, worst_rows, "eval-portfolio-worst-table", height="400px")

        return ts_fig, scatter_fig, residual_fig, pct_table, worst_table

    # ── Generic group-by sub-tab builder (pre-computed) ───────────
    def _build_group_view(
        split: str,
        group_col: str,
        selected_values: Optional[List[str]],
        id_prefix: str,
    ):
        """Build group sub-tab using pre-computed cluster summaries."""
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_summary
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS, TEXT_SECONDARY

        backend = get_backend()
        cluster_attrs = backend.get_cluster_attributes()
        cluster_data = get_cluster_summary(split)

        if not cluster_data:
            msg = html.Div("No data available.")
            return msg, msg, msg, msg

        # Group clusters by the requested attribute
        groups: dict = {}
        for cid, ca in cluster_attrs.items():
            val = ca.get(group_col)
            if val is not None:
                groups.setdefault(str(val), []).append(cid)

        if selected_values:
            groups = {g: cids for g, cids in groups.items() if g in selected_values}

        group_preds_dict = {}
        group_targets_dict = {}
        group_residuals = {}
        table_rows = []

        for grp in sorted(groups.keys()):
            cids = groups[grp]
            combined_pred = None
            combined_tgt = None
            n_trades = 0
            for cid in cids:
                cd = cluster_data.get(cid)
                if cd is None:
                    continue
                p = cd["predictions"]
                t = cd["targets"]
                combined_pred = p if combined_pred is None else combined_pred + p
                combined_tgt = t if combined_tgt is None else combined_tgt + t
                n_trades += cluster_attrs.get(cid, {}).get("n_trades", 0)

            if combined_pred is None:
                continue

            group_preds_dict[grp] = combined_pred
            group_targets_dict[grp] = combined_tgt
            residual = combined_pred - combined_tgt
            group_residuals[grp] = residual
            table_rows.append({
                "group": grp,
                "n_trades": int(n_trades),
                "mae": float(np.mean(np.abs(residual))),
                "rmse": float(np.sqrt(np.mean(residual ** 2))),
            })

        ts_fig = dcc.Graph(
            figure=overlaid_group_timeseries(
                group_preds_dict,
                title=f"PnL by {group_col.replace('_', ' ').title()} — {split.capitalize()}",
            ),
            config={"displayModeBar": False},
        )
        box_fig = dcc.Graph(
            figure=violin_overlay(
                group_residuals,
                title=f"Residual Distribution by {group_col.replace('_', ' ').title()}",
            ),
            config={"displayModeBar": False},
        )

        n_groups = len(group_preds_dict)
        scatter_grid = html.Div()
        if n_groups > 0:
            import plotly.graph_objects as go
            ncols = min(n_groups, 4)
            nrows = (n_groups + ncols - 1) // ncols
            titles = [t[:25] + "..." if len(t) > 25 else t for t in group_preds_dict.keys()]
            fig = make_subplots(rows=nrows, cols=ncols,
                                subplot_titles=titles,
                                vertical_spacing=0.12,
                                horizontal_spacing=0.08)
            for i, (grp, p) in enumerate(group_preds_dict.items()):
                t = group_targets_dict[grp]
                r, c = i // ncols + 1, i % ncols + 1
                fig.add_trace(go.Scattergl(
                    x=t, y=p, mode="markers",
                    marker=dict(size=2, color=CHART_COLORS[i % len(CHART_COLORS)], opacity=0.5),
                    showlegend=False,
                ), row=r, col=c)
                vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
                fig.add_trace(go.Scattergl(
                    x=[vmin, vmax], y=[vmin, vmax], mode="lines",
                    line=dict(color=TEXT_SECONDARY, dash="dash", width=1),
                    showlegend=False,
                ), row=r, col=c)
            fig.update_layout(height=280 * nrows, title="Pred vs Target — Small Multiples")
            scatter_grid = dcc.Graph(figure=fig, config={"displayModeBar": False})

        col_defs = [
            {"field": "group", "headerName": group_col.replace("_", " ").title()},
            {"field": "n_trades", "headerName": "# Trades"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "rmse", "headerName": "RMSE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        table = metric_table(col_defs, table_rows, f"{id_prefix}-metrics-table", height="300px")

        return ts_fig, box_fig, scatter_grid, table

    # ── By Desk ───────────────────────────────────────────────────
    @app.callback(
        Output("eval-desk-timeseries", "children"),
        Output("eval-desk-boxplot", "children"),
        Output("eval-desk-scatter-grid", "children"),
        Output("eval-desk-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-desk-filter-desk", "value"),
    )
    def update_desk(split, sub_tab, selected_desks):
        if sub_tab != EVAL_SUB_DESK:
            return no_update, no_update, no_update, no_update
        attr_key = next(
            (k for k, v in EVAL_GROUP_COLUMNS.items() if k == "desk"), "desk"
        )
        return _build_group_view(split, attr_key, selected_desks, "eval-desk")

    # ── By Product ────────────────────────────────────────────────
    @app.callback(
        Output("eval-product-timeseries", "children"),
        Output("eval-product-boxplot", "children"),
        Output("eval-product-scatter-grid", "children"),
        Output("eval-product-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-product-filter-product_type", "value"),
    )
    def update_product(split, sub_tab, selected_products):
        if sub_tab != EVAL_SUB_PRODUCT:
            return no_update, no_update, no_update, no_update
        attr_key = next(
            (k for k, v in EVAL_GROUP_COLUMNS.items() if k == "product_type"), "product_type"
        )
        return _build_group_view(split, attr_key, selected_products, "eval-product")

    # ── By CCY ────────────────────────────────────────────────────
    @app.callback(
        Output("eval-ccy-timeseries", "children"),
        Output("eval-ccy-boxplot", "children"),
        Output("eval-ccy-scatter-grid", "children"),
        Output("eval-ccy-correlation", "children"),
        Output("eval-ccy-table-container", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-ccy-filter-ccy", "value"),
    )
    def update_ccy(split, sub_tab, selected_ccys):
        if sub_tab != EVAL_SUB_CCY:
            return no_update, no_update, no_update, no_update, no_update

        attr_key = next(
            (k for k, v in EVAL_GROUP_COLUMNS.items() if k == "ccy"), "ccy"
        )
        ts_fig, box_fig, scatter_grid, table = _build_group_view(
            split, attr_key, selected_ccys, "eval-ccy",
        )

        # Cross-currency residual correlation (pre-computed)
        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_group_correlations

        corr_fig = html.Div("Insufficient data for correlation.")
        correlations = get_group_correlations(split)
        ccy_corr = correlations.get(attr_key)

        if ccy_corr and "columns" in ccy_corr and "values" in ccy_corr:
            cols = ccy_corr["columns"]
            vals = np.array(ccy_corr["values"])
            fig = go.Figure(go.Heatmap(
                z=vals, x=cols, y=cols,
                colorscale="RdBu_r", zmid=0,
                text=np.round(vals, 2).astype(str),
                texttemplate="%{text}",
            ))
            fig.update_layout(title="Cross-CCY Residual Correlation", height=400)
            corr_fig = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return ts_fig, box_fig, scatter_grid, corr_fig, table

    # ── By Cluster (on-demand single-cluster load) ────────────────
    @app.callback(
        Output("eval-cluster-scatter", "children"),
        Output("eval-cluster-timeseries", "children"),
        Output("eval-cluster-violin", "children"),
        Output("eval-cluster-heatmap", "children"),
        Output("eval-cluster-trade-table", "children"),
        Input("eval-split-toggle", "value"),
        Input("eval-sub-tabs", "value"),
        Input("eval-cluster-cluster-dropdown", "value"),
    )
    def update_by_cluster(split, sub_tab, cluster_id):
        _nu5 = (no_update,) * 5
        if sub_tab != EVAL_SUB_CLUSTER:
            return _nu5

        if isinstance(cluster_id, list):
            cluster_id = cluster_id[0] if cluster_id else None
        if not cluster_id and cluster_id != 0:
            return _nu5

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_predictions
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session

        session = get_session()
        cluster_id_str = str(cluster_id)
        data = get_cluster_predictions(cluster_id_str, split)
        if data is None:
            msg = html.Div("No prediction data for this cluster.")
            return msg, msg, msg, msg, msg

        preds = data["predictions"]
        targets = data["targets"]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            targets = targets.reshape(-1, 1)

        trade_ids = session.config.cluster_mapping.get(cluster_id_str, [])
        trade_ids = [str(t) for t in trade_ids[:preds.shape[1]]]

        cluster_pred = preds.sum(axis=1)
        cluster_target = targets.sum(axis=1)

        scatter = dcc.Graph(
            figure=pred_vs_target_scatter(
                cluster_pred, cluster_target,
                title=f"Pred vs Target — {cluster_id} ({split.capitalize()})",
            ),
            config={"displayModeBar": False},
        )

        ts = dcc.Graph(
            figure=pnl_timeseries(
                cluster_pred, cluster_target,
                title=f"PnL Timeseries — {cluster_id} ({split.capitalize()})",
            ),
            config={"displayModeBar": False},
        )

        trade_residuals = {}
        for j, tid in enumerate(trade_ids):
            if j < preds.shape[1]:
                trade_residuals[tid] = preds[:, j] - targets[:, j]
        violin_fig = violin_overlay(trade_residuals, title="Per-Trade Residual Distribution")
        violin = dcc.Graph(figure=violin_fig, config={"displayModeBar": False})

        residuals = preds - targets
        max_scenarios = 500
        heatmap_data = residuals[:max_scenarios] if residuals.shape[0] > max_scenarios else residuals
        hm_fig = go.Figure(go.Heatmap(
            z=heatmap_data, x=trade_ids,
            y=list(range(heatmap_data.shape[0])),
            colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="Residual"),
        ))
        hm_fig.update_layout(
            title=f"Per-Trade Residual Heatmap — {cluster_id}",
            xaxis_title="Trade", yaxis_title="Scenario",
            height=max(300, min(600, 2 * heatmap_data.shape[0])),
        )
        heatmap = dcc.Graph(figure=hm_fig, config={"displayModeBar": False})

        col_defs = [
            {"field": "trade_id", "headerName": "Trade ID"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "rmse", "headerName": "RMSE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            {"field": "max_ae", "headerName": "Max AE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        rows = []
        for j, tid in enumerate(trade_ids):
            if j < preds.shape[1]:
                r = preds[:, j] - targets[:, j]
                rows.append({
                    "trade_id": tid,
                    "mae": float(np.mean(np.abs(r))),
                    "rmse": float(np.sqrt(np.mean(r ** 2))),
                    "max_ae": float(np.max(np.abs(r))),
                })
        table = metric_table(col_defs, rows, "eval-cluster-trade-metrics-table", height="350px")

        return scatter, ts, violin, heatmap, table
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/cluster_deep_dive_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 3 — Cluster Deep Dive (DB-ready version).

Uses single-cluster on-demand prediction loading instead of the
``GlobalPredictionStore``.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, State, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import METRIC_DISPLAY_NAMES
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.scatter import pred_vs_target_scatter
from src.ui.apps.ensemble_analytics.figures.timeseries import pnl_timeseries
from src.ui.apps.ensemble_analytics.figures.distributions import residual_histogram
from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY


def register(app):
    """Register Cluster Deep Dive callbacks on *app*."""

    @app.callback(
        Output("deep-dive-cluster-dropdown", "options"),
        Output("deep-dive-cluster-dropdown", "value"),
        Input("main-tabs", "value"),
    )
    def populate_dd_cluster(tab):
        if tab != "tab-cluster-deep-dive":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        session = get_session()
        attrs = session.cluster_attributes
        opts = []
        for cid in session.config.cluster_ids:
            if attrs and cid in attrs:
                parts = [f"{k}={v}" for k, v in attrs[cid].items() if v is not None]
                label = f"{cid}  ({', '.join(parts)})" if parts else cid
            else:
                label = cid
            opts.append({"label": label, "value": cid})
        default = session.config.cluster_ids[0] if session.config.cluster_ids else None
        return opts, default

    @app.callback(
        Output("deep-dive-header", "children"),
        Output("deep-dive-split-table", "children"),
        Output("deep-dive-convergence", "children"),
        Output("deep-dive-scatter", "children"),
        Output("deep-dive-timeseries", "children"),
        Output("deep-dive-residual", "children"),
        Output("deep-dive-trade-dropdown", "options"),
        Output("deep-dive-trade-dropdown", "value"),
        Output("deep-dive-elementary", "children"),
        Output("deep-dive-model-config", "children"),
        Output("deep-dive-config", "children"),
        Input("deep-dive-cluster-dropdown", "value"),
        Input("deep-dive-split-toggle", "value"),
    )
    def update_deep_dive(cluster_id, split):
        n_out = 11
        if not cluster_id:
            return (no_update,) * n_out

        import json as _json
        from pathlib import Path
        import base64
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_predictions

        session = get_session()
        display = session.load_cluster_display(cluster_id)
        attrs = session.cluster_attributes.get(cluster_id, {})
        tu = display.trade_universe
        dc = display.eval_metrics.get("data_config", {})

        # ── Header ────────────────────────────────────────────────
        attr_str = ", ".join(f"{k}={v}" for k, v in attrs.items() if v)
        n_elem = len(tu.get("elementary_ids", []))
        n_target = len(tu.get("target_ids", session.config.cluster_mapping.get(cluster_id, [])))
        header = html.Div([
            html.Div(f"Cluster: {cluster_id}", style={"fontSize": "18px", "fontWeight": "700"}),
            html.Div(f"Version: {display.version}", style={"color": TEXT_SECONDARY, "fontSize": "13px"}),
            html.Div(attr_str, style={"color": TEXT_SECONDARY, "fontSize": "13px"}) if attr_str else None,
            html.Div(
                f"n_elementary: {n_elem} · n_target: {n_target} · "
                f"seq_length: {dc.get('seq_length', '?')} · "
                f"transform: {dc.get('transform_type', '?')}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
        ])

        # ── Split comparison table ────────────────────────────────
        ens_display = session.ensemble_display
        split_rows = []
        if ens_display:
            for s in ["train", "val", "test"]:
                pm = ens_display.per_member_metrics.get(s, {}).get(cluster_id, {})
                if pm:
                    row = {"split": s.capitalize()}
                    row.update(pm)
                    split_rows.append(row)

        split_col_defs = [{"field": "split", "headerName": "Split", "pinned": "left"}]
        if split_rows:
            for key in split_rows[0]:
                if key != "split":
                    split_col_defs.append({
                        "field": key,
                        "headerName": METRIC_DISPLAY_NAMES.get(key, key.upper()),
                        "valueFormatter": {"function": "d3.format('.4f')(params.value)"},
                    })
        split_table = metric_table(split_col_defs, split_rows, "deep-dive-split-comp-table", height="180px")

        # ── Training convergence PNG ──────────────────────────────
        convergence_content = html.Div("No convergence plot available.", style={"color": TEXT_SECONDARY})
        if session.artifacts_dir:
            plot_path = Path(session.artifacts_dir) / "training" / display.version / "training_plots.png"
            if plot_path.exists():
                with open(plot_path, "rb") as f:
                    encoded = base64.b64encode(f.read()).decode()
                convergence_content = html.Img(
                    src=f"data:image/png;base64,{encoded}",
                    style={"width": "100%", "maxHeight": "400px", "objectFit": "contain"},
                )

        # ── Predictions: scatter, timeseries, residual ────────────
        scatter_content = html.Div("No predictions available.")
        ts_content = html.Div("No predictions available.")
        residual_content = html.Div("No predictions available.")
        trade_opts = []
        trade_default = []
        elementary_content = html.Div("Elementary PnL data not available.", style={"color": TEXT_SECONDARY})

        data = get_cluster_predictions(str(cluster_id), split)
        if data is not None:
            preds = data["predictions"]
            targets = data["targets"]
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
                targets = targets.reshape(-1, 1)
            trade_ids = [str(t) for t in session.config.cluster_mapping.get(cluster_id, [])]
            trade_ids = trade_ids[:preds.shape[1]]

            cluster_pred = preds.sum(axis=1)
            cluster_target = targets.sum(axis=1)

            scatter_content = dcc.Graph(
                figure=pred_vs_target_scatter(cluster_pred, cluster_target,
                                              title=f"Pred vs Target — {split.capitalize()}"),
                config={"displayModeBar": False},
            )
            ts_content = dcc.Graph(
                figure=pnl_timeseries(cluster_pred, cluster_target,
                                      title=f"PnL Timeseries — {split.capitalize()}"),
                config={"displayModeBar": False},
            )
            residual_content = dcc.Graph(
                figure=residual_histogram(cluster_pred, cluster_target,
                                          title=f"Residuals — {split.capitalize()}"),
                config={"displayModeBar": False},
            )

            trade_opts = [{"label": tid, "value": tid} for tid in trade_ids]
            trade_default = trade_ids[:6]

        # ── Elementary PnL Explorer ───────────────────────────────
        version_dir = Path(session.registry_dir) / display.version
        elem_pnl_path = version_dir / "elementary_pnl.parquet"
        if elem_pnl_path.exists():
            import pandas as pd
            import plotly.graph_objects as go
            from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS

            elem_df = pd.read_parquet(elem_pnl_path)
            n_elem = elem_df.shape[1]
            n_scenarios = elem_df.shape[0]

            stats_row = html.Div(
                f"{n_elem} elementary trades × {n_scenarios} scenarios",
                style={"color": TEXT_SECONDARY, "fontSize": "13px", "marginBottom": "8px"},
            )

            show_cols = elem_df.columns[:10]
            fig = go.Figure()
            for i, col in enumerate(show_cols):
                fig.add_trace(go.Scattergl(
                    x=np.arange(n_scenarios), y=elem_df[col].values,
                    mode="lines", name=str(col)[:20],
                    line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=1),
                ))
            fig.update_layout(
                title="Elementary PnL (first 10 trades)",
                xaxis_title="Scenario", yaxis_title="PnL",
                height=350,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
            )

            summary_data = []
            for col in elem_df.columns:
                vals = elem_df[col].values
                summary_data.append({
                    "trade_id": str(col),
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                    "min": round(float(np.min(vals)), 6),
                    "max": round(float(np.max(vals)), 6),
                })
            sum_col_defs = [
                {"field": "trade_id", "headerName": "Elementary Trade"},
                {"field": "mean", "headerName": "Mean"},
                {"field": "std", "headerName": "Std"},
                {"field": "min", "headerName": "Min"},
                {"field": "max", "headerName": "Max"},
            ]
            sum_table = metric_table(sum_col_defs, summary_data, "deep-dive-elem-table", height="300px")

            elementary_content = html.Div([
                stats_row,
                dcc.Graph(figure=fig, config={"displayModeBar": False}),
                sum_table,
            ])

        # ── Model configuration ───────────────────────────────────
        member_cfg = session.config.member_configs.get(cluster_id, {})
        if member_cfg:
            filtered = {
                k: v for k, v in member_cfg.items()
                if k not in ("metadata", "data_config")
            }
            cfg_json = _json.dumps(filtered, indent=2, default=str)
            model_config_content = html.Details([
                html.Summary(
                    "Expand Model / Training Config",
                    style={"cursor": "pointer", "color": TEXT_SECONDARY, "fontSize": "13px"},
                ),
                dcc.Markdown(
                    f"```json\n{cfg_json}\n```",
                    style={"fontSize": "12px", "maxHeight": "500px", "overflow": "auto"},
                ),
            ])
        else:
            model_config_content = html.Div("No member config available.", style={"color": TEXT_SECONDARY})

        # ── Data configuration ────────────────────────────────────
        if dc:
            dc_json = _json.dumps(dc, indent=2, default=str)
            data_config_content = html.Details([
                html.Summary(
                    "Expand Data Config",
                    style={"cursor": "pointer", "color": TEXT_SECONDARY, "fontSize": "13px"},
                ),
                dcc.Markdown(
                    f"```json\n{dc_json}\n```",
                    style={"fontSize": "12px", "maxHeight": "500px", "overflow": "auto"},
                ),
            ])
        else:
            data_config_content = html.Div("No data config available.", style={"color": TEXT_SECONDARY})

        return (header, split_table, convergence_content, scatter_content,
                ts_content, residual_content,
                trade_opts, trade_default,
                elementary_content, model_config_content, data_config_content)

    # ── Per-trade scatter ─────────────────────────────────────────
    @app.callback(
        Output("deep-dive-scatter-matrix", "children"),
        Input("deep-dive-trade-dropdown", "value"),
        State("deep-dive-cluster-dropdown", "value"),
        State("deep-dive-split-toggle", "value"),
    )
    def update_trade_scatter(selected_trades, cluster_id, split):
        if not selected_trades or not cluster_id:
            return html.Div("Select trades above.", style={"color": TEXT_SECONDARY})

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_cluster_predictions
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.theme.colors import CHART_COLORS, TEXT_SECONDARY as TS

        session = get_session()
        data = get_cluster_predictions(str(cluster_id), split)
        if data is None:
            return html.Div("No prediction data.")

        preds = data["predictions"]
        targets = data["targets"]
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            targets = targets.reshape(-1, 1)
        trade_ids = [str(t) for t in session.config.cluster_mapping.get(cluster_id, [])]
        trade_ids = trade_ids[:preds.shape[1]]

        show = [t for t in selected_trades if t in trade_ids][:6]
        if not show:
            return html.Div("Selected trades not found in this cluster.")

        n_show = len(show)
        ncols = min(n_show, 3)
        nrows = (n_show + ncols - 1) // ncols
        titles = [t[:20] + "\u2026" if len(t) > 20 else t for t in show]
        fig = make_subplots(
            rows=nrows, cols=ncols,
            subplot_titles=titles,
            vertical_spacing=0.15,
            horizontal_spacing=0.10,
        )
        for j, tid in enumerate(show):
            idx = trade_ids.index(tid)
            r, c = j // ncols + 1, j % ncols + 1
            t = targets[:, idx]
            p = preds[:, idx]
            residuals = p - t
            fig.add_trace(go.Scattergl(
                x=t, y=p, mode="markers",
                marker=dict(size=3, color=CHART_COLORS[j % len(CHART_COLORS)], opacity=0.5),
                customdata=residuals,
                hovertemplate="Target: %{x:.4f}<br>Pred: %{y:.4f}<br>Residual: %{customdata:.4f}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)
            vmin, vmax = min(t.min(), p.min()), max(t.max(), p.max())
            fig.add_trace(go.Scattergl(
                x=[vmin, vmax], y=[vmin, vmax], mode="lines",
                line=dict(color=TS, dash="dash", width=1),
                showlegend=False,
            ), row=r, col=c)

        fig.update_layout(height=300 * nrows, title="Per-Trade Pred vs Target")
        fig.update_annotations(font_size=11)
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/trade_graph_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 5 — Trade Graph Explorer (DB-ready version).

Node analytics uses pre-computed trade metrics instead of the
``GlobalPredictionStore``.  Graph data and adjacency analysis are
unchanged (still reads from registry joblobs).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from dash import Input, Output, State, dcc, html, no_update, ctx, clientside_callback
import dash_cytoscape as cyto

from src.ui.apps.ensemble_analytics.config import (
    TG_SUB_GRAPH_VIEW,
    TG_SUB_ADJACENCY,
    TG_SUB_NODE_ANALYTICS,
    TG_SUB_CROSS_CLUSTER,
)
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.network import build_cytoscape_elements
from src.ui.apps.ensemble_analytics.figures.heatmaps import adjacency_spy
from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, BG_CARD, TEXT_PRIMARY, TEXT_SECONDARY


def register(app):
    """Register Trade Graph tab callbacks on *app*."""

    @app.callback(
        Output("tg-cluster-dropdown", "options"),
        Output("tg-cluster-dropdown", "value"),
        Input("main-tabs", "value"),
    )
    def populate_tg_cluster(tab):
        if tab != "tab-trade-graph":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        session = get_session()
        attrs = session.cluster_attributes
        opts = []
        for cid in session.config.cluster_ids:
            if attrs and cid in attrs:
                parts = [f"{k}={v}" for k, v in attrs[cid].items() if v is not None]
                label = f"{cid}  ({', '.join(parts)})" if parts else cid
            else:
                label = cid
            opts.append({"label": label, "value": cid})
        default = session.config.cluster_ids[0] if session.config.cluster_ids else None
        return opts, default

    _ROW_VISIBLE = {"display": "flex", "alignItems": "center", "marginRight": "20px"}
    _HIDDEN = {"display": "none"}

    @app.callback(
        Output("tg-graph-controls-layout", "style"),
        Output("tg-graph-controls-threshold", "style"),
        Output("tg-graph-controls-search", "style"),
        Input("tg-sub-tabs", "value"),
    )
    def toggle_tg_graph_controls(sub_tab):
        show = sub_tab == TG_SUB_GRAPH_VIEW
        return (
            _ROW_VISIBLE if show else _HIDDEN,
            {"width": "300px"} if show else _HIDDEN,
            _ROW_VISIBLE if show else _HIDDEN,
        )

    @app.callback(
        Output("tg-sub-tab-content", "children"),
        Input("tg-sub-tabs", "value"),
    )
    def render_tg_sub_tab(sub_tab):
        if sub_tab == TG_SUB_GRAPH_VIEW:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.graph_view import layout
            return layout()
        elif sub_tab == TG_SUB_ADJACENCY:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.adjacency_analysis import layout
            return layout()
        elif sub_tab == TG_SUB_NODE_ANALYTICS:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.node_analytics import layout
            return layout()
        elif sub_tab == TG_SUB_CROSS_CLUSTER:
            from src.ui.apps.ensemble_analytics.tabs.trade_graph.cross_cluster import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    # ── Graph View ────────────────────────────────────────────────
    @app.callback(
        Output("tg-graph-container", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
        Input("tg-layout-selector", "value"),
        Input("tg-weight-threshold", "value"),
        Input("tg-search-box", "value"),
    )
    def update_graph_view(cluster_id, sub_tab, layout_name, threshold, search_term):
        if sub_tab != TG_SUB_GRAPH_VIEW or not cluster_id:
            return no_update

        from src.ui.apps.ensemble_analytics_db.data.graph_data_loader import get_graph_data
        from src.ui.apps.ensemble_analytics_db.data.trade_catalogue import get_trade_catalogue

        gdata = get_graph_data(cluster_id)
        graph_results = gdata.get("graph_results", {})
        trade_universe = gdata.get("trade_universe", {})

        indices = graph_results.get("sparse_indices")
        values = graph_results.get("sparse_values")
        all_ids = trade_universe.get("elementary_ids", []) + trade_universe.get("target_ids", [])
        target_set = set(trade_universe.get("target_ids", []))

        if indices is None or values is None or not all_ids:
            return html.Div("No graph data available.", style={"color": TEXT_SECONDARY})

        indices = np.array(indices)
        values = np.array(values)

        catalogue = get_trade_catalogue()
        catalogue_lookup = {}
        if catalogue is not None and not catalogue.empty:
            for _, row in catalogue.iterrows():
                tid_key = row.get("trade_id") or row.get("id")
                if tid_key:
                    catalogue_lookup[str(tid_key)] = {
                        k: str(v) for k, v in row.items()
                        if k not in ("trade_id", "id") and v is not None and str(v) != "nan"
                    }

        node_attrs = {}
        for tid in all_ids:
            attrs = {"trade_type": "target" if tid in target_set else "elementary"}
            if tid in catalogue_lookup:
                attrs["trade_attrs"] = catalogue_lookup[tid]
            node_attrs[tid] = attrs

        elements = build_cytoscape_elements(
            all_ids, indices, values,
            node_attrs=node_attrs,
            weight_threshold=threshold or 0.01,
        )

        edge_weights = [abs(el["data"]["weight"]) for el in elements if "source" in el.get("data", {})]
        max_w = max(edge_weights) if edge_weights else 1.0
        for el in elements:
            d = el.get("data", {})
            if "source" in d:
                norm = abs(d["weight"]) / max_w if max_w > 0 else 0
                d["norm_weight"] = round(norm, 4)

        neighbour_map = defaultdict(list)
        for el in elements:
            d = el.get("data", {})
            if "source" in d and "target" in d:
                w = d.get("weight", 0.0)
                neighbour_map[d["source"]].append((d["target"], w))
                neighbour_map[d["target"]].append((d["source"], w))

        for el in elements:
            d = el.get("data", {})
            nid = d.get("id")
            if nid is None or "source" in d:
                continue
            nbrs = neighbour_map.get(nid, [])
            d["degree"] = len(nbrs)
            top5 = sorted(nbrs, key=lambda x: abs(x[1]), reverse=True)[:5]
            d["top_neighbours"] = [
                {"id": n_id, "type": node_attrs.get(n_id, {}).get("trade_type", "?"), "weight": round(w, 4)}
                for n_id, w in top5
            ]

        if search_term and search_term.strip():
            term = search_term.strip().lower()
            matched_targets = {
                el["data"]["id"] for el in elements
                if "source" not in el.get("data", {})
                and el["data"].get("trade_type") == "target"
                and term in el["data"]["id"].lower()
            }
            if matched_targets:
                connected_ids = set(matched_targets)
                kept_edges = []
                for el in elements:
                    d = el.get("data", {})
                    if "source" in d:
                        if d["source"] in matched_targets or d["target"] in matched_targets:
                            connected_ids.add(d["source"])
                            connected_ids.add(d["target"])
                            kept_edges.append(el)
                elements = [
                    el for el in elements
                    if "source" not in el.get("data", {}) and el["data"]["id"] in connected_ids
                ] + kept_edges

        layout_opts = {"name": layout_name or "cose", "animate": True}
        if (layout_name or "cose") == "cose":
            layout_opts.update({
                "nodeRepulsion": 8000,
                "idealEdgeLength": 80,
                "nodeOverlap": 20,
            })

        return cyto.Cytoscape(
            id="tg-cytoscape",
            elements=elements,
            layout=layout_opts,
            style={"width": "100%", "height": "550px", "backgroundColor": BG_CARD},
            userZoomingEnabled=False,
            stylesheet=[
                {
                    "selector": "node[trade_type='elementary']",
                    "style": {"background-color": ACCENT_BLUE, "width": 16, "height": 16},
                },
                {
                    "selector": "node[trade_type='target']",
                    "style": {"background-color": "#d29922", "width": 22, "height": 22},
                },
                {
                    "selector": ":selected",
                    "style": {
                        "label": "data(label)", "font-size": "10px",
                        "color": TEXT_PRIMARY,
                        "text-background-color": BG_CARD, "text-background-opacity": 0.8,
                        "text-background-padding": "3px",
                        "background-color": "#f85149", "border-width": 2, "border-color": "#fff",
                    },
                },
                {
                    "selector": "edge",
                    "style": {
                        "width": "mapData(norm_weight, 0, 1, 0.5, 3)",
                        "line-color": "mapData(norm_weight, 0, 1, #30363d, #f85149)",
                        "opacity": "mapData(norm_weight, 0, 1, 0.3, 0.9)",
                    },
                },
            ],
        )

    # ── Zoom controls ─────────────────────────────────────────────
    clientside_callback(
        """
        function(zoomIn, zoomOut, reset, currentZoom) {
            const triggered = dash_clientside.callback_context.triggered;
            if (!triggered || triggered.length === 0) return dash_clientside.no_update;
            const id = triggered[0].prop_id.split('.')[0];
            const z = currentZoom || 1;
            if (id === 'tg-zoom-in')    return Math.min(z * 1.3, 5);
            if (id === 'tg-zoom-out')   return Math.max(z / 1.3, 0.2);
            if (id === 'tg-zoom-reset') return 1;
            return dash_clientside.no_update;
        }
        """,
        Output("tg-cytoscape", "zoom"),
        Input("tg-zoom-in", "n_clicks"),
        Input("tg-zoom-out", "n_clicks"),
        Input("tg-zoom-reset", "n_clicks"),
        State("tg-cytoscape", "zoom"),
    )

    # ── Node detail panel ─────────────────────────────────────────
    @app.callback(
        Output("tg-node-detail", "children"),
        Input("tg-cytoscape", "tapNodeData"),
    )
    def show_node_detail(node_data):
        if not node_data:
            return html.Div("Click a node to see details.", style={"color": TEXT_SECONDARY, "fontSize": "13px"})

        import dash_bootstrap_components as dbc

        tid = node_data.get("id", "?")
        trade_type = node_data.get("trade_type", "unknown")
        degree = node_data.get("degree", 0)
        top_nbrs = node_data.get("top_neighbours", [])
        trade_attrs = node_data.get("trade_attrs", {})

        left_items = [
            html.Div(f"Trade: {tid}", style={"fontSize": "15px", "fontWeight": "600"}),
            html.Div(
                [
                    html.Span(f"Type: {trade_type}", style={"marginRight": "20px"}),
                    html.Span(f"Degree: {degree}"),
                ],
                style={"color": TEXT_SECONDARY, "fontSize": "13px", "marginBottom": "8px"},
            ),
        ]

        if top_nbrs:
            left_items.append(html.Div(
                "Top 5 Neighbours:",
                style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "4px"},
            ))
            for nbr in top_nbrs:
                nbr_id = nbr.get("id", "?") if isinstance(nbr, dict) else str(nbr)
                nbr_type = nbr.get("type", "?") if isinstance(nbr, dict) else "?"
                nbr_w = nbr.get("weight", 0.0) if isinstance(nbr, dict) else 0.0
                left_items.append(html.Div(
                    [
                        html.Span(nbr_id, style={"fontWeight": "500", "marginRight": "10px"}),
                        html.Span(f"({nbr_type})", style={"color": TEXT_SECONDARY, "marginRight": "10px"}),
                        html.Span(f"w={nbr_w:.4f}", style={"color": TEXT_SECONDARY}),
                    ],
                    style={"fontSize": "12px", "marginLeft": "12px", "marginBottom": "2px"},
                ))

        right_items = []
        if trade_attrs:
            right_items.append(html.Div(
                "Trade Attributes:",
                style={"fontSize": "13px", "fontWeight": "600", "marginBottom": "4px"},
            ))
            for k, v in trade_attrs.items():
                right_items.append(html.Div(
                    [
                        html.Span(f"{k}: ", style={"fontWeight": "500", "color": TEXT_SECONDARY}),
                        html.Span(str(v)),
                    ],
                    style={"fontSize": "12px", "marginBottom": "2px"},
                ))
        else:
            right_items.append(html.Div(
                "No trade attributes available.",
                style={"color": TEXT_SECONDARY, "fontSize": "12px"},
            ))

        return dbc.Row([
            dbc.Col(html.Div(left_items), md=6),
            dbc.Col(html.Div(right_items), md=6),
        ])

    # ── Adjacency Analysis ────────────────────────────────────────
    @app.callback(
        Output("tg-adj-stats", "children"),
        Output("tg-adj-weight-hist", "children"),
        Output("tg-adj-degree-dist", "children"),
        Output("tg-adj-spy", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
    )
    def update_adjacency(cluster_id, sub_tab):
        if sub_tab != TG_SUB_ADJACENCY or not cluster_id:
            return no_update, no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.graph_data_loader import get_graph_data
        import plotly.graph_objects as go

        gdata = get_graph_data(cluster_id)
        gr = gdata.get("graph_results", {})
        indices = gr.get("sparse_indices")
        values = gr.get("sparse_values")
        shape = gr.get("sparse_shape", [0, 0])

        if indices is None or values is None:
            msg = html.Div("No adjacency data.", style={"color": TEXT_SECONDARY})
            return msg, msg, msg, msg

        indices = np.array(indices)
        values = np.array(values)
        n_nodes = shape[0] if shape[0] > 0 else max(indices.max() + 1, 1)
        nnz = len(values)
        density = nnz / (n_nodes * n_nodes) if n_nodes > 0 else 0

        stats = html.Div([
            html.Span(f"Nodes: {n_nodes}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Edges: {nnz}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Density: {density:.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Mean weight: {values.mean():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Max weight: {values.max():.4f}", style={"fontSize": "13px"}),
        ])

        weight_hist = go.Figure(go.Histogram(x=values, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8))
        weight_hist.update_layout(title="Edge Weight Distribution", height=350)

        if indices.ndim == 2 and indices.shape[0] == 2:
            rows = indices[0]
        else:
            rows = indices[:, 0]
        degrees = np.bincount(rows.astype(int), minlength=n_nodes)
        degree_hist = go.Figure(go.Histogram(x=degrees, nbinsx=40, marker_color=ACCENT_BLUE, opacity=0.8))
        degree_hist.update_layout(title="Degree Distribution", height=350)

        spy_fig = adjacency_spy(indices, values, list(shape), title=f"Adjacency — Cluster {cluster_id}")

        return (
            stats,
            dcc.Graph(figure=weight_hist, config={"displayModeBar": False}),
            dcc.Graph(figure=degree_hist, config={"displayModeBar": False}),
            dcc.Graph(figure=spy_fig, config={"displayModeBar": False}),
        )

    # ── Node Analytics (uses pre-computed trade_metrics) ──────────
    @app.callback(
        Output("tg-node-degree-scatter", "children"),
        Output("tg-node-feature-table", "children"),
        Input("tg-cluster-dropdown", "value"),
        Input("tg-sub-tabs", "value"),
    )
    def update_node_analytics(cluster_id, sub_tab):
        if sub_tab != TG_SUB_NODE_ANALYTICS or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.graph_data_loader import get_graph_data
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_trade_metrics
        import plotly.graph_objects as go

        gdata = get_graph_data(cluster_id)
        gr = gdata.get("graph_results", {})
        tu = gdata.get("trade_universe", {})
        indices = gr.get("sparse_indices")
        shape = gr.get("sparse_shape", [0, 0])
        trade_ids = tu.get("target_ids", [])

        if indices is None or not trade_ids:
            msg = html.Div("Insufficient data.", style={"color": TEXT_SECONDARY})
            return msg, msg

        indices = np.array(indices)
        n_nodes = shape[0] if shape[0] > 0 else max(indices.max() + 1, 1)
        if indices.ndim == 2 and indices.shape[0] == 2:
            rows = indices[0]
        else:
            rows = indices[:, 0]
        degrees = np.bincount(rows.astype(int), minlength=n_nodes)

        # Use pre-computed trade metrics instead of GlobalPredictionStore
        all_metrics = get_trade_metrics("test")
        trade_mae_map = {
            m["trade_id"]: m["mae"]
            for m in all_metrics
            if m["cluster_id"] == cluster_id
        }
        trade_mae = np.array([trade_mae_map.get(tid, 0.0) for tid in trade_ids])

        n_plot = min(len(degrees), len(trade_mae), len(trade_ids))
        fig = go.Figure(go.Scattergl(
            x=degrees[:n_plot], y=trade_mae[:n_plot],
            mode="markers", marker=dict(size=5, color=ACCENT_BLUE, opacity=0.7),
            text=trade_ids[:n_plot], hoverinfo="text+x+y",
        ))
        fig.update_layout(title="Degree vs MAE", xaxis_title="Node Degree", yaxis_title="MAE", height=400)

        col_defs = [
            {"field": "trade_id", "headerName": "Trade ID"},
            {"field": "degree", "headerName": "Degree"},
            {"field": "mae", "headerName": "MAE", "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
        ]
        table_rows = [
            {"trade_id": trade_ids[i], "degree": int(degrees[i]) if i < len(degrees) else 0,
             "mae": float(trade_mae[i]) if i < len(trade_mae) else 0.0}
            for i in range(n_plot)
        ]
        table = metric_table(col_defs, table_rows, "tg-node-table", height="350px")

        return (
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
            table,
        )

    # ── Cross-Cluster (uses pre-computed graph_stats) ─────────────
    @app.callback(
        Output("tg-cross-cluster-table", "children"),
        Output("tg-cross-cluster-chart", "children"),
        Input("tg-sub-tabs", "value"),
    )
    def update_cross_cluster(sub_tab):
        if sub_tab != TG_SUB_CROSS_CLUSTER:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_backend
        import plotly.graph_objects as go

        backend = get_backend()
        graph_stats = backend.get_graph_stats()

        rows = []
        for cid in sorted(graph_stats.keys()):
            gs = graph_stats[cid]
            rows.append({
                "cluster_id": cid,
                "n_nodes": gs.get("n_nodes", 0),
                "n_edges": gs.get("n_edges", 0),
                "density": gs.get("density", 0),
                "mean_weight": gs.get("mean_weight", 0),
            })

        col_defs = [
            {"field": "cluster_id", "headerName": "Cluster"},
            {"field": "n_nodes", "headerName": "Nodes"},
            {"field": "n_edges", "headerName": "Edges"},
            {"field": "density", "headerName": "Density"},
            {"field": "mean_weight", "headerName": "Mean Weight"},
        ]
        table = metric_table(col_defs, rows, "tg-cross-table", height="300px")

        fig = go.Figure(go.Bar(
            x=[r["cluster_id"] for r in rows],
            y=[r["density"] for r in rows],
            marker_color=ACCENT_BLUE,
            text=[f"{r['density']:.4f}" for r in rows],
            textposition="auto",
        ))
        fig.update_layout(title="Graph Density by Cluster", yaxis_title="Density", height=350)

        return table, dcc.Graph(figure=fig, config={"displayModeBar": False})
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/inference_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 6 — Inference (DB-ready version).

Uses ``get_portfolio_summary("test")`` for the baseline stress
comparison instead of the ``GlobalPredictionStore``.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, State, dcc, html, no_update

from src.ui.apps.ensemble_analytics.components.loading_progress import loading_progress
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_GREEN, ACCENT_RED, TEXT_SECONDARY


def register(app):
    """Register Inference tab callbacks on *app*."""

    @app.callback(
        Output("inference-load-status", "children"),
        Output("inference-run-btn", "disabled"),
        Input("main-tabs", "value"),
    )
    def check_load_status(tab):
        if tab != "tab-inference":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        session = get_session()
        if session.all_inference_ready:
            return (
                html.Span("All models loaded.", style={"color": ACCENT_GREEN, "fontSize": "13px"}),
                False,
            )
        loaded = len(session.inference_ready_clusters)
        total = session.config.n_members
        return (
            html.Span(
                f"Models loaded: {loaded} / {total}",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            ),
            True,
        )

    @app.callback(
        Output("inference-progress-container", "children"),
        Input("inference-load-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def load_models(n_clicks):
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        session = get_session()

        if session.all_inference_ready:
            return html.Span("Already loaded.", style={"color": ACCENT_GREEN, "fontSize": "13px"})

        session.load_inference_state(parallel=True)

        total = session.config.n_members
        loaded = len(session.inference_ready_clusters)
        return loading_progress(total, loaded, id_prefix="inference")

    @app.callback(
        Output("inference-results-container", "children"),
        Output("inference-scenario-table-container", "children"),
        Output("inference-stress-comparison", "children"),
        Input("inference-run-btn", "n_clicks"),
        State("inference-mode", "value"),
        State("inference-scenario-dir", "value"),
        prevent_initial_call=True,
    )
    def run_inference(n_clicks, mode, scenario_dir):
        """Execute inference and display results."""
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics_db.data.prediction_store import get_portfolio_summary
        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE, ACCENT_GREEN

        session = get_session()

        if not session.all_inference_ready:
            err = html.Div("Models not loaded. Click 'Load Models' first.", style={"color": ACCENT_RED})
            return err, html.Div(), html.Div()

        try:
            result = session.run_inference(mode=mode)
        except Exception as exc:
            err = html.Div(f"Inference failed: {exc}", style={"color": ACCENT_RED})
            return err, html.Div(), html.Div()

        predictions = result.get("predictions")
        per_member = result.get("per_member", {})
        metadata = result.get("metadata", {})

        children = []
        children.append(html.Div([
            html.Span(f"Mode: {metadata.get('mode', 'N/A')}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Scenarios: {metadata.get('n_scenarios', 'N/A')}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Targets: {metadata.get('n_targets', 'N/A')}", style={"fontSize": "13px"}),
        ], style={"marginBottom": "16px"}))

        scenario_table = html.Div()
        stress_fig = html.Div()

        if predictions is not None and predictions.ndim >= 1:
            portfolio_pnl = predictions.sum(axis=1) if predictions.ndim > 1 else predictions
            hist = go.Figure(go.Histogram(
                x=portfolio_pnl, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8,
            ))
            hist.update_layout(title="Portfolio PnL Distribution (Inference)",
                               xaxis_title="PnL", yaxis_title="Count", height=350)
            children.append(dcc.Graph(figure=hist, config={"displayModeBar": False}))

            var_95 = float(np.percentile(portfolio_pnl, 5))
            es_95 = float(portfolio_pnl[portfolio_pnl <= var_95].mean()) if (portfolio_pnl <= var_95).any() else var_95
            children.append(html.Div([
                html.Span(f"VaR (95%): {var_95:.4f}  |  ", style={"fontSize": "13px"}),
                html.Span(f"ES (95%): {es_95:.4f}", style={"fontSize": "13px"}),
            ], style={"marginTop": "8px", "marginBottom": "16px"}))

            scenario_col_defs = [
                {"field": "scenario", "headerName": "Scenario"},
                {"field": "portfolio_pnl", "headerName": "Portfolio PnL",
                 "valueFormatter": {"function": "d3.format('.4f')(params.value)"}},
            ]
            for cid in sorted(per_member.keys()):
                scenario_col_defs.append({"field": cid, "headerName": cid,
                                          "valueFormatter": {"function": "d3.format('.4f')(params.value)"}})

            scenario_rows = []
            for i in range(min(len(portfolio_pnl), 200)):
                row = {"scenario": i, "portfolio_pnl": float(portfolio_pnl[i])}
                scenario_rows.append(row)
            scenario_table = metric_table(scenario_col_defs, scenario_rows,
                                          "inference-scenario-detail-table", height="400px")

            # Baseline from pre-computed portfolio summary (instant)
            baseline = get_portfolio_summary("test")
            if baseline is not None:
                baseline_pnl = baseline["predictions"]
                sfig = go.Figure()
                sfig.add_trace(go.Histogram(x=baseline_pnl, nbinsx=50, name="Baseline (Test)",
                                            marker_color=ACCENT_GREEN, opacity=0.6))
                sfig.add_trace(go.Histogram(x=portfolio_pnl, nbinsx=50, name="Inference (Stressed)",
                                            marker_color=ACCENT_BLUE, opacity=0.6))
                sfig.update_layout(title="Baseline vs Stressed Portfolio PnL",
                                   barmode="overlay", height=350)
                stress_fig = dcc.Graph(figure=sfig, config={"displayModeBar": False})

        if per_member:
            col_defs = [
                {"field": "cluster_id", "headerName": "Cluster"},
                {"field": "n_trades", "headerName": "# Trades"},
                {"field": "n_scenarios", "headerName": "# Scenarios"},
            ]
            row_data = [{"cluster_id": cid, **meta} for cid, meta in per_member.items()]
            children.append(metric_table(col_defs, row_data, "inference-member-table", height="250px"))

        return html.Div(children), scenario_table, stress_fig

    @app.callback(
        Output("inference-download-csv", "data"),
        Input("inference-download-csv-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def download_csv(n_clicks):
        return dcc.send_string("", filename="inference_results.csv")
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/governance_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 7 — Model Governance (DB-ready version).

Near-identical to the original — governance is already metadata-driven.
Only import paths to ``session_manager`` are changed.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(app):
    """Register Governance tab callbacks on *app*."""
    from dash import Input, Output, dcc, html, no_update

    @app.callback(
        Output("governance-compare-version", "options"),
        Input("main-tabs", "value"),
    )
    def populate_compare_dropdown(tab):
        if tab != "tab-governance":
            return no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session

        session = get_session()
        current_ver = session.ensemble_version
        versions = session._ens_registry.list_versions()

        opts = []
        for v in versions:
            ver = v["version"] if isinstance(v, dict) else str(v)
            if ver == current_ver:
                continue
            if isinstance(v, dict):
                lbl = f"{ver}  ({v.get('n_members', '?')} clusters, {v.get('n_trades', '?')} trades)"
            else:
                lbl = ver
            opts.append({"label": lbl, "value": ver})
        return opts

    @app.callback(
        Output("governance-comparison-content", "children"),
        Input("governance-compare-version", "value"),
    )
    def compare_versions(compare_version):
        if not compare_version:
            return html.Div(
                "Select a version above to compare.",
                style={"color": "#8b949e", "fontSize": "13px"},
            )

        import json as _json
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
        from src.ui.apps.ensemble_analytics.theme.colors import TEXT_SECONDARY
        import plotly.graph_objects as go

        session = get_session()
        current_ens = session.ensemble_display
        if not current_ens:
            return html.Div("Current display data not loaded.")

        current_metrics = current_ens.ensemble_metrics.get("test", {})
        if not current_metrics:
            return html.Div(
                "No test-split metrics for the current version.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        if session.artifacts_dir is None:
            return html.Div("Artifacts directory not configured.")

        compare_dir = session.artifacts_dir / "ensemble" / compare_version / "evaluation"
        compare_path = compare_dir / "ensemble_metrics.json"
        if not compare_path.exists():
            compare_path = compare_dir / "ensemble_metrics_test.json"
        if not compare_path.exists():
            return html.Div(
                f"No test metrics found for version '{compare_version}'.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        with open(compare_path) as f:
            compare_metrics = _json.load(f)

        rows = []
        all_keys = sorted(set(list(current_metrics.keys()) + list(compare_metrics.keys())))
        for mk in all_keys:
            cv = current_metrics.get(mk)
            ev = compare_metrics.get(mk)
            if cv is None or ev is None:
                continue
            try:
                cv_f = float(cv)
                ev_f = float(ev)
            except (TypeError, ValueError):
                continue
            delta = cv_f - ev_f
            pct = (delta / abs(ev_f) * 100) if ev_f != 0 else 0.0
            rows.append({
                "metric": mk.upper(),
                "current": round(cv_f, 6),
                "compare": round(ev_f, 6),
                "delta": round(delta, 6),
                "pct_change": round(pct, 2),
            })

        if not rows:
            return html.Div(
                "No overlapping numeric metrics found.",
                style={"color": TEXT_SECONDARY, "fontSize": "13px"},
            )

        col_defs = [
            {"field": "metric", "headerName": "Metric"},
            {"field": "current", "headerName": f"Current ({session.ensemble_version})"},
            {"field": "compare", "headerName": compare_version},
            {"field": "delta", "headerName": "Delta"},
            {"field": "pct_change", "headerName": "% Change"},
        ]
        table = metric_table(col_defs, rows, "governance-comparison-table", height="250px")

        labels = [r["metric"] for r in rows]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=f"Current ({session.ensemble_version})",
            x=labels, y=[r["current"] for r in rows],
            marker_color="#58a6ff",
        ))
        fig.add_trace(go.Bar(
            name=compare_version,
            x=labels, y=[r["compare"] for r in rows],
            marker_color="#d29922",
        ))
        fig.update_layout(
            title="Metric Comparison (Test Split)",
            barmode="group", height=350,
            xaxis_title="Metric", yaxis_title="Value",
        )

        return html.Div([
            table,
            dcc.Graph(figure=fig, config={"displayModeBar": False}),
        ])
```

---

## File: src/ui/apps/ensemble_analytics_db/callbacks/market_data_cb.py
**Action:** NEW

```python
"""
Callbacks for Tab 4 — Market Data (DB-ready version).

Identical to the original — market data reads from registry joblobs
and does not use the ``GlobalPredictionStore``.  Only import paths
to ``session_manager`` and ``market_data_loader`` are changed.
"""
from __future__ import annotations

import numpy as np
from dash import Input, Output, dcc, html, no_update

from src.ui.apps.ensemble_analytics.config import (
    MD_SUB_RF_SUMMARY,
    MD_SUB_SHOCK_EXPLORER,
    MD_SUB_SCENARIO_HEATMAP,
    MD_SUB_DISTRIBUTION,
)
from src.ui.apps.ensemble_analytics.components.metric_table import metric_table
from src.ui.apps.ensemble_analytics.figures.heatmaps import rf_scenario_heatmap
from src.ui.apps.ensemble_analytics.figures.distributions import violin_overlay, qq_plot


def register(app):
    """Register Market Data tab callbacks on *app*."""

    @app.callback(
        Output("md-cluster-dropdown", "options"),
        Output("md-cluster-dropdown", "value"),
        Input("main-tabs", "value"),
    )
    def populate_md_cluster(tab):
        if tab != "tab-market-data":
            return no_update, no_update
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session
        session = get_session()
        attrs = session.cluster_attributes
        opts = []
        for cid in session.config.cluster_ids:
            if attrs and cid in attrs:
                parts = [f"{k}={v}" for k, v in attrs[cid].items() if v is not None]
                label = f"{cid}  ({', '.join(parts)})" if parts else cid
            else:
                label = cid
            opts.append({"label": label, "value": cid})
        default = session.config.cluster_ids[0] if session.config.cluster_ids else None
        return opts, default

    _ROW_VISIBLE = {"display": "flex", "alignItems": "center", "marginRight": "20px"}
    _HIDDEN = {"display": "none"}

    @app.callback(
        Output("md-shock-asset-wrapper", "style"),
        Output("md-shock-rf-wrapper", "style"),
        Input("md-sub-tabs", "value"),
    )
    def toggle_md_shock_controls(sub_tab):
        show = sub_tab == MD_SUB_SHOCK_EXPLORER
        s = _ROW_VISIBLE if show else _HIDDEN
        return s, s

    @app.callback(
        Output("md-shock-asset-dd", "options"),
        Output("md-shock-asset-dd", "value"),
        Output("md-shock-rf-dd", "options"),
        Output("md-shock-rf-dd", "value"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def populate_md_shock_selectors(cluster_id, sub_tab):
        if sub_tab != MD_SUB_SHOCK_EXPLORER or not cluster_id:
            return no_update, no_update, no_update, no_update
        from src.ui.apps.ensemble_analytics_db.data.market_data_loader import get_market_data
        mdata = get_market_data(cluster_id)
        asset_names = sorted(mdata.keys())
        all_rfs = sorted({rf for rfs in mdata.values() for rf in rfs})
        a_opts = [{"label": a, "value": a} for a in asset_names]
        r_opts = [{"label": r, "value": r} for r in all_rfs]
        return (a_opts, asset_names[0] if asset_names else None,
                r_opts, all_rfs[0] if all_rfs else None)

    @app.callback(
        Output("md-sub-tab-content", "children"),
        Input("md-sub-tabs", "value"),
    )
    def render_md_sub_tab(sub_tab):
        if sub_tab == MD_SUB_RF_SUMMARY:
            from src.ui.apps.ensemble_analytics.tabs.market_data.rf_summary import layout
            return layout()
        elif sub_tab == MD_SUB_SHOCK_EXPLORER:
            from src.ui.apps.ensemble_analytics.tabs.market_data.shock_explorer import layout
            return layout()
        elif sub_tab == MD_SUB_SCENARIO_HEATMAP:
            from src.ui.apps.ensemble_analytics.tabs.market_data.scenario_heatmap import layout
            return layout()
        elif sub_tab == MD_SUB_DISTRIBUTION:
            from src.ui.apps.ensemble_analytics.tabs.market_data.distribution import layout
            return layout()
        return html.Div("Unknown sub-tab.")

    @app.callback(
        Output("md-rf-summary-table", "children"),
        Output("md-rf-coverage-heatmap", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_rf_summary(cluster_id, sub_tab):
        if sub_tab != MD_SUB_RF_SUMMARY or not cluster_id:
            return no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.market_data_loader import get_market_data
        from src.ui.apps.ensemble_analytics_db.data.session_manager import get_session

        session = get_session()
        all_rfs = {}
        for cid in session.config.cluster_ids:
            mdata = get_market_data(cid)
            for asset, rfs in mdata.items():
                for rf_name in rfs:
                    all_rfs.setdefault(rf_name, set()).add(cid)

        col_defs = [
            {"field": "rf_name", "headerName": "Risk Factor"},
            {"field": "n_clusters", "headerName": "# Clusters"},
            {"field": "clusters", "headerName": "Clusters"},
        ]
        rows = [
            {
                "rf_name": rf,
                "n_clusters": len(cids),
                "clusters": ", ".join(sorted(cids)),
            }
            for rf, cids in sorted(all_rfs.items())
        ]
        table = metric_table(col_defs, rows, "md-rf-inventory-table", height="400px")

        import plotly.graph_objects as go
        cluster_ids = session.config.cluster_ids
        rf_names = sorted(all_rfs.keys())
        coverage = np.zeros((len(rf_names), len(cluster_ids)))
        for i, rf in enumerate(rf_names):
            for j, cid in enumerate(cluster_ids):
                if cid in all_rfs.get(rf, set()):
                    coverage[i, j] = 1.0

        fig = go.Figure(go.Heatmap(
            z=coverage, x=cluster_ids, y=rf_names,
            colorscale=[[0, "#161b22"], [1, "#58a6ff"]],
            showscale=False,
        ))
        fig.update_layout(title="RF Coverage Matrix", height=max(300, 18 * len(rf_names) + 100))
        heatmap = dcc.Graph(figure=fig, config={"displayModeBar": False})

        return table, heatmap

    @app.callback(
        Output("md-shock-timeseries", "children"),
        Output("md-shock-distribution", "children"),
        Output("md-shock-stats", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-shock-asset-dd", "value"),
        Input("md-shock-rf-dd", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_shock_explorer(cluster_id, asset, rf, sub_tab):
        if sub_tab != MD_SUB_SHOCK_EXPLORER or not all([cluster_id, asset, rf]):
            return no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        shocks = mdata.get(asset, {}).get(rf)
        if shocks is None:
            msg = html.Div("No shock data for this asset/RF combination.")
            return msg, msg, msg

        import plotly.graph_objects as go
        from src.ui.apps.ensemble_analytics.theme.colors import ACCENT_BLUE

        ts = go.Figure(go.Scattergl(
            x=np.arange(len(shocks)), y=shocks, mode="lines",
            line=dict(color=ACCENT_BLUE, width=1.5),
        ))
        ts.update_layout(title=f"{asset} — {rf} Shocks", xaxis_title="Scenario", yaxis_title="Shock", height=350)

        hist = go.Figure(go.Histogram(x=shocks, nbinsx=60, marker_color=ACCENT_BLUE, opacity=0.8))
        hist.update_layout(title=f"{rf} Shock Distribution", height=350)

        stats_content = html.Div([
            html.Span(f"Mean: {shocks.mean():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Std: {shocks.std():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Min: {shocks.min():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"Max: {shocks.max():.4f}  |  ", style={"fontSize": "13px"}),
            html.Span(f"N: {len(shocks)}", style={"fontSize": "13px"}),
        ])

        return (
            dcc.Graph(figure=ts, config={"displayModeBar": False}),
            dcc.Graph(figure=hist, config={"displayModeBar": False}),
            stats_content,
        )

    @app.callback(
        Output("md-heatmap-container", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_scenario_heatmap(cluster_id, sub_tab):
        if sub_tab != MD_SUB_SCENARIO_HEATMAP or not cluster_id:
            return no_update

        from src.ui.apps.ensemble_analytics_db.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        if not mdata:
            return html.Div("No market data available.")

        all_rfs = {}
        for asset, rfs in mdata.items():
            for rf_name, arr in rfs.items():
                all_rfs[f"{asset}/{rf_name}"] = arr

        if not all_rfs:
            return html.Div("No RF shocks found.")

        rf_names = sorted(all_rfs.keys())
        n_scenarios = min(arr.shape[0] for arr in all_rfs.values())
        matrix = np.column_stack([all_rfs[rf][:n_scenarios] for rf in rf_names])

        fig = rf_scenario_heatmap(rf_names, matrix, title=f"Cluster {cluster_id} — RF × Scenario")
        return dcc.Graph(figure=fig, config={"displayModeBar": False})

    @app.callback(
        Output("md-dist-violin", "children"),
        Output("md-dist-qq", "children"),
        Output("md-dist-corr-heatmap", "children"),
        Input("md-cluster-dropdown", "value"),
        Input("md-sub-tabs", "value"),
    )
    def update_distribution(cluster_id, sub_tab):
        if sub_tab != MD_SUB_DISTRIBUTION or not cluster_id:
            return no_update, no_update, no_update

        from src.ui.apps.ensemble_analytics_db.data.market_data_loader import get_market_data

        mdata = get_market_data(cluster_id)
        if not mdata:
            msg = html.Div("No market data available.")
            return msg, msg, msg

        rf_arrays = {}
        for asset, rfs in mdata.items():
            for rf_name, arr in rfs.items():
                rf_arrays[f"{asset}/{rf_name}"] = arr

        if not rf_arrays:
            msg = html.Div("No RF shocks found.")
            return msg, msg, msg

        subset = dict(list(rf_arrays.items())[:10])
        violin_fig = dcc.Graph(
            figure=violin_overlay(subset, title="RF Shock Distributions"),
            config={"displayModeBar": False},
        )

        first_rf = next(iter(rf_arrays.values()))
        qq_fig = dcc.Graph(
            figure=qq_plot(first_rf, title=f"QQ — {next(iter(rf_arrays.keys()))}"),
            config={"displayModeBar": False},
        )

        import plotly.graph_objects as go
        import pandas as pd
        df = pd.DataFrame({k: v[:min(len(v) for v in rf_arrays.values())]
                           for k, v in list(rf_arrays.items())[:20]})
        corr = df.corr()
        heatmap_fig = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
            colorscale="RdBu_r", zmid=0,
        ))
        heatmap_fig.update_layout(title="RF Correlation Matrix", height=500)
        corr_content = dcc.Graph(figure=heatmap_fig, config={"displayModeBar": False})

        return violin_fig, qq_fig, corr_content
```

---

## File: examples/rade_ml_pt/hybrid_gnn_rnn/09_run_ensemble_dashboard_db.py
**Action:** NEW

```python
"""
Launch the DB-ready Ensemble Analytics dashboard.

This is the fast-loading version of the ensemble dashboard that reads
from pre-computed ``db_ready/`` files instead of building the full
``GlobalPredictionStore``.  Supports both ``file`` and ``sqlite``
backends.

Prerequisites
-------------
Run the ensemble evaluation pipeline with ``save_db_artifacts=True``
(the default) to generate the ``db_ready/`` directory.

Optionally, publish to SQLite for query-based access::

    from src.rade_ml_pt.ensemble.publish_to_db import publish_to_sqlite
    publish_to_sqlite("/path/to/evaluation/db_ready")
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = str(Path(__file__).resolve().parents[3])
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ── Configuration ─────────────────────────────────────────────────
REGISTRY_DIR = "/path/to/your/registry"
ARTIFACTS_DIR = "/path/to/your/artifacts"
VERSION = "latest"

# Backend: "file" reads from db_ready/ JSON+NPZ (zero deps, recommended)
#          "sqlite" reads from a local .db file (requires publish_to_sqlite first)
BACKEND = "file"

# Only needed for sqlite backend — path to the .db file.
# If None, defaults to db_ready/ensemble.db
DB_PATH = None


def main():
    from src.ui.apps.ensemble_analytics_db import create_app

    app = create_app(
        registry_dir=REGISTRY_DIR,
        artifacts_dir=ARTIFACTS_DIR,
        version=VERSION,
        backend=BACKEND,
        db_path=DB_PATH,
        debug=True,
    )

    print(f"\nStarting DB-ready Ensemble Analytics Dashboard")
    print(f"  Backend:   {BACKEND}")
    print(f"  Registry:  {REGISTRY_DIR}")
    print(f"  Artifacts: {ARTIFACTS_DIR}")
    print(f"  Version:   {VERSION}")
    print(f"  URL:       http://127.0.0.1:8052\n")

    app.run(host="127.0.0.1", port=8052, debug=True)


if __name__ == "__main__":
    main()
```
