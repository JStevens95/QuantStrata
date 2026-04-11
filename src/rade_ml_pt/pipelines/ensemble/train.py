"""
Ensemble training pipeline.

Orchestrates N member training runs (one per cluster) using each cluster's
configured pipeline class, then assembles and registers the ensemble.

Execution strategies
--------------------
``EnsembleConfig.execution_strategy`` controls how member training is
parallelised:

- ``"sequential"`` — one at a time in a for-loop (default, simplest).
- ``"process_pool"`` — ``concurrent.futures.ProcessPoolExecutor``.
- ``"gpu_parallel"`` — ``torch.multiprocessing`` with per-device affinity.
- ``"distributed"`` — external backend (Ray, Vertex AI, K8s).

Only ``"sequential"`` is implemented today.  The architecture is designed so
that adding a new strategy requires only a new ``_run_<strategy>`` method and
one line in ``_dispatch_training``.  The per-member work function
``train_single_member`` is module-level and fully picklable so it can be
shipped across process/machine boundaries.
"""
from __future__ import annotations

import importlib
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TYPE_CHECKING

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import TrainingResult
    from src.rade_ml_pt.pipelines.base import TrainPipeline

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.HybridGnnRnnTrainPipeline"

_SUPPORTED_STRATEGIES = {"sequential", "process_pool", "gpu_parallel"}
_PLANNED_STRATEGIES = {"distributed"}


def _import_pipeline_class(dotpath: str) -> type:
    """Dynamically import a TrainPipeline subclass from a dotpath string."""
    module_path, cls_name = dotpath.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


# ======================================================================
# Module-level helpers
# ======================================================================

def _read_trained_target_ids(version_dir: Path) -> Optional[List[str]]:
    """Read the actual target trade IDs from a member's saved trade_universe.json.

    After training the data pipeline may have filtered or reduced the
    original trade list, so the model's output dimension corresponds to
    these IDs — not the full ``cluster_mapping`` from the EnsembleConfig.
    """
    tu_path = version_dir / "trade_universe.json"
    if not tu_path.exists():
        return None
    try:
        with open(tu_path, "r") as f:
            universe = json.load(f)
        return universe.get("target_ids")
    except Exception as exc:
        logger.warning("Could not read target_ids from %s: %s", tu_path, exc)
        return None


# ======================================================================
# Module-level worker — picklable, portable across processes / nodes
# ======================================================================

def train_single_member(
    config: EnsembleConfig,
    cluster_id: str,
    default_pipeline: str = _DEFAULT_PIPELINE,
) -> Dict[str, Any]:
    """
    Train one cluster's member model and return the result.

    This is a **module-level function** (not a method) so that it can be
    pickled and dispatched to ``ProcessPoolExecutor``,
    ``torch.multiprocessing``, or ``ray.remote`` without any special
    serialisation logic.

    Parameters
    ----------
    config : EnsembleConfig
        Full ensemble config (picklable dataclass).
    cluster_id : str
        Which cluster to train.
    default_pipeline : str
        Fallback dotpath if no pipeline_class is specified for this cluster.

    Returns
    -------
    dict
        ``{"result": TrainingResult, "version": str}``.
    """
    dotpath = config.pipeline_class.get(cluster_id) or default_pipeline
    pipeline_cls = _import_pipeline_class(dotpath)

    member_config = config.get_member_pipeline_config(cluster_id)

    existing_tags = member_config.metadata.get("tags", [])
    member_config.metadata["tags"] = existing_tags + [
        f"{cluster_id}_latest",
        f"ensemble_member_{cluster_id}",
    ]

    pipeline: "TrainPipeline" = pipeline_cls(member_config)
    result = pipeline.run()

    version = "unknown"
    trained_target_ids: Optional[List[str]] = None
    if hasattr(pipeline, "_registered_entry") and pipeline._registered_entry is not None:
        version = pipeline._registered_entry.version
        trained_target_ids = _read_trained_target_ids(
            Path(pipeline._registered_entry.model_dir)
        )

    return {
        "result": result,
        "version": version,
        "trained_target_ids": trained_target_ids,
    }


def _train_member_with_threads(
    config: EnsembleConfig,
    cluster_id: str,
    num_threads: int,
    default_pipeline: str = _DEFAULT_PIPELINE,
) -> Dict[str, Any]:
    """Train one member with PyTorch's intra-op thread count limited.

    Prevents thread contention when multiple members train in parallel
    on a shared CPU.  Each spawned process receives a pickled copy of
    *config*, so calling ``torch.set_num_threads`` only affects this worker.
    """
    import torch
    torch.set_num_threads(num_threads)
    return train_single_member(config, cluster_id, default_pipeline)


def _train_member_on_device(
    config: EnsembleConfig,
    cluster_id: str,
    device_str: str,
    default_pipeline: str = _DEFAULT_PIPELINE,
) -> Dict[str, Any]:
    """Train one member with its training strategy overridden to *device_str*.

    Each spawned process receives a pickled copy of *config*, so the
    mutation below is safe and does not affect other workers.
    """
    raw = config.member_configs.get(cluster_id, {})
    tc = raw.get("training_config") or {}
    if isinstance(tc, dict):
        tc = {**tc, "strategy": device_str}
    config.member_configs.setdefault(cluster_id, {})["training_config"] = tc
    return train_single_member(config, cluster_id, default_pipeline)


# ======================================================================
# Pipeline
# ======================================================================

class EnsembleTrainPipeline:
    """
    Train N member models (one per cluster) and register the ensemble.

    Parameters
    ----------
    config : EnsembleConfig
        Full ensemble configuration including per-cluster member configs
        and ``execution_strategy``.
    tags : list of str or None
        Tags to apply to the registered ensemble version.
    """

    def __init__(
        self,
        config: EnsembleConfig,
        tags: Optional[List[str]] = None,
    ) -> None:
        self.config = config
        self.tags = tags or []

    def run(self) -> Dict[str, Any]:
        """
        Execute the full ensemble training pipeline.

        Steps
        -----
        1. Create artifacts and registry directories.
        2. Dispatch member training via the configured execution strategy.
        3. Build ensemble via ``EnsembleBuilder`` to validate trade coverage.
        4. Register ensemble in ``EnsembleRegistry``.
        5. Save per-member summary.

        Returns
        -------
        dict
            ``{ensemble_version, member_versions, member_results}``.
        """
        strategy = self.config.execution_strategy
        logger.info(
            "EnsembleTrainPipeline: starting (%d clusters, strategy='%s')",
            self.config.n_members, strategy,
        )
        t0 = time.perf_counter()

        if self.config.artifacts_dir:
            Path(self.config.artifacts_dir).mkdir(parents=True, exist_ok=True)
        if self.config.registry_dir:
            Path(self.config.registry_dir).mkdir(parents=True, exist_ok=True)

        member_versions, member_results = self._dispatch_training()

        # Build ensemble to validate coverage.
        if self.config.registry_dir:
            from src.rade_ml_pt.registry.store import ModelRegistry
            registry = ModelRegistry(self.config.registry_dir)
            builder = EnsembleBuilder(registry)
            _ = builder.build(self.config, member_versions)

        # Register ensemble.
        ensemble_version = self._register_ensemble(member_versions, member_results)

        elapsed = time.perf_counter() - t0
        logger.info(
            "EnsembleTrainPipeline: done (%.1fs). Ensemble version: '%s'",
            elapsed, ensemble_version,
        )

        return {
            "ensemble_version": ensemble_version,
            "member_versions": member_versions,
            "member_results": member_results,
        }

    # ------------------------------------------------------------------
    # Execution strategy dispatch
    # ------------------------------------------------------------------

    def _dispatch_training(self) -> tuple:
        """
        Route to the configured execution strategy.

        Adding a new strategy:
            1. Implement ``_run_<strategy>(self) -> (member_versions, member_results)``.
            2. Add the strategy name to ``_SUPPORTED_STRATEGIES``.
            3. Add an ``elif`` branch here.
        """
        strategy = self.config.execution_strategy

        if strategy == "sequential":
            return self._run_sequential()
        elif strategy == "process_pool":
            return self._run_process_pool()
        elif strategy == "gpu_parallel":
            return self._run_gpu_parallel()
        elif strategy in _PLANNED_STRATEGIES:
            raise NotImplementedError(
                f"Execution strategy '{strategy}' is planned but not yet "
                f"implemented. Available now: "
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

    def _run_sequential(self) -> tuple:
        """Train all members sequentially in a single process."""
        member_versions: Dict[str, str] = {}
        member_results: Dict[str, "TrainingResult"] = {}

        for cid in self.config.cluster_ids:
            logger.info("--- Training member '%s' ---", cid)
            out = train_single_member(self.config, cid)
            member_results[cid] = out["result"]
            member_versions[cid] = out["version"]
            self._sync_cluster_mapping(cid, out.get("trained_target_ids"))
            logger.info(
                "Member '%s' trained: version='%s', best_val_loss=%.6f",
                cid, out["version"],
                out["result"].best_val_loss if out["result"].best_val_loss is not None else float("nan"),
            )

        return member_versions, member_results

    # ------------------------------------------------------------------
    # Strategy: process_pool (multi-CPU)
    # ------------------------------------------------------------------

    def _run_process_pool(self) -> tuple:
        """Train members in parallel across CPU cores using ProcessPoolExecutor.

        PyTorch's intra-op thread count is divided evenly across workers to
        prevent over-subscription (e.g. 12 cores / 4 workers = 3 threads each).
        """
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed

        n_clusters = len(self.config.cluster_ids)
        max_w = self.config.max_workers or n_clusters
        total_cores = os.cpu_count() or 1
        threads_per_worker = max(1, total_cores // max_w)
        logger.info(
            "process_pool: %d members across %d workers (%d threads/worker on %d cores)",
            n_clusters, max_w, threads_per_worker, total_cores,
        )

        member_versions: Dict[str, str] = {}
        member_results: Dict[str, "TrainingResult"] = {}

        with ProcessPoolExecutor(max_workers=max_w) as pool:
            futures = {
                pool.submit(
                    _train_member_with_threads,
                    self.config, cid, threads_per_worker,
                ): cid
                for cid in self.config.cluster_ids
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    out = future.result()
                except Exception:
                    logger.exception("Member '%s' failed in process_pool", cid)
                    raise
                member_results[cid] = out["result"]
                member_versions[cid] = out["version"]
                self._sync_cluster_mapping(cid, out.get("trained_target_ids"))
                logger.info(
                    "Member '%s' trained: version='%s', best_val_loss=%.6f",
                    cid, out["version"],
                    out["result"].best_val_loss if out["result"].best_val_loss is not None else float("nan"),
                )

        return member_versions, member_results

    # ------------------------------------------------------------------
    # Strategy: gpu_parallel (multi-GPU)
    # ------------------------------------------------------------------

    def _resolve_gpu_ids(self) -> List[int]:
        """Return the list of CUDA device IDs to use for gpu_parallel."""
        if self.config.gpu_device_ids:
            return list(self.config.gpu_device_ids)
        import torch
        n = torch.cuda.device_count()
        if n == 0:
            raise RuntimeError(
                "gpu_parallel strategy requires at least one CUDA GPU, "
                "but torch.cuda.device_count() == 0."
            )
        return list(range(n))

    def _run_gpu_parallel(self) -> tuple:
        """Train members in parallel, each pinned to a specific CUDA device.

        Uses the ``spawn`` multiprocessing context so that each child process
        gets a fresh CUDA context.  Members are round-robin assigned to the
        available GPUs; ``max_workers`` is capped at the number of GPUs.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        gpu_ids = self._resolve_gpu_ids()
        n_gpus = len(gpu_ids)
        max_w = min(self.config.max_workers or n_gpus, n_gpus)
        ctx = mp.get_context("spawn")

        assignments = {
            cid: f"cuda:{gpu_ids[i % n_gpus]}"
            for i, cid in enumerate(self.config.cluster_ids)
        }
        logger.info(
            "gpu_parallel: %d members across %d GPUs (max_workers=%d). "
            "Assignments: %s",
            len(self.config.cluster_ids), n_gpus, max_w, assignments,
        )

        member_versions: Dict[str, str] = {}
        member_results: Dict[str, "TrainingResult"] = {}

        with ProcessPoolExecutor(max_workers=max_w, mp_context=ctx) as pool:
            futures = {
                pool.submit(
                    _train_member_on_device,
                    self.config, cid, assignments[cid],
                ): cid
                for cid in self.config.cluster_ids
            }
            for future in as_completed(futures):
                cid = futures[future]
                try:
                    out = future.result()
                except Exception:
                    logger.exception("Member '%s' failed on %s", cid, assignments[cid])
                    raise
                member_results[cid] = out["result"]
                member_versions[cid] = out["version"]
                self._sync_cluster_mapping(cid, out.get("trained_target_ids"))
                logger.info(
                    "Member '%s' trained on %s: version='%s', best_val_loss=%.6f",
                    cid, assignments[cid], out["version"],
                    out["result"].best_val_loss if out["result"].best_val_loss is not None else float("nan"),
                )

        return member_versions, member_results

    # ------------------------------------------------------------------
    # Cluster mapping sync
    # ------------------------------------------------------------------

    def _sync_cluster_mapping(
        self,
        cluster_id: str,
        trained_target_ids: Optional[List[str]],
    ) -> None:
        """Update ``cluster_mapping`` to reflect the trades the model was
        actually trained on.

        The data pipeline may filter or reduce the original trade list
        (dimensionality reduction, trade selection, etc.), so the model's
        output columns correspond to ``trained_target_ids`` — not the full
        list originally placed in ``cluster_mapping``.
        """
        if trained_target_ids is None:
            return
        original = self.config.cluster_mapping.get(cluster_id, [])
        if len(trained_target_ids) != len(original):
            logger.info(
                "Cluster '%s': updating cluster_mapping from %d -> %d trades "
                "(data pipeline filtered trades during training)",
                cluster_id, len(original), len(trained_target_ids),
            )
            self.config.cluster_mapping[cluster_id] = trained_target_ids

    # ------------------------------------------------------------------
    # Ensemble registration
    # ------------------------------------------------------------------

    def _register_ensemble(
        self,
        member_versions: Dict[str, str],
        member_results: Dict[str, "TrainingResult"],
    ) -> str:
        """Register the ensemble and save member summary."""
        if not self.config.registry_dir:
            logger.warning("No registry_dir set; skipping ensemble registration.")
            return "not_registered"

        member_summary: Dict[str, Dict[str, Any]] = {}
        for cid, result in member_results.items():
            member_summary[cid] = {
                "n_trades": len(self.config.cluster_mapping.get(cid, [])),
                "best_val_loss": result.best_val_loss,
                "best_train_loss": result.best_train_loss,
                "final_epoch": result.final_epoch,
                "stopped_early": result.stopped_early,
                "training_time_seconds": result.training_time_seconds,
            }

        ens_registry = EnsembleRegistry(self.config.registry_dir)
        version = ens_registry.register(
            config=self.config,
            member_versions=member_versions,
            member_summary=member_summary,
            tags=self.tags,
        )

        # Save member summary to artifacts too.
        if self.config.artifacts_dir:
            ens_dir = Path(self.config.artifacts_dir) / "ensemble" / version
            ens_dir.mkdir(parents=True, exist_ok=True)
            with open(ens_dir / "member_summary.json", "w") as f:
                json.dump(member_summary, f, indent=2)

        return version
