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
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

if TYPE_CHECKING:
    from src.rade_ml_pt.core.types import TrainingResult
    from src.rade_ml_pt.pipelines.base import TrainPipeline

logger = logging.getLogger(__name__)

_DEFAULT_PIPELINE = "src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train.HybridGnnRnnTrainPipeline"

_SUPPORTED_STRATEGIES = {"sequential"}
_PLANNED_STRATEGIES = {"process_pool", "gpu_parallel", "distributed"}


def _import_pipeline_class(dotpath: str) -> type:
    """Dynamically import a TrainPipeline subclass from a dotpath string."""
    module_path, cls_name = dotpath.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


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
    if hasattr(pipeline, "_registered_entry") and pipeline._registered_entry is not None:
        version = pipeline._registered_entry.version

    return {"result": result, "version": version}


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
        elif strategy in _PLANNED_STRATEGIES:
            raise NotImplementedError(
                f"Execution strategy '{strategy}' is planned but not yet "
                f"implemented. Contributions welcome — see ARCHITECTURE.md "
                f"for the extension point contract. Available now: "
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
            logger.info(
                "Member '%s' trained: version='%s', best_val_loss=%.6f",
                cid, out["version"],
                out["result"].best_val_loss if out["result"].best_val_loss is not None else float("nan"),
            )

        return member_versions, member_results

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
