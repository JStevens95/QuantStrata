"""
Ensemble training pipeline.

Orchestrates N member training runs (one per cluster) using each cluster's
configured pipeline class, then assembles and registers the ensemble.
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


def _import_pipeline_class(dotpath: str) -> type:
    """Dynamically import a TrainPipeline subclass from a dotpath string."""
    module_path, cls_name = dotpath.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


class EnsembleTrainPipeline:
    """
    Train N member models (one per cluster) and register the ensemble.

    Parameters
    ----------
    config : EnsembleConfig
        Full ensemble configuration including per-cluster member configs.
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
        2. For each cluster:
           a. Resolve pipeline class (from config or default).
           b. Build per-member ``PipelineConfig``.
           c. Run ``pipeline_cls(member_config).run()``.
           d. Record the registered model version.
        3. Build ensemble via ``EnsembleBuilder``.
        4. Register ensemble in ``EnsembleRegistry``.
        5. Save per-member summary.

        Returns
        -------
        dict
            ``{ensemble_version, member_versions, member_results}``.
        """
        logger.info("EnsembleTrainPipeline: starting (%d clusters)", self.config.n_members)
        t0 = time.perf_counter()

        if self.config.artifacts_dir:
            Path(self.config.artifacts_dir).mkdir(parents=True, exist_ok=True)
        if self.config.registry_dir:
            Path(self.config.registry_dir).mkdir(parents=True, exist_ok=True)

        member_versions: Dict[str, str] = {}
        member_results: Dict[str, "TrainingResult"] = {}

        for cid in self.config.cluster_ids:
            logger.info("--- Training member '%s' ---", cid)
            result, version = self._train_member(cid)
            member_results[cid] = result
            member_versions[cid] = version
            logger.info(
                "Member '%s' trained: version='%s', best_val_loss=%.6f",
                cid, version,
                result.best_val_loss if result.best_val_loss is not None else float("nan"),
            )

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
    # Member training
    # ------------------------------------------------------------------

    def _train_member(self, cluster_id: str) -> tuple:
        """Train a single cluster's model and return (TrainingResult, version_string)."""
        dotpath = self.config.pipeline_class.get(cluster_id) or _DEFAULT_PIPELINE
        pipeline_cls = _import_pipeline_class(dotpath)

        member_config = self.config.get_member_pipeline_config(cluster_id)

        # Inject a cluster-specific tag so the member version is retrievable.
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

        return result, version

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
