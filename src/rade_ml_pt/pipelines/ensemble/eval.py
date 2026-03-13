"""
Ensemble evaluation pipeline.

Loads an ensemble from the registry, routes test data to each member,
collects per-member and ensemble-level metrics, and saves evaluation
artifacts for the UI dashboard.
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
    """

    def __init__(
        self,
        ensemble_config: EnsembleConfig,
        ensemble_version: str = "latest",
    ) -> None:
        self.config = ensemble_config
        self.ensemble_version = ensemble_version

    def run(self) -> Dict[str, Any]:
        """
        Execute the full ensemble evaluation.

        Steps
        -----
        1. Load ensemble (all members + router) from registry.
        2. For each member: run evaluation via the base ``Evaluator``.
        3. Aggregate predictions across members.
        4. Compute ensemble-level metrics.
        5. Save evaluation artifacts.

        Returns
        -------
        dict
            ``{ensemble_metrics, per_member_metrics, member_summary}``.
        """
        logger.info("EnsembleEvalPipeline: starting")
        t0 = time.perf_counter()

        # Load ensemble.
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, resolved_version = ens_registry.load(self.ensemble_version)

        from src.rade_ml_pt.registry.store import ModelRegistry
        registry = ModelRegistry(self.config.registry_dir)
        builder = EnsembleBuilder(registry)
        ensemble = builder.build(config, member_versions)

        # Evaluate each member.
        from src.rade_ml_pt.evaluation.evaluator import Evaluator

        per_member_preds: Dict[str, np.ndarray] = {}
        per_member_targets: Dict[str, np.ndarray] = {}
        per_member_eval: Dict[str, Dict[str, float]] = {}

        for cid in config.cluster_ids:
            logger.info("--- Evaluating member '%s' ---", cid)
            member_result = self._evaluate_member(
                cid, ensemble.members[cid], config, registry, member_versions[cid],
            )
            if member_result is not None:
                per_member_preds[cid] = member_result["predictions"]
                per_member_targets[cid] = member_result["targets"]
                per_member_eval[cid] = member_result["metrics"]

        # Per-member metrics.
        per_member_metrics = compute_per_member_metrics(per_member_preds, per_member_targets)
        member_rollup = aggregate_member_metrics(per_member_metrics)

        # Ensemble-level metrics (from aggregated predictions).
        ensemble_metrics: Dict[str, float] = {}
        if per_member_preds:
            try:
                combined_preds = ensemble._combine(per_member_preds)
                combined_targets = ensemble._combine(per_member_targets)
                ensemble_metrics = compute_ensemble_metrics(combined_preds, combined_targets)
            except Exception as exc:
                logger.warning("Could not compute ensemble-level metrics: %s", exc)

        # Save artifacts.
        if self.config.artifacts_dir:
            self._save_artifacts(
                resolved_version, per_member_metrics, ensemble_metrics, member_rollup,
            )

        elapsed = time.perf_counter() - t0
        logger.info("EnsembleEvalPipeline: done (%.1fs)", elapsed)

        return {
            "ensemble_version": resolved_version,
            "ensemble_metrics": ensemble_metrics,
            "per_member_metrics": per_member_metrics,
            "member_summary": member_rollup,
        }

    # ------------------------------------------------------------------
    # Per-member evaluation
    # ------------------------------------------------------------------

    def _evaluate_member(
        self,
        cluster_id: str,
        model: Any,
        config: EnsembleConfig,
        registry: Any,
        member_version: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluate a single member model on its test data.

        Attempts to load cached datasets from the member's registry directory.
        Falls back to building data from config if no cache is available.
        """
        from src.rade_ml_pt.evaluation.evaluator import Evaluator

        # Try loading cached test data from the member version directory.
        version_dir = registry.root_dir / member_version
        ds_dir = version_dir / "datasets"

        test_ds = None
        if (ds_dir / "test.pt").exists():
            import torch
            from torch.utils.data import DataLoader
            try:
                from src.rade_ml_pt.data.dataset import _collate_dict_batch
                test_dataset = torch.load(str(ds_dir / "test.pt"), weights_only=False)
                test_ds = DataLoader(
                    test_dataset, batch_size=32, shuffle=False,
                    collate_fn=_collate_dict_batch,
                )
            except Exception as exc:
                logger.warning("Could not load cached test data for '%s': %s", cluster_id, exc)

        if test_ds is None:
            logger.warning(
                "No test data available for member '%s' (version '%s'). "
                "Skipping evaluation.", cluster_id, member_version,
            )
            return None

        evaluator = Evaluator(model=model)
        eval_result = evaluator.run(test_ds)

        return {
            "predictions": eval_result.predictions,
            "targets": eval_result.targets,
            "metrics": eval_result.metrics,
        }

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def _save_artifacts(
        self,
        version: str,
        per_member_metrics: Dict[str, Dict[str, float]],
        ensemble_metrics: Dict[str, float],
        member_rollup: Dict[str, Any],
    ) -> None:
        """Save evaluation artifacts to artifacts_dir/ensemble/{version}/."""
        eval_dir = Path(self.config.artifacts_dir) / "ensemble" / version / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        with open(eval_dir / "ensemble_metrics.json", "w") as f:
            json.dump(ensemble_metrics, f, indent=2)

        with open(eval_dir / "per_member_metrics.json", "w") as f:
            json.dump(per_member_metrics, f, indent=2)

        with open(eval_dir / "member_rollup.json", "w") as f:
            json.dump(member_rollup, f, indent=2)

        # Generate ensemble plots.
        try:
            from src.rade_ml_pt.ensemble.plots import save_ensemble_plots
            plots_dir = eval_dir / "plots"
            save_ensemble_plots(per_member_metrics, plots_dir, ensemble_metrics)
            logger.info("Saved ensemble evaluation plots to %s", plots_dir)
        except Exception as exc:
            logger.warning("Could not generate ensemble evaluation plots: %s", exc)

        logger.info("Ensemble evaluation artifacts saved to %s", eval_dir)
