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

_SUPPORTED_STRATEGIES = {"sequential"}
_PLANNED_STRATEGIES = {"process_pool", "gpu_parallel", "distributed"}


# ======================================================================
# Module-level worker — picklable, portable across processes / nodes
# ======================================================================

def evaluate_single_member(
    cluster_id: str,
    model: Any,
    registry_root_dir: str,
    member_version: str,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate one member model on its cached test data.

    Module-level so it can be dispatched to ``ProcessPoolExecutor`` or
    ``ray.remote`` in future parallel strategies.

    Parameters
    ----------
    cluster_id : str
        Cluster identifier.
    model : nn.Module
        Member model in eval mode.
    registry_root_dir : str
        Root registry directory (to locate cached test datasets).
    member_version : str
        Registry version string for this member.

    Returns
    -------
    dict or None
        ``{"predictions": ndarray, "targets": ndarray, "metrics": dict}``
        or ``None`` if no test data is available.
    """
    from src.rade_ml_pt.evaluation.evaluator import Evaluator

    version_dir = Path(registry_root_dir) / member_version
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
        2. Dispatch per-member evaluation via execution strategy.
        3. Aggregate predictions across members.
        4. Compute ensemble-level metrics.
        5. Save evaluation artifacts.

        Returns
        -------
        dict
            ``{ensemble_metrics, per_member_metrics, member_summary}``.
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

        # Dispatch per-member evaluation.
        per_member_preds, per_member_targets, per_member_eval = (
            self._dispatch_evaluation(config, ensemble, registry, member_versions)
        )

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
        """Evaluate all members sequentially in a single process."""
        per_member_preds: Dict[str, np.ndarray] = {}
        per_member_targets: Dict[str, np.ndarray] = {}
        per_member_eval: Dict[str, Dict[str, float]] = {}

        for cid in config.cluster_ids:
            logger.info("--- Evaluating member '%s' ---", cid)
            member_result = evaluate_single_member(
                cluster_id=cid,
                model=ensemble.members[cid],
                registry_root_dir=str(registry.root_dir),
                member_version=member_versions[cid],
            )
            if member_result is not None:
                per_member_preds[cid] = member_result["predictions"]
                per_member_targets[cid] = member_result["targets"]
                per_member_eval[cid] = member_result["metrics"]

        return per_member_preds, per_member_targets, per_member_eval

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
