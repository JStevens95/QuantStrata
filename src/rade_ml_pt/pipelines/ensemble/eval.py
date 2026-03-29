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
        eval_dir = Path(self.   config.artifacts_dir) / "ensemble" / version / "evaluation"
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
