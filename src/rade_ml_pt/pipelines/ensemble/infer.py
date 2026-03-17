"""
Ensemble inference pipeline.

Loads an ensemble from the registry, routes inputs to each member model,
runs predictions, aggregates, and performs post-inference analytics.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.core.types import InferenceResult

logger = logging.getLogger(__name__)


class EnsembleInferencePipeline:
    """
    Run inference through an ensemble of models.

    Supports two input modes (set in config.metadata["inference"]["input_mode"]):
    - ``new_scenarios``: Same trades, new risk-factor scenario data.
    - ``new_trades``: New trade attributes to route and predict.

    Parameters
    ----------
    ensemble_config : EnsembleConfig
        Must have ``registry_dir`` set.
    ensemble_version : str
        Ensemble version or tag to load.
    """

    def __init__(
        self,
        ensemble_config: EnsembleConfig,
        ensemble_version: str = "latest",
    ) -> None:
        self.config = ensemble_config
        self.ensemble_version = ensemble_version
        self._ensemble = None
        self._ens_config = None
        self._member_versions = None

    def run(self) -> InferenceResult:
        """
        Execute the full ensemble inference pipeline.

        Steps
        -----
        1. Load ensemble (all members + router) from registry.
        2. Resolve input_mode from config metadata.
        3. Route inputs to members and run predictions.
        4. Aggregate member predictions.
        5. ``post_infer()`` — save results, log summary.

        Returns
        -------
        InferenceResult
        """
        logger.info("EnsembleInferencePipeline: starting")
        t0 = time.perf_counter()

        self._load_ensemble()

        infer_meta = self.config.metadata.get("inference", {})
        input_mode = infer_meta.get("input_mode", "new_scenarios")

        if input_mode == "new_trades":
            result = self._run_new_trades(infer_meta)
        elif input_mode == "new_scenarios":
            result = self._run_new_scenarios(infer_meta)
        else:
            raise ValueError(
                f"Unknown input_mode: '{input_mode}'. Expected 'new_trades' or 'new_scenarios'."
            )

        result.latency_seconds = time.perf_counter() - t0
        self.post_infer(result)

        logger.info(
            "EnsembleInferencePipeline: done (%.3fs, %d samples)",
            result.latency_seconds, result.n_samples,
        )
        return result

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_ensemble(self) -> None:
        """Load ensemble config + members from the registry."""
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, version = ens_registry.load(self.ensemble_version)
        self._ens_config = config
        self._member_versions = member_versions

        from src.rade_ml_pt.registry.store import ModelRegistry
        registry = ModelRegistry(self.config.registry_dir)
        builder = EnsembleBuilder(registry)
        self._ensemble = builder.build(config, member_versions)

        logger.info("Loaded ensemble '%s' (%d members)", version, config.n_members)

    # ------------------------------------------------------------------
    # New scenarios mode
    # ------------------------------------------------------------------

    def _run_new_scenarios(self, infer_meta: Dict[str, Any]) -> InferenceResult:
        """
        New risk-factor scenarios for existing trades.

        Each member uses the same graph and trade features (unchanged from
        training) but receives new PnL history computed from the new scenarios.
        The caller is responsible for providing per-member input dicts in
        ``infer_meta["member_inputs"]``.
        """
        member_inputs = infer_meta.get("member_inputs", {})
        if not member_inputs:
            raise ValueError(
                "new_scenarios mode requires 'member_inputs' in config.metadata['inference']. "
                "Provide {cluster_id: model_input_dict} with new pnl_history."
            )

        combined = self._ensemble.predict(member_inputs)
        return self._build_result(combined, infer_meta)

    # ------------------------------------------------------------------
    # New trades mode
    # ------------------------------------------------------------------

    def _run_new_trades(self, infer_meta: Dict[str, Any]) -> InferenceResult:
        """
        New trade attributes routed to existing clusters.

        Each new trade is assigned to a cluster via the router.  The
        affected member's graph is extended (via ``build_graph_projection``
        on the member's saved graph_builder).  The caller provides the
        routed and prepared inputs in ``infer_meta["member_inputs"]``.
        """
        member_inputs = infer_meta.get("member_inputs", {})
        if not member_inputs:
            raise ValueError(
                "new_trades mode requires 'member_inputs' in config.metadata['inference']. "
                "Provide {cluster_id: model_input_dict} with extended graph inputs."
            )

        new_trade_assignments = infer_meta.get("new_trade_assignments", {})
        if new_trade_assignments:
            logger.info(
                "New trades routed: %s",
                {cid: len(tids) for cid, tids in new_trade_assignments.items()},
            )

        combined = self._ensemble.predict(member_inputs)
        result = self._build_result(combined, infer_meta)
        result.metadata["new_trade_assignments"] = new_trade_assignments
        return result

    # ------------------------------------------------------------------
    # Post-inference
    # ------------------------------------------------------------------

    def post_infer(self, result: InferenceResult) -> None:
        """
        Post-inference analytics: log summary, save predictions CSV.
        """
        if result.predictions is not None:
            preds = result.predictions
            logger.info(
                "Ensemble inference summary: n_samples=%d, mean=%.4f, std=%.4f, "
                "min=%.4f, max=%.4f",
                result.n_samples, np.mean(preds), np.std(preds),
                np.min(preds), np.max(preds),
            )

        if self.config.artifacts_dir and result.predictions is not None:
            self._save_predictions(result)

    def _save_predictions(self, result: InferenceResult) -> None:
        """Save predictions to CSV in the artifacts directory."""
        out_dir = Path(self.config.artifacts_dir) / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)

        csv_path = out_dir / "predictions.csv"
        preds = result.predictions

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            if preds.ndim == 2:
                header = [f"target_{i}" for i in range(preds.shape[1])]
                if result.sample_ids:
                    header = ["sample_id"] + header
                writer.writerow(header)
                for row_idx in range(preds.shape[0]):
                    row = list(preds[row_idx])
                    if result.sample_ids and row_idx < len(result.sample_ids):
                        row = [result.sample_ids[row_idx]] + row
                    writer.writerow(row)
            else:
                writer.writerow(["prediction"])
                for val in preds.flat:
                    writer.writerow([val])

        logger.info("Predictions saved to %s", csv_path)

        result.to_json(out_dir / "inference_result.json")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(
        self,
        combined: np.ndarray,
        infer_meta: Dict[str, Any],
    ) -> InferenceResult:
        return InferenceResult(
            predictions=combined,
            n_samples=combined.shape[0],
            sample_ids=infer_meta.get("sample_ids"),
            model_version=self.ensemble_version,
            metadata={
                "input_mode": infer_meta.get("input_mode", "new_scenarios"),
                "n_members": self._ensemble.router.cluster_ids if self._ensemble else [],
            },
        )
