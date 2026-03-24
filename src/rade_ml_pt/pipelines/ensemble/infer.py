"""
Ensemble inference pipeline.

Two usage patterns:

1. **Standalone (non-UI):** Construct with an ``EnsembleConfig``, call ``run()``.
   The pipeline loads the ensemble from the registry, builds per-member inputs
   via the single-cluster inference functions, predicts, and aggregates.

2. **Via EnsembleSession (UI):** The session pre-loads models + inference
   contexts.  Call ``run(session=session)`` or use ``EnsembleSession.run_inference()``
   directly.  The pipeline skips registry loading and uses the cached state.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.core.types import InferenceResult

if TYPE_CHECKING:
    from src.rade_ml_pt.ensemble.session import EnsembleSession

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
    ) -> None:
        self.config = ensemble_config
        self.ensemble_version = ensemble_version
        self._session = session

        self._ensemble = None
        self._ens_config = None
        self._member_versions = None
        self._inference_contexts: Dict[str, Dict[str, Any]] = {}

    def run(self) -> InferenceResult:
        """
        Execute the full ensemble inference pipeline.

        Steps
        -----
        1. Load ensemble (from registry or session).
        2. Resolve input_mode from config metadata.
        3. Build per-member inputs via single-cluster inference functions.
        4. Predict + aggregate.
        5. ``post_infer()`` — save results, log summary.

        Returns
        -------
        InferenceResult
        """
        logger.info("EnsembleInferencePipeline: starting")
        t0 = time.perf_counter()

        infer_meta = self.config.metadata.get("inference", {})
        input_mode = infer_meta.get("input_mode", "new_scenarios")
        if input_mode not in {"new_scenarios", "new_trades"}:
            raise ValueError(
                f"Unknown input_mode '{input_mode}'. "
                f"Supported modes: 'new_scenarios', 'new_trades'."
            )

        # If member_inputs are already prepared externally, skip model-specific
        # context loading and keep the pipeline model-agnostic.
        need_contexts = not bool(infer_meta.get("member_inputs"))
        if self._session is not None and self._session.all_inference_ready:
            self._load_from_session()
        else:
            self._load_from_registry(need_contexts=need_contexts)

        member_inputs = self._build_member_inputs(input_mode, infer_meta)
        combined = self._ensemble.predict(member_inputs)
        result = self._build_result(combined, infer_meta)

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

    def _load_from_registry(self, need_contexts: bool = True) -> None:
        """Cold-start: load ensemble + per-member inference contexts from registry."""
        ens_registry = EnsembleRegistry(self.config.registry_dir)
        config, member_versions, version = ens_registry.load(self.ensemble_version)
        self._ens_config = config
        self._member_versions = member_versions

        from src.rade_ml_pt.registry.store import ModelRegistry
        registry = ModelRegistry(self.config.registry_dir)
        builder = EnsembleBuilder(registry)
        self._ensemble = builder.build(config, member_versions)

        if need_contexts:
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import load_inference_context
            for cid in config.cluster_ids:
                ver = member_versions[cid]
                version_dir = Path(self.config.registry_dir) / ver
                try:
                    self._inference_contexts[cid] = load_inference_context({
                        "graph_builder_path": str(version_dir / "graph_builder.pkl"),
                        "encoder_path": str(version_dir / "encoder.pkl"),
                        "version_dir": str(version_dir),
                    })
                except Exception as exc:
                    raise ValueError(
                        f"Could not build member inputs for cluster '{cid}'. "
                        f"Provide metadata['inference']['member_inputs'] for "
                        f"model-agnostic ensemble inference, or ensure model-"
                        f"specific inference artifacts are present. Root cause: {exc}"
                    ) from exc

        logger.info("Loaded ensemble '%s' (%d members) from registry", version, config.n_members)

    def _load_from_session(self) -> None:
        """Warm-start: reuse the session's pre-loaded models + contexts."""
        self._ens_config = self._session.config
        self._member_versions = self._session.member_versions
        self._ensemble = self._session.ensemble_model

        for cid in self._ens_config.cluster_ids:
            state = self._session._inference[cid]
            self._inference_contexts[cid] = state.inference_context

        logger.info(
            "Using pre-loaded session (ensemble '%s', %d members)",
            self._session.ensemble_version, self._ens_config.n_members,
        )

    # ------------------------------------------------------------------
    # Input building (uses single-cluster inference functions)
    # ------------------------------------------------------------------

    def _build_member_inputs(
        self, input_mode: str, infer_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build the 7-key model input dict for each member.

        Uses the same ``build_static_dict`` + ``build_model_input_dict`` functions
        as the single-cluster pipeline.  Falls back to pre-built ``member_inputs``
        in infer_meta if provided (backward compatible).
        """
        pre_built = infer_meta.get("member_inputs")
        if pre_built:
            return pre_built

        if input_mode == "new_scenarios" and not self._inference_contexts:
            raise ValueError(
                "metadata['inference']['member_inputs'] is required for "
                "new_scenarios mode unless model-specific inference contexts are loaded."
            )

        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            build_static_dict,
            build_model_input_dict,
        )

        cluster_pnl = infer_meta.get("cluster_pnl_histories", {})
        new_trade_attribs = infer_meta.get("new_trade_attribs")

        routed_attribs: Dict[str, Any] = {}
        if input_mode == "new_trades" and new_trade_attribs is not None:
            first_key = next(iter(new_trade_attribs), None)
            if first_key in self._ens_config.cluster_ids:
                routed_attribs = new_trade_attribs
            else:
                cid = self._ensemble.router.assign_new_trade(new_trade_attribs)
                routed_attribs[cid] = new_trade_attribs

        member_inputs: Dict[str, Any] = {}
        for cid in self._ens_config.cluster_ids:
            context = self._inference_contexts[cid]
            data_config = self._resolve_data_config(cid)
            attribs = routed_attribs.get(cid)

            static_dict = build_static_dict(context, attribs, data_config)

            pnl = cluster_pnl.get(cid)
            if pnl is None:
                pnl = infer_meta.get("pnl_history")
            if pnl is None and self._session is not None:
                state = self._session._inference.get(cid)
                if state and state.baseline_pnl is not None:
                    pnl = state.baseline_pnl
            if pnl is None:
                raise ValueError(
                    f"No pnl_history for cluster '{cid}'. Provide "
                    f"cluster_pnl_histories['{cid}'] or a global pnl_history."
                )

            member_inputs[cid] = build_model_input_dict(
                static_dict, np.asarray(pnl, dtype=np.float32),
            )

        return member_inputs

    def _resolve_data_config(self, cluster_id: str) -> Any:
        """Resolve HybridGnnRnnDataConfig for a cluster."""
        if self._session is not None:
            state = self._session._inference.get(cluster_id)
            if state and state.data_config is not None:
                return state.data_config

        version = self._member_versions.get(cluster_id, "")
        dc_path = Path(self.config.registry_dir) / version / "data_config.json"
        if dc_path.exists():
            from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
            return HybridGnnRnnDataConfig.from_json(dc_path)

        from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
        return HybridGnnRnnDataConfig()

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
        meta = {
            "input_mode": infer_meta.get("input_mode", "new_scenarios"),
            "n_members": self._ensemble.router.cluster_ids if self._ensemble else [],
        }
        if "new_trade_assignments" in infer_meta:
            meta["new_trade_assignments"] = infer_meta["new_trade_assignments"]

        return InferenceResult(
            predictions=combined,
            n_samples=combined.shape[0],
            sample_ids=infer_meta.get("sample_ids"),
            model_version=self.ensemble_version,
            metadata=meta,
        )
