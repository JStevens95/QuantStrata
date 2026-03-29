"""
Ensemble inference pipeline.

Two usage patterns:

1. **Standalone (non-UI):** Construct with an ``EnsembleConfig``, call ``run()``.
   The pipeline loads the ensemble from the registry, builds per-member inputs
   via the single-cluster inference helpers, predicts, and aggregates.

2. **Via EnsembleSession (UI):** The session pre-loads models + inference
   contexts.  Call ``run(session=session)`` or use ``EnsembleSession.run_inference()``
   directly.  The pipeline skips registry loading and uses the cached state.
"""
from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
from src.rade_ml_pt.core.types import InferenceResult

if TYPE_CHECKING:
    from src.rade_ml_pt.ensemble.session import EnsembleSession
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _dict_to_inference_context(
    raw: Dict[str, Any],
    data_config_override: Any = None,
) -> "InferenceContext":
    """
    Convert a raw context dict (as returned by ``load_inference_context_from_dir``)
    into an ``InferenceContext`` dataclass.

    Parameters
    ----------
    raw : dict
        Keys match those produced by ``load_inference_context_from_dir``.
    data_config_override : optional
        If provided, replaces the ``data_config`` from *raw*.
    """
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

    dc = data_config_override if data_config_override is not None else raw.get("data_config")
    return InferenceContext(
        data_config=dc,
        encoder=raw.get("encoder"),
        encoder_results=raw.get("encoder_results"),
        graph_builder=raw.get("graph_builder"),
        graph_results=raw.get("graph_results"),
        elementary_pnl=raw.get("elementary_pnl"),
        elementary_scaler=raw.get("elementary_scaler"),
        elementary_attributes=raw.get("elementary_attribs"),
        target_scaler=raw.get("target_scaler"),
        target_attributes=raw.get("target_attribs"),
        trade_universe=raw.get("trade_universe"),
        cluster_info=raw.get("cluster_info"),
        cluster_assets=raw.get("cluster_assets"),
        cluster_elem_trades=raw.get("cluster_elem_trades"),
    )


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class EnsembleInferencePipeline:
    """
    Run inference through an ensemble of models.

    Supports two input modes (set in config.metadata["inference"]["input_mode"]):
    - ``new_scenarios``: Same trades, new risk-factor scenario data.
    - ``new_trades``: New trade attributes to route and predict (not yet implemented).

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
        self._ens_config: Optional[EnsembleConfig] = None
        self._member_versions: Optional[Dict[str, str]] = None
        self._inference_contexts: Dict[str, Any] = {}

    # ==================================================================
    # Orchestration
    # ==================================================================

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

        need_contexts = not bool(infer_meta.get("member_inputs"))
        if self._session is not None and self._session.all_inference_ready:
            self._load_from_session()
        else:
            self._load_from_registry(need_contexts=need_contexts)

        member_inputs, extra_meta = self._build_member_inputs(input_mode, infer_meta)
        combined = self._ensemble.predict(member_inputs)
        result = self._build_result(combined, infer_meta, extra_meta)

        result.latency_seconds = time.perf_counter() - t0
        self.post_infer(result)

        logger.info(
            "EnsembleInferencePipeline: done (%.3fs, %d samples)",
            result.latency_seconds, result.n_samples,
        )
        return result

    # ==================================================================
    # Loading
    # ==================================================================

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
            from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
                load_inference_context_from_dir,
            )

            for cid in config.cluster_ids:
                ver = member_versions[cid]
                version_dir = Path(self.config.registry_dir) / ver
                try:
                    raw = load_inference_context_from_dir(version_dir)
                    self._inference_contexts[cid] = _dict_to_inference_context(raw)
                except Exception as exc:
                    raise ValueError(
                        f"Could not load inference context for cluster '{cid}' "
                        f"(version '{ver}', dir={version_dir}). "
                        f"Provide metadata['inference']['member_inputs'] for "
                        f"model-agnostic ensemble inference, or ensure model-"
                        f"specific inference artifacts are present. "
                        f"Root cause: {exc}"
                    ) from exc

        logger.info(
            "Loaded ensemble '%s' (%d members) from registry",
            version, config.n_members,
        )

    def _load_from_session(self) -> None:
        """Warm-start: reuse the session's pre-loaded models + contexts."""
        self._ens_config = self._session.config
        self._member_versions = self._session.member_versions
        self._ensemble = self._session.ensemble_model

        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import InferenceContext

        for cid in self._ens_config.cluster_ids:
            state = self._session._inference[cid]
            ctx = state.inference_context
            if isinstance(ctx, InferenceContext):
                self._inference_contexts[cid] = ctx
            else:
                self._inference_contexts[cid] = _dict_to_inference_context(
                    ctx, data_config_override=state.data_config,
                )

        logger.info(
            "Using pre-loaded session (ensemble '%s', %d members)",
            self._session.ensemble_version, self._ens_config.n_members,
        )

    # ==================================================================
    # Input building
    # ==================================================================

    def _build_member_inputs(
        self,
        input_mode: str,
        infer_meta: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build the per-member model input dicts.

        Returns
        -------
        tuple of (member_inputs, extra_meta)
            *member_inputs*: ``{cluster_id: 7-key model input dict}`` ready
            for ``EnsembleModel.predict``.
            *extra_meta*: auxiliary info (sample_ids per member, etc.).
        """
        pre_built = infer_meta.get("member_inputs")
        if pre_built:
            return pre_built, {}

        if input_mode == "new_scenarios":
            return self._build_new_scenarios_inputs(infer_meta)

        if input_mode == "new_trades":
            raise NotImplementedError(
                "new_trades inference is not yet supported in the ensemble "
                "pipeline. The underlying HybridGnnRnnInferencePipeline does "
                "not implement _prepare_new_trade_inputs yet."
            )

        raise ValueError(f"Unknown input_mode: {input_mode}")

    def _build_new_scenarios_inputs(
        self,
        infer_meta: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Build per-member inputs for new-scenarios mode.

        Mirrors the flow of
        ``HybridGnnRnnInferencePipeline._prepare_new_scenarios_inputs``
        for every member, sharing the new-scenario shocks across clusters.

        Steps (per member)
        ------------------
        1. Load new risk-factor shock CSVs (shared, loaded once).
        2. Inject unchanged static inputs (trade_features, adjacency, indices).
        3. Deep-copy asset portfolio with new shocks injected.
        4. Filter elementary trades to match cluster's reduced population.
        5. Calculate elementary PnL for new scenarios.
        6. Concatenate elementary PnL across assets, reorder to training column order.
        7. Standardise elementary PnL using saved scaler.
        8. Store scaled PnL on InferenceInputData.
        9. Build PnL sequences with same windowing as training.
        10. Assemble 7-key model input dict.
        """
        from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.infer import (
            HybridGnnRnnInferencePipeline,
        )

        if not self._inference_contexts:
            raise ValueError(
                "Inference contexts not loaded. Provide "
                "metadata['inference']['member_inputs'] for model-agnostic "
                "ensemble inference, or load contexts via registry/session."
            )

        new_scenario_dir = infer_meta.get("new_scenario_dir")
        if not new_scenario_dir:
            raise ValueError(
                "metadata['inference']['new_scenario_dir'] is required for "
                "new_scenarios mode."
            )

        new_scenario_shocks = HybridGnnRnnInferencePipeline.load_new_scenarios(
            new_scenario_dir,
        )

        member_inputs: Dict[str, Any] = {}
        extra_meta: Dict[str, Any] = {"sample_ids": {}}

        for cid in self._ens_config.cluster_ids:
            context = self._inference_contexts[cid]

            inputs = HybridGnnRnnInferencePipeline._inject_unchanged_inputs(
                context, mode="new_scenarios",
            )

            new_asset_portfolio = HybridGnnRnnInferencePipeline._update_asset_portfolio(
                context.cluster_assets, new_scenario_shocks,
            )

            elem_trades = {
                z: [
                    x for x in context.cluster_elem_trades[z]
                    if x["id"] in inputs.elementary_ids
                ]
                for z in context.cluster_elem_trades.keys()
            }

            asset_elementary_pnl = HybridGnnRnnInferencePipeline.calculate_elementary_pnl(
                asset_portfolio=new_asset_portfolio,
                elementary_trades=elem_trades,
            )

            new_elementary_pnl = pd.concat(asset_elementary_pnl.values(), axis=1)
            new_elementary_pnl = new_elementary_pnl[
                context.elementary_attributes["trade_id"]
            ]

            new_elementary_pnl_scaled = HybridGnnRnnInferencePipeline._standardise_pnl(
                pnl_unscaled=new_elementary_pnl,
                scaler=context.elementary_scaler,
            )

            inputs.elementary_pnl = pd.DataFrame(
                new_elementary_pnl_scaled,
                columns=context.elementary_pnl.columns.tolist(),
                index=new_elementary_pnl.index.tolist(),
            )

            elem_seq = HybridGnnRnnInferencePipeline.build_new_pnl_sequences(
                elementary_pnl=inputs.elementary_pnl,
                seq_length=context.data_config.seq_length,
                n_targets=len(inputs.target_indices),
            )

            result = HybridGnnRnnInferencePipeline.build_model_inputs(
                elem_seq=elem_seq,
                inputs=inputs,
                seq_length=context.data_config.seq_length,
            )

            member_inputs[cid] = result["inputs"]
            extra_meta["sample_ids"][cid] = result.get("sample_ids")

            logger.info(
                "Built new-scenarios inputs for cluster '%s' "
                "(%d windows, seq_length=%d)",
                cid, elem_seq.shape[0], context.data_config.seq_length,
            )

        return member_inputs, extra_meta

    # ==================================================================
    # Post-inference
    # ==================================================================

    def post_infer(self, result: InferenceResult) -> None:
        """Post-inference analytics: log summary, save predictions CSV."""
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

    # ==================================================================
    # Result building
    # ==================================================================

    def _build_result(
        self,
        combined: np.ndarray,
        infer_meta: Dict[str, Any],
        extra_meta: Dict[str, Any],
    ) -> InferenceResult:
        """Wrap aggregated predictions into an ``InferenceResult``."""
        meta: Dict[str, Any] = {
            "input_mode": infer_meta.get("input_mode", "new_scenarios"),
            "cluster_ids": self._ensemble.router.cluster_ids if self._ensemble else [],
            "n_members": self._ens_config.n_members if self._ens_config else 0,
        }
        if extra_meta.get("sample_ids"):
            meta["per_member_sample_ids"] = extra_meta["sample_ids"]

        all_sample_ids = None
        per_member_ids = extra_meta.get("sample_ids", {})
        if per_member_ids:
            all_sample_ids = []
            for cid in sorted(per_member_ids.keys()):
                ids = per_member_ids[cid]
                if ids:
                    all_sample_ids.extend(ids)
            all_sample_ids = all_sample_ids or None

        return InferenceResult(
            predictions=combined,
            n_samples=combined.shape[0],
            sample_ids=all_sample_ids or infer_meta.get("sample_ids"),
            model_version=self.ensemble_version,
            metadata=meta,
        )
