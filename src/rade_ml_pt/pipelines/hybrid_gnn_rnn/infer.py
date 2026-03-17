"""
Inference pipeline for the Hybrid GNN-RNN model (PyTorch).

Loads a registered model, prepares inputs via prepare_inputs() which branches
on detected mode (new scenarios vs new trades), and returns PnL predictions.

Input mode is inferred from config.metadata["inference"]; the user does not
set input_mode explicitly.

Design: load inference context once (read-only), then build static_dict and
pnl_history per mode; a single _build_model_input_dict() assembles the 7-key
model input (same names as training, no targets).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np
import torch

from src.rade_ml_pt.pipelines.base import InferencePipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.core.types import InferenceResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Model forward() expects exactly these keys (no targets). Used by _build_model_input_dict.
MODEL_INPUT_KEYS = [
    "trade_features",
    "pnl_history",
    "adjacency_indices",
    "adjacency_values",
    "adjacency_dense_shape",
    "elementary_indices",
    "target_indices",
]
STATIC_KEYS = [k for k in MODEL_INPUT_KEYS if k != "pnl_history"]

# Keys that indicate scenario-like input (same trades, new risk-factor paths or pnl_history).
_SCENARIO_KEYS = frozenset({
    "scenario_dir", "scenario_paths", "scenario_files", "scenario_data",
    "graph_builder_path", "pnl_history",
})
# Keys that indicate trade-like input (new trade attributes to encode and add to graph).
_TRADE_KEYS = frozenset({
    "trades_path", "trades_csv", "trade_attributes", "new_trades", "new_trades_path",
    "new_trade_attribs",
})


def _detect_input_mode(infer_meta: Dict[str, Any]) -> str:
    """
    Infer input mode from config.metadata['inference']; no user-defined mode flag.

    - If any trade-like key is present and has a non-empty value (e.g. new_trade_attribs,
      trades_path, trade_attributes) → "new_trades".
    - Else if any scenario-like key is present (scenario_dir, scenario_paths, scenario_data,
      or graph_builder_path + pnl_history) → "new_scenarios".
    - If nothing sufficient is present, raise.
    """
    has_trade = any(
        infer_meta.get(k) not in (None, [], {})
        for k in _TRADE_KEYS
        if k in infer_meta
    )
    has_scenario = any(
        infer_meta.get(k) not in (None, [], {})
        for k in _SCENARIO_KEYS
        if k in infer_meta
    )
    # Classic single-model case: graph_builder_path + pnl_history (no new_trade_attribs) = new_scenarios
    if has_trade:
        return "new_trades"
    if has_scenario:
        return "new_scenarios"
    raise ValueError(
        "Could not detect inference input mode. Provide one of: "
        "scenario_dir / scenario_paths / scenario_data / pnl_history (new scenarios), or "
        "trades_path / trade_attributes / new_trade_attribs (new trades)."
    )


# ---------------------------------------------------------------------------
# Inference context (load once, read-only) and single assembly
# ---------------------------------------------------------------------------


def load_inference_context(infer_meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load all artifacts needed for inference into a single read-only context.

    Does not mutate any loaded objects. Caller uses context to build static_dict
    and pnl_history per mode, then _build_model_input_dict(static_dict, pnl_history).

    Parameters
    ----------
    infer_meta : dict
        config.metadata["inference"]; must contain graph_builder_path, encoder_path.

    Returns
    -------
    dict
        graph_builder : TradeGraphBuilder
        encoder : TradeAttributeEncoder
        Optional (if present in version_dir): scalers, portfolio, elementary_universe.
    """
    from src.rade_ml_pt.utilities.graph_builder import TradeGraphBuilder
    from src.rade_ml_pt.utilities.attribute_encoder import TradeAttributeEncoder

    graph_builder_path = infer_meta["graph_builder_path"]
    encoder_path = infer_meta["encoder_path"]

    graph_builder = TradeGraphBuilder.load(graph_builder_path)
    encoder = TradeAttributeEncoder.load(encoder_path)

    context: Dict[str, Any] = {
        "graph_builder": graph_builder,
        "encoder": encoder,
    }

    import joblib

    version_dir = infer_meta.get("version_dir")
    if version_dir is not None:
        vpath = Path(version_dir)
        for name, fname in [
            ("elementary_scaler", "elementary_scaler.joblib"),
            ("target_scaler", "target_scaler.joblib"),
            ("portfolio", "portfolio.joblib"),
            ("elementary_universe", "elementary_universe.joblib"),
        ]:
            p = vpath / fname
            if p.exists():
                context[name] = joblib.load(str(p))

    return context


def build_static_dict(
    context: Dict[str, Any],
    new_trade_attribs: Optional[Dict[str, Any]],
    data_config: Any,
) -> Dict[str, np.ndarray]:
    """
    Build the 6-key static part of the model input (same key names as training).

    Output matches the training static_inputs contract in data/hybrid_gnn_rnn/build.py:
    adjacency_indices [nnz, 2], adjacency_values [nnz], adjacency_dense_shape (2,) int64,
    elementary_indices / target_indices 0-based ranges. Model forward() expects these
    and reconstructs sparse adjacency from them.

    For new_trade_attribs is None: same graph, same trade_features and indices.
    For new_trade_attribs set: extend graph via build_graph_projection, new features and indices.

    Parameters
    ----------
    context : dict
        From load_inference_context() (graph_builder, encoder).
    new_trade_attribs : dict or None
        If None, use same graph; else merge with encoder attribs and extend graph.
    data_config : HybridGnnRnnDataConfig or dict
        Used for graph_builder.k when extending.

    Returns
    -------
    dict
        trade_features, adjacency_indices, adjacency_values, adjacency_dense_shape,
        elementary_indices, target_indices (numpy arrays; no pnl_history).
    """
    graph_builder = context["graph_builder"]
    encoder = context["encoder"]

    if new_trade_attribs is not None:
        n_new = len(new_trade_attribs.get("trade_id", []))
    else:
        n_new = 0

    if n_new > 0:
        all_attribs = _merge_attribs(encoder.last_attribs_, new_trade_attribs)
        encoded_trades = encoder.transform(all_attribs)
        k = getattr(data_config.graph_builder, "k", 5) if data_config is not None else 5
        graph_result = graph_builder.build_graph_projection(
            adjacency_matrix=graph_builder._adjacency_csr,
            encoded_trades=encoded_trades,
            new_targets=n_new,
            k=k,
        )
        trade_features = graph_builder._weighted_features(encoded_trades)
    else:
        encoded_trades = encoder.transform(encoder.last_attribs_)
        graph_result = graph_builder._pack_result(
            csr=graph_builder._adjacency_csr,
            indices=graph_builder.sparse_indices,
            values=graph_builder.sparse_values,
            is_target=graph_builder.is_target_trade,
        )
        trade_features = graph_builder.features

    # Use same adjacency format as training (build.py): sparse_indices [nnz, 2],
    # sparse_values, sparse_shape. The model expects indices [nnz, 2] and will
    # transpose to [2, nnz] internally for torch.sparse_coo_tensor.
    n_orig_elem = len(graph_builder.is_target_trade) - int(np.sum(graph_builder.is_target_trade))
    n_total = trade_features.shape[0]
    elementary_idx = np.arange(0, n_orig_elem, dtype=np.int64)
    target_idx = np.arange(n_orig_elem, n_total, dtype=np.int64)

    return {
        "trade_features": np.asarray(trade_features, dtype=np.float32),
        "adjacency_indices": np.asarray(graph_result["sparse_indices"], dtype=np.int64),
        "adjacency_values": np.asarray(graph_result["sparse_values"], dtype=np.float32),
        "adjacency_dense_shape": np.array(graph_result["sparse_shape"], dtype=np.int64),
        "elementary_indices": elementary_idx,
        "target_indices": target_idx,
    }


def build_model_input_dict(
    static_dict: Dict[str, np.ndarray],
    pnl_history: np.ndarray,
) -> Dict[str, torch.Tensor]:
    """
    Assemble the full model input dict (same 7 keys as training, no targets).

    Single assembly point: both new_scenarios and new_trades call this after
    building static_dict and pnl_history.

    Parameters
    ----------
    static_dict : dict
        From build_static_dict(): trade_features, adjacency_*, elementary_indices, target_indices.
    pnl_history : ndarray
        [n_scenarios, seq_len, n_elementary], float32.

    Returns
    -------
    dict
        All keys in MODEL_INPUT_KEYS, values as torch.Tensor, ready for model.forward().
    """
    pnl_history = np.asarray(pnl_history, dtype=np.float32)

    long_keys = ("adjacency_indices", "adjacency_dense_shape", "elementary_indices", "target_indices")
    float_keys = ("adjacency_values", "trade_features")

    out: Dict[str, torch.Tensor] = {}
    for key in STATIC_KEYS:
        v = static_dict[key]
        if not isinstance(v, torch.Tensor):
            v = torch.from_numpy(np.asarray(v))
        if key in long_keys and v.dtype != torch.long:
            v = v.long()
        elif key in float_keys and v.dtype != torch.float32:
            v = v.float()
        out[key] = v

    out["pnl_history"] = torch.from_numpy(pnl_history).float()
    return out


class HybridGnnRnnInferencePipeline(InferencePipeline):
    """
    Concrete inference pipeline for Hybrid GNN-RNN.

    prepare_inputs() detects mode from the data provided and branches to
    _prepare_new_scenarios_inputs() or _prepare_new_trade_inputs().
    """

    def get_result_cls(self) -> type:
        return InferenceResult

    def prepare_inputs(self, config: PipelineConfig) -> Dict[str, Any]:
        """
        Build model-ready inputs; branch by detected mode (new_scenarios vs new_trades).

        Mode is inferred from config.metadata["inference"] keys — no input_mode required.
        """
        infer_meta = config.metadata.get("inference", {})
        mode = _detect_input_mode(infer_meta)

        if mode == "new_scenarios":
            return self._prepare_new_scenarios_inputs(config, infer_meta)
        return self._prepare_new_trade_inputs(config, infer_meta)

    def _prepare_new_scenarios_inputs(
        self,
        config: PipelineConfig,
        infer_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build inputs for same trades, new risk-factor scenario data.

        Uses graph_builder_path, encoder_path, pnl_history (and optionally
        scenario_dir / scenario_paths / scenario_data to build pnl_history).
        """
        scenario_dir = infer_meta.get("scenario_dir")
        scenario_paths = infer_meta.get("scenario_paths") or infer_meta.get("scenario_files")
        scenario_data = infer_meta.get("scenario_data")

        if scenario_dir is not None:
            scenario_paths = (
                list(Path(scenario_dir).glob("*.csv"))
                if Path(scenario_dir).is_dir()
                else []
            )

        if scenario_paths or scenario_data is not None:
            return self._build_inputs_from_scenarios(
                config, infer_meta,
                scenario_paths=scenario_paths or [],
                scenario_data=scenario_data,
            )

        # Existing path: graph_builder + encoder + pnl_history (no new trades)
        return self._build_inputs_from_registry_and_pnl(config, infer_meta, new_trade_attribs=None)

    def _prepare_new_trade_inputs(
        self,
        config: PipelineConfig,
        infer_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build inputs for new trade attributes (extend graph, encode new trades).
        """
        new_trade_attribs = (
            infer_meta.get("new_trade_attribs")
            or infer_meta.get("trade_attributes")
            or infer_meta.get("new_trades")
        )
        trades_path = infer_meta.get("trades_path") or infer_meta.get("trades_csv") or infer_meta.get("new_trades_path")

        if trades_path is not None:
            return self._build_inputs_from_trades_path(config, infer_meta, trades_path)
        if new_trade_attribs is not None:
            return self._build_inputs_from_registry_and_pnl(config, infer_meta, new_trade_attribs=new_trade_attribs)

        raise ValueError(
            "New-trades input not found: provide new_trade_attribs, trade_attributes, or trades_path in config.metadata['inference']."
        )

    def _build_inputs_from_scenarios(
        self,
        config: PipelineConfig,
        infer_meta: Dict[str, Any],
        *,
        scenario_paths: List[Any],
        scenario_data: Any = None,
    ) -> Dict[str, Any]:
        """
        Build model inputs from scenario paths or in-memory scenario data.

        Load scenario CSVs / arrays, compute elementary PnLs from the trained job's
        asset portfolio and elementary trade objects, then build the model input dict.
        Override or extend for full implementation (e.g. load portfolio from registry).
        """
        # Stub: require pre-built pnl_history when using scenario paths/data until full build is implemented
        pnl_history = infer_meta.get("pnl_history")
        if pnl_history is None and scenario_data is None and not scenario_paths:
            raise ValueError(
                "New-scenarios input: provide pnl_history, scenario_data, or scenario_paths. "
                "Building pnl_history from scenario_paths requires asset portfolio and elementary trade objects (implement in subclass or pipeline)."
            )
        if pnl_history is not None:
            return self._build_inputs_from_registry_and_pnl(config, infer_meta, new_trade_attribs=None)
        raise NotImplementedError(
            "Building inputs from scenario_paths/scenario_data only is not yet implemented; "
            "provide pnl_history in config.metadata['inference'] or implement _build_inputs_from_scenarios."
        )

    def _build_inputs_from_trades_path(
        self,
        config: PipelineConfig,
        infer_meta: Dict[str, Any],
        trades_path: str,
    ) -> Dict[str, Any]:
        """
        Load trade attributes from a CSV path and build inputs (new_trades path).
        """
        import pandas as pd
        df = pd.read_csv(trades_path)
        # Assume same column layout as encoder expects; convert to dict of lists for encoder
        trade_attributes = df.to_dict("list") if hasattr(df, "to_dict") else {}
        return self._build_inputs_from_registry_and_pnl(config, infer_meta, new_trade_attribs=trade_attributes)

    def _get_inference_context(self, infer_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Load inference context once and cache by graph_builder_path + encoder_path."""
        cache_key = (infer_meta.get("graph_builder_path"), infer_meta.get("encoder_path"))
        if getattr(self, "_inference_context_cache", None) is None:
            self._inference_context_cache = {}
        if cache_key not in self._inference_context_cache:
            self._inference_context_cache[cache_key] = load_inference_context(infer_meta)
        return self._inference_context_cache[cache_key]

    def _build_inputs_from_registry_and_pnl(
        self,
        config: PipelineConfig,
        infer_meta: Dict[str, Any],
        *,
        new_trade_attribs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build model input dict via inference context and single assembly.

        Loads or reuses inference context (graph_builder, encoder), builds static_dict
        (same graph or extended), then _build_model_input_dict(static_dict, pnl_history).
        """
        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        context = self._get_inference_context(infer_meta)
        pnl_history = np.asarray(infer_meta["pnl_history"], dtype=np.float32)
        trade_ids = infer_meta.get("trade_ids")

        static_dict = build_static_dict(context, new_trade_attribs, data_config)
        inputs = build_model_input_dict(static_dict, pnl_history)

        graph_builder = context["graph_builder"]
        n_new = (
            len(new_trade_attribs.get("trade_id", []))
            if new_trade_attribs is not None
            else 0
        )
        n_total = static_dict["trade_features"].shape[0]

        return {
            "inputs": inputs,
            "sample_ids": trade_ids,
            "metadata": {
                "n_original_trades": len(graph_builder.is_target_trade),
                "n_new_trades": n_new,
                "n_total_trades": n_total,
            },
        }

    def post_infer(
        self,
        result: InferenceResult,
        config: PipelineConfig,
    ) -> None:
        if result.predictions is not None:
            pnl = result.predictions
            logger.info(
                f"Hybrid GNN-RNN inference | samples={result.n_samples} | "
                f"mean_pnl={np.mean(pnl):.4f} | std_pnl={np.std(pnl):.4f}"
            )


def _merge_attribs(
    original: Dict[str, Any],
    new: Dict[str, Any],
) -> Dict[str, Any]:
    """Append new trade attributes to the original set (preserves original order)."""
    merged: Dict[str, Any] = {}
    for key in original:
        orig_vals = list(original[key])
        new_vals = list(new.get(key, []))
        merged[key] = orig_vals + new_vals
    return merged
