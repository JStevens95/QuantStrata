"""
Inference pipeline for the Hybrid GNN-RNN model (PyTorch).

Loads a registered model, prepares trade inputs (including graph extension
for new trades), and returns PnL predictions via InferenceRunner.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, TYPE_CHECKING

import numpy as np
import torch

from src.rade_ml_pt.pipelines.base import InferencePipeline
from src.rade_ml_pt.pipelines.config import PipelineConfig
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import HybridGnnRnnDataConfig
from src.rade_ml_pt.core.types import InferenceResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class HybridGnnRnnInferencePipeline(InferencePipeline):
    """
    Concrete inference pipeline for Hybrid GNN-RNN.

    Implements prepare_inputs: loads the trained graph builder, encodes new
    trade attributes, extends the adjacency for new target trades, and
    assembles model-ready input tensors for the InferenceRunner.
    """

    def get_result_cls(self) -> type:
        return InferenceResult

    def prepare_inputs(self, config: PipelineConfig) -> Dict[str, Any]:
        from src.rade_ml_pt.utilities.graph_builder import TradeGraphBuilder
        from src.rade_ml_pt.utilities.attribute_encoder import TradeAttributeEncoder

        data_config = config.data_config
        if isinstance(data_config, dict):
            data_config = HybridGnnRnnDataConfig.from_dict(data_config)
        elif data_config is None:
            data_config = HybridGnnRnnDataConfig()

        infer_meta = config.metadata.get("inference", {})

        graph_builder_path = infer_meta["graph_builder_path"]
        encoder_path = infer_meta["encoder_path"]
        pnl_history = np.asarray(infer_meta["pnl_history"], dtype=np.float32)
        new_trade_attribs = infer_meta.get("new_trade_attribs")
        trade_ids = infer_meta.get("trade_ids")

        graph_builder = TradeGraphBuilder.load(graph_builder_path)
        encoder = TradeAttributeEncoder.load(encoder_path)

        if new_trade_attribs is not None:
            n_new = len(new_trade_attribs.get("trade_id", []))
        else:
            n_new = 0

        if n_new > 0:
            all_attribs = _merge_attribs(encoder.last_attribs_, new_trade_attribs)
            encoded_trades = encoder.transform(all_attribs)

            graph_result = graph_builder.build_graph_projection(
                adjacency_matrix=graph_builder._adjacency_csr,
                encoded_trades=encoded_trades,
                new_targets=n_new,
                k=data_config.graph_builder.k,
            )
        else:
            encoded_trades = encoder.transform(encoder.last_attribs_)
            graph_result = graph_builder._pack_result(
                csr=graph_builder._adjacency_csr,
                indices=graph_builder.sparse_indices,
                values=graph_builder.sparse_values,
                is_target=graph_builder.is_target_trade,
            )

        trade_features = graph_builder.features if n_new == 0 else graph_builder._weighted_features(encoded_trades)
        adj_sp = graph_result["adjacency_matrix"]

        n_orig_elem = len(graph_builder.is_target_trade) - int(np.sum(graph_builder.is_target_trade))
        n_total = trade_features.shape[0]
        elementary_idx = np.arange(0, n_orig_elem, dtype=np.int32)
        target_idx = np.arange(n_orig_elem, n_total, dtype=np.int32)

        # build model-ready input dict using PyTorch tensors
        inputs = {
            "trade_features": torch.tensor(trade_features, dtype=torch.float32),
            "adjacency_indices": torch.tensor(adj_sp.indices, dtype=torch.long),
            "adjacency_values": torch.tensor(adj_sp.values, dtype=torch.float32),
            "adjacency_dense_shape": torch.tensor(adj_sp.dense_shape, dtype=torch.long),
            "pnl_history": torch.tensor(pnl_history, dtype=torch.float32),
            "elementary_indices": torch.tensor(elementary_idx, dtype=torch.long),
            "target_indices": torch.tensor(target_idx, dtype=torch.long),
        }

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
