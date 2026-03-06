"""
HybridGnnRnn model: PyTorch port of the TensorFlow Hybrid GNN-RNN architecture.

This model combines graph neural network (GNN) and recurrent neural network (RNN) streams
to predict target trade PnL. It processes trade features through a GNN over the trade
graph, temporal PnL history through an RNN, fuses both streams, applies target attention,
and projects to per-target PnL predictions.

Architecture flow:
    1. GnnBlock     — structural features from trade graph [T, gnn_dim]
    2. RnnBlock     — temporal features from PnL history [B, rnn_dim]
    3. FusionLayer  — cross-attention fusion [B, T, fusion_dim]
    4. TargetAttentionLayer — self-attention over targets [B, n_targets, attn_dim]
    5. TargetPnlOutput — per-target PnL logits [B, n_targets]

LayerNorm is applied after each block (GNN, RNN, Fusion) for training stability.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from typing import Any, Dict

from src.rade_ml_pt.core.base import BaseModel
from src.rade_ml_pt.validation.base import validate_dict_keys
from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.gnn_layers import GnnBlock
from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.rnn_layers import RnnBlock
from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.fusion_layer import FusionLayer
from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.attention_layer import TargetAttentionLayer
from src.rade_ml_pt.models.hybrid_gnn_rnn.layers.projection_layer import TargetPnlOutput

# Required keys in the input dict for forward() validation.
_REQUIRED_KEYS = [
    "trade_features",
    "pnl_history",
    "adjacency_indices",
    "adjacency_values",
    "adjacency_dense_shape",
    "elementary_indices",
    "target_indices",
]


class HybridGnnRnn(BaseModel):
    """
    Hybrid GNN-RNN model for trade PnL prediction.

    Extends BaseModel and composes GnnBlock, RnnBlock, FusionLayer, TargetAttentionLayer,
    and TargetPnlOutput. Uses LayerNorm between blocks for training stability.

    The model receives a dict of inputs including sparse adjacency components
    (indices, values, dense_shape) which are reconstructed into a torch.sparse_coo_tensor.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        name: str = "hybrid_gnn_rnn",
        **kwargs,
    ) -> None:
        """
        Initialise HybridGnnRnn from a configuration dictionary.

        :param config: Nested dict with keys: general, gnn_layer, rnn_layer,
            fusion_layer, attention_layer, projection_layer. Each layer config
            should have 'general' and 'parameters' subdicts.
        :param name: Human-readable model name for logging and serialization.
        """
        super().__init__(name=name, **kwargs)

        self.model_config = config
        self.general_config = config.get("general", {})
        self.gnn_config = config.get("gnn_layer", {})
        self.rnn_config = config.get("rnn_layer", {})
        self.fusion_config = config.get("fusion_layer", {})
        self.attention_config = config.get("attention_layer", {})
        self.projection_config = config.get("projection_layer", {})

        # Build submodules: GNN, RNN, Fusion, Attention, Projection.
        self.gnn_block = GnnBlock(
            layer_config=self.gnn_config,
            name=f"{name}_gnn_block",
        )
        self.rnn_block = RnnBlock(
            layer_config=self.rnn_config,
        )
        self.fusion_layer = FusionLayer(
            layer_config=self.fusion_config,
        )
        self.attention_layer = TargetAttentionLayer(
            layer_config=self.attention_config,
            name=f"{name}_attn_layer",
        )
        self.projection_layer = TargetPnlOutput(
            layer_config=self.projection_config,
            name=f"{name}_proj_layer",
        )

        # LayerNorm after each block output for training stability.
        # Read units from each layer's parameters config.
        gnn_units = self.gnn_config.get("parameters", {}).get("units", 128)
        rnn_units = self.rnn_config.get("parameters", {}).get("units", 128)
        fusion_units = self.fusion_config.get("parameters", {}).get("units", 64)

        self.gnn_block_ln = nn.LayerNorm(gnn_units, eps=1e-5)
        self.rnn_block_ln = nn.LayerNorm(rnn_units, eps=1e-5)
        self.fusion_ln = nn.LayerNorm(fusion_units, eps=1e-5)

    def forward(self, inputs: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Forward pass: validate inputs, reconstruct sparse adjacency, run the model.

        Training/eval behaviour (dropout, etc.) is controlled by ``model.train()``
        and ``model.eval()``, not by a parameter.

        :param inputs: Dict with required keys: trade_features, pnl_history,
            adjacency_indices, adjacency_values, adjacency_dense_shape,
            elementary_indices, target_indices.
        :return: PnL predictions [batch, num_targets].
        """
        # Ensure all required keys are present in the input dict.
        validate_dict_keys(input_dict=inputs, keys=_REQUIRED_KEYS)

        # Reconstruct sparse adjacency from stored components.
        # Data pipeline stores indices as [nnz, 2]; PyTorch sparse_coo_tensor
        # expects indices as [2, nnz].
        adj_indices = inputs["adjacency_indices"]
        adj_values = inputs["adjacency_values"]
        adj_shape = inputs["adjacency_dense_shape"]

        # Convert numpy arrays to tensors if needed.
        if not isinstance(adj_indices, torch.Tensor):
            adj_indices = torch.as_tensor(adj_indices, dtype=torch.long)
        if not isinstance(adj_values, torch.Tensor):
            adj_values = torch.as_tensor(adj_values, dtype=torch.float32)
        if not isinstance(adj_shape, torch.Tensor):
            adj_shape = torch.as_tensor(adj_shape, dtype=torch.long)

        # Transpose indices from [nnz, 2] to [2, nnz] for PyTorch sparse COO format.
        if adj_indices.dim() == 2 and adj_indices.shape[1] == 2:
            adj_indices = adj_indices.t().contiguous()

        # Build sparse COO tensor and coalesce for efficient operations.
        adjacency = torch.sparse_coo_tensor(
            indices=adj_indices,
            values=adj_values,
            size=tuple(adj_shape.tolist()),
        ).coalesce()

        return self.run_model(
            inputs=(
                inputs["trade_features"],
                inputs["pnl_history"],
                adjacency,
                inputs["target_indices"],
            ),
        )

    def run_model(self, inputs: tuple) -> torch.Tensor:
        """
        Core model logic: GNN -> RNN -> Fusion -> Attention -> Projection.

        Kept separate from forward() for clarity and to allow direct invocation
        when adjacency is already constructed.

        :param inputs: Tuple of (trade_features, pnl_history, adjacency, target_indices).
        :return: PnL predictions [batch, num_targets].
        """
        trade_features, elementary_pnl, adjacency, target_indices = inputs

        gnn_features = self.gnn_block(trade_features, adjacency)
        gnn_features = self.gnn_block_ln(gnn_features)

        rnn_features = self.rnn_block(elementary_pnl)
        rnn_features = self.rnn_block_ln(rnn_features)

        fused_features = self.fusion_layer(
            inputs=(gnn_features, rnn_features, adjacency),
        )
        fused_features = self.fusion_ln(fused_features)

        attended_features = self.attention_layer(
            fused_features, adjacency, target_indices,
        )

        return self.projection_layer(
            inputs=(trade_features, attended_features, target_indices),
        )

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config["model_config"] = self.model_config
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "HybridGnnRnn":
        """
        Create model instance from a configuration dictionary.

        :param config: Dict produced by get_config() (includes model_config, metadata).
        """
        metadata = config.pop("metadata", {})
        model_config = config.pop("model_config", {})
        name = config.pop("name", "hybrid_gnn_rnn")
        model = cls(config=model_config, name=name, **kwargs)
        model._model_metadata.update(metadata)
        return model
