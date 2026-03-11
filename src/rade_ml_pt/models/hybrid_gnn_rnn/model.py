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

LayerNorm is applied after GNN and RNN blocks for training stability.
    FusionLayer applies its own internal LayerNorm.
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

        rnn_layer_type = self.rnn_config.get("general", {}).get("layer_type", "lstm")
        rnn_ln_dim = rnn_units * 2 if rnn_layer_type == "bilstm" else rnn_units

        self.gnn_block_ln = nn.LayerNorm(gnn_units, eps=1e-5)
        self.rnn_block_ln = nn.LayerNorm(rnn_ln_dim, eps=1e-5)

        # Cache for static GNN outputs (trade_features + adjacency are batch-invariant).
        # Cleared on train()/eval() mode switch and by invalidate_gnn_cache().
        self._gnn_cache: Dict[str, Any] = {}

    def train(self, mode: bool = True):
        """Override to clear GNN cache on mode switch (train <-> eval)."""
        self._gnn_cache.clear()
        return super().train(mode)

    def invalidate_gnn_cache(self) -> None:
        """Explicitly clear the cached GNN features (call if graph structure changes)."""
        self._gnn_cache.clear()

    def _get_adjacency(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """Build or retrieve the cached sparse adjacency matrix.

        The adjacency is static (batch-invariant) so we build it once on the
        first forward call and cache it for all subsequent batches within the
        same train/eval epoch.  The cache is cleared on mode switch via
        ``train()`` and can be manually invalidated with ``invalidate_gnn_cache()``.
        """
        if "adjacency" in self._gnn_cache:
            return self._gnn_cache["adjacency"]

        indices = inputs["adjacency_indices"]
        values = inputs["adjacency_values"]
        shape = inputs["adjacency_dense_shape"]

        # Ensure raw numpy arrays from the data pipeline are converted.
        if not isinstance(indices, torch.Tensor):
            indices = torch.as_tensor(indices, dtype=torch.long)
        if not isinstance(values, torch.Tensor):
            values = torch.as_tensor(values, dtype=torch.float32)
        if not isinstance(shape, torch.Tensor):
            shape = torch.as_tensor(shape, dtype=torch.long)

        # Data pipeline stores indices as [nnz, 2]; sparse_coo_tensor expects [2, nnz].
        if indices.dim() == 2 and indices.shape[1] == 2:
            indices = indices.t().contiguous()

        # Expect 1-D shape [2] from _collate_dict_batch; fail fast if not.
        if shape.dim() != 1:
            raise ValueError(
                f"adjacency_dense_shape should be 1-D but got shape {tuple(shape.shape)}. "
                f"Check that DataLoader uses _collate_dict_batch."
            )
        # int() cast handles float tensors that can arise from torch.load().
        size = tuple(int(s) for s in shape.tolist())
        adjacency = torch.sparse_coo_tensor(indices, values, size).coalesce()
        self._gnn_cache["adjacency"] = adjacency
        return adjacency

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
        if not getattr(self, "_keys_validated", False):
            validate_dict_keys(input_dict=inputs, keys=_REQUIRED_KEYS)
            self._keys_validated = True

        adjacency = self._get_adjacency(inputs)

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

        GNN features and the target adjacency submatrix depend only on
        trade_features, adjacency, and target_indices -- all of which are
        static (batch-invariant) inputs.  They are computed once and cached
        for the duration of the epoch, avoiding redundant work across batches.

        :param inputs: Tuple of (trade_features, pnl_history, adjacency, target_indices).
        :return: PnL predictions [batch, num_targets].
        """
        trade_features, elementary_pnl, adjacency, target_indices = inputs

        # GNN features depend only on static inputs (trade_features + adjacency).
        # During eval/inference we cache them to avoid redundant computation
        # across batches.  During training we recompute every batch so that
        # gradients flow correctly back through the GNN block.
        if not self.training and "gnn_features" in self._gnn_cache:
            gnn_features = self._gnn_cache["gnn_features"]
        else:
            gnn_features = self.gnn_block(trade_features, adjacency)
            gnn_features = self.gnn_block_ln(gnn_features)
            if not self.training:
                self._gnn_cache["gnn_features"] = gnn_features

        rnn_features = self.rnn_block(elementary_pnl)
        rnn_features = self.rnn_block_ln(rnn_features)

        fused_features = self.fusion_layer(
            inputs=(gnn_features, rnn_features, adjacency),
        )

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
