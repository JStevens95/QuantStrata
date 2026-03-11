"""
Target attention layer for the Hybrid GNN-RNN model.

Self-attention over target trades plus feed-forward network sublayer:
    1. Gathers target features from fused features using target_idx.
    2. Extracts [n_tgt, n_tgt] adjacency submatrix (sparse or dense).
    3. Computes multi-head self-attention masked by adjacency.
    4. FFN with residual + LayerNorm.

Supports both sparse COO and dense adjacency matrices. When the adjacency
submatrix is sparse, an efficient padded-neighbor attention path is used
instead of materializing the full [n_tgt, n_tgt] score matrix.
"""
import logging
import torch
import torch.nn as nn

from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Mapping from activation name strings to nn.Module constructors.
_ACTIVATION_MODULE_MAP: Dict[Optional[str], type] = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "linear": nn.Identity,
    None: nn.Identity,
}


def _build_activation_module(name: Optional[str]) -> nn.Module:
    """Return an nn.Module activation instance for the given name string."""
    if name in _ACTIVATION_MODULE_MAP:
        return _ACTIVATION_MODULE_MAP[name]()
    raise ValueError(f"Unsupported activation: {name}. Choose from {list(_ACTIVATION_MODULE_MAP.keys())}")


class TargetAttentionLayer(nn.Module):
    """
    Self-attention over target trades plus FFN sublayer.

    Flow (per forward call):
        fused_features [B, T, D], adjacency [T, T], target_idx [n_tgt]
        -> gather target features            [B, n_tgt, D]
        -> extract target adjacency          [n_tgt, n_tgt]
        -> multi-head masked self-attention  [B, n_tgt, units]
        -> residual + LayerNorm              [B, n_tgt, units]
        -> FFN (linear -> activation -> linear)
        -> residual + LayerNorm              [B, n_tgt, units]
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "target_attention") -> None:
        """
        Initialise TargetAttentionLayer from a configuration dictionary.

        :param layer_config: Dict with 'general' (num_heads, dropout_rate, k_nbrs, …)
            and 'parameters' (units, activation, kernel_initializer, bias_initializer).
        :param name: human-readable layer name for logging / serialization.
        """
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        # --- General hyper-parameters (defaults overridden by config) ---
        self.layer_type: str = "standard"
        self.use_residual: bool = True
        self.use_layer_norm: bool = True
        self.attention_mode: bool = True
        self.num_heads: int = 1
        self.dropout_rate: float = 0.0
        self.k_nbrs: int = 10
        self._unpack_configuration(config=layer_config.get("general", {}))

        # --- Layer parameters ---
        self.units: Optional[int] = None
        self.activation: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters", {}))

        # Validate that units is evenly divisible by num_heads for multi-head splits.
        if self.units % self.num_heads != 0:
            raise ValueError(
                f"units ({self.units}) must be divisible by num_heads ({self.num_heads})"
            )

        # Dimension per attention head.
        self.head_dim: int = self.units // self.num_heads

        # --- Multi-head attention projections (Q, K, V, output) ---
        # LazyLinear infers in_features on first forward call.
        self.query_proj = nn.LazyLinear(self.units)
        self.key_proj = nn.LazyLinear(self.units)
        self.value_proj = nn.LazyLinear(self.units)
        self.output_proj = nn.LazyLinear(self.units)

        # --- Feed-forward network: two linear layers with activation in between ---
        self.ffn_linear1 = nn.LazyLinear(self.units * 4)
        self.ffn_linear2 = nn.LazyLinear(self.units)
        self.ffn_activation = _build_activation_module(self.activation)

        # --- Layer normalization for attention and FFN sublayers ---
        self.attn_layer_norm = nn.LayerNorm(self.units) if self.use_layer_norm else None
        self.ffn_layer_norm = nn.LayerNorm(self.units) if self.use_layer_norm else None

        # --- Residual projection: maps input dim D -> units for skip connection ---
        self.input_proj = nn.LazyLinear(self.units, bias=False) if self.use_residual else None

        # --- Dropout applied to attention weights and FFN output ---
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        fused_features: torch.Tensor,
        adjacency: torch.Tensor,
        target_idx: torch.Tensor,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: gather targets, attend, FFN.

        :param fused_features: [B, T, D] fused node features (batched).
        :param adjacency: [T, T] adjacency matrix (sparse COO or dense), shared across batch.
        :param target_idx: [n_tgt] long tensor of target node indices.
        :param return_attention: if True, also return attention weights.
        :return: [B, n_tgt, units], or (output, attn_weights) when return_attention=True.
        """
        target_features = fused_features[:, target_idx, :]  # [B, n_tgt, D]

        target_adj = self._extract_target_submatrix(adjacency, target_idx)  # [n_tgt, n_tgt]

        attn_out, attn_weights = self._multi_head_attention(target_features, target_adj)

        if self.use_residual:
            residual = self.input_proj(target_features)  # [B, n_tgt, units]
            attn_out = attn_out + residual
        if self.use_layer_norm:
            attn_out = self.attn_layer_norm(attn_out)

        ffn_input = attn_out
        ffn_out = self.ffn_linear1(ffn_input)        # [B, n_tgt, units]
        ffn_out = self.ffn_activation(ffn_out)
        if self.dropout is not None:
            ffn_out = self.dropout(ffn_out)
        ffn_out = self.ffn_linear2(ffn_out)            # [B, n_tgt, units]

        if self.use_residual:
            ffn_out = ffn_out + ffn_input
        if self.use_layer_norm:
            ffn_out = self.ffn_layer_norm(ffn_out)

        if return_attention:
            return ffn_out, attn_weights
        return ffn_out

    # ------------------------------------------------------------------
    # Target submatrix extraction
    # ------------------------------------------------------------------

    def _extract_target_submatrix(
        self, adjacency: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract the [n_tgt, n_tgt] adjacency submatrix for target nodes.

        Dispatches to sparse or dense extraction based on the input tensor type.

        :param adjacency: [T, T] full adjacency (sparse COO or dense).
        :param target_idx: [n_tgt] target node indices.
        :return: [n_tgt, n_tgt] submatrix (sparse if input is sparse, dense otherwise).
        """
        if adjacency.is_sparse:
            return self._extract_sparse_submatrix(adjacency, target_idx)
        # Dense path: simple row then column index slicing.
        return adjacency[target_idx][:, target_idx]

    def _extract_sparse_submatrix(
        self, adjacency: torch.Tensor, target_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract a sparse [n_tgt, n_tgt] submatrix from a sparse COO adjacency.

        Uses a lookup table to remap global node indices to local target indices,
        keeping only edges where both endpoints are in the target set.

        :param adjacency: [T, T] sparse COO adjacency.
        :param target_idx: [n_tgt] target node indices (long).
        :return: [n_tgt, n_tgt] sparse COO submatrix, coalesced in row-major order.
        """
        adj = adjacency.coalesce()
        indices = adj.indices()   # [2, nnz] — global row and column indices
        values = adj.values()     # [nnz] — edge weights
        T = adj.shape[0]          # total number of nodes
        n_tgt = target_idx.shape[0]

        rows, cols = indices[0], indices[1]

        # Build a lookup table: global index -> local target index (-1 if not a target).
        lookup = torch.full((T,), -1, dtype=torch.long, device=adjacency.device)
        lookup[target_idx] = torch.arange(n_tgt, dtype=torch.long, device=adjacency.device)

        # Remap global row/col indices to local target indices.
        local_rows = lookup[rows]
        local_cols = lookup[cols]

        # Retain only edges where both endpoints are in the target set.
        keep = (local_rows >= 0) & (local_cols >= 0)

        new_indices = torch.stack([local_rows[keep], local_cols[keep]], dim=0)
        new_values = values[keep]

        # Build the submatrix sparse tensor and coalesce for row-major ordering.
        return torch.sparse_coo_tensor(
            new_indices, new_values, size=(n_tgt, n_tgt)
        ).coalesce()

    # ------------------------------------------------------------------
    # Multi-head self-attention
    # ------------------------------------------------------------------

    def _multi_head_attention(
        self, x: torch.Tensor, target_adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-head self-attention masked by the target adjacency.

        :param x: [B, n_tgt, D] target features.
        :param target_adj: [n_tgt, n_tgt] adjacency submatrix (sparse or dense).
        :return: (output [B, n_tgt, units], attn_weights [B, H, n_tgt, ...]).
        """
        B, n_tgt, _ = x.shape

        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        q = q.view(B, n_tgt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, n_tgt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, n_tgt, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if target_adj.is_sparse:
            attn_out, attn_weights = self._sparse_attention(q, k, v, target_adj)
        else:
            attn_out, attn_weights = self._dense_attention(q, k, v, target_adj)

        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, n_tgt, self.units)

        return self.output_proj(attn_out), attn_weights

    def _dense_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full masked attention using a dense adjacency matrix.

        :param q: [B, H, n_tgt, d_h] query vectors.
        :param k: [B, H, n_tgt, d_h] key vectors.
        :param v: [B, H, n_tgt, d_h] value vectors.
        :param adj: [n_tgt, n_tgt] dense adjacency.
        :return: (output [B, H, n_tgt, d_h], attn_weights [B, H, n_tgt, n_tgt]).
        """
        scale = self.head_dim ** 0.5

        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        mask = adj.unsqueeze(0).unsqueeze(0)  # [1, 1, n_tgt, n_tgt]
        very_neg = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
        scores = torch.where(mask > 0, scores, very_neg)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        return torch.matmul(attn_weights, v), attn_weights

    def _sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        adj: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Padded-neighbor attention using a sparse adjacency matrix.

        :param q: [B, H, n_tgt, d_h] query vectors.
        :param k: [B, H, n_tgt, d_h] key vectors.
        :param v: [B, H, n_tgt, d_h] value vectors.
        :param adj: [n_tgt, n_tgt] sparse COO adjacency submatrix.
        :return: (output [B, H, n_tgt, d_h], attn_weights [B, H, n_tgt, k_max]).
        """
        B, H, n_tgt, d_h = q.shape
        scale = d_h ** 0.5

        adj_c = adj.coalesce()
        indices = adj_c.indices()  # [2, nnz]
        rows, cols = indices[0], indices[1]

        row_counts = torch.bincount(rows.int(), minlength=n_tgt).long()
        k_max = row_counts.max().item()

        if k_max == 0:
            zeros = torch.zeros_like(q)
            empty_weights = torch.zeros(B, H, n_tgt, 0, device=q.device, dtype=q.dtype)
            return zeros, empty_weights

        row_starts = torch.zeros(n_tgt, dtype=torch.long, device=q.device)
        if n_tgt > 1:
            row_starts[1:] = row_counts[:-1].cumsum(0)

        edge_positions = torch.arange(rows.shape[0], device=q.device) - row_starts[rows]

        nbr_idx = torch.zeros(n_tgt, k_max, dtype=torch.long, device=q.device)
        nbr_idx[rows, edge_positions] = cols

        nbr_mask = torch.arange(k_max, device=q.device).unsqueeze(0) < row_counts.unsqueeze(1)

        k_nbr = k[:, :, nbr_idx, :]
        v_nbr = v[:, :, nbr_idx, :]

        scores = torch.matmul(
            q.unsqueeze(-2), k_nbr.transpose(-2, -1)
        ).squeeze(-2) / scale

        very_neg = torch.tensor(-1e9, dtype=scores.dtype, device=scores.device)
        mask_expanded = nbr_mask.unsqueeze(0).unsqueeze(0)
        scores = torch.where(mask_expanded, scores, very_neg)

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        if self.dropout is not None:
            attn_weights = self.dropout(attn_weights)

        attn_out = torch.matmul(
            attn_weights.unsqueeze(-2), v_nbr
        ).squeeze(-2)

        return attn_out, attn_weights

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return the layer configuration dictionary for serialization."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TargetAttentionLayer":
        """Reconstruct a TargetAttentionLayer from a configuration dictionary."""
        return cls(**config)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from a config sub-dict (e.g. num_heads=4, units=32)."""
        for k, v in config.items():
            setattr(self, k, v)
