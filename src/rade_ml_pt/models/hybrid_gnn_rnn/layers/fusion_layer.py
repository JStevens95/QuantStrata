"""
Fusion layer combining GNN and RNN streams via multi-head cross-attention.

Uses sparse neighborhood attention (O(T*k)) when adjacency is a SparseTensor,
so each trade only attends to its k-NN neighbors. Supports gate or add fusion modes.
"""
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# Mapping from activation name strings to callable torch functions.
_ACTIVATION_MAP = {
    "relu": F.relu,
    "leaky_relu": lambda x: F.leaky_relu(x, negative_slope=0.2),
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": F.elu,
    "selu": F.selu,
    "gelu": F.gelu,
    "linear": lambda x: x,
    None: lambda x: x,
}


def _get_activation(name: Optional[str]):
    """Return the torch activation function for a given name string."""
    if name in _ACTIVATION_MAP:
        return _ACTIVATION_MAP[name]
    raise ValueError(f"Unsupported activation: {name}. Choose from {list(_ACTIVATION_MAP.keys())}")


class FusionLayer(nn.Module):
    """
    Cross-attention fusion: GNN (structural) + RNN (temporal) -> fused features [B, T, d_f].

    Query = W_q_rnn(RNN) + W_q_gnn(GNN); Key, Value = GNN. Attention is masked by adjacency.
    Output is gated with RNN: gate*fusion + (1-gate)*rnn.
    """

    def __init__(self, layer_config: Dict[str, Any], **kwargs) -> None:
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config

        # --- General hyper-parameters (fusion_mode, dropout, heads, k_nbrs) ---
        self.dropout_rate: float = 0.0
        self.fusion_mode: Optional[str] = None
        self.num_heads: int = 1
        self.k_nbrs: int = 50
        self._unpack_configuration(config=layer_config.get("general"))

        # --- Layer parameters (units, activation, initialisers) ---
        self.units: Optional[int] = None
        self.activation: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters"))

        assert self.units % self.num_heads == 0, "Fusion units must be divisible by num_heads."
        self.head_units: int = self.units // self.num_heads

        # --- Projection layers: map GNN/RNN inputs to common dimensionality ---
        self.gnn_projection = nn.LazyLinear(out_features=self.units)
        self.rnn_projection = nn.LazyLinear(out_features=self.units)

        # --- Q/K/V projections for multi-head cross-attention ---
        # Query is formed from both streams: Q = W_q_rnn(rnn) + W_q_gnn(gnn)
        self.W_q_rnn = nn.LazyLinear(out_features=self.units)
        self.W_q_gnn = nn.LazyLinear(out_features=self.units)
        # Key and Value come from the GNN (structural) stream only
        self.W_k = nn.LazyLinear(out_features=self.units)
        self.W_v = nn.LazyLinear(out_features=self.units)

        # --- Output projection after multi-head attention concatenation ---
        self.output_projection = nn.LazyLinear(out_features=self.units)

        # --- Gate projection (only used in "gate" fusion mode) ---
        # Input is [fusion; rnn_proj] (2*units) -> sigmoid gate (units)
        if self.fusion_mode == "gate":
            self.gate_projection = nn.LazyLinear(out_features=self.units)

        # --- Layer normalization applied after fusion ---
        self.layer_norm = nn.LayerNorm(self.units)

        # --- Dropout applied to attention output before fusion ---
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None

    def forward(
        self,
        inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass: (GNN features, RNN features, adjacency) -> fused [B, T, units].

        :param inputs: Tuple of (gnn_features [T, d_g], rnn_features [B, d_r], adjacency [T, T]).
        :param return_attention: If True, also return attention weights.
        :return: Fused features [B, T, units], optionally with attention weights.
        """
        gnn_features, rnn_features, adjacency = inputs

        B = rnn_features.shape[0]   # batch size
        T = gnn_features.shape[0]   # number of trades / nodes
        is_sparse = adjacency.is_sparse

        # --- Broadcast inputs to [B, T, d] for batch-wise processing ---
        # GNN features are shared across the batch: [T, d_g] -> [B, T, d_g]
        gnn_broadcast = gnn_features.unsqueeze(0).expand(B, -1, -1)
        # RNN features are per-sample: [B, d_r] -> [B, T, d_r]
        rnn_broadcast = rnn_features.unsqueeze(1).expand(-1, T, -1)

        # --- Project both streams to common dimensionality (units) ---
        gnn_proj = self.gnn_projection(gnn_broadcast)   # [B, T, units]
        rnn_proj = self.rnn_projection(rnn_broadcast)    # [B, T, units]

        # --- Compute Q, K, V for multi-head cross-attention ---
        # Query combines both streams so the model attends from joint context
        Q = self.W_q_rnn(rnn_proj) + self.W_q_gnn(gnn_proj)   # [B, T, units]
        # Keys and values come from the GNN (structural) stream only
        K = self.W_k(gnn_proj)   # [B, T, units]
        V = self.W_v(gnn_proj)   # [B, T, units]

        # --- Reshape for multi-head: [B, T, units] -> [B, H, T, d_h] ---
        Q = Q.view(B, T, self.num_heads, self.head_units).permute(0, 2, 1, 3)
        K = K.view(B, T, self.num_heads, self.head_units).permute(0, 2, 1, 3)
        V = V.view(B, T, self.num_heads, self.head_units).permute(0, 2, 1, 3)

        # --- Compute masked attention (dense or sparse path) ---
        if is_sparse:
            attn_out, attn_weights = self._sparse_attention(Q, K, V, adjacency, B, T)
        else:
            attn_out, attn_weights = self._dense_attention(Q, K, V, adjacency, B, T)

        # --- Reassemble heads: [B, H, T, d_h] -> [B, T, H*d_h] = [B, T, units] ---
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, self.units)

        # --- Output projection ---
        fusion = self.output_projection(attn_out)   # [B, T, units]

        if self.dropout is not None:
            fusion = self.dropout(fusion)

        # --- Fuse attention output with RNN stream ---
        if self.fusion_mode == "gate":
            # Concatenate fusion and RNN projected features, project through sigmoid gate
            gate_input = torch.cat([fusion, rnn_proj], dim=-1)   # [B, T, 2*units]
            gate = torch.sigmoid(self.gate_projection(gate_input))   # [B, T, units]
            # Soft blend: gate * attention_result + (1 - gate) * rnn_stream
            out = gate * fusion + (1.0 - gate) * rnn_proj
        elif self.fusion_mode == "add":
            # Simple additive fusion of attention result and RNN stream
            out = fusion + rnn_proj
        else:
            raise ValueError(
                f"Unsupported fusion_mode: '{self.fusion_mode}'. Choose from ['gate', 'add']."
            )

        # --- Final layer normalization ---
        out = self.layer_norm(out)

        if return_attention:
            return out, attn_weights
        return out

    def _dense_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        adjacency: torch.Tensor,
        B: int,
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full dense attention masked by the adjacency matrix.

        Computes all T x T attention scores, then zeros out non-neighbor pairs.
        Complexity: O(T^2) per head.

        :param Q: Query tensor [B, H, T, d_h].
        :param K: Key tensor [B, H, T, d_h].
        :param V: Value tensor [B, H, T, d_h].
        :param adjacency: Dense adjacency matrix [T, T].
        :param B: Batch size.
        :param T: Number of nodes / trades.
        :return: (attention output [B, H, T, d_h], attention weights [B, H, T, T]).
        """
        # Scaled dot-product scores: [B, H, T, T]
        scale = math.sqrt(self.head_units)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # Create mask from adjacency: zero entries mean no edge -> fill with -inf
        # adjacency [T, T] -> [1, 1, T, T] for broadcasting across batch and heads
        adj_dense = adjacency.to_dense() if adjacency.is_sparse else adjacency
        mask = (adj_dense == 0).unsqueeze(0).unsqueeze(0)   # [1, 1, T, T]
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax over key dimension (last axis) to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)   # [B, H, T, T]
        # Replace NaN values (from rows where all scores are -inf) with zeros
        attn_weights = torch.where(
            torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )

        # Weighted sum of values: [B, H, T, T] @ [B, H, T, d_h] -> [B, H, T, d_h]
        attn_out = torch.matmul(attn_weights, V)
        return attn_out, attn_weights

    def _sparse_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        adjacency: torch.Tensor,
        B: int,
        T: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse neighborhood attention using padded neighbor indices.

        Each node only attends to its k-nearest neighbors from the sparse adjacency,
        reducing complexity from O(T^2) to O(T*k) per head.

        :param Q: Query tensor [B, H, T, d_h].
        :param K: Key tensor [B, H, T, d_h].
        :param V: Value tensor [B, H, T, d_h].
        :param adjacency: Sparse COO adjacency matrix [T, T].
        :param B: Batch size.
        :param T: Number of nodes / trades.
        :return: (attention output [B, H, T, d_h], attention weights [B, H, T, k]).
        """
        device = Q.device
        adjacency = adjacency.coalesce()

        # Extract edge indices from sparse COO format: [2, nnz]
        rows = adjacency.indices()[0]   # source nodes [nnz]
        cols = adjacency.indices()[1]   # target / neighbor nodes [nnz]

        # Count how many neighbors each node has
        row_counts = torch.bincount(rows, minlength=T)   # [T]
        # Cap the neighbor count at k_nbrs to limit attention window
        k = min(self.k_nbrs, int(row_counts.max().item())) if row_counts.numel() > 0 else 0

        # Edge case: no edges in the graph -> return zero attention
        if k == 0:
            attn_out = torch.zeros(B, self.num_heads, T, self.head_units, device=device, dtype=Q.dtype)
            attn_weights = torch.zeros(B, self.num_heads, T, 0, device=device, dtype=Q.dtype)
            return attn_out, attn_weights

        # --- Build padded neighbor index tensor [T, k] ---
        # coalesce() already sorts indices lexicographically (row-major),
        # so rows is non-decreasing and cols are sorted within each row.

        # Cumulative start offset for each node's neighbor list
        cum_counts = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            torch.cumsum(row_counts, dim=0)[:-1],
        ])
        # Position of each edge within its source node's neighbor group
        within_group_pos = torch.arange(len(rows), device=device) - cum_counts[rows]

        # Keep only the first k neighbors per node (truncate if a node has more than k)
        keep_mask = within_group_pos < k
        kept_rows = rows[keep_mask]
        kept_cols = cols[keep_mask]
        kept_pos = within_group_pos[keep_mask]

        # Scatter neighbor column indices into the padded [T, k] tensor
        nbr_indices = torch.zeros(T, k, dtype=torch.long, device=device)
        nbr_indices[kept_rows, kept_pos] = kept_cols

        # Boolean validity mask: True where a real neighbor exists
        # mask[i, j] = True if node i has at least (j+1) neighbors
        capped_counts = torch.clamp(row_counts, max=k)   # [T], capped at k
        nbr_mask = torch.arange(k, device=device).unsqueeze(0) < capped_counts.unsqueeze(1)   # [T, k]

        # --- Gather K, V for each node's neighbors ---
        # Flatten [T, k] -> [T*k] and index into K / V along the node dimension
        nbr_flat = nbr_indices.reshape(-1)   # [T*k]
        K_nbr = K[:, :, nbr_flat, :].view(B, self.num_heads, T, k, self.head_units)   # [B, H, T, k, d_h]
        V_nbr = V[:, :, nbr_flat, :].view(B, self.num_heads, T, k, self.head_units)   # [B, H, T, k, d_h]

        # --- Compute attention scores: Q dot K_nbr for each node's neighbors ---
        # Q: [B, H, T, 1, d_h] @ K_nbr^T: [B, H, T, d_h, k] -> [B, H, T, 1, k] -> squeeze -> [B, H, T, k]
        scale = math.sqrt(self.head_units)
        scores = torch.matmul(
            Q.unsqueeze(-2), K_nbr.transpose(-2, -1)
        ).squeeze(-2) / scale   # [B, H, T, k]

        # Mask out padded (invalid) neighbor positions with -inf before softmax
        mask_exp = nbr_mask.unsqueeze(0).unsqueeze(0)   # [1, 1, T, k]
        scores = scores.masked_fill(~mask_exp, float("-inf"))

        # Softmax over the neighbor dimension to get attention weights
        attn_weights = torch.softmax(scores, dim=-1)   # [B, H, T, k]
        # Replace NaN (from nodes with zero neighbors) with zeros
        attn_weights = torch.where(
            torch.isnan(attn_weights), torch.zeros_like(attn_weights), attn_weights
        )

        # --- Weighted sum of neighbor values ---
        # attn_weights: [B, H, T, 1, k] @ V_nbr: [B, H, T, k, d_h] -> [B, H, T, 1, d_h] -> squeeze
        attn_out = torch.matmul(
            attn_weights.unsqueeze(-2), V_nbr
        ).squeeze(-2)   # [B, H, T, d_h]

        return attn_out, attn_weights

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration to a plain dict."""
        return {"layer_config": self.layer_config}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FusionLayer":
        """Reconstruct layer from a serialized config dict."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict (e.g. fusion_mode='gate', units=64)."""
        for k, v in config.items():
            setattr(self, k, v)
