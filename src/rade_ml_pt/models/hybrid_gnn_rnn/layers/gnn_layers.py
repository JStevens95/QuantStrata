"""
GNN layers for the Hybrid GNN-RNN model: GnnBlock, GraphSage, MixedGraphSage, GraphormerLayer.

Design (see ARCHITECTURE.md):
- GNN sublayers (GraphSage, MixedGraphSage, GraphormerLayer) are LINEAR primitives—they
  perform message passing / attention and a linear transform only. Activation is applied
  by the block.
- GnnBlock stacks L sublayers with LayerNorm, activation, and dropout between layers,
  plus a residual connection and final activation.
"""
import copy
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)

def _leaky_relu(x: torch.Tensor) -> torch.Tensor:
    return F.leaky_relu(x, negative_slope=0.2)


def _identity(x: torch.Tensor) -> torch.Tensor:
    return x


# Mapping from activation name strings to callable torch functions.
_ACTIVATION_MAP = {
    "relu": F.relu,
    "leaky_relu": _leaky_relu,
    "tanh": torch.tanh,
    "sigmoid": torch.sigmoid,
    "elu": F.elu,
    "selu": F.selu,
    "gelu": F.gelu,
    "linear": _identity,
    None: _identity,
}


def _get_activation(name: Optional[str]):
    """Return the torch activation function for a given name string."""
    if name in _ACTIVATION_MAP:
        return _ACTIVATION_MAP[name]
    raise ValueError(f"Unsupported activation: {name}. Choose from {list(_ACTIVATION_MAP.keys())}")


def _sparse_max_aggregation(x: torch.Tensor, adjacency: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Element-wise max aggregation over neighbors using a sparse adjacency matrix.

    For each node i, computes max over {x[j] : (i, j) is an edge} across each feature.
    Isolated nodes (no neighbors) get zeros instead of -inf.
    """
    indices = adjacency.indices() if adjacency.is_coalesced() else adjacency.coalesce().indices()
    rows, cols = indices[0], indices[1]

    # Gather neighbor features for every edge, then scatter-reduce to max per row.
    gathered = x[cols]  # [nnz, d_in] — feature vectors of neighbor nodes
    out = torch.full((num_nodes, x.shape[1]), float("-inf"), dtype=x.dtype, device=x.device)
    out.scatter_reduce_(0, rows.unsqueeze(1).expand_as(gathered), gathered, reduce="amax")

    # Replace -inf (isolated nodes with no neighbors) with zeros.
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


def _dense_max_aggregation(x: torch.Tensor, adjacency: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Element-wise max aggregation over neighbors using a dense adjacency matrix.

    Converts nonzero entries to edge index pairs, then delegates to scatter-reduce.
    """
    rows, cols = torch.where(adjacency > 0)  # edge indices from dense matrix
    gathered = x[cols]  # [num_edges, d_in]
    out = torch.full((num_nodes, x.shape[1]), float("-inf"), dtype=x.dtype, device=x.device)
    out.scatter_reduce_(0, rows.unsqueeze(1).expand_as(gathered), gathered, reduce="amax")

    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


class GraphSage(nn.Module):
    """
    Inductive GraphSAGE layer: h' = W_self·h + W_neigh·AGG(h | neighbors).

    Aggregation options: mean (default), max. When used inside GnnBlock, activation
    is typically None (block applies it). Standalone use can pass activation in config.
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "graph_sage") -> None:
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        # Defaults overridden by _unpack_configuration.
        self.layers: int = 1
        self.layer_type: Optional[str] = None
        self.dropout_rate: float = 0.0
        self.use_bias: Optional[bool] = None
        self.aggregation_op: str = "mean"
        self._unpack_configuration(config=layer_config.get("general"))
        # Config may use 'aggregator_op'; normalise to 'aggregation_op'.
        self.aggregation_op = getattr(self, "aggregator_op", self.aggregation_op)

        self.units: Optional[int] = None
        self.activation: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters"))

        # W_self: transforms the node's own features.
        self.dense_self = nn.LazyLinear(out_features=self.units, bias=self.use_bias)
        # W_neigh: transforms aggregated neighbor features.
        self.dense_neigh = nn.LazyLinear(out_features=self.units, bias=self.use_bias)
        # Dropout applied to input features before aggregation.
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None

        # Resolve activation function once.
        self._act_fn = _get_activation(self.activation)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward: h' = activation(W_self·h + W_neigh·AGG(h)).

        :param features: Node features [T, d_in].
        :param adjacency: Adjacency matrix [T, T], sparse COO or dense.
        :return: Updated node features [T, d_out].
        """
        x = features
        num_nodes = x.shape[0]
        is_sparse = adjacency.is_sparse

        if self.dropout is not None:
            x = self.dropout(x)

        # --- Aggregation: compute neighbor summary per node ---
        if self.aggregation_op.lower() == "mean":
            # Mean: for row-normalised A, A @ x = weighted mean of neighbors per node.
            if is_sparse:
                neigh_summary = torch.sparse.mm(adjacency, x)
            else:
                neigh_summary = torch.matmul(adjacency, x)

        elif self.aggregation_op.lower() == "max":
            # Max: per-feature max over neighbors.
            if is_sparse:
                neigh_summary = _sparse_max_aggregation(x, adjacency, num_nodes)
            else:
                neigh_summary = _dense_max_aggregation(x, adjacency, num_nodes)
        else:
            raise ValueError(f"Unsupported aggregator: {self.aggregation_op.lower()}...")

        # --- Linear transform and combine ---
        h_self = self.dense_self(x)           # W_self · h
        h_neigh = self.dense_neigh(neigh_summary)  # W_neigh · AGG(h)
        out = h_self + h_neigh
        return self._act_fn(out)

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration to a plain dict."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphSage":
        """Reconstruct layer from a serialized config dict."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict (e.g. layers=2, dropout_rate=0.1)."""
        for k, v in config.items():
            setattr(self, k, v)


class MixedGraphSage(nn.Module):
    """
    Inductive mixed-aggregation GraphSage: h' = activation(W·[h || mean(h_neigh) || max(h_neigh)]).

    Concatenates self, mean-aggregated neighbors, and max-aggregated neighbors (3*d_in),
    then applies a single fusion Linear. Captures both smooth (mean) and salient (max) signals.
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "mixed_graph_sage") -> None:
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        self.layers: int = 1
        self.layer_type: Optional[str] = None
        self.dropout_rate: float = 0.0
        self.use_bias: Optional[bool] = None
        self._unpack_configuration(config=layer_config.get("general"))

        self.units: Optional[int] = None
        self.activation: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters"))

        # W_fuse: R^{3*d_in} -> R^{d_out}  (input = self || mean || max concatenated).
        self.fusion_dense = nn.LazyLinear(out_features=self.units, bias=self.use_bias)
        # Dropout applied to input features.
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None

        self._act_fn = _get_activation(self.activation)

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward: concat [h, mean(h_neigh), max(h_neigh)] -> W_fuse -> activation.

        :param features: Node features [T, d_in].
        :param adjacency: Adjacency matrix [T, T], sparse COO or dense.
        :return: Updated node features [T, d_out].
        """
        x = features
        num_nodes = x.shape[0]
        is_sparse = adjacency.is_sparse

        if self.dropout is not None:
            x = self.dropout(x)

        # --- Mean aggregation: A @ x (row-normalised A => mean per node) ---
        if is_sparse:
            neigh_mean = torch.sparse.mm(adjacency, x)
        else:
            neigh_mean = torch.matmul(adjacency, x)

        # --- Max aggregation: per-feature max over neighbors ---
        if is_sparse:
            neigh_max = _sparse_max_aggregation(x, adjacency, num_nodes)
        else:
            neigh_max = _dense_max_aggregation(x, adjacency, num_nodes)

        # --- Concatenate [self, mean, max] and fuse through linear layer ---
        concat_feats = torch.cat([x, neigh_mean, neigh_max], dim=1)  # [T, 3*d_in]
        out = self.fusion_dense(concat_feats)  # [T, d_out]
        return self._act_fn(out)

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration to a plain dict."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MixedGraphSage":
        """Reconstruct layer from a serialized config dict."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict."""
        for k, v in config.items():
            setattr(self, k, v)


class GraphormerLayer(nn.Module):
    """
    Sparse Graph Transformer sublayer with centrality encoding (Graphormer-style).

    Replaces the fixed aggregation of GraphSAGE (mean / max) with learned,
    data-dependent multi-head attention over the k-NN neighborhood, augmented
    by centrality encoding that distinguishes structurally important (high
    in-degree) nodes from peripheral ones.

    Per-node update:
        Q_v = W_q · x_v  +  z_q(deg_in(v))       query with centrality bias
        K_u = W_k · x_u  +  z_k(deg_in(u))       key with centrality bias
        V_u = W_v · x_u                            value (no centrality)

        α(v,u) = softmax_u( Q_v^T K_u / √d_h  +  b_spatial )   over u ∈ N(v)
        h_v    = W_out · concat_heads( Σ_u α(v,u) V_u )

    Centrality embeddings z_q, z_k are initialised to zero so the layer starts
    equivalent to vanilla multi-head attention and learns structural biases
    during training.

    Follows Ying et al. (2021) "Do Transformers Really Perform Bad for Graph
    Representation?" adapted for sparse k-NN trade graphs.  When used inside
    GnnBlock, activation is typically None (block applies it).

    Neighbor indices and in-degrees are cached internally since they depend
    only on the adjacency topology (not on features).
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "graphormer") -> None:
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        # Defaults overridden by _unpack_configuration.
        self.layers: int = 1
        self.layer_type: Optional[str] = None
        self.dropout_rate: float = 0.0
        self.use_bias: Optional[bool] = None
        self.num_heads: int = 4
        self.k_nbrs: int = 50
        self.max_degree: int = 512
        self._unpack_configuration(config=layer_config.get("general"))

        self.units: Optional[int] = None
        self.activation: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters"))

        assert self.units % self.num_heads == 0, (
            f"Graphormer units ({self.units}) must be divisible by num_heads ({self.num_heads})"
        )
        self.head_dim: int = self.units // self.num_heads

        # Centrality encoding: separate learned embeddings for query and key
        # so the model can learn asymmetric effects (hub-as-query ≠ hub-as-key).
        # Zero-initialised → starts equivalent to vanilla attention.
        self.centrality_enc_q = nn.Embedding(self.max_degree + 1, self.units)
        self.centrality_enc_k = nn.Embedding(self.max_degree + 1, self.units)
        nn.init.zeros_(self.centrality_enc_q.weight)
        nn.init.zeros_(self.centrality_enc_k.weight)

        # Q / K / V projections (LazyLinear to handle variable d_in).
        self.W_q = nn.LazyLinear(out_features=self.units, bias=self.use_bias)
        self.W_k = nn.LazyLinear(out_features=self.units, bias=self.use_bias)
        self.W_v = nn.LazyLinear(out_features=self.units, bias=self.use_bias)

        # Output projection after head concatenation.
        self.out_proj = nn.Linear(self.units, self.units, bias=self.use_bias)

        # Per-head spatial bias added to every attention score within the
        # k-NN window.  Allows each head to learn a different baseline
        # attention strength toward neighbors vs. suppression.
        self.spatial_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None
        self._act_fn = _get_activation(self.activation)

        # Topology-dependent caches (depend only on adjacency, not features).
        self._cached_in_degrees: Optional[torch.Tensor] = None
        self._cached_nbr_indices: Optional[torch.Tensor] = None
        self._cached_nbr_mask: Optional[torch.Tensor] = None
        self._cached_k: int = 0
        self._cached_T: int = 0

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward: multi-head attention with centrality encoding over k-NN neighbors.

        :param features: Node features [T, d_in].
        :param adjacency: Adjacency matrix [T, T], sparse COO or dense.
        :return: Updated node features [T, d_out].
        """
        T = features.shape[0]
        device = features.device

        x = features
        if self.dropout is not None:
            x = self.dropout(x)

        # --- Structural encodings (topology-only, cached across forward calls) ---
        in_degrees = self._get_in_degrees(adjacency, T, device)
        nbr_indices, nbr_mask, k = self._get_neighbor_indices(adjacency, T, device)

        # --- Q, K, V with centrality bias on Q and K ---
        Q = self.W_q(x) + self.centrality_enc_q(in_degrees)   # [T, units]
        K = self.W_k(x) + self.centrality_enc_k(in_degrees)   # [T, units]
        V = self.W_v(x)                                        # [T, units]

        # --- Reshape for multi-head: [T, units] -> [H, T, d_h] ---
        H = self.num_heads
        d_h = self.head_dim
        Q = Q.view(T, H, d_h).permute(1, 0, 2)   # [H, T, d_h]
        K = K.view(T, H, d_h).permute(1, 0, 2)
        V = V.view(T, H, d_h).permute(1, 0, 2)

        # Edge case: graph has no edges.
        if k == 0:
            out = torch.zeros(T, self.units, dtype=features.dtype, device=device)
            return self._act_fn(self.out_proj(out))

        # --- Gather neighbor K, V from padded index tensor ---
        nbr_flat = nbr_indices.reshape(-1)                          # [T*k]
        K_nbr = K[:, nbr_flat, :].view(H, T, k, d_h)              # [H, T, k, d_h]
        V_nbr = V[:, nbr_flat, :].view(H, T, k, d_h)

        # --- Scaled dot-product scores + spatial bias ---
        scale = math.sqrt(d_h)
        scores = torch.matmul(
            Q.unsqueeze(2), K_nbr.transpose(-2, -1),
        ).squeeze(2) / scale                                        # [H, T, k]
        scores = scores + self.spatial_bias.view(H, 1, 1)

        # --- Masked softmax over valid neighbors ---
        mask_exp = nbr_mask.unsqueeze(0)                            # [1, T, k]
        scores = scores.masked_fill(~mask_exp, float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # --- Weighted aggregation of neighbor values ---
        attn_out = torch.matmul(
            attn_weights.unsqueeze(2), V_nbr,
        ).squeeze(2)                                                # [H, T, d_h]

        # --- Concat heads -> output projection ---
        attn_out = attn_out.permute(1, 0, 2).reshape(T, self.units) # [T, units]
        out = self.out_proj(attn_out)
        return self._act_fn(out)

    # ── Structural encoding helpers (cached) ─────────────────────────

    def _get_in_degrees(
        self, adjacency: torch.Tensor, T: int, device: torch.device,
    ) -> torch.Tensor:
        """Compute clamped in-degree per node; cached across forward calls."""
        if self._cached_in_degrees is not None and self._cached_T == T:
            return self._cached_in_degrees

        if adjacency.is_sparse:
            adj = adjacency.coalesce()
            cols = adj.indices()[1]
            in_deg = torch.bincount(cols, minlength=T).clamp(max=self.max_degree)
        else:
            in_deg = (adjacency > 0).sum(dim=0).long().clamp(max=self.max_degree)

        self._cached_in_degrees = in_deg
        self._cached_T = T
        return in_deg

    def _get_neighbor_indices(
        self, adjacency: torch.Tensor, T: int, device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Build padded neighbor index tensor [T, k] and validity mask; cached."""
        if (
            self._cached_nbr_indices is not None
            and self._cached_T == T
            and self._cached_nbr_indices.device == device
        ):
            return self._cached_nbr_indices, self._cached_nbr_mask, self._cached_k

        if adjacency.is_sparse:
            adj = adjacency.coalesce()
            rows = adj.indices()[0]
            cols = adj.indices()[1]
        else:
            rows, cols = torch.where(adjacency > 0)

        row_counts = torch.bincount(rows, minlength=T)
        k = min(self.k_nbrs, int(row_counts.max().item())) if row_counts.numel() > 0 else 0

        if k == 0:
            empty_idx = torch.zeros(T, 0, dtype=torch.long, device=device)
            empty_mask = torch.zeros(T, 0, dtype=torch.bool, device=device)
            self._cached_nbr_indices = empty_idx
            self._cached_nbr_mask = empty_mask
            self._cached_k = 0
            return empty_idx, empty_mask, 0

        cum_counts = torch.cat([
            torch.tensor([0], device=device, dtype=torch.long),
            torch.cumsum(row_counts, dim=0)[:-1],
        ])
        within_group_pos = torch.arange(len(rows), device=device) - cum_counts[rows]

        keep_mask = within_group_pos < k
        kept_rows = rows[keep_mask]
        kept_cols = cols[keep_mask]
        kept_pos = within_group_pos[keep_mask]

        nbr_indices = torch.zeros(T, k, dtype=torch.long, device=device)
        nbr_indices[kept_rows, kept_pos] = kept_cols

        capped_counts = torch.clamp(row_counts, max=k)
        nbr_mask = torch.arange(k, device=device).unsqueeze(0) < capped_counts.unsqueeze(1)

        self._cached_nbr_indices = nbr_indices
        self._cached_nbr_mask = nbr_mask
        self._cached_k = k
        return nbr_indices, nbr_mask, k

    def invalidate_cache(self) -> None:
        """Clear cached structural encodings (call if graph topology changes)."""
        self._cached_in_degrees = None
        self._cached_nbr_indices = None
        self._cached_nbr_mask = None
        self._cached_k = 0
        self._cached_T = 0

    def get_config(self) -> Dict[str, Any]:
        """Serialize layer configuration to a plain dict."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GraphormerLayer":
        """Reconstruct layer from a serialized config dict."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict."""
        for k, v in config.items():
            setattr(self, k, v)


class GnnBlock(nn.Module):
    """
    Graph neural network block stacking L GNN sublayers (GraphSAGE, MixedGraphSAGE,
    or Graphormer) with residual connections, LayerNorm, activation, and dropout.

    Flow (2-layer example):
        X -> [GNN1 (linear)] -> LN -> σ -> Dropout -> [GNN2 (linear)] -> LN -> (Z + W_proj·X) -> σ -> H

    Sublayers are linear; this block applies all activation (between layers and after residual).
    Matches PyG/DGL convention and ResNet residual formulation.
    """

    def __init__(self, layer_config: Dict[str, Any], name: str = "gnn_block") -> None:
        """
        :param layer_config: Dict with keys 'general' and 'parameters'.
            general: layers, layer_type, dropout_rate, use_bias, use_residual, batch_norm
            parameters: units, activation, kernel_initializer, bias_initializer
        """
        super().__init__()

        self.layer_config: Dict[str, Any] = layer_config
        self.layer_name: str = name

        # From 'general': number of sublayers, type (graph_sage / mixed_graph_sage), etc.
        self.num_layers: int = 1
        self.layer_type: Optional[str] = None
        self.dropout_rate: float = 0.0
        self.use_bias: Optional[bool] = None
        self.use_residual: Optional[bool] = None
        self.batch_norm: bool = False
        self._unpack_configuration(config=layer_config.get("general"))
        # Config uses 'layers'; store as num_layers to avoid shadowing nn.Module internals.
        self.num_layers = getattr(self, "layers", self.num_layers)

        # From 'parameters': hidden size, activation name, initialisers.
        self.units: Optional[int] = None
        self.activation_name: Optional[str] = None
        self.kernel_initializer: Optional[str] = None
        self.bias_initializer: Optional[str] = None
        self._unpack_configuration(config=layer_config.get("parameters"))
        # Config key is 'activation'; rename to avoid collision with nn.Module method.
        self.activation_name = getattr(self, "activation", self.activation_name)

        # Build GNN sublayers, norm layers, projection, and dropout eagerly.
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self._build_gnn_layers()

        # Project input to match GNN output dim for residual: W_proj : R^{p} -> R^{d_g}.
        self.input_projection = nn.LazyLinear(out_features=self.units, bias=False) if self.use_residual else None

        # Dropout applied between sublayers.
        self.dropout = nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0.0 else None

    def _build_gnn_layers(self) -> None:
        """
        Build L GNN sublayers and optionally LayerNorm.

        Supported sublayer types:
            - ``graph_sage``       : GraphSAGE (Hamilton et al. 2017)
            - ``mixed_graph_sage`` : Mixed mean+max aggregation
            - ``graphormer``       : Sparse Graph Transformer with centrality encoding

        Deep-copies config and forces activation=None so each sublayer is linear. The block
        applies activation between layers and after the residual add.
        """
        for i in range(self.num_layers):
            sub_config = copy.deepcopy(self.layer_config)
            sub_config["parameters"]["activation"] = None
            sub_config["general"]["dropout_rate"] = 0.0

            layer_type_lower = self.layer_type.lower()
            sub_name = f"{self.layer_name}_{layer_type_lower}_{i}"

            if layer_type_lower == "graph_sage":
                layer = GraphSage(layer_config=sub_config, name=sub_name)
            elif layer_type_lower == "mixed_graph_sage":
                layer = MixedGraphSage(layer_config=sub_config, name=sub_name)
            elif layer_type_lower == "graphormer":
                layer = GraphormerLayer(layer_config=sub_config, name=sub_name)
            else:
                raise ValueError(
                    f"Undefined GNN layer type: {layer_type_lower!r}. "
                    f"Choose from: graph_sage, mixed_graph_sage, graphormer"
                )

            self.gnn_layers.append(layer)

            if self.batch_norm:
                self.norm_layers.append(nn.LayerNorm(self.units, eps=1e-5))

    def forward(self, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: X, A -> H.

        :param features: Node features [T, p].
        :param adjacency: Adjacency matrix [T, T], sparse COO or dense.
        :return: H [T, d_g], node embeddings.
        """
        residual = features
        if self.use_residual:
            residual = self.input_projection(residual)

        x = features
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, adjacency)

            if self.batch_norm and i < len(self.norm_layers):
                x = self.norm_layers[i](x)

            if i < self.num_layers - 1:
                x = self._activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)

        # Residual add: H = Z^(L-1) + W_proj · X
        if self.use_residual:
            x = x + residual

        # Final activation: H = σ(Z + residual)
        x = self._activation(x)
        return x

    def _activation(self, output: torch.Tensor) -> torch.Tensor:
        """Apply the configured activation (relu, leaky_relu, tanh, etc.) or identity if None."""
        return _get_activation(self.activation_name)(output)

    def get_config(self) -> Dict[str, Any]:
        """Serialize block configuration to a plain dict."""
        return {"layer_config": self.layer_config, "name": self.layer_name}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GnnBlock":
        """Reconstruct block from a serialized config dict."""
        return cls(**config)

    def _unpack_configuration(self, config: Dict[str, Any]) -> None:
        """Set instance attributes from config dict (e.g. layers=2, dropout_rate=0.1)."""
        for k, v in config.items():
            setattr(self, k, v)
