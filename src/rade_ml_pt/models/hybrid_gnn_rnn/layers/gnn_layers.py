"""
GNN layers for the Hybrid GNN-RNN model: GnnBlock, GraphSage, MixedGraphSage.

Design (see ARCHITECTURE.md):
- GNN sublayers (GraphSage, MixedGraphSage) are LINEAR primitives—they perform message
  passing, aggregation, and a linear transform only. Activation is applied by the block.
- GnnBlock stacks L sublayers with LayerNorm, activation, and dropout between layers,
  plus a residual connection and final activation.
"""
import copy
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Union

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


def _sparse_max_aggregation(x: torch.Tensor, adjacency: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    Element-wise max aggregation over neighbors using a sparse adjacency matrix.

    For each node i, computes max over {x[j] : (i, j) is an edge} across each feature.
    Isolated nodes (no neighbors) get zeros instead of -inf.
    """
    indices = adjacency.coalesce().indices()  # [2, nnz] — row and column edge indices
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


class GnnBlock(nn.Module):
    """
    Graph neural network block stacking L GNN sublayers (GraphSAGE or MixedGraphSAGE)
    with residual connections, LayerNorm, activation, and dropout.

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
        Build L GNN sublayers (GraphSage or MixedGraphSage) and optionally LayerNorm.

        Deep-copies config and forces activation=None so each sublayer is linear. The block
        applies activation between layers and after the residual add.
        """
        for i in range(self.num_layers):
            # Force sublayers to be linear (no activation).
            sub_config = copy.deepcopy(self.layer_config)
            sub_config["parameters"]["activation"] = None

            layer_type_lower = self.layer_type.lower()
            sub_name = f"{self.layer_name}_{layer_type_lower}_{i}"

            if layer_type_lower == "graph_sage":
                layer = GraphSage(layer_config=sub_config, name=sub_name)
            elif layer_type_lower == "mixed_graph_sage":
                layer = MixedGraphSage(layer_config=sub_config, name=sub_name)
            else:
                raise ValueError(f"Undefined layer type, got {layer_type_lower}")

            self.gnn_layers.append(layer)

            if self.batch_norm:
                # LayerNorm on last dimension (feature axis); epsilon for numerical stability.
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
