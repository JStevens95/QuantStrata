"""
Message passing over the instrument graph.

What this computes, and why a risk model wants it
--------------------------------------------------
Each instrument starts as its own encoded attributes -- strike, maturity,
delta, vega, product type. That vector says what the instrument *is*, and
nothing about what it sits next to. Message passing fixes that: each round
replaces an instrument's representation with a function of itself and its
neighbours, so after two rounds an instrument carries information from
everything within two hops of it in attribute space.

That is the point for a book of trades. A thinly traded instrument has
little history of its own, but its neighbours -- same underlying, similar
maturity, similar moneyness -- have plenty, and the graph is what lets their
behaviour inform its prediction.

Why mean *and* max
------------------
``mixed_graph_sage`` concatenates three things: the instrument itself, the
mean over its neighbours, and the element-wise maximum over them.

The mean alone is the standard choice and it is the wrong one here on its
own. Averaging is a low-pass filter: it describes the typical neighbour and
discards the extreme one. For a risk graph the extreme neighbour is
frequently the informative one -- the instrument in the group that actually
moved. Keeping the maximum alongside the mean lets the layer see both the
consensus and the outlier, and lets the network learn which matters where.

Keeping the instrument's own vector in the concatenation matters for the
same reason the graph has self-loops: without it, one round would replace
each instrument by a summary of its neighbours, and its own attributes
would be gone.

Why the sublayers are linear
----------------------------
Each round applies a linear map and nothing else. The activation lives on
the block, applied *between* rounds and once more after the residual add.

Two reasons. The residual addition has to happen before the final
non-linearity, which is the ResNet formulation and the reason deep stacks
train at all -- an activation inside each round would put a non-linearity
between the residual and its destination. And it keeps each round's
mathematics trivially inspectable, which matters when a layer is being
checked against a recorded baseline tensor by tensor.

Eager construction
------------------
Every parameter is sized from the input signature at build time rather than
materialised by a first forward pass. The original used lazy layers, which
is defect 6: a lazy parameter that materialises after the optimiser has been
constructed leaves the optimiser holding the placeholder, and the layer
silently never trains. Building eagerly removes the failure rather than
sequencing around it.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import ActivationName, HybridModelSpec

__all__ = ["GnnBlock", "GraphSage", "MixedGraphSage", "activation_function"]

#: Epsilon for every normalisation in the model, matching the original and
#: PyTorch's own default. Named once so the five layers cannot drift apart.
LAYER_NORM_EPS = 1e-5

#: Negative slope for ``leaky_relu``, matching the original.
_LEAKY_RELU_SLOPE = 0.01

#: How many vectors ``mixed_graph_sage`` concatenates per node: the node, the
#: neighbour mean, and the neighbour maximum.
_MIXED_AGGREGATES = 3


def activation_function(name: ActivationName) -> Callable[[Tensor], Tensor]:
    """
    Return the activation the given name refers to.

    Parameters
    ----------
    name
        One of the names :data:`~..spec.ActivationName` allows.

    Returns
    -------
    Callable
        The function.

    Raises
    ------
    ContractError
        If the name is unknown. Unreachable through a validated spec, and
        kept because this function is also reachable from a saved
        configuration written by an older version.
    """
    functions: dict[str, Callable[[Tensor], Tensor]] = {
        "relu": torch.relu,
        "leaky_relu": lambda x: torch.nn.functional.leaky_relu(x, _LEAKY_RELU_SLOPE),
        "tanh": torch.tanh,
        "sigmoid": torch.sigmoid,
        "elu": torch.nn.functional.elu,
        "selu": torch.nn.functional.selu,
        "gelu": torch.nn.functional.gelu,
        # Identity. Named "linear" rather than represented by ``None`` so
        # that "no activation here" is something a configuration states
        # rather than something it omits.
        "linear": lambda x: x,
    }
    if name not in functions:
        raise ContractError(f"unknown activation {name!r}; the model supports {sorted(functions)}")
    return functions[name]


def neighbour_mean(features: Tensor, adjacency: Tensor) -> Tensor:
    """
    Average each node's neighbours, weighted by the edge weights.

    A plain matrix product, because the adjacency is already row-normalised:
    each row sums to one, so multiplying by it *is* the weighted mean. Doing
    the normalisation here instead would make the layer's behaviour depend
    on how the graph was built.

    Parameters
    ----------
    features
        Node features, shape ``(n_nodes, n_features)``.
    adjacency
        Row-normalised adjacency, sparse or dense.

    Returns
    -------
    torch.Tensor
        Shape ``(n_nodes, n_features)``.
    """
    if adjacency.is_sparse:
        return torch.sparse.mm(adjacency, features)
    return torch.matmul(adjacency, features)


def neighbour_max(features: Tensor, adjacency: Tensor) -> Tensor:
    """
    Take the element-wise maximum over each node's neighbours.

    Unweighted, deliberately: the maximum asks "what is the most extreme
    thing near this instrument", and scaling a neighbour's features by its
    edge weight before the comparison would make a close neighbour look
    smaller than a distant one.

    Parameters
    ----------
    features
        Node features, shape ``(n_nodes, n_features)``.
    adjacency
        Adjacency, sparse or dense. Only its sparsity pattern is used.

    Returns
    -------
    torch.Tensor
        Shape ``(n_nodes, n_features)``. A node with no neighbours gets
        zeros rather than negative infinity -- see below.
    """
    if adjacency.is_sparse:
        indices = (adjacency if adjacency.is_coalesced() else adjacency.coalesce()).indices()
        rows, columns = indices[0], indices[1]
    else:
        rows, columns = torch.where(adjacency > 0)

    gathered = features[columns]
    # Seeded with negative infinity so the first real value always wins. A
    # seed of zero would silently floor every output at zero, turning the
    # maximum into a ReLU of the maximum.
    reduced = torch.full(
        (features.shape[0], features.shape[1]),
        float("-inf"),
        dtype=features.dtype,
        device=features.device,
    )
    reduced.scatter_reduce_(0, rows.unsqueeze(1).expand_as(gathered), gathered, reduce="amax")
    # An isolated node keeps its negative-infinity seed, which would turn
    # into NaN at the first subtraction downstream. Zero is the neutral
    # value for a node with nothing to aggregate. Self-loops mean this
    # cannot currently happen, but the layer must not depend on that.
    return torch.where(reduced.isfinite(), reduced, torch.zeros_like(reduced))


class GraphSage(nn.Module):
    """
    One round of GraphSAGE: a node's own map plus its neighbourhood's.

    Keeps the self and neighbour transforms in two separate matrices rather
    than one over the concatenation. The two are equivalent in capacity, and
    separate matrices make "how much does this model rely on the graph at
    all" answerable by comparing their norms.

    Parameters
    ----------
    in_features
        Width of the incoming node features.
    out_features
        Width to produce.
    aggregation
        ``mean`` or ``max``.
    use_bias
        Whether the linear maps carry a bias. Only the neighbour map does
        when both are present, since two biases added together are one bias.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        aggregation: str = "mean",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.aggregation = aggregation
        self.dense_self = nn.Linear(in_features, out_features, bias=use_bias)
        self.dense_neigh = nn.Linear(in_features, out_features, bias=use_bias)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Apply one round of message passing.

        Parameters
        ----------
        features
            Node features, shape ``(n_nodes, in_features)``.
        adjacency
            Row-normalised adjacency.

        Returns
        -------
        torch.Tensor
            Shape ``(n_nodes, out_features)``. Linear: the block applies
            the activation.

        Raises
        ------
        ContractError
            If the aggregation is not one this layer implements.
        """
        if self.aggregation == "mean":
            summary = neighbour_mean(features, adjacency)
        elif self.aggregation == "max":
            summary = neighbour_max(features, adjacency)
        else:
            raise ContractError(
                f"unknown GraphSAGE aggregation {self.aggregation!r}; expected 'mean' or 'max'"
            )
        return self.dense_self(features) + self.dense_neigh(summary)


class MixedGraphSage(nn.Module):
    """
    One round over the node, its neighbour mean, and its neighbour maximum.

    The flagship's default. See this module's docstring for why both
    aggregates are kept.

    Parameters
    ----------
    in_features
        Width of the incoming node features. The fusion matrix is three
        times this wide, since it consumes the concatenation.
    out_features
        Width to produce.
    use_bias
        Whether the fusion map carries a bias.
    dropout
        Dropout applied to the *input* features, before aggregation, so a
        dropped feature is dropped consistently from the node and from
        every neighbour summary that reads it. Applying it afterwards would
        drop a feature from the node's own copy while its neighbours' copies
        survived, which is not what dropping a feature means.
    """

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.fusion_dense = nn.Linear(_MIXED_AGGREGATES * in_features, out_features, bias=use_bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Apply one round of mixed-aggregation message passing.

        Parameters
        ----------
        features
            Node features, shape ``(n_nodes, in_features)``.
        adjacency
            Row-normalised adjacency.

        Returns
        -------
        torch.Tensor
            Shape ``(n_nodes, out_features)``. Linear, as above.
        """
        if self.dropout is not None:
            features = self.dropout(features)
        combined = torch.cat(
            [
                features,
                neighbour_mean(features, adjacency),
                neighbour_max(features, adjacency),
            ],
            dim=1,
        )
        return self.fusion_dense(combined)


class GnnBlock(nn.Module):
    """
    A stack of message-passing rounds with a residual connection.

    The flow, for the default two rounds::

        X ──▶ round 1 ──▶ norm ──▶ act ──▶ drop ──▶ round 2 ──▶ norm ──┐
        │                                                              ▼
        └──────────────── W_proj · X ─────────────────────────────▶ (+) ──▶ act ──▶ H

    The residual is what makes more than two rounds usable. Message passing
    is repeated averaging, and repeated averaging converges: without the
    skip connection, a deep stack drives every instrument towards the graph
    mean and they stop being distinguishable from one another. This is
    over-smoothing, and it is the reason graph networks are shallow.

    The projection on the skip path exists because the input is the raw
    attribute width and the output is the block's width. It carries no bias
    -- a bias on the skip path is added to a result that is about to be
    normalised, so it would have no effect beyond consuming parameters.

    Parameters
    ----------
    spec
        The model specification.
    in_features
        Width of the encoded attribute matrix, taken from the input
        signature so that every parameter is sized at build time.
    """

    def __init__(self, spec: HybridModelSpec, *, in_features: int) -> None:
        super().__init__()
        width = spec.width("gnn")

        self.n_layers = spec.gnn_layers
        self.use_residual = spec.use_residual
        self.activation = activation_function(spec.gnn_activation)

        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        # The first round consumes the attribute width; every later round
        # consumes the block's own width.
        for position in range(spec.gnn_layers):
            incoming = in_features if position == 0 else width
            self.gnn_layers.append(self._make_layer(spec, incoming=incoming, width=width))
            if spec.gnn_normalise:
                self.norm_layers.append(nn.LayerNorm(width, eps=LAYER_NORM_EPS))

        self.input_projection = (
            nn.Linear(in_features, width, bias=False) if spec.use_residual else None
        )
        self.dropout = nn.Dropout(spec.dropout) if spec.dropout > 0.0 else None

    @staticmethod
    def _make_layer(spec: HybridModelSpec, *, incoming: int, width: int) -> nn.Module:
        """
        Build one message-passing round.

        Dropout is held by the block rather than by the round, except for
        ``mixed_graph_sage``'s input dropout, which has to happen before
        aggregation.

        Parameters
        ----------
        spec
            The model specification.
        incoming
            Input width for this round.
        width
            Output width.

        Returns
        -------
        torch.nn.Module
            The round.

        Raises
        ------
        ContractError
            If the layer type is not implemented.
        """
        if spec.gnn_type == "mixed_graph_sage":
            return MixedGraphSage(in_features=incoming, out_features=width)
        if spec.gnn_type == "graph_sage":
            return GraphSage(
                in_features=incoming, out_features=width, aggregation=spec.gnn_aggregation
            )
        raise ContractError(
            f"unknown GNN layer type {spec.gnn_type!r}; the model implements "
            f"'mixed_graph_sage' and 'graph_sage'"
        )

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Run every round and add the residual.

        Parameters
        ----------
        features
            Encoded instrument attributes, shape
            ``(n_nodes, in_features)``. No batch dimension: the graph and
            the attributes are the same for every sample, which is what
            declaring them static in the signature means.
        adjacency
            Row-normalised adjacency.

        Returns
        -------
        torch.Tensor
            Node embeddings, shape ``(n_nodes, width)``.
        """
        residual = self.input_projection(features) if self.use_residual else None

        hidden = features
        for position, layer in enumerate(self.gnn_layers):
            hidden = layer(hidden, adjacency)
            if position < len(self.norm_layers):
                hidden = self.norm_layers[position](hidden)
            # Between rounds only. The last round's activation comes after
            # the residual add, which is the ResNet ordering.
            if position < self.n_layers - 1:
                hidden = self.activation(hidden)
                if self.dropout is not None:
                    hidden = self.dropout(hidden)

        if residual is not None:
            hidden = hidden + residual
        return self.activation(hidden)
