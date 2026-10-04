# `src/rade_qnet/models/hybrid_gnn_rnn/layers`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 44 | 1544 | `f2731bbc54fbfd3c` |
| 2 | `attention.py` | 319 | 12172 | `8965de3a68e489f0` |
| 3 | `fusion.py` | 296 | 12085 | `f48de1d670a94a44` |
| 4 | `gnn.py` | 452 | 16534 | `7b1e2fec9b208fe6` |
| 5 | `projection.py` | 292 | 11354 | `40485abc1ee3cd42` |
| 6 | `rnn.py` | 143 | 5550 | `f69273be0c78e06d` |

---

## 1. `src/rade_qnet/models/hybrid_gnn_rnn/layers/__init__.py`

1544 bytes · SHA-256 `f2731bbc54fbfd3c`

```python
"""
The architectural blocks of the hybrid graph-temporal network.

One block per file, each independently testable in isolation: given an input
of a known shape, assert the output shape, the gradient flow and the
invariances the block is supposed to have. Testing an architecture only
end-to-end makes every shape bug a bisection exercise.

Modules
-------
``gnn.py``
    Propagates each instrument's attributes along the similarity graph, so
    its representation reflects its neighbourhood and not only itself.
``rnn.py``
    Reads the recent window of elementary P&L into one fixed-width summary
    of the current regime.
``fusion.py``
    Crosses the two streams: graph-masked attention plus a learned gate,
    producing one vector per instrument per sample.
``attention.py``
    Lets the target instruments attend to each other over the target
    sub-graph, so correlated targets are predicted consistently.
``projection.py``
    Turns each target's representation into a P&L number: a per-target
    fitted baseline plus a shared residual correction, with unfitted
    targets borrowing a baseline from their nearest neighbours.
"""

from .attention import TargetAttentionLayer
from .fusion import FusionLayer
from .gnn import GnnBlock, GraphSage, MixedGraphSage, activation_function
from .projection import ProjectionLayer
from .rnn import RnnBlock

__all__ = [
    "FusionLayer",
    "GnnBlock",
    "GraphSage",
    "MixedGraphSage",
    "ProjectionLayer",
    "RnnBlock",
    "TargetAttentionLayer",
    "activation_function",
]
```

---

## 2. `src/rade_qnet/models/hybrid_gnn_rnn/layers/attention.py`

12172 bytes · SHA-256 `8965de3a68e489f0`

```python
"""
Lets the predicted instruments attend to each other.

Everything before this treats each target instrument separately. That is
incomplete in a way that matters for a book: two targets on the same
underlying are not independent, and a model that predicts each in isolation
will produce a pair of forecasts whose implied correlation is whatever the
features happen to induce rather than whatever the market actually shows.

This layer is a transformer block restricted to the targets. Each target
attends over the other targets it is connected to in the graph, so the
prediction for a EUR swaption can be informed by what the model is about to
say for the other EUR swaptions.

Why only the targets
--------------------
The attention runs over the target sub-graph, not the whole universe. The
elementary instruments have already done their work: the GNN propagated
their structure and the fusion layer mixed it with the market state. What
remains is coordination between the things actually being predicted, and
restricting the attention to them keeps it at ``O(n_targets²)`` rather than
``O(n_universe²)`` -- which for a cluster is two orders of magnitude.

Why masked rather than full attention
--------------------------------------
Two targets with no edge between them are not related in attribute space,
and letting them attend to each other would invite the model to invent a
dependency the graph says is not there. The mask is the same discipline as
in the fusion layer: structure constrains what may talk to what.

The block's shape
-----------------
A standard pre-residual transformer block: masked self-attention, residual,
normalise; then a feed-forward network, residual, normalise. The
feed-forward expands to four times the width before projecting back, which
is the usual ratio and is where most of a transformer's capacity lives --
attention decides *what* to combine, and the feed-forward decides what to
do with the result.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import HybridModelSpec
from .gnn import LAYER_NORM_EPS, activation_function

__all__ = ["TargetAttentionLayer"]

#: Expansion ratio of the feed-forward network, the transformer default.
_FFN_EXPANSION = 4

#: Stand-in for negative infinity in the attention mask.
#:
#: A true ``-inf`` would be mathematically cleaner, but a row that is
#: entirely masked then softmaxes to NaN rather than to zero, and one NaN
#: reaches the loss and destroys the run. A large finite number softmaxes
#: to a uniform distribution over an empty neighbourhood, which is
#: meaningless but harmless, and the ``nan_to_num`` below still catches the
#: cases this does not. The value matches the original.
_MASKED_SCORE = -1e9


class TargetAttentionLayer(nn.Module):
    """
    Masked self-attention over the target instruments, plus a feed-forward block.

    Parameters
    ----------
    spec
        The model specification.
    in_features
        Width of the fused representation arriving from the fusion layer.

    Raises
    ------
    ContractError
        If the width does not divide evenly by the head count.
    """

    def __init__(self, spec: HybridModelSpec, *, in_features: int) -> None:
        super().__init__()
        width = spec.width("attention")
        self.n_heads = spec.attention_heads
        self.width = width

        if width % spec.attention_heads:
            raise ContractError(
                f"attention width {width} is not divisible by {spec.attention_heads} "
                f"head(s); the head reshape would silently drop features"
            )
        self.head_width = width // spec.attention_heads

        self.query_proj = nn.Linear(in_features, width)
        self.key_proj = nn.Linear(in_features, width)
        self.value_proj = nn.Linear(in_features, width)
        self.output_proj = nn.Linear(width, width)

        self.ffn_linear1 = nn.Linear(width, width * _FFN_EXPANSION)
        self.ffn_linear2 = nn.Linear(width * _FFN_EXPANSION, width)
        self.ffn_activation = activation_function(spec.attention_activation)

        self.use_layer_norm = spec.attention_normalise
        self.attn_layer_norm = (
            nn.LayerNorm(width, eps=LAYER_NORM_EPS) if spec.attention_normalise else None
        )
        self.ffn_layer_norm = (
            nn.LayerNorm(width, eps=LAYER_NORM_EPS) if spec.attention_normalise else None
        )

        # No bias on the skip path: its output is added to a result that is
        # immediately normalised, so a bias there has no effect beyond
        # consuming parameters.
        self.use_residual = spec.use_residual
        self.input_proj = nn.Linear(in_features, width, bias=False) if spec.use_residual else None
        self.dropout = nn.Dropout(spec.dropout) if spec.dropout > 0.0 else None

    def forward(self, fused: Tensor, adjacency: Tensor, target_indices: Tensor) -> Tensor:
        """
        Attend among the targets and run the feed-forward block.

        Parameters
        ----------
        fused
            Fused representations, shape ``(batch, n_nodes, in_features)``.
        adjacency
            Full-universe adjacency, sparse or dense.
        target_indices
            Which rows of ``fused`` hold target instruments.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets, width)``.
        """
        targets = fused[:, target_indices, :]
        sub_adjacency = self._target_submatrix(adjacency, target_indices)

        attended = self._attend(targets, sub_adjacency)
        if self.input_proj is not None:
            attended = attended + self.input_proj(targets)
        if self.attn_layer_norm is not None:
            attended = self.attn_layer_norm(attended)

        hidden = self.ffn_activation(self.ffn_linear1(attended))
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        hidden = self.ffn_linear2(hidden)

        if self.use_residual:
            hidden = hidden + attended
        if self.ffn_layer_norm is not None:
            hidden = self.ffn_layer_norm(hidden)
        return hidden

    @staticmethod
    def _target_submatrix(adjacency: Tensor, target_indices: Tensor) -> Tensor:
        """
        Restrict the adjacency to edges with both endpoints in the target set.

        Parameters
        ----------
        adjacency
            Full-universe adjacency.
        target_indices
            Which nodes are targets.

        Returns
        -------
        torch.Tensor
            Shape ``(n_targets, n_targets)``, sparse if the input was.
        """
        if not adjacency.is_sparse:
            return adjacency[target_indices][:, target_indices]

        adjacency = adjacency.coalesce()
        rows, columns = adjacency.indices()[0], adjacency.indices()[1]

        # A lookup from global node index to local target position, with
        # -1 marking a non-target. Built as a table rather than by
        # searching, so the relabelling is one gather instead of a scan
        # per edge.
        lookup = torch.full((adjacency.shape[0],), -1, dtype=torch.long, device=adjacency.device)
        lookup[target_indices] = torch.arange(
            target_indices.shape[0], dtype=torch.long, device=adjacency.device
        )
        local_rows, local_columns = lookup[rows], lookup[columns]

        keep = (local_rows >= 0) & (local_columns >= 0)
        return torch.sparse_coo_tensor(
            torch.stack([local_rows[keep], local_columns[keep]], dim=0),
            adjacency.values()[keep],
            size=(target_indices.shape[0], target_indices.shape[0]),
        ).coalesce()

    def _attend(self, features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Run masked multi-head self-attention.

        Parameters
        ----------
        features
            Shape ``(batch, n_targets, in_features)``.
        adjacency
            Target sub-graph, sparse or dense.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets, width)``.
        """
        n_samples, n_targets, _ = features.shape

        queries, keys, values = (
            projection(features)
            .view(n_samples, n_targets, self.n_heads, self.head_width)
            .permute(0, 2, 1, 3)
            for projection in (self.query_proj, self.key_proj, self.value_proj)
        )

        attended = (
            self._sparse_attention(queries, keys, values, adjacency)
            if adjacency.is_sparse
            else self._dense_attention(queries, keys, values, adjacency)
        )
        attended = attended.permute(0, 2, 1, 3).reshape(n_samples, n_targets, self.width)
        # Mixes across heads. Without it each head's slice of the output
        # stays in its own channels and the heads never interact, which
        # makes multi-head attention no better than one narrow head.
        return self.output_proj(attended)

    def _dense_attention(
        self, queries: Tensor, keys: Tensor, values: Tensor, adjacency: Tensor
    ) -> Tensor:
        """
        Score every target pair and mask the non-edges.

        Parameters
        ----------
        queries, keys, values
            Shape ``(batch, heads, n_targets, head_width)``.
        adjacency
            Dense target sub-graph.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, heads, n_targets, head_width)``.
        """
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.head_width**0.5
        scores = torch.where(
            adjacency.unsqueeze(0).unsqueeze(0) > 0,
            scores,
            torch.tensor(_MASKED_SCORE, dtype=scores.dtype, device=scores.device),
        )
        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return torch.matmul(weights, values)

    def _sparse_attention(
        self, queries: Tensor, keys: Tensor, values: Tensor, adjacency: Tensor
    ) -> Tensor:
        """
        Score only each target's own neighbours.

        Parameters
        ----------
        queries, keys, values
            Shape ``(batch, heads, n_targets, head_width)``.
        adjacency
            Sparse target sub-graph.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, heads, n_targets, head_width)``.
        """
        n_targets = queries.shape[2]
        device = queries.device

        adjacency = adjacency.coalesce()
        rows, columns = adjacency.indices()[0], adjacency.indices()[1]
        degrees = torch.bincount(rows, minlength=n_targets)

        width = int(degrees.max()) if degrees.numel() else 0
        if width == 0:
            # Every target isolated. Returning zeros rather than raising,
            # because a single-target cluster is legitimate and its one
            # target has nothing to attend to.
            return torch.zeros_like(queries)

        row_starts = torch.zeros(n_targets, dtype=torch.long, device=device)
        row_starts[1:] = degrees[:-1].cumsum(0)
        position = torch.arange(rows.shape[0], device=device) - row_starts[rows]

        neighbours = torch.zeros(n_targets, width, dtype=torch.long, device=device)
        neighbours[rows, position] = columns
        valid = torch.arange(width, device=device).unsqueeze(0) < degrees.unsqueeze(1)

        gathered_keys = keys[:, :, neighbours, :]
        gathered_values = values[:, :, neighbours, :]

        scores = (
            torch.matmul(queries.unsqueeze(-2), gathered_keys.transpose(-2, -1)).squeeze(-2)
            / self.head_width**0.5
        )
        scores = torch.where(
            valid.unsqueeze(0).unsqueeze(0),
            scores,
            torch.tensor(_MASKED_SCORE, dtype=scores.dtype, device=scores.device),
        )

        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        if self.dropout is not None:
            weights = self.dropout(weights)
        return torch.matmul(weights.unsqueeze(-2), gathered_values).squeeze(-2)
```

---

## 3. `src/rade_qnet/models/hybrid_gnn_rnn/layers/fusion.py`

12085 bytes · SHA-256 `f48de1d670a94a44`

```python
"""
Combines what an instrument *is* with what the book has been *doing*.

The GNN block produces one vector per instrument and the same vector for
every sample -- it describes structure, which does not change between
scenarios. The RNN block produces one vector per sample and the same vector
for every instrument -- it describes the recent period, which does not
change between instruments.

Neither is a prediction. This layer is where they meet, and it produces one
vector per instrument *per sample*: what this instrument looks like, given
what the book has just done.

Why cross-attention rather than concatenation
----------------------------------------------
Concatenating the two streams would work and would be cheaper. It would
also give every instrument the same view of the market state, which is
wrong in a specific and expensive way: a recent move in EUR rates is
informative for a EUR swaption and close to noise for a USD FX forward,
and a concatenation has no mechanism to say so.

Attention does. Each instrument forms a query from both streams, and
attends over its *neighbours'* structural representations. The attention
is masked by the graph, so an instrument can only draw on instruments it
is actually related to -- which is the whole reason for building the graph.

Why the query comes from both streams
-------------------------------------
The query is ``W_q_rnn(rnn) + W_q_gnn(gnn)``, while keys and values come
from the graph stream alone.

An instrument asks "which of my neighbours should I be listening to", and
the honest answer depends on both who it is and what the market just did.
A query built from structure alone would produce the same attention
pattern in every regime, so the layer would reduce to a fixed re-weighting
of the graph and the temporal stream would reach the output only through
the gate.

The gate
--------
The attended result and the recurrent stream are blended per feature by a
learned sigmoid gate rather than summed. A sum fixes the mixture at one to
one for every instrument in every regime. The gate lets the model learn
that a liquid instrument's own history deserves more weight than its
neighbours', and a thinly traded one's deserves less -- which is exactly
the distinction the graph exists to exploit.

Sparse and dense paths
----------------------
Two implementations of the same mathematics. The dense one scores every
pair and masks; the sparse one gathers each node's neighbours and scores
only those. Dense is ``O(n²)`` and fine for a cluster; sparse is ``O(n·k)``
and is what makes a large universe tractable. They agree to floating-point
tolerance, and a test asserts it rather than assuming it.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import HybridModelSpec
from .gnn import LAYER_NORM_EPS

__all__ = ["FusionLayer"]

#: How many vectors the gate consumes: the attended result and the
#: recurrent stream.
_GATE_INPUTS = 2


class FusionLayer(nn.Module):
    """
    Graph-masked cross-attention between the structural and temporal streams.

    Parameters
    ----------
    spec
        The model specification.
    gnn_features, rnn_features
        Widths of the two incoming streams. Taken from the blocks that
        produce them rather than assumed, because a bidirectional
        recurrence doubles its own width.

    Raises
    ------
    ContractError
        If the fusion width does not divide evenly by the head count,
        which would silently drop features on the head reshape.
    """

    def __init__(self, spec: HybridModelSpec, *, gnn_features: int, rnn_features: int) -> None:
        super().__init__()
        width = spec.width("fusion")
        self.n_heads = spec.fusion_heads
        self.width = width
        self.mode = spec.fusion_mode
        self.neighbour_cap = spec.neighbour_cap

        if width % spec.fusion_heads:
            raise ContractError(
                f"fusion width {width} is not divisible by {spec.fusion_heads} head(s). "
                f"The head reshape would drop {width % spec.fusion_heads} feature(s) "
                f"per head without raising"
            )
        self.head_width = width // spec.fusion_heads

        # Bring both streams to a common width before anything is mixed.
        self.gnn_projection = nn.Linear(gnn_features, width)
        self.rnn_projection = nn.Linear(rnn_features, width)

        # The query reads both streams; keys and values read structure only.
        self.W_q_rnn = nn.Linear(width, width)
        self.W_q_gnn = nn.Linear(width, width)
        self.W_k = nn.Linear(width, width)
        self.W_v = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)

        if spec.fusion_mode == "gate":
            self.gate_projection = nn.Linear(_GATE_INPUTS * width, width)

        self.layer_norm = nn.LayerNorm(width, eps=LAYER_NORM_EPS)
        self.dropout = nn.Dropout(spec.dropout) if spec.dropout > 0.0 else None

    def forward(self, gnn_features: Tensor, rnn_features: Tensor, adjacency: Tensor) -> Tensor:
        """
        Fuse the two streams into one representation per instrument per sample.

        Parameters
        ----------
        gnn_features
            Node embeddings, shape ``(n_nodes, gnn_features)``. No batch
            dimension: structure is shared across the batch.
        rnn_features
            Window summaries, shape ``(batch, rnn_features)``. No node
            dimension: the period is shared across instruments.
        adjacency
            Row-normalised adjacency, sparse or dense.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_nodes, width)``.

        Raises
        ------
        ContractError
            If the fusion mode is not one this layer implements.
        """
        n_samples = rnn_features.shape[0]
        n_nodes = gnn_features.shape[0]

        # `expand` rather than `repeat`: both streams are broadcast to the
        # full (batch, node) grid, and expand creates a view rather than
        # materialising batch copies of the node table.
        graph_stream = self.gnn_projection(gnn_features.unsqueeze(0).expand(n_samples, -1, -1))
        time_stream = self.rnn_projection(rnn_features.unsqueeze(1).expand(-1, n_nodes, -1))

        queries = self.W_q_rnn(time_stream) + self.W_q_gnn(graph_stream)
        keys = self.W_k(graph_stream)
        values = self.W_v(graph_stream)

        queries, keys, values = (
            tensor.view(n_samples, n_nodes, self.n_heads, self.head_width).permute(0, 2, 1, 3)
            for tensor in (queries, keys, values)
        )

        attended = (
            self._sparse_attention(queries, keys, values, adjacency)
            if adjacency.is_sparse
            else self._dense_attention(queries, keys, values, adjacency)
        )
        # Reassemble the heads into one vector per node.
        attended = attended.permute(0, 2, 1, 3).reshape(n_samples, n_nodes, self.width)

        fused = self.output_projection(attended)
        if self.dropout is not None:
            fused = self.dropout(fused)

        if self.mode == "gate":
            gate = torch.sigmoid(self.gate_projection(torch.cat([fused, time_stream], dim=-1)))
            combined = gate * fused + (1.0 - gate) * time_stream
        elif self.mode == "add":
            combined = fused + time_stream
        else:
            raise ContractError(f"unknown fusion mode {self.mode!r}; expected 'gate' or 'add'")
        return self.layer_norm(combined)

    def _dense_attention(
        self, queries: Tensor, keys: Tensor, values: Tensor, adjacency: Tensor
    ) -> Tensor:
        """
        Score every pair and mask out the non-edges.

        Parameters
        ----------
        queries, keys, values
            Shape ``(batch, heads, n_nodes, head_width)``.
        adjacency
            Adjacency, dense or sparse.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, heads, n_nodes, head_width)``.
        """
        dense = adjacency.to_dense() if adjacency.is_sparse else adjacency
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_width)
        # Masked before the softmax, not after. Zeroing afterwards would
        # leave the non-edges' probability mass removed but the remaining
        # weights no longer summing to one, so the output would be scaled
        # by each node's degree.
        scores = scores.masked_fill((dense == 0).unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        # A node with no edges has an all-`-inf` row, whose softmax is NaN.
        # Self-loops make that unreachable today; the guard is here because
        # one NaN propagates through every later layer and the symptom --
        # a loss of NaN after several epochs -- names nothing.
        weights = torch.nan_to_num(weights, nan=0.0)
        return torch.matmul(weights, values)

    def _sparse_attention(
        self, queries: Tensor, keys: Tensor, values: Tensor, adjacency: Tensor
    ) -> Tensor:
        """
        Score only each node's own neighbours.

        The same mathematics as the dense path at ``O(n·k)`` instead of
        ``O(n²)``, by gathering each node's neighbours into a padded
        ``(n_nodes, k)`` table and attending within rows.

        Parameters
        ----------
        queries, keys, values
            Shape ``(batch, heads, n_nodes, head_width)``.
        adjacency
            Sparse adjacency.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, heads, n_nodes, head_width)``.
        """
        n_samples, _, n_nodes, _ = queries.shape
        device = queries.device

        adjacency = adjacency.coalesce()
        rows, columns = adjacency.indices()[0], adjacency.indices()[1]
        degrees = torch.bincount(rows, minlength=n_nodes)

        # The table is as wide as the busiest node, capped. Sizing it to
        # the cap regardless would waste memory on a sparse graph; sizing
        # it to the busiest node uncapped would lose the bound the cap
        # exists to provide.
        width = min(self.neighbour_cap, int(degrees.max())) if degrees.numel() else 0
        if width == 0:
            return torch.zeros_like(queries)

        # Each edge's position within its own node's run. The adjacency is
        # coalesced, so edges are already grouped by row and this is just
        # the offset from where the row began.
        row_starts = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), degrees.cumsum(0)[:-1]]
        )
        position = torch.arange(rows.numel(), device=device) - row_starts[rows]
        keep = position < width

        neighbours = torch.zeros(n_nodes, width, dtype=torch.long, device=device)
        neighbours[rows[keep], position[keep]] = columns[keep]
        # Marks which cells of the padded table hold a real neighbour.
        # Without it, the padding -- all of which points at node zero --
        # would be attended over, so every node would partly attend to node
        # zero in proportion to how few neighbours it has.
        valid = torch.arange(width, device=device).unsqueeze(0) < degrees.clamp(
            max=width
        ).unsqueeze(1)

        flat = neighbours.reshape(-1)
        gathered_keys = keys[:, :, flat, :].view(
            n_samples, self.n_heads, n_nodes, width, self.head_width
        )
        gathered_values = values[:, :, flat, :].view(
            n_samples, self.n_heads, n_nodes, width, self.head_width
        )

        scores = torch.matmul(queries.unsqueeze(-2), gathered_keys.transpose(-2, -1)).squeeze(
            -2
        ) / math.sqrt(self.head_width)
        scores = scores.masked_fill(~valid.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
        return torch.matmul(weights.unsqueeze(-2), gathered_values).squeeze(-2)
```

---

## 4. `src/rade_qnet/models/hybrid_gnn_rnn/layers/gnn.py`

16534 bytes · SHA-256 `7b1e2fec9b208fe6`

```python
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
```

---

## 5. `src/rade_qnet/models/hybrid_gnn_rnn/layers/projection.py`

11354 bytes · SHA-256 `40485abc1ee3cd42`

```python
"""
Turns one representation per target into one PnL number per target.

Everything upstream is shared machinery: the same GNN, the same recurrence,
the same attention, applied to every instrument. This layer is where the
model stops being general and becomes specific to the instruments in front
of it, and it is the only place in the network that holds per-instrument
parameters.

Baseline plus residual
----------------------
The prediction is a sum of two terms, and the split is the point of the
layer.

The *baseline* is a dedicated weight vector and bias per target: a linear
read of that one instrument's representation. It is the model saying "this
particular swaption's PnL is roughly this function of its state", and it is
what you would get from fitting each target separately.

The *residual* is a small shared network over the representation
concatenated with the instrument's static attributes. It corrects the
baseline using patterns learnt across the whole book -- the part a
per-instrument fit cannot see.

Together: per-instrument accuracy where there is history to fit, shared
structure where there is not. Neither alone is adequate. A purely shared
head cannot express that two instruments with near-identical attributes
behave differently; purely per-instrument baselines cannot say anything
at all about an instrument that was not in the training set.

Instruments that were not in the training set
----------------------------------------------
Which is the harder half of the problem. A trade booked after the model
was fitted has no baseline kernel, and the whole point of a replication
model is to price it anyway.

Targets arrive with the fitted ones first and the new ones after. A new
target borrows a baseline from its ``k`` nearest fitted targets in
attribute space -- a 10y EUR payer inherits from the other 10y EUR payers
-- and its residual is damped, because the residual network never saw it
and its correction is extrapolation rather than fit.

This is interpolation in attribute space, and it is only as good as the
attributes. If a new trade is genuinely unlike anything fitted, the blend
will return a confident-looking number built from poor neighbours. The
graph diagnostics are what tell you whether that is happening; this layer
cannot know.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import HybridModelSpec
from .gnn import activation_function

__all__ = ["ProjectionLayer"]

#: Guard against dividing by a zero norm or a zero distance.
_EPSILON = 1e-8

#: Keeps the inverse-distance weights from summing to zero.
_WEIGHT_SUM_FLOOR = 1e-12


class ProjectionLayer(nn.Module):
    """
    Per-target output head: a fitted baseline plus a shared residual correction.

    Parameters
    ----------
    spec
        The model specification.
    in_features
        Width of the attended representation.
    attribute_features
        Width of the static attribute vector.
    n_fitted_targets
        How many targets get a dedicated baseline. Sized here rather than
        on the first forward pass so that the optimiser sees every
        parameter when it is constructed.
    """

    def __init__(
        self,
        spec: HybridModelSpec,
        *,
        in_features: int,
        attribute_features: int,
        n_fitted_targets: int,
    ) -> None:
        super().__init__()
        self.n_fitted_targets = n_fitted_targets
        self.blend = spec.new_target_blend
        self.n_neighbours = spec.new_target_neighbours
        self.temperature = spec.new_target_temperature
        self.distance_power = spec.new_target_distance_power
        self.residual_damping = spec.new_target_residual_damping
        self.use_weight_norm = spec.baseline_weight_norm

        # One row per fitted target. Xavier rather than the default
        # initialisation because each row is read in isolation by a single
        # dot product, so its scale sets that target's output scale
        # directly.
        self._baseline_kernels = nn.Parameter(torch.empty(n_fitted_targets, in_features))
        nn.init.xavier_uniform_(self._baseline_kernels)
        self._baseline_biases = nn.Parameter(torch.zeros(n_fitted_targets))

        # Initialised so softplus(gain) == 1, leaving the layer equivalent
        # to the plain dot product at step zero. Starting a reparametrised
        # layer anywhere else changes the function the optimiser begins
        # from, which is not what a reparametrisation is for.
        self._baseline_gain = (
            nn.Parameter(torch.full((n_fitted_targets,), math.log(math.expm1(1.0))))
            if spec.baseline_weight_norm
            else None
        )

        width = spec.width("projection")
        self._residual_fc_1 = nn.Linear(in_features + attribute_features, width)
        self._residual_fc_2 = nn.Linear(width, 1)
        self._activation = activation_function(spec.projection_activation)
        self._residual_dropout = nn.Dropout(spec.dropout) if spec.dropout > 0.0 else None

    def forward(self, attended: Tensor, attributes: Tensor) -> Tensor:
        """
        Predict PnL for every target.

        Parameters
        ----------
        attended
            Attended representations, shape ``(batch, n_targets, in_features)``.
        attributes
            Static attributes of the targets, shape
            ``(n_targets, attribute_features)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets)``, in scaled space.

        Raises
        ------
        ContractError
            If the two inputs disagree on how many targets there are.
        """
        n_samples, n_targets, _ = attended.shape
        if attributes.shape[0] != n_targets:
            raise ContractError(
                f"{n_targets} attended target(s) but {attributes.shape[0]} attribute "
                f"row(s); the attribute table must be restricted to the targets"
            )

        # Attributes are shared across the batch, so `expand` gives a view
        # rather than materialising a copy per sample.
        residual = self._residual_fc_1(
            torch.cat([attended, attributes.unsqueeze(0).expand(n_samples, -1, -1)], dim=-1)
        )
        residual = self._activation(residual)
        if self._residual_dropout is not None:
            residual = self._residual_dropout(residual)
        residual = self._residual_fc_2(residual).squeeze(-1)

        # Fewer targets than kernels is legitimate: a cluster can be
        # evaluated on a subset of what it was fitted on.
        n_fitted = min(self.n_fitted_targets, n_targets)
        fitted_baseline = self._fitted_baseline(attended[:, :n_fitted, :], n_fitted)

        n_new = n_targets - n_fitted
        if n_new > 0:
            new_baseline = self._borrowed_baseline(
                fitted_baseline,
                fitted_attributes=attributes[:n_fitted, :],
                new_attributes=attributes[n_fitted:, :],
            )
            residual = torch.cat(
                [residual[:, :n_fitted], residual[:, n_fitted:] * self.residual_damping],
                dim=1,
            )
        else:
            new_baseline = attended.new_zeros((n_samples, 0))

        return torch.cat([fitted_baseline, new_baseline], dim=1) + residual

    def _fitted_baseline(self, attended: Tensor, n_fitted: int) -> Tensor:
        """
        Read each fitted target's own kernel.

        Parameters
        ----------
        attended
            Shape ``(batch, n_fitted, in_features)``.
        n_fitted
            How many kernels to use.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_fitted)``.
        """
        kernels = self._baseline_kernels[:n_fitted, :]
        if self._baseline_gain is not None:
            direction = kernels / (torch.norm(kernels, dim=1, keepdim=True) + _EPSILON)
            kernels = direction * nn.functional.softplus(self._baseline_gain[:n_fitted]).unsqueeze(
                1
            )
        # einsum rather than a matmul because each target reads only its
        # own kernel: this is a row-wise dot product, not a matrix product.
        return (
            torch.einsum("bna,na->bn", attended, kernels) + self._baseline_biases[None, :n_fitted]
        )

    def _borrowed_baseline(
        self,
        fitted_baseline: Tensor,
        *,
        fitted_attributes: Tensor,
        new_attributes: Tensor,
    ) -> Tensor:
        """
        Blend fitted targets' baselines for targets that have none.

        Parameters
        ----------
        fitted_baseline
            Shape ``(batch, n_fitted)``.
        fitted_attributes, new_attributes
            Attribute rows, shape ``(n_fitted, p)`` and ``(n_new, p)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_new)``.
        """
        neighbours, weights = self._neighbour_weights(new_attributes, fitted_attributes)
        # The blend is over baselines, not over kernels. Mixing kernels
        # and then reading the new target's own representation would
        # apply a neighbour's weights to a different instrument's state;
        # mixing outputs asks "what would my neighbours have predicted",
        # which is the question actually being answered.
        return torch.sum(fitted_baseline[:, neighbours] * weights[None, :, :], dim=-1)

    def _neighbour_weights(
        self, new_attributes: Tensor, fitted_attributes: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Find each new target's nearest fitted targets and how much to weight them.

        Parameters
        ----------
        new_attributes, fitted_attributes
            Attribute rows.

        Returns
        -------
        tuple of torch.Tensor
            Neighbour indices and weights, both ``(n_new, k)``; the weights
            sum to one along the last axis.

        Raises
        ------
        ContractError
            If the blend mode is not one this layer implements.
        """
        k = min(self.n_neighbours, fitted_attributes.shape[0])

        if self.blend == "cosine_softmax":
            # Cosine rather than Euclidean: attribute vectors carry one-hot
            # blocks whose magnitude reflects how many categories a field
            # has, so direction is comparable across instruments where
            # distance is not.
            similarity = (
                nn.functional.normalize(new_attributes, p=2, dim=1)
                @ nn.functional.normalize(fitted_attributes, p=2, dim=1).T
            )
            scores, neighbours = torch.topk(similarity, k=k, dim=1, sorted=False)
            return neighbours, torch.softmax(scores * self.temperature, dim=1)

        if self.blend == "inverse_distance":
            distances = torch.cdist(new_attributes, fitted_attributes)
            nearest, neighbours = torch.topk(-distances, k=k, dim=1, sorted=False)
            raw = 1.0 / torch.clamp(-nearest, min=_EPSILON).pow(self.distance_power)
            return neighbours, raw / (raw.sum(dim=1, keepdim=True) + _WEIGHT_SUM_FLOOR)

        raise ContractError(
            f"unknown new-target blend {self.blend!r}; expected 'cosine_softmax' "
            f"or 'inverse_distance'"
        )
```

---

## 6. `src/rade_qnet/models/hybrid_gnn_rnn/layers/rnn.py`

5550 bytes · SHA-256 `f69273be0c78e06d`

```python
"""
Compresses a window of P&L history into one vector per sample.

Where the GNN answers "what is this instrument like", this block answers
"what has the book been doing lately". It consumes the scaled P&L of every
selected elementary instrument over the window and returns a single
fixed-width summary of the whole period.

Why a recurrence rather than a flattened window
------------------------------------------------
The obvious alternative is to flatten the window into one long vector and
feed it to a dense layer. That works, and it has two properties that make
it the wrong default here.

It makes the parameter count scale with the window length, so lengthening
the history from four days to twenty multiplies the first layer by five and
the model starts overfitting the length rather than learning from it. And
it gives the model no notion of ordering: a flattened window is a bag of
numbers, so the network has to learn from scratch that column 3 is one day
after column 2. A recurrence has that structure built in and its parameter
count does not depend on the window length at all.

Why the last hidden state
-------------------------
The recurrence emits one output per timestep; this block keeps only the
final hidden state. That state is the network's running summary of
everything it has read, and it is the only one conditioned on the whole
window. Taking the mean over timesteps instead would weight the oldest day
as heavily as the most recent, which for predicting tomorrow is backwards.

For a bidirectional recurrence the final states of both directions are
concatenated, which is why that setting doubles the output width -- the
fusion block accounts for it rather than assuming a width.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from ....core.lifecycle.errors import ContractError
from ..spec import HybridModelSpec

__all__ = ["RnnBlock"]

#: Recurrences whose final state is a ``(hidden, cell)`` pair rather than a
#: bare tensor. GRU keeps no cell state, so it returns the tensor directly.
_CARRIES_CELL_STATE = frozenset({"lstm", "bilstm"})


class RnnBlock(nn.Module):
    """
    A stacked recurrence over the P&L window.

    Parameters
    ----------
    spec
        The model specification.
    in_features
        How many instruments the window carries, taken from the input
        signature so that the recurrence is sized at build time rather
        than materialised by a first forward pass -- defect 6.

    Notes
    -----
    Stacking is delegated to PyTorch's ``num_layers`` rather than looped
    here. That is not only shorter: the fused multi-layer kernel is
    substantially faster than a Python loop over single-layer modules, and
    on CUDA it is the difference between using cuDNN's recurrent kernels and
    not.
    """

    def __init__(self, spec: HybridModelSpec, *, in_features: int) -> None:
        super().__init__()
        self.rnn_type = spec.rnn_type
        width = spec.width("rnn")

        # PyTorch applies this dropout *between* stacked layers, so it is
        # meaningless with one layer and warns if passed anyway.
        between_layers = spec.dropout if spec.rnn_layers > 1 else 0.0

        shared = {
            "input_size": in_features,
            "hidden_size": width,
            "num_layers": spec.rnn_layers,
            # The window arrives as (batch, time, instrument), which is the
            # layout the data module produces and the one a reader expects.
            "batch_first": True,
            "dropout": between_layers,
        }
        if spec.rnn_type == "lstm":
            self.rnn: nn.Module = nn.LSTM(**shared)
        elif spec.rnn_type == "bilstm":
            self.rnn = nn.LSTM(**shared, bidirectional=True)
        elif spec.rnn_type == "gru":
            self.rnn = nn.GRU(**shared)
        else:
            raise ContractError(
                f"unknown recurrence type {spec.rnn_type!r}; the model implements "
                f"'lstm', 'bilstm' and 'gru'"
            )

    @property
    def out_features(self) -> int:
        """
        Width of the summary this block produces.

        Doubled for a bidirectional recurrence, since the two directions'
        final states are concatenated. Exposed as a property so the fusion
        block can size itself from it rather than re-deriving the doubling
        rule and getting it wrong once.
        """
        width = self.rnn.hidden_size
        return width * 2 if self.rnn_type == "bilstm" else width

    def forward(self, history: Tensor) -> Tensor:
        """
        Summarise the window.

        Parameters
        ----------
        history
            Scaled P&L, shape ``(batch, window, n_instruments)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, out_features)``.
        """
        _, final = self.rnn(history)

        # An LSTM returns (hidden, cell); the cell state is the memory the
        # gates write to, not the output, so only the hidden state leaves.
        hidden = final[0] if self.rnn_type in _CARRIES_CELL_STATE else final

        if self.rnn_type == "bilstm":
            # `hidden` is (num_layers * 2, batch, width), with the two
            # directions of the final layer in the last two positions:
            # -2 is the forward pass, -1 the backward one.
            return torch.cat([hidden[-2], hidden[-1]], dim=-1)
        # The top layer's state, which is the one conditioned on everything
        # the stack has read.
        return hidden[-1]
```

