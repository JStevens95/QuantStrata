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

from ....core.runtime.errors import ContractError
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
