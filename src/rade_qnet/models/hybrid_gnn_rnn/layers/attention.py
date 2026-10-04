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
