"""
The hybrid graph-temporal network.

This file is the mathematics and nothing else. How the model plugs into the
framework -- its registration, its specs, its pipelines -- lives in
``register.py``, so a reader asking *what does this compute?* finds only the
answer to that question here.

What it computes
----------------
Given a book of elementary instruments with recent P&L, and a set of target
instruments to replicate, predict each target's P&L.

Two questions have to be answered at once, and the architecture is one block
per question::

    attributes ──► GNN ──► what each instrument IS
                            (one vector per instrument,
                             identical across samples)
                                        │
                                        ├──► fusion ──► attention ──► head ──► P&L
                                        │
    P&L history ──► RNN ──► what the book just DID
                            (one vector per sample,
                             identical across instruments)

The GNN propagates each instrument's attributes along the similarity graph,
so an instrument's representation reflects its neighbourhood rather than
only itself. The RNN reads the recent window of elementary P&L into a
summary of the current regime. Fusion crosses the two into one vector per
instrument per sample. Target attention lets the predicted instruments
coordinate. The head turns each into a number.

Why the two streams are separate
---------------------------------
It would be simpler to concatenate attributes onto the P&L history and run
one network. That would be a worse model and a far more expensive one.

Structure does not vary across samples, so computing it per sample repeats
identical work -- for a 500-instrument universe and a 2,000-sample epoch
that is three orders of magnitude of waste, and it is why
:meth:`HybridGnnRnn.precompute` exists. Market state does not vary across
instruments, so computing it per instrument repeats identical work too.
Keeping the streams separate until they must meet is what makes the model
tractable on a real book, and the asymmetry is in the tensor shapes: the
graph stream has no batch axis and the temporal stream has no node axis
until the fusion layer gives them both.

Why nothing here is cached
---------------------------
The network holds parameters and no other state. The reusable encoding of
the static inputs is produced on demand by :meth:`precompute` and handed
back to the caller, rather than being stashed on the module.

This is deliberate and it is a fix. Caching the graph embedding inside the
module means two jobs sharing a process can see each other's results
through it, and a cache that is invalidated on mode changes but not on
parameter changes goes stale in a way that produces plausible numbers
rather than an error. A pure module cannot do either.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from ...core.lifecycle.errors import ContractError
from .layers.attention import TargetAttentionLayer
from .layers.fusion import FusionLayer
from .layers.gnn import LAYER_NORM_EPS, GnnBlock
from .layers.projection import ProjectionLayer
from .layers.rnn import RnnBlock
from .spec import HybridModelSpec

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.signature import InputSignature

__all__ = ["HybridGnnRnn"]

#: The static inputs that describe the graph, in the order the sparse
#: tensor constructor wants them.
_ADJACENCY_KEYS = ("adjacency_indices", "adjacency_values", "adjacency_shape")

#: What `precompute` returns, and what `forward_with_precomputed` expects.
_PRECOMPUTED_KEY = "graph_embedding"


class HybridGnnRnn(nn.Module):
    """
    Graph-temporal replication network.

    Every layer is sized from the signature at construction, so the module
    is fully parameterised before it is returned. Lazy layers would defer
    that to the first forward pass, by which point an optimiser built over
    the module is tracking an empty parameter list and the model trains
    nothing -- silently, because the loss still decreases via the layers
    that did exist.

    Parameters
    ----------
    spec
        The architecture specification.
    signature
        The declared input interface, which supplies every width.

    Raises
    ------
    ContractError
        If the signature is missing a tensor the network needs, or declares
        one with a shape it cannot use.
    """

    def __init__(self, spec: HybridModelSpec, signature: InputSignature) -> None:
        super().__init__()
        self.spec = spec

        n_attributes = _width(signature, "trade_features", axis=-1, static=True)
        n_elementary = _width(signature, "pnl_history", axis=-1, static=False)
        n_targets = _width(signature, "target_indices", axis=0, static=True)

        self.gnn_block = GnnBlock(spec, in_features=n_attributes)
        self.rnn_block = RnnBlock(spec, in_features=n_elementary)

        # Each block is normalised before the two streams meet. Without
        # this, whichever stream happens to come out larger dominates the
        # fusion for the first several epochs purely through scale, and
        # the model spends that time learning to undo the imbalance.
        self.gnn_block_ln = nn.LayerNorm(spec.width("gnn"), eps=LAYER_NORM_EPS)
        self.rnn_block_ln = nn.LayerNorm(self.rnn_block.out_features, eps=LAYER_NORM_EPS)

        self.fusion_layer = FusionLayer(
            spec,
            gnn_features=spec.width("gnn"),
            rnn_features=self.rnn_block.out_features,
        )
        self.attention_layer = TargetAttentionLayer(spec, in_features=spec.width("fusion"))
        self.projection_layer = ProjectionLayer(
            spec,
            in_features=spec.width("attention"),
            attribute_features=n_attributes,
            n_fitted_targets=n_targets,
        )

    def forward(
        self,
        *,
        pnl_history: Tensor,
        trade_features: Tensor,
        adjacency_indices: Tensor,
        adjacency_values: Tensor,
        adjacency_shape: Tensor,
        target_indices: Tensor,
    ) -> Tensor:
        """
        Predict each target's P&L.

        Parameters
        ----------
        pnl_history
            Recent elementary P&L, shape
            ``(batch, sequence_length, n_elementary)``.
        trade_features
            Encoded attributes for every instrument, shape
            ``(n_nodes, n_attributes)``.
        adjacency_indices, adjacency_values, adjacency_shape
            The similarity graph in sparse COO form.
        target_indices
            Which rows of ``trade_features`` are targets.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets)``, in scaled P&L units. Returning
            them to money is the fitted state's job, not the network's.
        """
        adjacency = _sparse_adjacency(adjacency_indices, adjacency_values, adjacency_shape)
        graph_embedding = self.gnn_block_ln(self.gnn_block(trade_features, adjacency))
        return self._head(
            graph_embedding,
            pnl_history=pnl_history,
            trade_features=trade_features,
            adjacency=adjacency,
            target_indices=target_indices,
        )

    def precompute(self, static: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """
        Encode the graph once for reuse across an evaluation pass.

        The graph stream depends only on the static inputs, so its result
        is identical for every batch. The framework calls this once at the
        start of an evaluation or inference pass, and never during
        training -- where the parameters change each step and a reused
        embedding would be one step stale.

        Parameters
        ----------
        static
            The static inputs, already on the correct device.

        Returns
        -------
        dict
            The graph embedding, under a key this class also understands
            in :meth:`forward_with_precomputed`.
        """
        adjacency = _sparse_adjacency(*(static[key] for key in _ADJACENCY_KEYS))
        return {
            _PRECOMPUTED_KEY: self.gnn_block_ln(self.gnn_block(static["trade_features"], adjacency))
        }

    def forward_with_precomputed(
        self, batch: Mapping[str, Tensor], precomputed: Mapping[str, Tensor]
    ) -> Tensor:
        """
        Run a forward pass reusing an already-computed graph embedding.

        Parameters
        ----------
        batch
            One batch, including the static inputs.
        precomputed
            The mapping returned by :meth:`precompute`.

        Returns
        -------
        torch.Tensor
            The same values :meth:`forward` would return, up to
            floating-point reassociation.
        """
        return self._head(
            precomputed[_PRECOMPUTED_KEY],
            pnl_history=batch["pnl_history"],
            trade_features=batch["trade_features"],
            adjacency=_sparse_adjacency(*(batch[key] for key in _ADJACENCY_KEYS)),
            target_indices=batch["target_indices"],
        )

    @property
    def supports_unseen_entities(self) -> bool:
        """
        Whether targets absent from training can be predicted.

        True: the output head gives an unfitted target a baseline borrowed
        from its nearest fitted neighbours in attribute space. The quality
        of that borrow is only as good as the attributes, which is a
        question for the graph diagnostics rather than for the network.
        """
        return True

    def _head(
        self,
        graph_embedding: Tensor,
        *,
        pnl_history: Tensor,
        trade_features: Tensor,
        adjacency: Tensor,
        target_indices: Tensor,
    ) -> Tensor:
        """
        Run everything downstream of the graph stream.

        Shared by the ordinary and the precomputed forward paths, so the
        two cannot drift apart: the only difference between them is where
        the graph embedding came from.

        Parameters
        ----------
        graph_embedding
            Normalised node embeddings, shape ``(n_nodes, gnn_width)``.
        pnl_history, trade_features, adjacency, target_indices
            As for :meth:`forward`, with the graph already assembled.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, n_targets)``.
        """
        temporal = self.rnn_block_ln(self.rnn_block(pnl_history))
        fused = self.fusion_layer(graph_embedding, temporal, adjacency)
        attended = self.attention_layer(fused, adjacency, target_indices)
        return self.projection_layer(attended, trade_features[target_indices])


def _sparse_adjacency(indices: Tensor, values: Tensor, shape: Tensor) -> Tensor:
    """
    Assemble the similarity graph as a sparse tensor.

    Rebuilt per call rather than held on the module. It is three cheap
    tensor operations, and the alternative -- a cached sparse tensor
    belonging to the model -- is state that outlives the batch it was
    built for and silently follows the model into the next job.

    Parameters
    ----------
    indices
        Edge endpoints, shape ``(n_edges, 2)``, row-major.
    values
        Edge weights, shape ``(n_edges,)``.
    shape
        Dense shape, as a two-element tensor.

    Returns
    -------
    torch.Tensor
        A coalesced sparse COO tensor.
    """
    # Transposed because the signature stores one edge per row, which is
    # how the fixture and every diagnostic reads it, while Torch wants one
    # axis per row.
    return torch.sparse_coo_tensor(
        indices.t().contiguous(), values, tuple(int(size) for size in shape)
    ).coalesce()


def _width(signature: InputSignature, name: str, *, axis: int, static: bool) -> int:
    """
    Read one fixed axis length from the signature.

    Parameters
    ----------
    signature
        The declared interface.
    name
        Which tensor to read.
    axis
        Which axis of it.
    static
        Whether to look among the static inputs or the dynamic ones.

    Returns
    -------
    int
        The axis length.

    Raises
    ------
    ContractError
        If the tensor is absent, or that axis is declared variable. A
        variable axis here is not something to default around: it means
        the data module and the network disagree about what is fixed.
    """
    declared = signature.static if static else signature.dynamic
    if name not in declared:
        raise ContractError(
            f"the signature declares no {'static' if static else 'dynamic'} input "
            f"{name!r}; the hybrid network requires it. Declared: "
            f"{sorted(declared)}"
        )
    size = declared[name].shape[axis]
    if size is None:
        raise ContractError(
            f"axis {axis} of {name!r} is declared variable, but the network must "
            f"size a layer from it. Only the batch axis may vary"
        )
    return int(size)
