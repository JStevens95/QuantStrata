"""Tests for the fusion layer."""

from __future__ import annotations

import pytest
import torch

from src.rade_xl.core.runtime.errors import ContractError
from src.rade_xl.models.hybrid_gnn_rnn.layers.fusion import FusionLayer
from src.rade_xl.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import BATCH, N_NODES, UNITS


@pytest.fixture
def streams() -> tuple[torch.Tensor, torch.Tensor]:
    """Return a graph stream and a temporal stream at the expected widths."""
    generator = torch.Generator().manual_seed(0)
    return (
        torch.randn(N_NODES, UNITS, generator=generator),
        torch.randn(BATCH, UNITS, generator=generator),
    )


def layer(**overrides: object) -> FusionLayer:
    """Build a fusion layer at the shared width."""
    spec = HybridModelSpec(units=UNITS, dropout=0.0, **overrides)
    return FusionLayer(spec, gnn_features=UNITS, rnn_features=UNITS)


class TestShape:
    """What comes out."""

    def test_the_two_streams_cross_into_a_grid(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """
        One vector per instrument per sample.

        This is the layer's whole purpose. The graph stream arrives with
        no batch axis and the temporal stream with no node axis; what
        leaves has both, which is what lets the model say something
        different about the same instrument in two different regimes.
        """
        graph, temporal = streams
        assert layer()(graph, temporal, adjacency).shape == (BATCH, N_NODES, UNITS)

    def test_a_width_not_divisible_by_the_heads_is_refused(self) -> None:
        """
        Rather than dropping the remainder on the head reshape.

        The reshape would succeed for some combinations and silently
        discard features for others, which is the kind of fault that
        shows up as a mildly worse model six weeks later.
        """
        spec = HybridModelSpec(units=10, fusion_heads=4)
        with pytest.raises(ContractError, match="divisible"):
            FusionLayer(spec, gnn_features=UNITS, rnn_features=UNITS)

    def test_a_bidirectional_temporal_stream_is_accepted(self, adjacency: torch.Tensor) -> None:
        """
        The incoming widths are taken from the blocks, not assumed.

        A bidirectional recurrence doubles its own width, so a fusion
        layer that assumed ``units`` would be built wrong whenever that
        option was chosen.
        """
        spec = HybridModelSpec(units=UNITS, dropout=0.0)
        fusion = FusionLayer(spec, gnn_features=UNITS, rnn_features=2 * UNITS)
        graph = torch.randn(N_NODES, UNITS)
        temporal = torch.randn(BATCH, 2 * UNITS)
        assert fusion(graph, temporal, adjacency).shape == (BATCH, N_NODES, UNITS)


class TestAttention:
    """The graph-masked cross-attention."""

    def test_the_sparse_and_dense_paths_agree(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """
        Two implementations of the same mathematics.

        The sparse path is what makes a large universe tractable and the
        dense path is what a small cluster uses, so in practice a run
        exercises one or the other and a divergence would go unnoticed.
        """
        graph, temporal = streams
        fusion = layer()
        fusion.eval()
        with torch.no_grad():
            sparse = fusion(graph, temporal, adjacency)
            dense = fusion(graph, temporal, adjacency.to_dense())
        torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=0.0)

    def test_an_instrument_cannot_attend_beyond_its_neighbourhood(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """
        Changing a non-neighbour's features leaves a node's output alone.

        This is the claim the graph exists to make, and it is the one
        thing no shape check can see. If the mask were applied after the
        softmax instead of before, or dropped entirely, every instrument
        would quietly be reading the whole book.
        """
        graph, temporal = streams
        dense = adjacency.to_dense()
        # Node 0's ring neighbours are itself and the two adjacent nodes,
        # so node 3 is a non-neighbour by construction.
        stranger = 3
        assert dense[0, stranger] == 0.0

        fusion = layer()
        fusion.eval()
        perturbed = graph.clone()
        perturbed[stranger] += 10.0
        with torch.no_grad():
            before = fusion(graph, temporal, adjacency)[:, 0, :]
            after = fusion(perturbed, temporal, adjacency)[:, 0, :]
        torch.testing.assert_close(before, after)

    def test_the_neighbour_cap_bounds_the_gathered_table(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """
        A capped layer still runs and still produces the right shape.

        The cap is what keeps the sparse path's memory bounded on a
        universe where one instrument is related to hundreds. Capping
        changes the answer, which is the trade being made; what it must
        not do is change the interface.
        """
        graph, temporal = streams
        capped = layer(neighbour_cap=1)
        assert capped(graph, temporal, adjacency).shape == (BATCH, N_NODES, UNITS)


class TestMixing:
    """How the attended result and the temporal stream combine."""

    def test_the_gate_lies_between_the_two_streams(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """
        Every parameter still receives a gradient under the gate.

        A gate that saturated at initialisation would cut one stream out
        of the backward pass entirely, so the blocks feeding it would
        never train -- and the loss would still fall, through the other
        stream.
        """
        graph, temporal = streams
        fusion = layer(fusion_mode="gate")
        fusion(graph, temporal, adjacency).sum().backward()
        unused = [name for name, parameter in fusion.named_parameters() if parameter.grad is None]
        assert not unused

    def test_the_additive_mode_has_no_gate_parameters(self) -> None:
        """A disabled option costs nothing, rather than being bypassed."""
        assert not hasattr(layer(fusion_mode="add"), "gate_projection")
        assert hasattr(layer(fusion_mode="gate"), "gate_projection")

    def test_an_unknown_mode_is_refused(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """Rather than falling through to one of the two silently."""
        graph, temporal = streams
        fusion = layer()
        fusion.mode = "concatenate"
        with pytest.raises(ContractError, match="fusion mode"):
            fusion(graph, temporal, adjacency)

    def test_the_two_streams_both_reach_the_output(
        self, streams: tuple[torch.Tensor, torch.Tensor], adjacency: torch.Tensor
    ) -> None:
        """
        Perturbing either stream changes the result.

        The failure this guards against is a gate that collapses to one
        side, which produces a model that looks like a graph-temporal
        hybrid and behaves like whichever half survived.
        """
        graph, temporal = streams
        fusion = layer()
        fusion.eval()
        with torch.no_grad():
            base = fusion(graph, temporal, adjacency)
            moved_graph = fusion(graph + 1.0, temporal, adjacency)
            moved_time = fusion(graph, temporal + 1.0, adjacency)
        assert not torch.allclose(base, moved_graph)
        assert not torch.allclose(base, moved_time)
