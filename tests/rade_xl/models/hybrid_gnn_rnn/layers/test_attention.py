"""Tests for the target attention layer."""

from __future__ import annotations

import pytest
import torch

from src.rade_xl.core.runtime.errors import ContractError
from src.rade_xl.models.hybrid_gnn_rnn.layers.attention import TargetAttentionLayer
from src.rade_xl.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import BATCH, N_NODES, N_TARGETS, UNITS


@pytest.fixture
def fused() -> torch.Tensor:
    """Return a fused representation, one vector per instrument per sample."""
    return torch.randn(BATCH, N_NODES, UNITS, generator=torch.Generator().manual_seed(0))


def layer(**overrides: object) -> TargetAttentionLayer:
    """Build an attention layer at the shared width."""
    spec = HybridModelSpec(units=UNITS, dropout=0.0, **overrides)
    return TargetAttentionLayer(spec, in_features=UNITS)


class TestShape:
    """What comes out."""

    def test_only_the_targets_survive(
        self, fused: torch.Tensor, adjacency: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """
        The node axis narrows to the targets.

        Everything upstream runs over the full universe because the
        elementary instruments carry the information; from here on only
        the instruments being predicted matter, and keeping the rest
        would make the attention quadratic in the universe rather than in
        the target count.
        """
        output = layer()(fused, adjacency, target_indices)
        assert output.shape == (BATCH, N_TARGETS, UNITS)

    def test_a_width_not_divisible_by_the_heads_is_refused(self) -> None:
        """Rather than dropping the remainder on the head reshape."""
        spec = HybridModelSpec(units=10, attention_heads=4)
        with pytest.raises(ContractError, match="divisible"):
            TargetAttentionLayer(spec, in_features=UNITS)

    @pytest.mark.parametrize("n_heads", [1, 2, 4])
    def test_any_dividing_head_count_works(
        self,
        n_heads: int,
        fused: torch.Tensor,
        adjacency: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> None:
        """The head split and the reassembly are inverses."""
        output = layer(attention_heads=n_heads)(fused, adjacency, target_indices)
        assert output.shape == (BATCH, N_TARGETS, UNITS)


class TestSubmatrix:
    """Restricting the graph to the targets."""

    def test_the_sparse_and_dense_extractions_agree(
        self, adjacency: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """
        Two routes to the same sub-graph.

        The sparse route relabels global node indices to local target
        positions through a lookup table, which is where an off-by-one
        would live; the dense route is a pair of index slices and is
        obviously right. Comparing them is how the sparse one earns trust.
        """
        sparse = TargetAttentionLayer._target_submatrix(adjacency, target_indices)
        dense = TargetAttentionLayer._target_submatrix(adjacency.to_dense(), target_indices)
        torch.testing.assert_close(sparse.to_dense(), dense)

    def test_an_edge_to_a_non_target_is_dropped(
        self, adjacency: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """
        Only edges with both endpoints among the targets survive.

        Keeping a half-edge would mean a target attending to a position
        that, after relabelling, points at an unrelated target -- the
        worst kind of bug, because the shapes stay right.
        """
        sub = TargetAttentionLayer._target_submatrix(adjacency, target_indices)
        full = adjacency.to_dense()
        for local_row, row in enumerate(target_indices.tolist()):
            for local_column, column in enumerate(target_indices.tolist()):
                assert sub.to_dense()[local_row, local_column] == full[row, column]


class TestAttention:
    """The masked self-attention over the targets."""

    def test_the_sparse_and_dense_paths_agree(
        self, fused: torch.Tensor, adjacency: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """Same mathematics, two implementations, as in the fusion layer."""
        attention = layer()
        attention.eval()
        with torch.no_grad():
            sparse = attention(fused, adjacency, target_indices)
            dense = attention(fused, adjacency.to_dense(), target_indices)
        torch.testing.assert_close(sparse, dense, atol=1e-6, rtol=0.0)

    def test_a_target_with_no_target_neighbours_still_produces_a_value(
        self, fused: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """
        An isolated target returns a number, not a NaN.

        The mask fills empty rows before the softmax, so a target whose
        only edges go to elementary instruments has an all-masked row.
        Left as a true negative infinity that row softmaxes to NaN, and
        one NaN reaches the loss and destroys the run several epochs
        later, with nothing in the logs naming the cause.
        """
        isolated = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (N_NODES, N_NODES)
        ).coalesce()
        output = layer()(fused, isolated, target_indices)
        assert torch.isfinite(output).all()

    def test_every_parameter_receives_a_gradient(
        self, fused: torch.Tensor, adjacency: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """
        No weight is disconnected from the output.

        This is the test that caught the output projection being
        constructed and never applied -- a fault invisible to any check
        on parameter names and shapes, because an unused parameter still
        appears in the state dict at the right size.
        """
        attention = layer()
        attention(fused, adjacency, target_indices).sum().backward()
        unused = [
            name for name, parameter in attention.named_parameters() if parameter.grad is None
        ]
        assert not unused

    def test_the_targets_are_not_predicted_independently(
        self, fused: torch.Tensor, adjacency: torch.Tensor, target_indices: torch.Tensor
    ) -> None:
        """
        Perturbing one target moves its neighbour's representation.

        This is the layer's entire reason to exist. Without it each
        target is predicted in isolation and the implied correlation
        between two forecasts on the same underlying is an accident of
        their features rather than anything the model was asked to get
        right.
        """
        first, second = int(target_indices[0]), int(target_indices[1])
        assert adjacency.to_dense()[first, second] > 0

        attention = layer()
        attention.eval()
        perturbed = fused.clone()
        perturbed[:, second, :] += 5.0
        with torch.no_grad():
            before = attention(fused, adjacency, target_indices)[:, 0, :]
            after = attention(perturbed, adjacency, target_indices)[:, 0, :]
        assert not torch.allclose(before, after)


class TestOptions:
    """Switches that change the block's shape."""

    def test_switching_off_normalisation_removes_both_norm_layers(self) -> None:
        """A transformer block has two, and neither should linger."""
        without = layer(attention_normalise=False)
        assert without.attn_layer_norm is None
        assert without.ffn_layer_norm is None

    def test_the_feed_forward_expands_fourfold(self) -> None:
        """
        The transformer default, and where most of the capacity lives.

        Attention decides what to combine; the feed-forward decides what
        to do with the result, and it is the wider of the two by a factor
        of four.
        """
        assert layer().ffn_linear1.out_features == 4 * UNITS
        assert layer().ffn_linear2.in_features == 4 * UNITS
