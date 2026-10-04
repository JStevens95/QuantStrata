"""Tests for the graph block."""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.layers.gnn import (
    GnnBlock,
    GraphSage,
    MixedGraphSage,
    activation_function,
    neighbour_max,
    neighbour_mean,
)
from src.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import N_ATTRIBUTES, N_NODES, UNITS


class TestAggregation:
    """The two neighbourhood reductions, independent of any layer."""

    def test_the_mean_agrees_between_sparse_and_dense(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """
        A sparse and a dense adjacency describe the same graph.

        The two code paths exist for performance, not for different
        mathematics, and nothing else in the suite would notice if they
        diverged -- a run uses one or the other, never both.
        """
        sparse = neighbour_mean(node_features, adjacency)
        dense = neighbour_mean(node_features, adjacency.to_dense())
        torch.testing.assert_close(sparse, dense)

    def test_the_max_agrees_between_sparse_and_dense(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """The same, for the maximum."""
        sparse = neighbour_max(node_features, adjacency)
        dense = neighbour_max(node_features, adjacency.to_dense())
        torch.testing.assert_close(sparse, dense)

    def test_the_mean_is_a_weighted_average_of_the_neighbours(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """
        Each row is its neighbourhood's weighted mean, computed by hand.

        Asserting against an independent calculation rather than against
        another call of the same function, so the test would catch the
        matrix product being transposed -- which agrees with itself.
        """
        dense = adjacency.to_dense()
        expected = torch.stack(
            [
                sum(dense[node, other] * node_features[other] for other in range(N_NODES))
                for node in range(N_NODES)
            ]
        )
        torch.testing.assert_close(neighbour_mean(node_features, adjacency), expected)

    def test_a_node_with_no_neighbours_reduces_to_zero_under_the_max(
        self, node_features: torch.Tensor
    ) -> None:
        """
        An edgeless node scores zero rather than negative infinity.

        The maximum is seeded with ``-inf`` so that genuinely negative
        features still win; a node with nothing to reduce over would keep
        that seed and poison every later layer with an infinity.
        """
        empty = torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long), torch.zeros(0), (N_NODES, N_NODES)
        ).coalesce()
        assert torch.all(neighbour_max(node_features, empty) == 0.0)


class TestLayers:
    """The two layer types, in isolation."""

    def test_graph_sage_projects_self_and_neighbourhood_separately(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """
        A node's own features and its neighbourhood get their own weights.

        Keeping the self term is what stops the block from being a pure
        smoother: without it, several rounds of averaging drive every node
        towards the same vector. Two summed projections rather than one
        over a concatenation -- algebraically identical, and the form the
        original uses, so the parameter names match.
        """
        layer = GraphSage(in_features=N_ATTRIBUTES, out_features=UNITS)
        assert layer.dense_self.in_features == N_ATTRIBUTES
        assert layer.dense_neigh.in_features == N_ATTRIBUTES
        assert layer(node_features, adjacency).shape == (N_NODES, UNITS)

    def test_graph_sage_refuses_an_unknown_aggregation(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """Rather than silently reducing one way and reporting the other."""
        layer = GraphSage(in_features=N_ATTRIBUTES, out_features=UNITS)
        layer.aggregation = "median"
        with pytest.raises(ContractError, match="aggregation"):
            layer(node_features, adjacency)

    def test_mixed_graph_sage_reads_three_aggregates(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """
        The mixed layer adds the neighbourhood maximum.

        The mean says what the neighbourhood looks like on average; the
        maximum says whether any neighbour is extreme. For a book, that
        distinction is the difference between a quiet cluster and a quiet
        cluster containing one blown-up position.
        """
        layer = MixedGraphSage(in_features=N_ATTRIBUTES, out_features=UNITS)
        assert layer.fusion_dense.in_features == 3 * N_ATTRIBUTES
        assert layer(node_features, adjacency).shape == (N_NODES, UNITS)


class TestBlock:
    """The stacked block the model actually uses."""

    def test_the_output_is_one_row_per_instrument(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """There is no batch axis: structure is shared across samples."""
        block = GnnBlock(HybridModelSpec(units=UNITS), in_features=N_ATTRIBUTES)
        assert block(node_features, adjacency).shape == (N_NODES, UNITS)

    def test_every_parameter_receives_a_gradient(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """
        No parameter is disconnected from the output.

        A layer that is constructed and never applied still appears in the
        state dict with the right shape, so a parity check on names and
        shapes passes while the layer contributes nothing. Only the
        gradient notices.
        """
        block = GnnBlock(HybridModelSpec(units=UNITS), in_features=N_ATTRIBUTES)
        block(node_features, adjacency).sum().backward()
        unused = [name for name, parameter in block.named_parameters() if parameter.grad is None]
        assert not unused

    def test_the_layer_count_is_what_the_spec_asked_for(
        self, node_features: torch.Tensor, adjacency: torch.Tensor
    ) -> None:
        """Depth comes from the spec, not from a constant in the block."""
        block = GnnBlock(HybridModelSpec(units=UNITS, gnn_layers=3), in_features=N_ATTRIBUTES)
        assert len(block.gnn_layers) == 3

    def test_switching_off_normalisation_removes_the_norm_layers(self) -> None:
        """A disabled option costs no parameters, rather than being bypassed."""
        block = GnnBlock(
            HybridModelSpec(units=UNITS, gnn_normalise=False), in_features=N_ATTRIBUTES
        )
        assert len(block.norm_layers) == 0

    def test_the_residual_projection_exists_only_when_asked_for(self) -> None:
        """The same, for the skip connection."""
        without = GnnBlock(
            HybridModelSpec(units=UNITS, use_residual=False), in_features=N_ATTRIBUTES
        )
        with_skip = GnnBlock(
            HybridModelSpec(units=UNITS, use_residual=True), in_features=N_ATTRIBUTES
        )
        assert without.input_projection is None
        assert with_skip.input_projection is not None

    def test_the_block_is_built_eagerly(self) -> None:
        """
        Every parameter exists before the first forward pass.

        This is defect 6. A lazily shaped layer materialises on its first
        forward call, which is after the optimiser was constructed -- so
        the optimiser tracks an empty list and those parameters never
        update. The loss still falls, via the layers that were eager, and
        nothing reports an error.
        """
        block = GnnBlock(HybridModelSpec(units=UNITS), in_features=N_ATTRIBUTES)
        assert all(
            not isinstance(parameter, torch.nn.UninitializedParameter)
            for parameter in block.parameters()
        )
        assert list(block.parameters())


class TestActivations:
    """The activation lookup."""

    @pytest.mark.parametrize(
        "name", ["relu", "leaky_relu", "tanh", "sigmoid", "elu", "selu", "gelu", "linear"]
    )
    def test_every_declared_activation_resolves(self, name: str) -> None:
        """
        The spec's literal and the lookup agree.

        They are two lists in two files, and a name in one but not the
        other fails at model-construction time on a configuration that
        validated cleanly.
        """
        assert activation_function(name)(torch.zeros(3)).shape == (3,)

    def test_an_unknown_activation_is_refused(self) -> None:
        """Rather than silently falling back to the identity."""
        with pytest.raises(ContractError, match="activation"):
            activation_function("swish")
