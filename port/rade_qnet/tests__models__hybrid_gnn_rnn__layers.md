# `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers`

7 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 9 | 433 | `74ed9a32e29b7cd6` |
| 2 | `conftest.py` | 78 | 2501 | `22199b76430ed4c2` |
| 3 | `test_attention.py` | 194 | 7953 | `ccff859641dcee9d` |
| 4 | `test_fusion.py` | 192 | 7814 | `fb100899a532acb1` |
| 5 | `test_gnn.py` | 215 | 8934 | `c101232729666724` |
| 6 | `test_projection.py` | 252 | 10275 | `0576d467a97225b6` |
| 7 | `test_rnn.py` | 125 | 5265 | `e6647b6a8262d8f7` |

---

## 1. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/__init__.py`

433 bytes · SHA-256 `74ed9a32e29b7cd6`

```python
"""
Tests for the network's architectural blocks, one module per block.

Each block is tested in isolation: given an input of a known shape, assert
the output shape, that gradients reach every parameter, and the invariances
the block is supposed to have. Testing an architecture only end to end makes
every shape bug a bisection exercise, and makes a block that quietly
contributes nothing indistinguishable from one that works.
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/conftest.py`

2501 bytes · SHA-256 `22199b76430ed4c2`

```python
"""Shared fixtures for the layer tests: a small graph and matching tensors."""

from __future__ import annotations

import numpy as np
import pytest
import torch

#: Instruments in the toy universe. Small enough to reason about by hand,
#: large enough that a head reshape or a neighbour gather has somewhere to
#: go wrong.
N_NODES = 6

#: How many of those are targets. They are the last rows, which is the
#: ordering the data module guarantees and the attention layer relies on.
N_TARGETS = 2

#: Samples per batch.
BATCH = 4

#: Width of the encoded attribute vector.
N_ATTRIBUTES = 7

#: Width of the selected elementary basis.
N_ELEMENTARY = 3

#: Window length of the P&L history.
SEQUENCE = 5

#: Layer width used throughout. Divisible by two so a multi-head test has a
#: valid head count available.
UNITS = 8


@pytest.fixture
def target_indices() -> torch.Tensor:
    """Which rows of the node table are targets."""
    return torch.arange(N_NODES - N_TARGETS, N_NODES)


@pytest.fixture
def adjacency() -> torch.Tensor:
    """
    Return a sparse ring graph with self-loops, row-normalised.

    A ring rather than a random graph because every node then has exactly
    the same degree, so a test that fails has failed for a reason other
    than one node happening to be unusual. Row-normalised because that is
    what the graph builder produces and what the layers assume.
    """
    rows, columns = [], []
    for node in range(N_NODES):
        for neighbour in (node, (node - 1) % N_NODES, (node + 1) % N_NODES):
            rows.append(node)
            columns.append(neighbour)
    indices = torch.tensor([rows, columns], dtype=torch.long)
    values = torch.full((len(rows),), 1.0 / 3.0, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, (N_NODES, N_NODES)).coalesce()


@pytest.fixture
def dense_adjacency(adjacency: torch.Tensor) -> torch.Tensor:
    """Return the same graph densified, for the dense-path comparisons."""
    return adjacency.to_dense()


@pytest.fixture
def node_features() -> torch.Tensor:
    """Return encoded attributes, one row per instrument."""
    generator = np.random.default_rng(0)
    return torch.tensor(generator.normal(size=(N_NODES, N_ATTRIBUTES)), dtype=torch.float32)


@pytest.fixture
def pnl_history() -> torch.Tensor:
    """Return a batch of elementary P&L windows."""
    generator = np.random.default_rng(1)
    return torch.tensor(generator.normal(size=(BATCH, SEQUENCE, N_ELEMENTARY)), dtype=torch.float32)
```

---

## 3. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_attention.py`

7953 bytes · SHA-256 `ccff859641dcee9d`

```python
"""Tests for the target attention layer."""

from __future__ import annotations

import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.layers.attention import (
    TargetAttentionLayer,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

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
```

---

## 4. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_fusion.py`

7814 bytes · SHA-256 `fb100899a532acb1`

```python
"""Tests for the fusion layer."""

from __future__ import annotations

import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.layers.fusion import FusionLayer
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

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
```

---

## 5. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_gnn.py`

8934 bytes · SHA-256 `c101232729666724`

```python
"""Tests for the graph block."""

from __future__ import annotations

import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.layers.gnn import (
    GnnBlock,
    GraphSage,
    MixedGraphSage,
    activation_function,
    neighbour_max,
    neighbour_mean,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

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
```

---

## 6. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_projection.py`

10275 bytes · SHA-256 `0576d467a97225b6`

```python
"""Tests for the output head."""

from __future__ import annotations

import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.layers.projection import (
    ProjectionLayer,
)
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import BATCH, N_ATTRIBUTES, N_TARGETS, UNITS


@pytest.fixture
def attended() -> torch.Tensor:
    """Attended representations, one per target per sample."""
    return torch.randn(BATCH, N_TARGETS, UNITS, generator=torch.Generator().manual_seed(0))


@pytest.fixture
def attributes() -> torch.Tensor:
    """Return static attributes for the targets."""
    return torch.randn(N_TARGETS, N_ATTRIBUTES, generator=torch.Generator().manual_seed(1))


def head(n_fitted: int = N_TARGETS, **overrides: object) -> ProjectionLayer:
    """Build an output head at the shared widths."""
    spec = HybridModelSpec(units=UNITS, dropout=0.0, **overrides)
    return ProjectionLayer(
        spec,
        in_features=UNITS,
        attribute_features=N_ATTRIBUTES,
        n_fitted_targets=n_fitted,
    )


class TestShape:
    """What comes out."""

    def test_one_number_per_target_per_sample(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """The representation collapses to a P&L prediction."""
        assert head()(attended, attributes).shape == (BATCH, N_TARGETS)

    def test_mismatched_inputs_are_refused(self, attended: torch.Tensor) -> None:
        """
        The attended targets and the attribute rows must be the same set.

        Broadcasting would otherwise silently pair each target with the
        wrong instrument's attributes, which produces plausible numbers
        and no error at all.
        """
        with pytest.raises(ContractError, match="attribute row"):
            head()(attended, torch.randn(N_TARGETS + 1, N_ATTRIBUTES))

    def test_the_head_is_built_eagerly(self) -> None:
        """
        Every per-target parameter exists before the first forward pass.

        This head is where the original was laziest -- its baseline
        kernels materialised on the first call, after the optimiser had
        already been constructed over nothing. That is defect 6, and it
        meant the per-target baselines never trained.
        """
        layer = head()
        assert all(
            not isinstance(parameter, torch.nn.UninitializedParameter)
            for parameter in layer.parameters()
        )
        assert layer._baseline_kernels.shape == (N_TARGETS, UNITS)


class TestBaselineAndResidual:
    """The two terms the prediction is made of."""

    def test_each_target_reads_only_its_own_kernel(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """
        The baseline is a row-wise dot product, not a matrix product.

        If it were a matrix product, every target's baseline would read
        every other target's kernel -- which is exactly the per-instrument
        specificity this half of the head exists to provide.
        """
        layer = head()
        layer.eval()
        with torch.no_grad():
            base = layer._fitted_baseline(attended, N_TARGETS)
            layer._baseline_kernels[1] += 10.0
            moved = layer._fitted_baseline(attended, N_TARGETS)
        torch.testing.assert_close(base[:, 0], moved[:, 0])
        assert not torch.allclose(base[:, 1], moved[:, 1])

    def test_the_residual_reads_the_attributes(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """
        Changing a target's attributes changes its prediction.

        The residual is the only path by which a static attribute reaches
        the output, so if it ignored them the head would reduce to a
        per-target linear read and could say nothing about an instrument
        it had not fitted.
        """
        layer = head()
        layer.eval()
        with torch.no_grad():
            before = layer(attended, attributes)
            after = layer(attended, attributes + 1.0)
        assert not torch.allclose(before, after)

    def test_every_parameter_receives_a_gradient(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """No weight is disconnected from the output."""
        layer = head()
        layer(attended, attributes).sum().backward()
        unused = [name for name, parameter in layer.named_parameters() if parameter.grad is None]
        assert not unused

    def test_weight_normalisation_starts_from_the_same_function(
        self, attended: torch.Tensor, attributes: torch.Tensor
    ) -> None:
        """
        The gain is initialised so that the layer begins where it would have.

        A reparametrisation that also moves the starting point is not a
        reparametrisation; it is a different model with a confusing name.
        The gain starts at the value whose softplus is one, so the first
        step is taken from the identical function.
        """
        plain, normed = head(), head(baseline_weight_norm=True)
        normed.load_state_dict(plain.state_dict(), strict=False)
        normed._baseline_kernels.data = plain._baseline_kernels.data.clone()

        # Unit-norm times a gain of one is the original kernel only when
        # the original was already unit-norm, so compare the directions.
        with torch.no_grad():
            gain = torch.nn.functional.softplus(normed._baseline_gain)
        torch.testing.assert_close(gain, torch.ones_like(gain))


class TestUnseenTargets:
    """Instruments that had no baseline fitted."""

    def test_an_unfitted_target_still_gets_a_prediction(self) -> None:
        """
        A trade booked after training is priced, not rejected.

        The whole point of a replication model is to price new business.
        A head that could only speak about the instruments it was fitted
        on would be useless for the job it exists to do.
        """
        layer = head(n_fitted=1)
        output = layer(torch.randn(BATCH, N_TARGETS, UNITS), torch.randn(N_TARGETS, N_ATTRIBUTES))
        assert output.shape == (BATCH, N_TARGETS)
        assert torch.isfinite(output).all()

    def test_the_borrowed_baseline_follows_the_nearest_neighbour(self) -> None:
        """
        A new target that matches one fitted target inherits its baseline.

        Constructed so the answer is known: the new instrument's
        attributes are a copy of the first fitted one's, and the cosine
        blend is sharp enough that the softmax puts essentially all its
        weight there. If the neighbour lookup were indexing the wrong
        axis this would inherit from the wrong instrument.
        """
        n_fitted = 3
        layer = ProjectionLayer(
            HybridModelSpec(units=UNITS, dropout=0.0, new_target_temperature=50.0),
            in_features=UNITS,
            attribute_features=N_ATTRIBUTES,
            n_fitted_targets=n_fitted,
        )
        layer.eval()

        fitted_attributes = torch.eye(n_fitted, N_ATTRIBUTES)
        # The new row is the first fitted row exactly.
        attributes = torch.cat([fitted_attributes, fitted_attributes[:1]], dim=0)
        attended = torch.randn(BATCH, n_fitted + 1, UNITS)

        with torch.no_grad():
            fitted = layer._fitted_baseline(attended[:, :n_fitted, :], n_fitted)
            borrowed = layer._borrowed_baseline(
                fitted,
                fitted_attributes=fitted_attributes,
                new_attributes=attributes[n_fitted:],
            )
        torch.testing.assert_close(borrowed[:, 0], fitted[:, 0], atol=1e-5, rtol=0.0)

    @pytest.mark.parametrize("blend", ["cosine_softmax", "inverse_distance"])
    def test_the_blend_weights_sum_to_one(self, blend: str) -> None:
        """
        Whichever blend is chosen, the borrow is an average.

        Weights that did not sum to one would scale the borrowed baseline
        by the number of neighbours, so a new instrument's prediction
        would depend on how many neighbours it happened to have.
        """
        layer = head(new_target_blend=blend)
        _, weights = layer._neighbour_weights(
            torch.randn(2, N_ATTRIBUTES), torch.randn(5, N_ATTRIBUTES)
        )
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))

    def test_an_unknown_blend_is_refused(self) -> None:
        """Rather than falling through to one of the two silently."""
        layer = head()
        layer.blend = "nearest"
        with pytest.raises(ContractError, match="blend"):
            layer._neighbour_weights(torch.randn(1, N_ATTRIBUTES), torch.randn(2, N_ATTRIBUTES))

    def test_the_residual_is_damped_for_unfitted_targets_only(self) -> None:
        """
        The correction is trusted less where it was never fitted.

        The residual network saw the fitted targets and did not see the
        new ones, so for a new instrument its output is extrapolation.
        Damping it is an explicit statement of that, and the damping must
        not touch the targets that *were* fitted.
        """
        n_fitted = 2
        full = ProjectionLayer(
            HybridModelSpec(units=UNITS, dropout=0.0, new_target_residual_damping=1.0),
            in_features=UNITS,
            attribute_features=N_ATTRIBUTES,
            n_fitted_targets=n_fitted,
        )
        damped = ProjectionLayer(
            HybridModelSpec(units=UNITS, dropout=0.0, new_target_residual_damping=0.0),
            in_features=UNITS,
            attribute_features=N_ATTRIBUTES,
            n_fitted_targets=n_fitted,
        )
        damped.load_state_dict(full.state_dict())
        full.eval()
        damped.eval()

        attended = torch.randn(BATCH, n_fitted + 1, UNITS)
        attributes = torch.randn(n_fitted + 1, N_ATTRIBUTES)
        with torch.no_grad():
            undamped_out = full(attended, attributes)
            damped_out = damped(attended, attributes)

        torch.testing.assert_close(undamped_out[:, :n_fitted], damped_out[:, :n_fitted])
        assert not torch.allclose(undamped_out[:, n_fitted:], damped_out[:, n_fitted:])
```

---

## 7. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/layers/test_rnn.py`

5265 bytes · SHA-256 `e6647b6a8262d8f7`

```python
"""Tests for the recurrent block."""

from __future__ import annotations

import pytest
import torch

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.layers.rnn import RnnBlock
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.spec import HybridModelSpec

from .conftest import BATCH, N_ELEMENTARY, SEQUENCE, UNITS


class TestShape:
    """What comes out, for each recurrence type."""

    @pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
    def test_one_summary_per_sample(self, rnn_type: str, pnl_history: torch.Tensor) -> None:
        """
        The window collapses to a single vector.

        There is no node axis: the recent period is the same period for
        every instrument, and giving each one its own copy here would
        multiply the work by the universe size for no information gained.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS, rnn_type=rnn_type), in_features=N_ELEMENTARY)
        assert block(pnl_history).shape == (BATCH, UNITS)

    def test_a_bidirectional_recurrence_is_twice_as_wide(self, pnl_history: torch.Tensor) -> None:
        """
        Both directions' final states are kept, so the output doubles.

        The width is read from the block rather than assumed by whatever
        consumes it, which is why ``out_features`` exists. A consumer that
        assumed ``units`` would build a layer of the wrong shape and fail
        only on the first forward pass.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS, rnn_type="bilstm"), in_features=N_ELEMENTARY)
        assert block.out_features == 2 * UNITS
        assert block(pnl_history).shape == (BATCH, 2 * UNITS)

    def test_an_unknown_recurrence_is_refused(self) -> None:
        """
        Rather than defaulting to one and reporting another.

        The spec's literal already excludes this, so reaching it means a
        spec was bypassed -- and silently substituting a GRU for a
        requested temporal convolution would be worse than failing.
        """
        spec = HybridModelSpec(units=UNITS)
        object.__setattr__(spec, "rnn_type", "tcn")
        with pytest.raises(ContractError, match="tcn"):
            RnnBlock(spec, in_features=N_ELEMENTARY)


class TestBehaviour:
    """What the block does with its input."""

    def test_every_parameter_receives_a_gradient(self, pnl_history: torch.Tensor) -> None:
        """No weight is disconnected from the output."""
        block = RnnBlock(HybridModelSpec(units=UNITS), in_features=N_ELEMENTARY)
        block(pnl_history).sum().backward()
        unused = [name for name, parameter in block.named_parameters() if parameter.grad is None]
        assert not unused

    def test_the_summary_depends_on_the_order_of_the_window(
        self, pnl_history: torch.Tensor
    ) -> None:
        """
        Reversing time changes the answer.

        The whole claim of the recurrent stream is that *when* a move
        happened matters. A block whose output survived a time reversal
        would be an expensive way to compute a summary statistic, and
        nothing else in the suite would notice.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS), in_features=N_ELEMENTARY)
        block.eval()
        with torch.no_grad():
            forward = block(pnl_history)
            backward = block(pnl_history.flip(dims=(1,)))
        assert not torch.allclose(forward, backward)

    def test_the_last_step_of_the_window_matters_most(self, pnl_history: torch.Tensor) -> None:
        """
        Perturbing the most recent day moves the summary more than the oldest.

        Not a law of recurrences -- an untrained network could behave
        either way -- but at initialisation an LSTM's forget gate damps
        earlier steps, and if that ordering inverted it would mean the
        window is being read backwards. That is the bug this catches.
        """
        block = RnnBlock(HybridModelSpec(units=UNITS), in_features=N_ELEMENTARY)
        block.eval()
        with torch.no_grad():
            base = block(pnl_history)

            oldest = pnl_history.clone()
            oldest[:, 0, :] += 1.0
            newest = pnl_history.clone()
            newest[:, SEQUENCE - 1, :] += 1.0

            moved_by_oldest = (block(oldest) - base).abs().mean()
            moved_by_newest = (block(newest) - base).abs().mean()
        assert moved_by_newest > moved_by_oldest

    def test_dropout_between_layers_only_applies_to_a_stack(self) -> None:
        """
        A single-layer recurrence gets no inter-layer dropout.

        Torch warns and ignores it, which is noise in every log of every
        single-layer run. Suppressing it here rather than tolerating the
        warning means a real warning stays visible.
        """
        single = RnnBlock(
            HybridModelSpec(units=UNITS, rnn_layers=1, dropout=0.5),
            in_features=N_ELEMENTARY,
        )
        stacked = RnnBlock(
            HybridModelSpec(units=UNITS, rnn_layers=2, dropout=0.5),
            in_features=N_ELEMENTARY,
        )
        assert single.rnn.dropout == 0.0
        assert stacked.rnn.dropout == 0.5
```

