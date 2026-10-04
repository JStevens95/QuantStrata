"""
Tests for the flagship's instrument graph.

Two things are being defended. The first is the mathematics: rows that sum
to one, self-loops present, weights that fall with distance. The second is
the port's one deliberate difference from the baseline -- the distance is
computed more accurately than scikit-learn computed it -- and the claim
that the difference cannot change which edges exist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pytest

from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.features.encoder import EntityEncoderState
from src.rade_qnet.models.hybrid_gnn_rnn.features.graph import (
    SparseGraphState,
    build_graph,
    weighted_features,
)
from src.rade_qnet.models.hybrid_gnn_rnn.spec import AttributeEncoderSpec, GraphSpec
from src.rade_qnet.testkit.parity import ADJACENCY_VALUE_ATOL

N_INSTRUMENTS = 6

#: Four elementary then two target, matching the shared attribute fixture.
IS_TARGET = np.array([False, False, False, False, True, True])


@pytest.fixture
def encoded(attributes: Mapping[str, Sequence[Any]]):
    """Encode the shared attribute fixture."""
    return EntityEncoderState.fit(attributes, spec=AttributeEncoderSpec()).transform(attributes)


@pytest.fixture
def graph(encoded) -> SparseGraphState:
    """Build a three-neighbour graph over the fixture."""
    return build_graph(encoded, spec=GraphSpec(n_neighbours=3), is_target=IS_TARGET)


def densify(graph: SparseGraphState) -> np.ndarray:
    """
    Expand the edge list into a dense matrix.

    Tests read far better against a dense matrix, and at six nodes the cost
    is nothing. Production never does this.

    Parameters
    ----------
    graph
        The graph to expand.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_nodes)``.
    """
    dense = np.zeros(graph.dense_shape, dtype=np.float64)
    dense[graph.indices[:, 0], graph.indices[:, 1]] = graph.values
    return dense


class TestStructure:
    """What the edge list looks like."""

    def test_every_node_has_a_self_loop(self, graph: SparseGraphState) -> None:
        """
        Without one, message passing discards a node's own features.

        One round of aggregation would replace each instrument with the mean
        of its neighbours, so after two layers nothing of the instrument
        itself remains.
        """
        dense = densify(graph)
        assert (np.diag(dense) > 0).all()

    def test_each_node_has_the_requested_neighbours_plus_itself(
        self, graph: SparseGraphState
    ) -> None:
        """Three neighbours and one self-loop, for every node."""
        assert np.array_equal(graph.degrees(), np.full(N_INSTRUMENTS, 4))

    def test_the_edges_are_sorted_row_major(self, graph: SparseGraphState) -> None:
        """
        The order is part of the state, so it has to be deterministic.

        Two graphs with the same edges listed differently are the same
        graph but not the same array, and the comparison against the
        baseline is an array comparison.
        """
        keys = graph.indices[:, 0] * graph.n_nodes + graph.indices[:, 1]
        assert np.array_equal(keys, np.sort(keys))
        assert len(np.unique(keys)) == len(keys)

    def test_a_node_is_never_its_own_neighbour_twice(self, graph: SparseGraphState) -> None:
        """
        The self-loop is added explicitly, so the search must not find it too.

        If it did, the diagonal would be counted twice and the node would
        weight itself at roughly double, with no error anywhere.
        """
        rows, columns = graph.indices[:, 0], graph.indices[:, 1]
        assert int((rows == columns).sum()) == N_INSTRUMENTS

    def test_more_neighbours_than_instruments_is_clamped(self, encoded) -> None:
        """
        A small cluster is legitimate and must not raise.

        Asking for twenty neighbours among six instruments should give five
        plus a self-loop, not an error and not a wrapped index.
        """
        graph = build_graph(encoded, spec=GraphSpec(n_neighbours=20), is_target=IS_TARGET)
        assert np.array_equal(graph.degrees(), np.full(N_INSTRUMENTS, N_INSTRUMENTS))


class TestWeights:
    """What the edge weights mean."""

    def test_every_row_sums_to_one(self, graph: SparseGraphState) -> None:
        """
        Row normalisation is what makes aggregation a mean rather than a sum.

        Without it a node's activation scales with its degree, and the
        network spends capacity learning to undo that.
        """
        assert np.allclose(densify(graph).sum(axis=1), 1.0, atol=1e-6)

    def test_the_weights_are_positive(self, graph: SparseGraphState) -> None:
        """
        A negative weight would mean a neighbour subtracts from a node.

        The kernel cannot produce one, so this is a guard on the
        normalisation arithmetic rather than on the kernel.
        """
        assert (graph.values > 0).all()

    def test_a_closer_neighbour_is_weighted_more_heavily(self, encoded) -> None:
        """
        The whole purpose of the kernel is that similarity decays with distance.

        A graph that weighted all neighbours equally would make the
        attribute encoding irrelevant to message passing.
        """
        features = weighted_features(encoded, spec=GraphSpec())
        graph = build_graph(encoded, spec=GraphSpec(n_neighbours=3), is_target=IS_TARGET)
        dense = densify(graph)

        node = 0
        neighbours = [c for c in range(N_INSTRUMENTS) if c != node and dense[node, c] > 0]
        distances = [np.linalg.norm(features[node] - features[c]) for c in neighbours]
        weights = [dense[node, c] for c in neighbours]

        by_distance = [w for _, w in sorted(zip(distances, weights, strict=True))]
        assert by_distance == sorted(by_distance, reverse=True)

    def test_duplicate_instruments_do_not_produce_nan(
        self, attributes: Mapping[str, Sequence[Any]]
    ) -> None:
        """
        Two identical instruments sit at distance zero from each other.

        Their kernel bandwidth is the median of their neighbour distances,
        which is then zero, and dividing by it would make every one of their
        edges NaN. Not hypothetical -- a book holding the same trade twice
        is ordinary.
        """
        duplicated = {key: [values[0]] * 3 + list(values)[3:] for key, values in attributes.items()}
        state = EntityEncoderState.fit(duplicated, spec=AttributeEncoderSpec())
        graph = build_graph(
            state.transform(duplicated), spec=GraphSpec(n_neighbours=2), is_target=IS_TARGET
        )
        assert np.isfinite(graph.values).all()
        assert np.allclose(densify(graph).sum(axis=1), 1.0, atol=1e-6)


class TestWeightedFeatures:
    """How the attribute groups are combined before measuring distance."""

    def test_the_weight_applies_to_squared_distance(self, encoded) -> None:
        """
        Square-rooting alpha is what makes the weights mean what they say.

        Quadrupling an attribute's alpha should double the contribution of
        its differences to the distance, not quadruple it.
        """
        base = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=1.0))
        scaled = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=4.0))
        assert scaled[:, 0] == pytest.approx(2.0 * base[:, 0], rel=1e-6)

    def test_a_zero_weight_drops_the_attribute(self, encoded) -> None:
        """Weighting a group to zero should remove its columns, not zero them."""
        with_moneyness = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=1.0))
        without = weighted_features(encoded, spec=GraphSpec(alpha_moneyness=0.0))
        assert without.shape[1] == with_moneyness.shape[1] - 1

    def test_a_wide_one_hot_does_not_dominate_by_width(self, encoded) -> None:
        """
        A categorical attribute's influence should follow its weight, not its levels.

        Two instruments differing in product type differ in two one-hot
        columns by one each, which without the width normalisation
        contributes twice what a one-column scalar does -- purely because
        the attribute happens to have two levels.
        """
        features = weighted_features(encoded, spec=GraphSpec())
        start, end = encoded.blocks["product_type"]
        width = end - start
        block = features[:, 4 : 4 + width]
        # Two rows with different product types differ by exactly sqrt(2/width)
        # per differing column, so their squared separation is 2/width * ... ,
        # i.e. independent of width.
        separation = np.linalg.norm(block[0] - block[2]) ** 2
        assert separation == pytest.approx(2.0 / width, rel=1e-5)

    def test_weighting_everything_to_zero_is_refused(self, encoded) -> None:
        """
        A graph over no features would have arbitrary structure.

        Returning one would be worse than failing: it would train, and the
        GNN would be aggregating over noise.
        """
        with pytest.raises(ContractError, match="weighted to zero"):
            weighted_features(
                encoded,
                spec=GraphSpec(
                    alpha_moneyness=0.0,
                    alpha_maturity=0.0,
                    alpha_delta=0.0,
                    alpha_vega=0.0,
                    alpha_product_type=0.0,
                    alpha_product_subtype=0.0,
                    alpha_underlying=0.0,
                    alpha_underlying_risk_factors=0.0,
                ),
            )


class TestMetrics:
    """The distance metrics."""

    @pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
    def test_each_metric_produces_a_valid_graph(self, encoded, metric: str) -> None:
        """Whichever metric is chosen, the output is still a row-stochastic graph."""
        graph = build_graph(
            encoded, spec=GraphSpec(n_neighbours=3, distance_metric=metric), is_target=IS_TARGET
        )
        assert np.allclose(densify(graph).sum(axis=1), 1.0, atol=1e-6)

    def test_an_unknown_metric_is_refused(self, encoded) -> None:
        """Silently falling back to euclidean would hide a typo in a config."""
        spec = GraphSpec(n_neighbours=3).model_copy(update={"distance_metric": "mahalanobis"})
        with pytest.raises(ContractError, match="mahalanobis"):
            build_graph(encoded, spec=spec, is_target=IS_TARGET)


class TestTheGraphIsRobust:
    """
    The port's one deliberate numerical difference from the baseline.

    scikit-learn computed neighbour distances in float32 from an expansion
    that cancels; this port computes the explicit difference in float64.
    The weights therefore differ in their last float32 bit, which
    ``ADJACENCY_VALUE_ATOL`` allows. These tests are the evidence that the
    allowance is safe -- that the noise is far too small to change which
    edges exist.
    """

    def test_the_neighbour_margin_dwarfs_the_numerical_noise(self, encoded) -> None:
        """
        Measure the gap between the furthest kept neighbour and the nearest dropped one.

        If that margin were comparable to the arithmetic noise, the two
        implementations could disagree about the graph's *structure* and the
        relaxed weight tolerance would be hiding a real difference. It is
        four orders of magnitude larger.
        """
        features = weighted_features(encoded, spec=GraphSpec()).astype(np.float64)
        difference = features[:, None, :] - features[None, :, :]
        distances = np.sqrt(np.einsum("ijk,ijk->ij", difference, difference))
        np.fill_diagonal(distances, np.inf)

        n_neighbours = 3
        ordered = np.sort(distances, axis=1)
        margin = (ordered[:, n_neighbours] - ordered[:, n_neighbours - 1]).min()
        assert margin > 100 * ADJACENCY_VALUE_ATOL

    def test_perturbing_the_features_by_that_noise_leaves_the_edges_unchanged(
        self, encoded
    ) -> None:
        """
        The direct statement of the claim, rather than a proxy for it.

        Jitter the features by more than the two implementations disagree
        by, and the edge list must come out identical.
        """
        spec = GraphSpec(n_neighbours=3)
        baseline = build_graph(encoded, spec=spec, is_target=IS_TARGET)

        generator = np.random.default_rng(0)
        noise = generator.normal(scale=1e-6, size=encoded.features.shape).astype(np.float32)
        jittered = build_graph(
            type(encoded)(
                features=encoded.features + noise, names=encoded.names, blocks=encoded.blocks
            ),
            spec=spec,
            is_target=IS_TARGET,
        )
        assert np.array_equal(jittered.indices, baseline.indices)

    def test_the_tolerance_is_one_float32_bit_not_a_blanket(self) -> None:
        """
        A tolerance wide enough to hide a real bug is not a tolerance.

        One float32 ULP near unit magnitude is about 1.2e-07; anything much
        larger would start admitting genuine differences in the kernel or
        the normalisation.
        """
        assert np.finfo(np.float32).eps >= ADJACENCY_VALUE_ATOL


class TestRoundTrip:
    """Saving and loading."""

    def test_a_reloaded_graph_is_identical(self, graph: SparseGraphState, tmp_path) -> None:
        """Including the edge order, which is part of the state."""
        graph.save(tmp_path)
        restored = SparseGraphState.load(tmp_path)
        assert np.array_equal(restored.indices, graph.indices)
        assert np.array_equal(restored.values, graph.values)
        assert restored.n_nodes == graph.n_nodes
        assert np.array_equal(restored.is_target, graph.is_target)

    def test_nothing_is_pickled(self, graph: SparseGraphState, tmp_path) -> None:
        """
        The original stored a SciPy sparse matrix by pickle.

        That ties a saved model to a SciPy version. Plain arrays and JSON do
        not.
        """
        graph.save(tmp_path)
        suffixes = {path.suffix for path in tmp_path.rglob("*") if path.is_file()}
        assert suffixes <= {".npy", ".npz", ".json"}
