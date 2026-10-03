"""Tests for the model's own figures."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_xl.core.runtime.errors import ContractError
from src.rade_xl.models.hybrid_gnn_rnn.features.graph import SparseGraphState
from src.rade_xl.models.hybrid_gnn_rnn.visuals import (
    edge_weight_figure,
    neighbour_similarity_figure,
    node_degree_figure,
)

#: Instruments in the toy graph.
N_NODES = 6


def ring(n_nodes: int = N_NODES) -> SparseGraphState:
    """
    Build a row-normalised ring graph with self-loops.

    Parameters
    ----------
    n_nodes
        How many instruments.

    Returns
    -------
    SparseGraphState
        The graph.
    """
    pairs = [
        (node, neighbour)
        for node in range(n_nodes)
        for neighbour in (node, (node - 1) % n_nodes, (node + 1) % n_nodes)
    ]
    return SparseGraphState(
        indices=np.array(pairs, dtype=np.int64),
        values=np.full(len(pairs), 1.0 / 3.0, dtype=np.float32),
        n_nodes=n_nodes,
        is_target=np.array([False] * (n_nodes - 2) + [True] * 2),
    )


def empty(n_nodes: int = N_NODES) -> SparseGraphState:
    """
    Build a graph with no edges at all.

    Parameters
    ----------
    n_nodes
        How many instruments.

    Returns
    -------
    SparseGraphState
        The graph.
    """
    return SparseGraphState(
        indices=np.zeros((0, 2), dtype=np.int64),
        values=np.zeros(0, dtype=np.float32),
        n_nodes=n_nodes,
        is_target=np.zeros(n_nodes, dtype=np.bool_),
    )


class TestFigures:
    """Each factory builds a figure and writes nothing."""

    @pytest.mark.parametrize(
        "factory", [node_degree_figure, edge_weight_figure, neighbour_similarity_figure]
    )
    def test_a_figure_comes_back(self, factory) -> None:
        """
        The rule the whole visuals package obeys.

        A factory that also saved its output would need a separate
        variant for every caller: a report writer, a notebook, a
        dashboard and a test all want the figure and disagree about
        where it should go.
        """
        figure = factory(ring())
        assert isinstance(figure, Figure)
        assert figure.axes

    def test_the_degree_figure_marks_the_median(self) -> None:
        """
        So a reader can see immediately whether the graph is uniform.

        A fixed-neighbour graph is a vertical line and the median says
        nothing; the figure earns its place when a threshold was used
        and the distribution has a left tail.
        """
        axes = node_degree_figure(ring()).axes[0]
        assert axes.get_legend() is not None
        assert axes.get_xlabel() == "neighbours"

    def test_the_weight_figure_marks_the_uniform_level(self) -> None:
        """
        The reference the distribution should be compared against.

        Mass piled at ``1/degree`` means the kernel has flattened into
        an unweighted average, which is the failure the figure exists to
        make visible -- and it is invisible without the reference line.
        """
        axes = edge_weight_figure(ring()).axes[0]
        lines = [line for line in axes.get_lines() if line.get_linestyle() == "--"]
        assert lines

    def test_the_similarity_figure_is_sorted(self) -> None:
        """
        So the worst-covered instruments are at one end and easy to read.

        The question this figure answers is "how bad is the bottom of
        the distribution", and an unsorted scatter answers it far less
        directly.
        """
        axes = neighbour_similarity_figure(ring()).axes[0]
        values = axes.get_lines()[0].get_ydata()
        assert np.all(np.diff(values) >= 0)


class TestEmptyGraph:
    """What happens when there is nothing to plot."""

    @pytest.mark.parametrize("factory", [edge_weight_figure, neighbour_similarity_figure])
    def test_an_edgeless_graph_is_refused(self, factory) -> None:
        """
        Loudly, rather than by drawing an empty pair of axes.

        A graph with no edges is a far more serious finding than a
        missing figure, and an empty plot in a report reads as "nothing
        interesting here" rather than "the graph build failed".
        """
        with pytest.raises(ContractError, match="no edges"):
            factory(empty())

    def test_the_degree_figure_still_works(self) -> None:
        """
        Because a degree of zero is exactly what it should be showing.

        The other two have nothing to put on an axis; this one has the
        most important thing it could possibly report.
        """
        assert node_degree_figure(empty()).axes
