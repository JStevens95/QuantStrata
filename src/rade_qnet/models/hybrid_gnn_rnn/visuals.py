"""
Figures only this model can produce.

The shared figures in :mod:`rade_qnet.analysis.visuals` work for any model
because they only look at predictions and losses. These look at the graph,
which no other model has.

The same rule applies here as there: **a figure factory returns a figure and
writes nothing.** A test asserts on the axes without a temporary directory,
a notebook displays the result without producing a file, and the report
writer decides where it lands.

Why the graph deserves its own figures
---------------------------------------
The graph is the model's single largest modelling assumption and the one
least visible in any metric. A poorly built graph does not announce itself:
the loss still falls, the metrics still look reasonable, and the damage
shows up only on instruments the training set under-covered -- which is
precisely the population the model exists to price.

These three figures are the diagnostics that make that failure visible
before it costs anything:

- the degree distribution says whether the neighbourhood size is uniform or
  whether a handful of instruments are carrying the whole book;
- the edge-weight distribution says whether neighbours are genuinely
  similar or whether the kernel has flattened into an average;
- the neighbour-similarity profile says, per instrument, how good its
  *best* neighbour actually is -- which is what an unfitted target's
  borrowed baseline will be built from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from ...analysis.visuals.style import figure_style
from ...core.runtime.errors import ContractError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .features.graph import SparseGraphState

__all__ = [
    "edge_weight_figure",
    "neighbour_similarity_figure",
    "node_degree_figure",
]

#: Default figure size, in inches. Matches the shared primitives so a report
#: mixing both does not have figures of two different widths.
DEFAULT_FIGSIZE = (8.0, 4.5)

#: How many histogram bins. Enough to show a bimodal weight distribution --
#: the signature of a graph that has split into tight cliques -- without
#: being so fine that a 200-instrument cluster shows mostly empty bins.
_N_BINS = 40

#: Quantiles reported alongside the neighbour-similarity profile. The low
#: tail is the interesting half: the median instrument's best neighbour is
#: rarely the problem.
_QUANTILES = (0.01, 0.05, 0.25, 0.50)


def node_degree_figure(graph: SparseGraphState, *, title: str = "Neighbourhood size") -> Figure:
    """
    Plot how many neighbours each instrument has.

    A graph built with a fixed neighbour count produces a near-vertical
    line and is uninteresting; the figure earns its place when a threshold
    or a kernel cutoff was used, where it reveals the instruments that
    ended up nearly isolated. Those instruments get almost no information
    from the graph, so for them the model degenerates to the temporal
    stream alone.

    Parameters
    ----------
    graph
        The fitted graph.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.
    """
    degrees = graph.degrees()
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        # Integer-aligned bins: a degree histogram on continuous bins
        # straddles whole numbers and reads as though degrees were
        # fractional.
        edges = np.arange(degrees.min(), degrees.max() + 2) - 0.5
        axes.hist(degrees, bins=edges, edgecolor="white", linewidth=0.5)
        axes.axvline(
            float(np.median(degrees)),
            color="#808080",
            linestyle="--",
            linewidth=1.0,
            label=f"median ({int(np.median(degrees))})",
        )
        axes.set_xlabel("neighbours")
        axes.set_ylabel("instruments")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def edge_weight_figure(graph: SparseGraphState, *, title: str = "Edge weights") -> Figure:
    """
    Plot the distribution of normalised edge weights.

    Rows are normalised, so the weights on one instrument's edges sum to
    one. Mass concentrated near ``1/degree`` means the kernel is treating
    every neighbour alike, which makes the graph an unweighted average and
    discards the similarity information it was built to carry. A long right
    tail is the healthy shape: each instrument has a few neighbours that
    genuinely matter.

    Parameters
    ----------
    graph
        The fitted graph.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the graph has no edges, which would make the figure empty and
        is a far more serious finding than a missing plot.
    """
    weights = np.asarray(graph.values, dtype=np.float64)
    if weights.size == 0:
        raise ContractError(
            "the graph has no edges, so there are no weights to plot; this is a "
            "failure of the graph build rather than of the figure"
        )

    uniform = 1.0 / np.maximum(graph.degrees().mean(), 1.0)
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.hist(weights, bins=_N_BINS, edgecolor="white", linewidth=0.5)
        axes.axvline(
            uniform,
            color="#808080",
            linestyle="--",
            linewidth=1.0,
            label=f"uniform ({uniform:.3f})",
        )
        axes.set_xlabel("normalised edge weight")
        axes.set_ylabel("edges")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def neighbour_similarity_figure(
    graph: SparseGraphState, *, title: str = "Best-neighbour weight by instrument"
) -> Figure:
    """
    Plot each instrument's largest edge weight, sorted.

    This is the figure to read before trusting a prediction for an
    instrument that was not in the training set. Such an instrument has no
    fitted baseline and borrows one from its nearest neighbours, so the
    quality of that borrow is bounded by how close its closest neighbour
    actually is. The left end of this curve is the population at risk.

    Parameters
    ----------
    graph
        The fitted graph.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the graph has no edges.
    """
    best = _best_weight_per_node(graph)
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.plot(np.sort(best), marker="")
        for quantile in _QUANTILES:
            value = float(np.quantile(best, quantile))
            axes.axhline(
                value,
                color="#808080",
                linestyle=":",
                linewidth=0.8,
            )
            # Annotated on the axis rather than in a legend: four legend
            # entries of the same style are harder to read than four
            # labels sitting on the lines they describe.
            axes.annotate(
                f"q{int(quantile * 100):02d} = {value:.3f}",
                xy=(0, value),
                xytext=(4, 2),
                textcoords="offset points",
                fontsize="small",
                color="#505050",
            )
        axes.set_xlabel("instrument (sorted)")
        axes.set_ylabel("largest edge weight")
        axes.set_title(title)
        figure.tight_layout()
        return figure


def _best_weight_per_node(graph: SparseGraphState) -> NDArray[np.float64]:
    """
    Find each node's largest edge weight.

    Parameters
    ----------
    graph
        The fitted graph.

    Returns
    -------
    numpy.ndarray
        One value per node. A node with no edges scores zero, which is the
        honest answer: it has no best neighbour.

    Raises
    ------
    ContractError
        If the graph has no edges.
    """
    values = np.asarray(graph.values, dtype=np.float64)
    if values.size == 0:
        raise ContractError(
            "the graph has no edges, so no instrument has a best neighbour; this "
            "is a failure of the graph build rather than of the figure"
        )
    rows = np.asarray(graph.indices, dtype=np.int64)[:, 0]
    best = np.zeros(graph.n_nodes, dtype=np.float64)
    np.maximum.at(best, rows, values)
    return best
