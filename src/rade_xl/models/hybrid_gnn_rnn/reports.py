"""
Reports only this model can produce.

Every model gets the shared reports: a run summary, training curves, data
quality, baseline comparisons. Those answer *how well did it do?*. This one
answers a question that only applies to a graph model and that no metric
will raise on its own: *is the graph any good?*

Why this is not optional
------------------------
A badly built graph is silent. The loss still falls, because the temporal
stream alone can fit a book reasonably well. The headline metrics still
look acceptable, because they are dominated by instruments the training set
covered densely. What degrades is accuracy on instruments the training set
covered thinly -- and those are exactly the instruments a replication model
is bought for.

So the graph diagnostics are not a nice-to-have appendix. They are the only
place in the run where you can see the model's largest assumption and
decide whether to believe it. This report is registered under
``hybrid_graph`` and is added to the enabled set by the model's own train
pipeline, so it renders whether or not the user thought to ask for it.

What to look for
----------------
Three numbers decide it, and the report states each with the threshold it
should be compared against rather than leaving the reader to judge:

- **isolated instruments** -- nodes whose only edge is their own self-loop.
  They get nothing from the graph, so for them the model is the temporal
  stream and a bias term.
- **weight concentration** -- the share of each node's weight sitting on
  its single best neighbour. Near ``1/degree`` means the kernel has
  flattened into an unweighted average and the similarity information has
  been discarded.
- **worst-covered instruments** -- the bottom of the best-neighbour
  distribution, named individually. This is the list to read before
  pricing an unseen trade, because an unseen target's baseline is borrowed
  from precisely these neighbourhoods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...analysis.reports.base import Report
from ...analysis.visuals.export import save_figure
from ...core.runtime.components import report
from .state import HybridState
from .visuals import (
    edge_weight_figure,
    neighbour_similarity_figure,
    node_degree_figure,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from ...analysis.reports.base import ReportContext
    from .features.graph import SparseGraphState

__all__ = ["HybridGraphReport"]

#: Filename of the Markdown page.
GRAPH_FILENAME = "graph.md"

#: How many of the worst-covered instruments to name. Enough to spot a
#: pattern -- one currency, one product type -- and few enough to read.
_N_WORST = 10

#: Below this, an instrument's best neighbour is weak enough that a
#: borrowed baseline should not be trusted without a second look. It is a
#: prompt to investigate, not a hard rule, and the report says so.
_WEAK_NEIGHBOUR = 0.10


@report("hybrid_graph")
class HybridGraphReport(Report):
    """
    A Markdown page describing the instrument graph the model was trained on.

    Reads only the bundle's fitted state, so it re-renders from a saved run
    without the data and without the weights.
    """

    def render(self, context: ReportContext) -> Sequence[Path]:
        """
        Write the graph diagnostics page and its figures.

        Parameters
        ----------
        context
            The bundle to report on, and where to write.

        Returns
        -------
        Sequence of Path
            The Markdown page followed by the figures.

        Raises
        ------
        LookupError
            If the bundle's state is not a hybrid state. A deliberate skip
            rather than an incidental failure: this report is enabled by
            the model's own pipeline, so reaching it with another model's
            state means the run was assembled by hand and the reader
            deserves an explanation rather than a stack trace.
        """
        state = context.bundle.state
        if not isinstance(state, HybridState):
            raise LookupError(
                f"the hybrid graph report needs a HybridState, but this bundle "
                f"holds a {type(state).__name__}; nothing to describe"
            )

        figures = self._write_figures(state.graph, context)
        sections = [
            "# Instrument graph",
            self._shape_section(state),
            self._coverage_section(state),
            self._worst_section(state),
            self._figures_section(figures),
        ]
        page = context.directory / GRAPH_FILENAME
        page.write_text("\n\n".join(sections).rstrip() + "\n", encoding="utf-8")
        return (page, *figures)

    @staticmethod
    def _shape_section(state: HybridState) -> str:
        """
        Return the graph's dimensions.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        str
            Markdown text.
        """
        graph = state.graph
        degrees = graph.degrees()
        # Self-loops are excluded from the quoted neighbour count because
        # "three neighbours" meaning "two neighbours and itself" is the
        # kind of off-by-one that survives into a conversation with a desk.
        neighbours = degrees - 1
        return "\n".join(
            [
                "## Shape",
                "",
                "| property | value |",
                "| --- | --- |",
                f"| instruments | {graph.n_nodes} |",
                f"| elementary | {state.universe.n_elementary} |",
                f"| targets | {state.universe.n_targets} |",
                f"| edges (self-loops included) | {graph.n_edges} |",
                f"| neighbours per instrument (median) | {int(np.median(neighbours))} |",
                f"| neighbours per instrument (min) | {int(neighbours.min())} |",
                f"| density | {graph.n_edges / max(graph.n_nodes**2, 1):.4f} |",
            ]
        )

    @classmethod
    def _coverage_section(cls, state: HybridState) -> str:
        """
        Return the two findings that decide whether the graph is usable.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        str
            Markdown text, with a verdict line per finding.
        """
        graph = state.graph
        neighbours = graph.degrees() - 1
        isolated = int((neighbours <= 0).sum())
        best = cls._best_weight_per_node(graph)
        weak = int((best < _WEAK_NEIGHBOUR).sum())

        lines = [
            "## Coverage",
            "",
            "| finding | count | share |",
            "| --- | --- | --- |",
            f"| isolated (self-loop only) | {isolated} | {isolated / max(graph.n_nodes, 1):.1%} |",
            f"| best neighbour below {_WEAK_NEIGHBOUR:.2f} | {weak} | "
            f"{weak / max(graph.n_nodes, 1):.1%} |",
            "",
        ]
        if isolated:
            lines.append(
                f"{isolated} instrument(s) have no neighbour but themselves. For "
                f"these the graph contributes nothing and the model reduces to "
                f"the temporal stream. Either the similarity threshold is too "
                f"strict or these instruments are genuinely unlike the rest of "
                f"the book, and the two call for different responses."
            )
        else:
            lines.append("Every instrument has at least one neighbour.")
        if weak:
            lines.append(
                f"{weak} instrument(s) have no neighbour carrying more than "
                f"{_WEAK_NEIGHBOUR:.0%} of their edge weight. A target in this "
                f"group that was not in the training set will borrow its "
                f"baseline from neighbours that are not very close, so treat "
                f"its prediction as indicative rather than priced."
            )
        return "\n".join(lines)

    @classmethod
    def _worst_section(cls, state: HybridState) -> str:
        """
        Name the instruments with the weakest best neighbour.

        Named individually rather than counted, because the action this
        prompts is per instrument: look at it, decide whether its
        attributes are wrong or whether it is genuinely an outlier.

        Parameters
        ----------
        state
            The fitted state.

        Returns
        -------
        str
            Markdown text.
        """
        best = cls._best_weight_per_node(state.graph)
        identifiers = (
            *state.universe.elementary_ids[: state.graph.n_nodes],
            *state.universe.target_ids,
        )
        order = np.argsort(best, kind="stable")[:_N_WORST]

        rows = [
            "## Worst-covered instruments",
            "",
            f"The {min(_N_WORST, best.size)} instruments whose single best "
            f"neighbour carries the least weight.",
            "",
            "| instrument | role | best neighbour weight | neighbours |",
            "| --- | --- | --- | --- |",
        ]
        neighbours = state.graph.degrees() - 1
        for node in order:
            name = identifiers[node] if node < len(identifiers) else f"node {node}"
            role = "target" if state.graph.is_target[node] else "elementary"
            rows.append(f"| `{name}` | {role} | {best[node]:.4f} | {int(neighbours[node])} |")
        return "\n".join(rows)

    @staticmethod
    def _write_figures(graph: SparseGraphState, context: ReportContext) -> tuple[Path, ...]:
        """
        Render and save the three graph figures.

        Parameters
        ----------
        graph
            The fitted graph.
        context
            Where to write, and in what format.

        Returns
        -------
        tuple of Path
            The saved figures.
        """
        factories = {
            "graph_degrees": node_degree_figure,
            "graph_edge_weights": edge_weight_figure,
            "graph_neighbour_similarity": neighbour_similarity_figure,
        }
        return tuple(
            save_figure(
                factory(graph),
                context.directory,
                stem,
                figure_format=context.figure_format,
                dpi=context.figure_dpi,
            )
            for stem, factory in factories.items()
        )

    @staticmethod
    def _figures_section(paths: Sequence[Path]) -> str:
        """
        Link the figures from the page.

        Parameters
        ----------
        paths
            The saved figures.

        Returns
        -------
        str
            Markdown text.
        """
        # Linked by filename rather than by absolute path, so the page
        # survives the directory being copied or served from elsewhere.
        return "\n".join(["## Figures", "", *(f"![{path.stem}]({path.name})" for path in paths)])

    @staticmethod
    def _best_weight_per_node(graph: SparseGraphState) -> np.ndarray:
        """
        Find each node's largest edge weight, ignoring its self-loop.

        The self-loop is excluded because every node has one and it says
        nothing about that node's relationship to the rest of the book --
        including it would give an isolated instrument a perfect score.

        Parameters
        ----------
        graph
            The fitted graph.

        Returns
        -------
        numpy.ndarray
            One value per node, zero where a node has no neighbour.
        """
        rows, columns = graph.indices[:, 0], graph.indices[:, 1]
        off_diagonal = rows != columns
        best = np.zeros(graph.n_nodes, dtype=np.float64)
        np.maximum.at(
            best,
            rows[off_diagonal],
            np.asarray(graph.values, dtype=np.float64)[off_diagonal],
        )
        return best
