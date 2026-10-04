"""Tests for the graph diagnostics report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.rade_qnet.analysis.reports.base import ReportContext
from src.rade_qnet.core.spec.data import (
    ModelSourceSpec,
    ReductionSpec,
    SequenceSpec,
    TransformsSpec,
)
from src.rade_qnet.models.hybrid_gnn_rnn.data import HybridDataModule
from src.rade_qnet.models.hybrid_gnn_rnn.features.graph import SparseGraphState
from src.rade_qnet.models.hybrid_gnn_rnn.reports import GRAPH_FILENAME, HybridGraphReport
from src.rade_qnet.testkit.fixtures import StandardisingState

FIXTURE = Path("tests/fixtures/rade_qnet/golden/hybrid_gnn_rnn/input")

#: Window length. Short, because this report never looks at a window.
SEQUENCE_LENGTH = 4


@pytest.fixture(scope="module")
def state():
    """Fit the fixture's cluster once and return the resulting state."""
    spec = ModelSourceSpec(
        params={"directory": str(FIXTURE), "graph": {"n_neighbours": 5}},
        transforms=TransformsSpec(
            reduction=ReductionSpec(method="basis_selection"),
            sequence=SequenceSpec(length=SEQUENCE_LENGTH),
        ),
    )
    module = HybridDataModule()
    raw = module.load(spec)
    splits = module.split(raw, spec)
    return module.fit_state(raw, spec, train_indices=splits.train)


def context(tmp_path: Path, bundle_state) -> ReportContext:
    """
    Build a report context around a state.

    Only the bundle's state is read, so the rest of the bundle is a
    placeholder -- which is itself worth noting: the report re-renders
    from a saved run without the data and without the weights.

    Parameters
    ----------
    tmp_path
        Where to write.
    bundle_state
        The fitted state to report on.

    Returns
    -------
    ReportContext
        The context.
    """

    class _Bundle:
        state = bundle_state

    return ReportContext(bundle=_Bundle(), directory=tmp_path)  # type: ignore[arg-type]


class TestRendering:
    """What the report writes."""

    def test_a_page_and_three_figures(self, tmp_path: Path, state) -> None:
        """The Markdown page first, so a reader opening the list finds it."""
        paths = HybridGraphReport().render(context(tmp_path, state))
        assert paths[0].name == GRAPH_FILENAME
        assert len(paths) == 4
        assert all(path.is_file() for path in paths)

    def test_the_page_reports_the_graph_dimensions(self, tmp_path: Path, state) -> None:
        """
        Counted from the state rather than restated from the spec.

        A report that echoed the configuration would agree with it even
        when the graph did not.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert f"| instruments | {state.graph.n_nodes} |" in page
        assert f"| edges (self-loops included) | {state.graph.n_edges} |" in page

    def test_the_quoted_neighbour_count_excludes_the_self_loop(self, tmp_path: Path, state) -> None:
        """
        The self-loop is excluded from the quoted count.

        "Five neighbours" meaning "four and itself" survives into a
        conversation with a desk, and is wrong there.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        expected = int(np.median(state.graph.degrees() - 1))
        assert f"| neighbours per instrument (median) | {expected} |" in page

    def test_the_worst_covered_instruments_are_named(self, tmp_path: Path, state) -> None:
        """
        Named, not counted.

        The action this prompts is per instrument -- look at it, decide
        whether its attributes are wrong or whether it is genuinely an
        outlier -- and a count supports none of that.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert "## Worst-covered instruments" in page
        universe = json.loads((FIXTURE / "universe.json").read_text())
        named = [name for name in universe["target_ids"] if f"`{name}`" in page]
        assert named

    def test_the_figures_are_linked_by_filename(self, tmp_path: Path, state) -> None:
        """
        So the page survives the directory being copied or served elsewhere.

        An absolute path would work on the machine that produced it and
        nowhere else, which is the opposite of what a report is for.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert "![graph_degrees](graph_degrees.png)" in page
        assert str(tmp_path) not in page


class TestCoverageFindings:
    """The two numbers that decide whether the graph is usable."""

    def test_a_healthy_graph_says_so(self, tmp_path: Path, state) -> None:
        """
        An explicit verdict, not a table the reader has to interpret.

        A report that only prints numbers leaves the judgement to
        whoever reads it, and the whole value here is that the person
        who knows the threshold is the person writing the report.
        """
        HybridGraphReport().render(context(tmp_path, state))
        page = (tmp_path / GRAPH_FILENAME).read_text()
        assert "Every instrument has at least one neighbour." in page

    def test_the_self_loop_does_not_count_as_coverage(self) -> None:
        """
        An isolated instrument scores zero, not one.

        Every node has a self-loop, and its weight says nothing about
        that node's relationship to the rest of the book. Counting it
        would give the instruments most in need of attention a perfect
        score.
        """
        only_self = SparseGraphState(
            indices=np.array([[0, 0], [1, 1]], dtype=np.int64),
            values=np.ones(2, dtype=np.float32),
            n_nodes=2,
            is_target=np.array([False, True]),
        )
        best = HybridGraphReport._best_weight_per_node(only_self)
        assert best.tolist() == [0.0, 0.0]


class TestSkipping:
    """What happens when the report cannot apply."""

    def test_another_model_s_state_is_skipped_with_a_reason(self, tmp_path: Path) -> None:
        """
        A deliberate skip, not an incidental one.

        The pipeline turns any exception into a skipped report, so an
        accidental ``AttributeError`` would also be survivable -- but it
        would leave a stack trace in a log instead of a sentence anyone
        can act on.
        """
        outcome = HybridGraphReport().render_safely(
            context(tmp_path, StandardisingState.fit(np.zeros((4, 2))))
        )
        assert not outcome.succeeded
        assert "HybridState" in str(outcome.skipped_reason)
