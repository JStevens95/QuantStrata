"""
Tests for the search figures.

The substantive assertions are about honesty rather than appearance: that a
failed trial leaves a gap rather than being compacted away, that the running
best is monotone in the right direction, and that a parameter which did
nothing is not ranked as though it did something. Each of those, got wrong,
produces a chart that looks entirely normal and says something untrue.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.tuning import (
    parallel_coordinates_figure,
    parameter_importance_figure,
    trial_history_figure,
)
from src.rade_qnet.core.runtime.errors import ContractError

#: Enough trials to clear the minimum the influence chart insists on, and
#: few enough to read in a failure message.
N_TRIALS = 12


def labels(figure: Figure) -> list[str]:
    """
    Read the category labels off a horizontal bar chart.

    Returns
    -------
    list of str
        Bottom to top, so the last entry is at the top of the page.
    """
    return [text.get_text() for text in figure.axes[0].get_yticklabels()]


def bar_widths(figure: Figure) -> list[float]:
    """
    Read the bar lengths off a horizontal bar chart.

    Returns
    -------
    list of float
        One length per bar, bottom to top.
    """
    return [patch.get_width() for patch in figure.axes[0].patches]


@pytest.fixture
def search():
    """
    Produce a search in which one parameter matters and one does not.

    The objective is a clean function of ``lr`` alone, with ``noise``
    varying freely and affecting nothing. That separation is what makes the
    influence assertions meaningful: a fixture in which both parameters
    mattered could not distinguish a correct ranking from an arbitrary one.

    Returns
    -------
    tuple
        The proposals and their objectives.
    """
    rng = np.random.default_rng(0)
    overrides = [
        {"lr": float(value), "noise": float(rng.uniform(0.0, 1.0))}
        for value in np.linspace(0.001, 0.1, N_TRIALS)
    ]
    objectives = [proposal["lr"] * 10.0 for proposal in overrides]
    return overrides, objectives


class TestTrialHistory:
    """Whether the search converged, or merely ran out of budget."""

    def test_every_scored_trial_is_a_point(self, search):
        """Nothing is aggregated away."""
        _, objectives = search
        figure = trial_history_figure(objectives)
        points = figure.axes[0].collections[0].get_offsets()

        assert points.shape[0] == N_TRIALS

    def test_the_running_best_only_improves(self, search):
        """
        Which is what makes its shape readable.

        A line still descending at the last trial means the budget was the
        binding constraint, and that conclusion depends on the line being
        monotone.
        """
        _, objectives = search
        figure = trial_history_figure(objectives, direction="minimise")
        step = next(line for line in figure.axes[0].lines if line.get_drawstyle() != "default")
        values = np.asarray(step.get_ydata(), dtype=np.float64)

        assert np.all(np.diff(values) <= 0.0)

    def test_the_running_best_rises_when_maximising(self, search):
        """One implementation, honouring whichever way the metric runs."""
        _, objectives = search
        figure = trial_history_figure(objectives, direction="maximise")
        step = next(line for line in figure.axes[0].lines if line.get_drawstyle() != "default")
        values = np.asarray(step.get_ydata(), dtype=np.float64)

        assert np.all(np.diff(values) >= 0.0)

    def test_a_failed_trial_leaves_a_gap(self, search):
        """
        Rather than being compacted away.

        A search with failures and a clean-looking curve is a different
        object from one with none, and compacting the x-axis hides the
        difference entirely.
        """
        _, objectives = search
        with_failures = list(objectives)
        with_failures[3] = None
        with_failures[7] = None

        figure = trial_history_figure(with_failures)
        points = figure.axes[0].collections[0].get_offsets()

        assert points.shape[0] == N_TRIALS - 2
        # The x positions still span the full trial range, so the gaps show.
        assert float(points[:, 0].max()) == float(N_TRIALS - 1)

    def test_the_failure_count_is_stated(self, search):
        """Because a reader should not have to count the gaps."""
        _, objectives = search
        with_failures = list(objectives)
        with_failures[3] = None

        assert "1 failed" in trial_history_figure(with_failures).axes[0].get_xlabel()

    def test_a_clean_search_carries_no_failure_clause(self, search):
        """No caveat where none is needed."""
        _, objectives = search

        assert trial_history_figure(objectives).axes[0].get_xlabel() == "Trial"

    def test_an_empty_search_is_refused(self):
        """A blank chart reads as a rendering fault."""
        with pytest.raises(ContractError, match=r"no trials to draw"):
            trial_history_figure([])

    def test_a_search_in_which_everything_failed_is_refused(self):
        """
        Rather than drawn as an empty set of axes.

        Blank axes read as a search that found nothing good, which is a
        very different statement from a search that produced nothing.
        """
        with pytest.raises(ContractError, match=r"none of the 3 trial"):
            trial_history_figure([None, None, None])


class TestParameterImportance:
    """Which knobs mattered."""

    def test_the_influential_parameter_ranks_highest(self, search):
        """The objective is a clean function of ``lr`` and ignores ``noise``."""
        overrides, objectives = search

        assert labels(parameter_importance_figure(overrides, objectives))[-1] == "lr"

    def test_a_perfect_relationship_scores_one(self, search):
        """
        Rank correlation, so a monotone non-linear effect is still perfect.

        A linear coefficient would report a log-scaled learning rate as
        only partially influential, which is the case this measure exists
        to handle.
        """
        overrides, objectives = search
        widths = bar_widths(parameter_importance_figure(overrides, objectives))

        assert max(widths) == pytest.approx(1.0)

    def test_a_monotone_nonlinear_effect_still_scores_one(self):
        """The reason ranks are used rather than values."""
        overrides = [{"lr": float(value)} for value in np.linspace(1e-4, 1e-1, N_TRIALS)]
        objectives = [float(np.log(proposal["lr"])) for proposal in overrides]

        widths = bar_widths(parameter_importance_figure(overrides, objectives))
        assert max(widths) == pytest.approx(1.0)

    def test_a_categorical_parameter_is_skipped(self, search):
        """
        Rather than encoded as integers.

        Correlating against arbitrary integers produces a number that
        depends entirely on the order the values were listed in.
        """
        overrides, objectives = search
        with_categorical = [{**proposal, "sampler": "adam"} for proposal in overrides]

        assert "sampler" not in labels(
            parameter_importance_figure(with_categorical, objectives)
        )

    def test_a_constant_parameter_is_skipped(self, search):
        """A parameter that never varied cannot have mattered."""
        overrides, objectives = search
        with_constant = [{**proposal, "fixed": 1.0} for proposal in overrides]

        assert "fixed" not in labels(parameter_importance_figure(with_constant, objectives))

    def test_too_few_scored_trials_is_refused(self):
        """
        Below the minimum the coefficient is decided by two points.

        A chart built on four trials would be read with the confidence of
        one built on four hundred, which is how a search gets over-read.
        """
        overrides = [{"lr": float(index)} for index in range(4)]

        with pytest.raises(ContractError, match=r"at least 5"):
            parameter_importance_figure(overrides, [0.1, 0.2, 0.3, 0.4])

    def test_mismatched_lengths_are_refused(self):
        """Each trial needs both a proposal and an objective."""
        with pytest.raises(ContractError, match=r"cannot be paired"):
            parameter_importance_figure([{"lr": 1.0}], [0.1, 0.2])

    def test_an_entirely_categorical_search_is_refused(self):
        """With a message that says why, rather than drawing nothing."""
        overrides = [{"sampler": "adam"} for _ in range(N_TRIALS)]

        with pytest.raises(ContractError, match=r"no searched parameter is numeric"):
            parameter_importance_figure(overrides, [float(i) for i in range(N_TRIALS)])


class TestParallelCoordinates:
    """What the good trials had in common."""

    def test_one_line_per_scored_trial(self, search):
        """Nothing is aggregated away."""
        overrides, objectives = search
        figure = parallel_coordinates_figure(overrides, objectives)

        assert len(figure.axes[0].lines) == N_TRIALS

    def test_failed_trials_are_not_drawn(self, search):
        """
        A failed trial has no objective.

        So it has no colour and no position on the scale the figure is
        read by.
        """
        overrides, objectives = search
        with_failures = list(objectives)
        with_failures[0] = None

        figure = parallel_coordinates_figure(overrides, with_failures)
        assert len(figure.axes[0].lines) == N_TRIALS - 1

    def test_each_axis_is_scaled_to_its_own_range(self, search):
        """
        Because the parameters have incomparable units.

        A learning rate of 0.001 and a hidden size of 256 on one axis would
        render the first as a flat line at zero.
        """
        overrides, objectives = search
        figure = parallel_coordinates_figure(overrides, objectives)
        drawn = np.concatenate([line.get_ydata() for line in figure.axes[0].lines])

        assert float(drawn.min()) == pytest.approx(0.0)
        assert float(drawn.max()) == pytest.approx(1.0)

    def test_one_axis_per_numeric_parameter(self, search):
        """Categoricals and constants are excluded, as in the bar chart."""
        overrides, objectives = search
        figure = parallel_coordinates_figure(overrides, objectives)

        assert [text.get_text() for text in figure.axes[0].get_xticklabels()] == [
            "lr",
            "noise",
        ]

    def test_a_colour_bar_is_drawn(self, search):
        """Because the question is read from the colour, not the geometry."""
        overrides, objectives = search
        figure = parallel_coordinates_figure(overrides, objectives)

        assert len(figure.axes) == 2

    def test_fewer_than_two_axes_is_refused(self):
        """With one it is a strip chart, and with none it is blank."""
        overrides = [{"lr": float(index)} for index in range(N_TRIALS)]

        with pytest.raises(ContractError, match=r"at least two numeric parameters"):
            parallel_coordinates_figure(overrides, [float(i) for i in range(N_TRIALS)])

    def test_mismatched_lengths_are_refused(self):
        """Each trial needs both a proposal and an objective."""
        with pytest.raises(ContractError, match=r"cannot be paired"):
            parallel_coordinates_figure([{"a": 1.0, "b": 2.0}], [0.1, 0.2])
