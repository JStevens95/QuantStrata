"""
Tests for the cross-job figures.

These functions take data and return a figure. They save nothing, show
nothing and configure nothing global, which is what makes them assertable:
a test can read the bar count, the ordering and the axis labels without
rendering a file and comparing pixels.

The ordering assertions are the substantive ones. A ranking chart whose bars
are in the wrong order is not a cosmetic bug -- it is a chart that tells a
reader the opposite of the truth, and nothing about the image looks wrong.
"""

from __future__ import annotations

import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.jobset import (
    job_status_figure,
    metric_dispersion_figure,
    metric_ranking_figure,
    wall_time_figure,
)
from src.rade_qnet.core.runtime.errors import ContractError

#: Three jobs with clearly separated scores, so an ordering mistake cannot
#: hide behind near-ties.
SCORES = {"middle": 0.5, "best": 0.9, "worst": -0.3}


def labels(figure: Figure) -> list[str]:
    """
    Read the category labels off a horizontal bar chart.

    Bottom to top, which is matplotlib's drawing order -- so the *last*
    entry is the one at the top of the page.

    Parameters
    ----------
    figure
        The figure.

    Returns
    -------
    list of str
        The labels.
    """
    return [text.get_text() for text in figure.axes[0].get_yticklabels()]


class TestRanking:
    """The figure anybody opens first."""

    def test_there_is_one_bar_per_job(self):
        """Three jobs in, three bars out."""
        figure = metric_ranking_figure(SCORES, metric_name="r2")

        assert len(figure.axes[0].patches) == 3

    def test_the_best_job_is_drawn_at_the_top(self):
        """
        Where a reader looks first.

        Matplotlib draws the first entry at the bottom, so "best last" in
        the sorted list puts it first on the page. Getting this backwards
        produces a chart that reads as the exact inverse of the truth
        while looking entirely normal.
        """
        assert labels(metric_ranking_figure(SCORES, metric_name="r2"))[-1] == "best"

    def test_ascending_inverts_the_order(self):
        """
        For metrics where lower is better.

        No guess is made from the metric's name: `loss` and `r2` are both
        common, and a wrong guess silently inverts the conclusion.
        """
        assert labels(metric_ranking_figure(SCORES, ascending=True))[-1] == "worst"

    def test_highlighted_jobs_are_drawn_differently(self):
        """
        So a baseline cluster can be found among forty bars.

        Asserted through the bar colours rather than an eyeball.
        """
        figure = metric_ranking_figure(SCORES, highlight=["worst"])

        colours = {patch.get_facecolor() for patch in figure.axes[0].patches}

        assert len(colours) == 2

    def test_a_zero_reference_line_is_drawn(self):
        """
        "Worse than predicting the mean" is the most useful threshold.

        It is also invisible without a line, because a chart of negative
        r-squared values looks much like a chart of positive ones.
        """
        assert figure_has_vertical_line(metric_ranking_figure(SCORES), 0.0)

    def test_the_metric_name_reaches_the_axis(self):
        """A chart with an unlabelled axis is a chart nobody can cite."""
        figure = metric_ranking_figure(SCORES, metric_name="unexplained_pnl")

        assert figure.axes[0].get_xlabel() == "unexplained_pnl"

    def test_the_figure_grows_with_the_job_count(self):
        """
        Forty cluster names do not fit in a fixed height.

        A compressed chart overlaps its labels at exactly the size where
        the figure stops being readable and starts being decorative.
        """
        small = metric_ranking_figure({"a": 1.0})
        large = metric_ranking_figure({f"job{index}": float(index) for index in range(40)})

        assert large.get_figheight() > small.get_figheight()

    def test_the_height_is_capped(self):
        """
        A four-hundred-job set must still produce an openable file.

        Unbounded growth turns a slow set into a figure nobody can render.
        """
        figure = metric_ranking_figure({f"job{index}": float(index) for index in range(400)})

        assert figure.get_figheight() <= 20.0


class TestDispersion:
    """Whether the spread is the finding."""

    def test_a_histogram_is_drawn(self):
        """One patch per bin, over the supplied scores."""
        figure = metric_dispersion_figure(SCORES)

        assert figure.axes[0].patches

    def test_the_median_is_marked(self):
        """
        The median, not the mean.

        One catastrophic cluster drags a mean somewhere no job actually
        is, and a set with such a cluster is exactly when this is opened.
        """
        figure = metric_dispersion_figure({"a": 0.0, "b": 1.0, "c": 100.0})

        assert figure_has_vertical_line(figure, 1.0)

    def test_bins_do_not_exceed_the_sample_count(self):
        """
        A histogram with more bins than points draws noise as structure.

        The default bin count is right for forty jobs and badly wrong for
        three, which is a real size for a job set.
        """
        figure = metric_dispersion_figure({"a": 0.1, "b": 0.2}, bins=20)

        assert len(figure.axes[0].patches) <= 2

    def test_the_job_count_is_in_the_title(self):
        """
        How many jobs the distribution covers.

        A dispersion chart over four jobs means something different from
        one over four hundred, and the image alone does not say which.
        """
        assert "3 job(s)" in metric_dispersion_figure(SCORES).axes[0].get_title()


class TestStatus:
    """For the sets that did not all succeed."""

    def test_successes_and_failures_are_counted(self):
        """Two bars, whatever the mix."""
        figure = job_status_figure({"a": "succeeded", "b": "failed", "c": "failed"})

        heights = [patch.get_height() for patch in figure.axes[0].patches]

        assert heights == [1, 2]

    def test_an_unknown_status_counts_as_a_failure(self):
        """
        Not as a success.

        A status vocabulary that grows later should not silently start
        rendering new states as though everything were fine.
        """
        figure = job_status_figure({"a": "cancelled"})

        assert [patch.get_height() for patch in figure.axes[0].patches] == [0, 1]

    def test_the_counts_are_written_on_the_bars(self):
        """
        The whole content is two numbers.

        Making a reader estimate them off an axis would be a strange thing
        to do with a chart this simple.
        """
        figure = job_status_figure({"a": "succeeded", "b": "failed"})

        assert {text.get_text() for text in figure.axes[0].texts} == {"1"}

    def test_the_title_states_the_outcome(self):
        """So the figure can be read without its axes."""
        figure = job_status_figure({"a": "succeeded", "b": "failed"})

        assert figure.axes[0].get_title() == "1 of 2 job(s) succeeded"


class TestWallTime:
    """Where a set's time went."""

    def test_the_slowest_job_is_at_the_top(self):
        """The long pole is the thing being looked for."""
        figure = wall_time_figure({"quick": 1.0, "slow": 100.0, "medium": 10.0})

        assert labels(figure)[-1] == "slow"

    def test_only_the_slowest_jobs_are_drawn(self):
        """
        A four-hundred-job set's time profile is carried by its head.

        Drawing the tail makes the head unreadable, which inverts the
        point of the figure.
        """
        figure = wall_time_figure({f"job{index}": float(index) for index in range(50)}, top=5)

        assert len(figure.axes[0].patches) == 5

    def test_the_title_reports_the_total_and_the_truncation(self):
        """
        A reader seeing five bars needs to know fifty jobs ran.

        Otherwise the total and the bars do not reconcile, and the figure
        looks wrong rather than abbreviated.
        """
        title = (
            wall_time_figure({f"job{index}": 1.0 for index in range(50)}, top=5).axes[0].get_title()
        )

        assert "slowest 5 of 50" in title
        assert "50.0s total" in title

    def test_an_untruncated_set_says_so(self):
        """No "slowest N of N", which would imply something was hidden."""
        assert "by job" in wall_time_figure({"a": 1.0}).axes[0].get_title()


class TestEmptyInput:
    """A blank image reads as a rendering fault."""

    @pytest.mark.parametrize(
        "factory",
        [metric_ranking_figure, metric_dispersion_figure, job_status_figure, wall_time_figure],
    )
    def test_every_figure_refuses_an_empty_set(self, factory):
        """
        Raised rather than drawn.

        The real cause -- every job failed, so there are no metrics to
        compare -- is worth saying, and an empty chart does not say it.
        """
        with pytest.raises(ContractError):
            factory({})


def figure_has_vertical_line(figure: Figure, position: float) -> bool:
    """
    Report whether a vertical reference line is drawn at a position.

    Parameters
    ----------
    figure
        The figure.
    position
        The x coordinate.

    Returns
    -------
    bool
        True when a line is drawn there.
    """
    return any(
        line.get_xdata()[0] == pytest.approx(position)
        for line in figure.axes[0].lines
        if len(set(line.get_xdata())) == 1
    )
