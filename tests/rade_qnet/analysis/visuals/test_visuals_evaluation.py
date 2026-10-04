"""
Tests for the evaluation figures.

These functions take data and return a figure. They save nothing, show
nothing and configure nothing global, which is what makes them assertable:
a test can read the bar heights, the ordering and the axis labels without
rendering a file and comparing pixels.

The ordering and content assertions are the substantive ones. A drift chart
that omits its most severe bar, or an error chart whose buckets are cut on
the wrong variable, is not a cosmetic bug -- it is a chart that tells a
reader something untrue while looking entirely normal.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.evaluation import (
    drift_figure,
    error_by_bucket_figure,
    residual_against_prediction_figure,
)
from src.rade_qnet.core.lifecycle.errors import ContractError


def bar_heights(figure: Figure) -> list[float]:
    """
    Read the bar heights off a vertical bar chart.

    Parameters
    ----------
    figure
        The figure.

    Returns
    -------
    list of float
        One height per bar, in drawing order.
    """
    return [patch.get_height() for patch in figure.axes[0].patches]


def bar_widths(figure: Figure) -> list[float]:
    """
    Read the bar lengths off a horizontal bar chart.

    Parameters
    ----------
    figure
        The figure.

    Returns
    -------
    list of float
        One length per bar, bottom to top.
    """
    return [patch.get_width() for patch in figure.axes[0].patches]


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


@pytest.fixture
def paired():
    """
    Produce predictions and targets with error that grows with the target.

    Deliberately heteroscedastic, so a figure that genuinely cuts by
    observed value shows a rising profile and one that does not shows a
    flat one. A homoscedastic fixture would make both look the same.

    Returns
    -------
    tuple
        Predictions and targets.
    """
    rng = np.random.default_rng(0)
    targets = np.linspace(-10.0, 10.0, 500)
    predictions = targets + rng.normal(size=targets.size) * (1.0 + np.abs(targets))
    return predictions, targets


class TestErrorByBucket:
    """Where the error lives, which a single mean cannot say."""

    def test_one_bar_per_bucket(self, paired):
        """The chart shows what it was asked for."""
        figure = error_by_bucket_figure(*paired, buckets=10)

        assert len(bar_heights(figure)) == 10

    def test_the_profile_follows_the_observed_value(self, paired):
        """
        Which is the whole point of cutting the error up.

        A model can be excellent on the typical case and useless on the
        tail, and the headline metric is the same as one that is mediocre
        everywhere.
        """
        heights = bar_heights(error_by_bucket_figure(*paired, buckets=10))

        # Error grows with |target|, so the outer buckets beat the middle.
        assert heights[0] > heights[len(heights) // 2]
        assert heights[-1] > heights[len(heights) // 2]

    def test_buckets_hold_equal_counts(self):
        """
        Cut on rank rather than value, because a P&L series is skewed.

        Cutting on value would put almost every observation in one bucket
        and draw nine bars from a handful of points each.
        """
        skewed = np.concatenate([np.zeros(90), np.linspace(1.0, 100.0, 10)])
        figure = error_by_bucket_figure(skewed, skewed, buckets=10)

        # Every bucket drew a bar, which a value-cut would not manage.
        assert len(bar_heights(figure)) == 10

    def test_the_overall_mean_is_drawn_for_reference(self, paired):
        """
        So a bucket's distance from the headline number is readable.

        Without it the chart shows which bucket is worst and not whether
        any of them is unusual.
        """
        figure = error_by_bucket_figure(*paired)

        assert len(figure.axes[0].lines) >= 1

    def test_too_few_values_to_bucket_is_refused(self):
        """Rather than drawing single observations as though they were groups."""
        with pytest.raises(ContractError, match=r"looks like structure and is noise"):
            error_by_bucket_figure(np.zeros(4), np.zeros(4), buckets=10)

    def test_mismatched_lengths_are_refused(self):
        """
        Because NumPy would broadcast rather than complain.

        An ``(n, 1)`` against an ``(n,)`` expands into an ``(n, n)``, and
        the chart drawn from that looks entirely plausible.
        """
        with pytest.raises(ContractError, match=r"equal counts are required"):
            error_by_bucket_figure(np.zeros(20), np.zeros(19))


class TestResidualAgainstPrediction:
    """Whether the error grows with the size of the prediction."""

    def test_one_point_per_sample(self, paired):
        """Nothing is dropped or aggregated."""
        figure = residual_against_prediction_figure(*paired)
        collection = figure.axes[0].collections[0]

        assert collection.get_offsets().shape[0] == paired[0].size

    def test_the_zero_line_is_drawn(self, paired):
        """
        Because the eye reads a residual plot by deviation from zero.

        Without the line, a reader uses the lowest point as the reference
        and concludes the model is biased when it is not.
        """
        figure = residual_against_prediction_figure(*paired)

        assert any(line.get_ydata()[0] == 0.0 for line in figure.axes[0].lines)

    def test_the_axes_are_labelled_with_the_direction(self, paired):
        """
        Spelled out rather than labelled "residual".

        The sign convention is one nobody agrees on, so the label states
        it: ``observed - predicted``.
        """
        figure = residual_against_prediction_figure(*paired)

        assert "observed - predicted" in figure.axes[0].get_ylabel()

    def test_empty_input_is_refused(self):
        """A blank chart reads as a rendering fault."""
        with pytest.raises(ContractError, match=r"nothing to plot"):
            residual_against_prediction_figure(np.array([]), np.array([]))


class TestDrift:
    """Ranking features by how far their distribution has moved."""

    def test_one_bar_per_feature(self):
        """Nothing is aggregated away."""
        metrics = {"a": {"psi": 0.05}, "b": {"psi": 0.9}, "c": {"psi": 0.3}}

        assert len(bar_widths(drift_figure(metrics))) == 3

    def test_the_worst_feature_is_at_the_top(self):
        """
        Ranked, so the reader's eye lands on the worst one.

        Bottom to top is matplotlib's drawing order, so the last label is
        the one at the top of the page.
        """
        metrics = {"calm": {"psi": 0.05}, "moved": {"psi": 0.9}, "mild": {"psi": 0.3}}

        assert labels(drift_figure(metrics))[-1] == "moved"

    def test_an_infinite_value_is_still_drawn(self):
        """
        The single most important bar must not be the one omitted.

        An infinite measure means a feature that was constant in training
        and is not now -- total drift -- and matplotlib silently drops a
        bar of infinite height.
        """
        metrics = {"a": {"psi": 0.1}, "total": {"psi": float("inf")}}
        figure = drift_figure(metrics)
        widths = bar_widths(figure)

        assert all(np.isfinite(width) for width in widths)
        assert max(widths) > 0.1

    def test_capping_is_stated_in_the_title(self):
        """So a reader is never shown a capped bar without being told."""
        metrics = {"a": {"psi": 0.1}, "total": {"psi": float("inf")}}

        assert "capped" in drift_figure(metrics).axes[0].get_title()

    def test_nothing_is_capped_when_nothing_is_infinite(self):
        """The ordinary chart carries no caveat it does not need."""
        metrics = {"a": {"psi": 0.1}, "b": {"psi": 0.9}}

        assert "capped" not in drift_figure(metrics).axes[0].get_title()

    def test_the_threshold_is_drawn(self):
        """Because a stability index means nothing to a reader without one."""
        figure = drift_figure({"a": {"psi": 0.1}})

        assert len(figure.axes[0].lines) >= 1

    def test_another_measure_can_be_ranked_on(self):
        """The index is the default, not the only choice."""
        metrics = {"a": {"psi": 0.9, "ks": 0.01}, "b": {"psi": 0.01, "ks": 0.9}}

        assert labels(drift_figure(metrics, measure="ks", threshold=0.2))[-1] == "b"

    def test_an_absent_measure_names_the_available_ones(self):
        """So a typo is diagnosable from the message alone."""
        with pytest.raises(ContractError, match=r"available measures are"):
            drift_figure({"a": {"psi": 0.1}}, measure="wasserstein")

    def test_empty_metrics_are_refused(self):
        """An empty chart reads as a rendering fault rather than no data."""
        with pytest.raises(ContractError, match=r"no drift metrics to draw"):
            drift_figure({})
