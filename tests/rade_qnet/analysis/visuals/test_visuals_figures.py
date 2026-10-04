"""
Tests for the shared figure factories.

One rule governs every function here: a figure factory returns a figure and
writes nothing. That is what lets the same function serve a report writer, a
notebook, a dashboard and these tests -- the moment a plotting function also
saved its output, each of those callers would need its own variant.

So these tests assert on figure *contents*: how many lines were drawn, what
data they carry, whether a gap is a gap. Asserting that a file appeared would
test the wrong thing and would need a temporary directory to do it.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.figures import (
    metric_comparison_figure,
    prediction_scatter_figure,
    residual_histogram_figure,
    training_curve_figure,
)
from src.rade_qnet.core.lifecycle.errors import ContractError

#: Each factory, pre-bound to a valid minimal call, so the shared-contract
#: tests below can be written once rather than four times.
FACTORIES = {
    "training_curve": lambda: training_curve_figure([1.0, 0.5]),
    "prediction_scatter": lambda: prediction_scatter_figure(
        np.array([1.0, 2.0]), np.array([1.1, 2.1])
    ),
    "residual_histogram": lambda: residual_histogram_figure(
        np.array([1.0, 2.0]), np.array([1.1, 2.1])
    ),
    "metric_comparison": lambda: metric_comparison_figure({"mae": 0.2}),
}

#: The factories whose horizontal axis is numeric, and so needs a label. The
#: metric chart is excluded: its ticks already read ``mae`` and ``rmse``.
CONTINUOUS_X_FACTORIES = {
    name: factory for name, factory in FACTORIES.items() if name != "metric_comparison"
}


def _axes(figure):
    """
    Return a figure's single axes.

    Parameters
    ----------
    figure
        The figure.

    Returns
    -------
    matplotlib.axes.Axes
        The axes.
    """
    return figure.axes[0]


class TestTheSharedContract:
    """Every factory behaves the same way about what it returns."""

    @pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
    def test_a_figure_is_returned(self, factory):
        """Not a path, not ``None``, not an axes."""
        assert isinstance(factory(), Figure)

    @pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
    def test_nothing_is_written(self, factory, tmp_path, monkeypatch):
        """
        The rule that makes these reusable.

        Checked by running from an empty directory and confirming it stays
        empty, which catches a stray ``savefig`` with a relative path.
        """
        monkeypatch.chdir(tmp_path)
        factory()
        assert list(tmp_path.iterdir()) == []

    @pytest.mark.parametrize("factory", FACTORIES.values(), ids=FACTORIES)
    def test_every_figure_is_titled_and_has_a_value_axis(self, factory):
        """
        A figure in a report with no title or value axis is decoration.

        Someone reading it six months later has no way to know what it
        shows. The x-axis is excluded deliberately: the metric chart's
        categorical ticks already read ``mae``, ``rmse``, ``bias``, and
        labelling that axis "metric" would be noise.
        """
        axes = _axes(factory())
        assert axes.get_title()
        assert axes.get_ylabel()

    @pytest.mark.parametrize("factory", CONTINUOUS_X_FACTORIES.values(), ids=CONTINUOUS_X_FACTORIES)
    def test_a_continuous_horizontal_axis_is_labelled(self, factory):
        """
        A numeric axis is not self-describing.

        "Epoch", "observed" and "residual" each have to be stated, or the
        figure cannot be read without the code that made it.
        """
        assert _axes(factory()).get_xlabel()


class TestTrainingCurve:
    """The plot every run produces, for every engine."""

    def test_the_training_series_is_plotted(self):
        """One line when there is no validation split."""
        axes = _axes(training_curve_figure([1.0, 0.7, 0.5]))
        assert len(axes.lines) == 1

    def test_the_validation_series_is_plotted_beside_it(self):
        """Two lines when there is."""
        axes = _axes(training_curve_figure([1.0, 0.7], [1.1, 0.8]))
        assert len(axes.lines) == 2

    def test_the_plotted_values_are_the_values_given(self):
        """
        The test that would catch an off-by-one in the epoch axis.

        Which is otherwise invisible: a curve shifted by one epoch looks
        exactly like a curve.
        """
        axes = _axes(training_curve_figure([1.0, 0.7, 0.5]))
        assert list(axes.lines[0].get_ydata()) == [1.0, 0.7, 0.5]

    def test_epochs_are_numbered_from_zero(self):
        """Matching ``EpochRecord.epoch``, which is zero-based."""
        axes = _axes(training_curve_figure([1.0, 0.7, 0.5]))
        assert list(axes.lines[0].get_xdata()) == [0, 1, 2]

    def test_a_missing_value_becomes_a_gap_not_a_shift(self):
        """
        ``None`` renders as ``NaN``, so matplotlib leaves a hole.

        Dropping the point would shift every later epoch left and silently
        misalign the curve against the epoch axis -- a wrong plot that looks
        right.
        """
        axes = _axes(training_curve_figure([1.0, None, 0.5]))
        values = axes.lines[0].get_ydata()
        assert len(values) == 3
        assert np.isnan(values[1])

    def test_the_best_epoch_is_marked(self):
        """
        A vertical rule marks the epoch whose weights were kept.

        That makes the gap between "where training stopped" and "which
        weights were kept" visible, which a loss curve otherwise hides
        entirely.
        """
        axes = _axes(training_curve_figure([1.0, 0.7, 0.9], best_epoch=1))
        assert any("best epoch" in str(text.get_text()) for text in axes.get_legend().get_texts())

    def test_mismatched_series_lengths_are_rejected(self):
        """
        Two curves of different lengths cannot share an epoch axis.

        It would mean an epoch's two losses were recorded against different
        indices, so the plot would be comparing different epochs.
        """
        with pytest.raises(ContractError, match="different lengths"):
            training_curve_figure([1.0, 0.7], [1.1])

    def test_an_empty_history_still_produces_a_figure(self):
        """
        A run that failed before its first epoch still gets a report.

        Reports are never load-bearing, so a factory that raised here would
        turn a training failure into two failures.
        """
        assert isinstance(training_curve_figure([]), Figure)


class TestPredictionScatter:
    """Diagnostic because of the parity line, not despite it."""

    def test_the_points_are_plotted(self):
        """One collection holding every point."""
        figure = prediction_scatter_figure(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]))
        assert len(_axes(figure).collections[0].get_offsets()) == 3

    def test_observed_is_the_horizontal_axis(self):
        """
        The convention that makes the parity line readable.

        Reversed, a shallow cloud would mean over-confidence rather than
        regression to the mean, and the reader would draw the opposite
        conclusion.
        """
        figure = prediction_scatter_figure(np.array([1.0]), np.array([2.0]))
        assert _axes(figure).get_xlabel() == "observed"

    def test_the_parity_line_is_drawn(self):
        """
        What makes the plot diagnostic rather than decorative.

        A cloud tilted shallower than the line is the signature of a model
        that has regressed towards the mean, which no error metric reveals.
        """
        figure = prediction_scatter_figure(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert len(_axes(figure).lines) == 1

    def test_the_axes_share_limits_and_aspect(self):
        """
        So a visual departure from parity is a real departure.

        With independent scaling, any cloud can be made to look like it sits
        on the line.
        """
        figure = prediction_scatter_figure(np.array([0.0, 1.0]), np.array([0.0, 10.0]))
        axes = _axes(figure)
        assert axes.get_xlim() == axes.get_ylim()

    def test_a_column_vector_is_accepted(self):
        """
        Model output commonly arrives as ``(n, 1)``.

        Requiring the caller to flatten would put the same reshape in every
        report.
        """
        figure = prediction_scatter_figure(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))
        assert len(_axes(figure).collections[0].get_offsets()) == 2

    def test_mismatched_lengths_are_rejected(self):
        """Predictions and targets from different rows cannot be plotted."""
        with pytest.raises(ContractError, match="different lengths"):
            prediction_scatter_figure(np.zeros(3), np.zeros(4))


class TestResidualHistogram:
    """Where the bias becomes visible rather than merely reported."""

    def test_the_residual_is_predicted_minus_observed(self):
        """
        Sign convention, stated in the axis label and tested here.

        Reversed, a model that over-predicts would appear to under-predict.
        """
        figure = residual_histogram_figure(np.array([3.0, 3.0]), np.array([1.0, 1.0]))
        assert "predicted - observed" in _axes(figure).get_xlabel()

    def test_zero_and_the_mean_are_both_marked(self):
        """
        The distance between them *is* the bias.

        Showing it makes a systematic error immediately visible as
        systematic rather than as noise.
        """
        figure = residual_histogram_figure(np.array([3.0, 3.0]), np.array([1.0, 1.0]))
        assert len(_axes(figure).lines) >= 2

    def test_the_mean_marker_reports_its_value(self):
        """So the reader gets the number as well as the picture."""
        figure = residual_histogram_figure(np.array([3.0, 3.0]), np.array([1.0, 1.0]))
        labels = [text.get_text() for text in _axes(figure).get_legend().get_texts()]
        assert any("+2" in label for label in labels)

    def test_mismatched_lengths_are_rejected(self):
        """As everywhere else."""
        with pytest.raises(ContractError, match="different lengths"):
            residual_histogram_figure(np.zeros(3), np.zeros(4))


class TestMetricComparison:
    """A headline beside its reference point."""

    def test_one_bar_per_metric(self):
        """Without a baseline, a single series."""
        figure = metric_comparison_figure({"mae": 0.2, "rmse": 0.3})
        assert len(_axes(figure).containers) == 1

    def test_a_baseline_adds_a_second_series(self):
        """
        Which is the whole point of the figure.

        A mean absolute error of 0.004 is excellent or useless depending
        entirely on the target's scale.
        """
        figure = metric_comparison_figure({"mae": 0.2}, {"mae": 0.5})
        assert len(_axes(figure).containers) == 2

    def test_metrics_are_ordered_consistently(self):
        """
        Sorted, so two runs' figures can be compared side by side.

        Dictionary order would make the bars move between runs.
        """
        figure = metric_comparison_figure({"rmse": 0.3, "bias": 0.1, "mae": 0.2})
        labels = [text.get_text() for text in _axes(figure).get_xticklabels()]
        assert labels == ["bias", "mae", "rmse"]

    def test_a_missing_baseline_metric_does_not_shift_the_bars(self):
        """
        It plots as zero height instead.

        Dropping it would slide the remaining bars out of alignment with
        their labels, which mislabels every metric after the gap.
        """
        figure = metric_comparison_figure({"mae": 0.2, "r2": 0.9}, {"mae": 0.5})
        labels = [text.get_text() for text in _axes(figure).get_xticklabels()]
        assert labels == ["mae", "r2"]

    def test_a_zero_rule_is_drawn(self):
        """
        Bias and r2 are legitimately negative.

        Without a zero line, a small negative bar reads as a small positive
        one.
        """
        figure = metric_comparison_figure({"bias": -0.1})
        assert len(_axes(figure).lines) == 1

    def test_no_metrics_is_rejected(self):
        """An empty bar chart communicates nothing and looks like a bug."""
        with pytest.raises(ContractError, match="no metrics"):
            metric_comparison_figure({})
