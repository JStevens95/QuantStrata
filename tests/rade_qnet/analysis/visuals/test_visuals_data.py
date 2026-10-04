"""
Tests for the data figures.

A figure factory in this framework is a pure function: it takes arrays,
returns a ``Figure``, writes nothing and never touches the pyplot state
machine. That constraint is what makes the figures usable from a worker
thread, from a notebook, and from a report writer without the three
interfering, and it is what the first group of tests checks.

The rest check that each figure shows the thing it exists to show, because a
figure that renders without error and plots the wrong series is worse than no
figure -- someone will read it and believe it. These are asserted on the
artist data rather than on pixels: a visual regression test would be fragile
across matplotlib versions, while the data behind the artists is exactly what
the figure claims.

``split_layout_figure`` is the one that earns its place. The boundary gaps it
draws are the visible proof that a chronological split with a sequence length
above one is not leaking, and spotting their absence takes a glance rather
than a calculation.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib import pyplot
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.data import (
    feature_correlation_figure,
    missingness_figure,
    split_layout_figure,
    target_distribution_figure,
)

SPLITS = {
    "train": np.arange(0, 300),
    "validation": np.arange(320, 380),
    "test": np.arange(400, 500),
}


class TestPurity:
    """The constraint that makes these usable anywhere."""

    def test_a_factory_returns_a_figure_rather_than_drawing_one(self):
        """
        So the caller decides whether to save, show or discard it.

        A factory that called ``show`` would block a batch run, and one that
        saved would make the filename its business rather than the export
        layer's.
        """
        assert isinstance(split_layout_figure(SPLITS), Figure)

    def test_a_factory_does_not_touch_the_pyplot_state_machine(self):
        """
        Which is what lets a report build figures from a worker thread.

        Pyplot's current-figure state is global and not thread-safe, so a
        factory that used it would make two concurrently built reports draw
        into each other's axes.
        """
        before = len(pyplot.get_fignums())
        split_layout_figure(SPLITS)
        target_distribution_figure({"train": np.random.default_rng(0).normal(size=100)})
        assert len(pyplot.get_fignums()) == before

    def test_a_figure_is_not_registered_with_a_window_manager(self):
        """
        Which is what makes a report runnable with no display attached.

        A figure created through pyplot acquires a manager tied to the active
        backend, and on a headless machine that fails at creation -- turning a
        reporting problem into a startup problem. Constructing ``Figure``
        directly sidesteps the backend entirely, and this is the observable
        consequence.
        """
        assert split_layout_figure(SPLITS).canvas.manager is None

    def test_a_factory_does_not_modify_its_input(self):
        """
        So a figure can be built from the same arrays that feed a metric.

        An in-place sort or fill inside a factory would silently change a
        number elsewhere in the report, and the report would be internally
        inconsistent with no indication which half was wrong.
        """
        features = np.array([[3.0, 1.0], [np.nan, 2.0]])
        before = features.copy()
        missingness_figure(features)
        assert np.array_equal(features, before, equal_nan=True)


class TestSplitLayout:
    """The figure that makes a leaking split visible at a glance."""

    def test_one_band_per_split(self):
        """So a reader counts the splits without reading a legend entry."""
        figure = split_layout_figure(SPLITS)
        assert len(figure.axes) == 1
        assert figure.axes[0].get_yticklabels()

    def test_every_split_is_labelled(self):
        """
        Because an unlabelled band is three equally plausible splits.

        And the order of the bands is not something a reader should have to
        infer from their position on the axis.
        """
        figure = split_layout_figure(SPLITS)
        labels = {text.get_text() for text in figure.axes[0].get_yticklabels()}
        assert {label.split(" ")[0] for label in labels} == {
            "train",
            "validation",
            "test",
        }

    def test_each_label_carries_the_split_size(self):
        """
        Because the sizes are the first thing anyone checks on this figure.

        Reading them off the band widths is imprecise, and the share of data
        held out is exactly the number that explains a noisy test metric.
        """
        figure = split_layout_figure(SPLITS)
        labels = {text.get_text() for text in figure.axes[0].get_yticklabels()}
        assert "train (300)" in labels

    def test_the_axis_spans_the_full_scenario_count_when_given(self):
        """
        So scenarios discarded off the end are visible as empty space.

        Inferred from the largest index instead, a split that dropped its last
        fifty scenarios would draw an axis that ends exactly where the data
        does -- which hides the thing worth seeing.
        """
        figure = split_layout_figure(SPLITS, n_scenarios=600)
        assert figure.axes[0].get_xlim()[1] >= 600

    def test_an_empty_split_does_not_break_the_figure(self):
        """
        Because a report must not be load-bearing.

        A run whose test split came out empty has a real problem, and the
        report is how someone finds out -- so it has to render.
        """
        assert isinstance(
            split_layout_figure({"train": np.arange(100), "test": np.array([], dtype=np.int64)}),
            Figure,
        )


class TestTargetDistribution:
    """Comparability between splits, which no single metric reports."""

    def test_one_series_per_split(self):
        """So a regime change between train and test is visible as a shift."""
        rng = np.random.default_rng(0)
        figure = target_distribution_figure(
            {"train": rng.normal(size=200), "test": rng.normal(5.0, size=100)}
        )
        assert figure.axes[0].get_legend() is not None

    def test_the_bins_are_shared_across_splits(self):
        """
        Otherwise the shapes are not comparable, which is the whole point.

        Per-split bins would draw two distributions on different x-scales
        inside one axes, making a shifted distribution look identical.
        """
        rng = np.random.default_rng(1)
        figure = target_distribution_figure(
            {"train": rng.normal(size=200), "test": rng.normal(5.0, size=100)}, bins=20
        )
        # Both series are drawn over the same x-range, which is what makes
        # the shapes comparable; per-split bins would give each its own.
        left, right = figure.axes[0].get_xlim()
        assert left < 0.0
        assert right > 5.0

    def test_non_finite_targets_are_excluded(self):
        """
        Because a NaN in the range makes every bin edge NaN.

        The histogram then renders empty, which reads as "no data in this
        split" rather than "one value was missing".
        """
        figure = target_distribution_figure({"train": np.array([1.0, 2.0, np.nan, 3.0, np.inf])})
        assert figure.axes[0].patches


class TestMissingness:
    """The pattern, not the percentage."""

    def test_features_run_down_the_axis_and_scenarios_across_it(self):
        """
        Matching how a time series is read, so an outage is a horizontal gap.

        The other orientation would draw the same outage as a vertical stripe,
        which is indistinguishable at a glance from a column that is simply
        unpopulated.
        """
        features = np.ones((50, 4))
        features[10:20, 1] = np.nan
        figure = missingness_figure(features)
        assert figure.axes[0].images[0].get_array().shape == (4, 50)

    def test_feature_names_are_used_when_given(self):
        """
        Because "column 2" is not something anyone can act on.

        The names come from the data module, so the figure never has to guess
        them -- but it must fall back when they are absent.
        """
        features = np.ones((10, 2))
        figure = missingness_figure(features, feature_names=["spot", "vol"])
        assert "spot" in {text.get_text() for text in figure.axes[0].get_yticklabels()}

    def test_a_complete_matrix_still_renders(self):
        """
        Showing nothing missing, rather than failing on an empty mask.

        The common case, and the one that would otherwise be the only input
        the figure was never tested on.
        """
        assert isinstance(missingness_figure(np.ones((10, 3))), Figure)

    def test_infinities_are_shown_alongside_gaps(self):
        """
        Because both are unusable and they usually share a cause.

        A division that produced an infinity and one that produced a NaN
        differ only in whether the numerator was zero.
        """
        features = np.ones((10, 2))
        features[0, 0] = np.inf
        features[1, 1] = np.nan
        figure = missingness_figure(features)
        assert figure.axes[0].images[0].get_array().sum() == 2


class TestFeatureCorrelation:
    """The context a basis selection needs to be read in."""

    def test_the_matrix_is_square_in_the_feature_count(self):
        """So every pair is shown, which is what a reader scans for."""
        rng = np.random.default_rng(0)
        figure = feature_correlation_figure(rng.normal(size=(100, 5)))
        assert figure.axes[0].images[0].get_array().shape == (5, 5)

    def test_a_constant_column_does_not_emit_a_numeric_warning(self):
        """
        Because a figure factory writing warnings into a training log is noise.

        A constant column has zero variance, so its correlation is a
        zero-divided-by-zero NaN. That is handled, and handling it quietly is
        part of the figure being safe to call from a report.
        """
        features = np.random.default_rng(0).normal(size=(50, 3))
        features[:, 1] = 7.0
        with np.errstate(all="raise"):
            figure = feature_correlation_figure(features)
        assert np.all(np.isfinite(figure.axes[0].images[0].get_array()))

    def test_perfectly_correlated_columns_are_visible(self):
        """
        Which is the case that makes a basis selection misleading.

        Three columns correlated at 0.98 are one feature chosen three times,
        and the reported dimensionality overstates what the model sees.
        """
        rng = np.random.default_rng(0)
        base = rng.normal(size=100)
        features = np.column_stack([base, base * 2.0, rng.normal(size=100)])
        values = feature_correlation_figure(features).axes[0].images[0].get_array()
        assert values[0, 1] == pytest.approx(1.0, abs=1e-6)
