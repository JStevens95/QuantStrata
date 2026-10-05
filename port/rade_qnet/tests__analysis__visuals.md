# `tests/rade_qnet/analysis/visuals`

9 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 34 | 1461 | `abc60d222b39299f` |
| 2 | `test_visuals_data.py` | 278 | 11024 | `7a56dedb933736a8` |
| 3 | `test_visuals_evaluation.py` | 271 | 9331 | `ccfa5fe9d0da520e` |
| 4 | `test_visuals_export.py` | 174 | 6309 | `692334725d62e0f8` |
| 5 | `test_visuals_figures.py` | 324 | 12357 | `715c04c3860dc524` |
| 6 | `test_visuals_jobset.py` | 289 | 9538 | `f128864e10da8635` |
| 7 | `test_visuals_style.py` | 185 | 6592 | `072bcbbeaf080d05` |
| 8 | `test_visuals_training.py` | 337 | 11903 | `df167c0059d128ce` |
| 9 | `test_visuals_tuning.py` | 301 | 11420 | `52554c0b2f02024a` |

---

## 1. `tests/rade_qnet/analysis/visuals/__init__.py`

1461 bytes · SHA-256 `abc60d222b39299f`

```python
"""
Tests for ``rade_qnet.analysis.visuals``.

Every visual is checked for the two properties that make it reusable: it
returns a figure, and it writes nothing. The second is asserted explicitly
against a temporary working directory, because a plotting function that saves
as a side effect cannot be used from a dashboard, a notebook or another test.

Planned modules
---------------
``test_visuals_style.py``
    The house style applies and -- critically -- restores global plotting state
    on exit, so one figure cannot change the appearance of the next.
    [Phase 1]
``test_visuals_primitives.py``
    The shared building blocks, including empty and single-point inputs.
    [Phase 1]
``test_visuals_data.py``
    Split layout, target distribution and missingness diagnostics.  [Phase 2]
``test_visuals_training.py``
    Learning curves, including a run that stopped early.  [Phase 2]
``test_visuals_graph.py``
    Sparsity, degree distribution and subgraph layout.  [Phase 3]
``test_visuals_jobset.py``
    Cross-job comparison, including a set with a failed job.  [Phase 4]
``test_visuals_evaluation.py``
    Predicted against actual, residuals and baseline comparison.  [Phase 5]
``test_visuals_tuning.py``
    Trial history and parameter importance.  [Phase 5]
``test_visuals_episodes.py``
    Reward curves and action trajectories.  [Phase 7]
``test_visuals_export.py``
    Writing a figure: format, resolution and atomic replacement.  [Phase 1]
"""
```

---

## 2. `tests/rade_qnet/analysis/visuals/test_visuals_data.py`

11024 bytes · SHA-256 `7a56dedb933736a8`

```python
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
```

---

## 3. `tests/rade_qnet/analysis/visuals/test_visuals_evaluation.py`

9331 bytes · SHA-256 `ccfa5fe9d0da520e`

```python
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
```

---

## 4. `tests/rade_qnet/analysis/visuals/test_visuals_export.py`

6309 bytes · SHA-256 `692334725d62e0f8`

```python
"""
Tests for figure export.

This is the only module in ``visuals`` permitted to touch the filesystem, so
it is the only place these tests look for files. The division is what lets one
figure factory serve a report, a notebook and a test.

The behaviour most worth pinning is the clearing of figures after writing. A
figure built directly is an ordinary object and will be collected eventually,
but "eventually" is doing a lot of work when a job set produces thousands of
small objects pointing at large array buffers -- the interpreter has no
pressure to collect them promptly.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.export import SUPPORTED_FORMATS, save_figure
from src.rade_qnet.core.lifecycle.errors import SpecError


@pytest.fixture
def figure():
    """
    Provide a small figure with one line.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.
    """
    built = Figure(figsize=(4, 3))
    built.add_subplot().plot(np.arange(10), np.arange(10))
    return built


class TestWriting:
    """Where the file goes, and what it is called."""

    def test_the_written_path_is_returned(self, figure, tmp_path):
        """
        So a caller can record it as an artifact without reconstructing it.

        Rebuilding the path at the call site is how a report ends up
        recording an artifact that does not exist.
        """
        assert save_figure(figure, tmp_path, "curve") == tmp_path / "curve.png"

    def test_the_file_is_actually_written(self, figure, tmp_path):
        """And is not empty."""
        assert save_figure(figure, tmp_path, "curve").stat().st_size > 0

    def test_the_directory_is_created(self, figure, tmp_path):
        """
        A report's figure directory does not exist before its first figure.

        Requiring the caller to create it would put the same two lines in
        every report.
        """
        destination = tmp_path / "reports" / "figures"
        assert save_figure(figure, destination, "curve").is_file()

    @pytest.mark.parametrize("figure_format", sorted(SUPPORTED_FORMATS))
    def test_each_supported_format_writes(self, figure, tmp_path, figure_format):
        """
        All three are lossless or vector.

        A figure that will be read carefully should not be a JPEG, which is
        why the set is restricted rather than passed through.
        """
        written = save_figure(figure, tmp_path, "curve", figure_format=figure_format)
        assert written.suffix == f".{figure_format}"
        assert written.is_file()

    def test_an_unsupported_format_is_rejected(self, figure, tmp_path):
        """
        Rather than handed to matplotlib to fail less helpfully.

        And the message lists what is available, since the usual cause is a
        configuration typo.
        """
        with pytest.raises(SpecError, match="pdf"):
            save_figure(figure, tmp_path, "curve", figure_format="jpeg")

    def test_an_extension_in_the_name_is_stripped(self, figure, tmp_path):
        """
        The format argument decides the extension, not the name.

        So ``"curve.png"`` with ``figure_format="svg"`` is not
        ``curve.png.svg``, which is an easy mistake to make from a report
        that builds its figure names from a template.
        """
        written = save_figure(figure, tmp_path, "curve.png", figure_format="svg")
        assert written.name == "curve.svg"

    def test_an_existing_file_is_replaced(self, figure, tmp_path):
        """
        Re-running a report over the same directory is normal.

        Refusing would mean a second run produces a half-updated report,
        which is worse than either outcome.
        """
        save_figure(figure, tmp_path, "curve")
        second = Figure(figsize=(4, 3))
        second.add_subplot().plot([0, 1], [1, 0])
        assert save_figure(second, tmp_path, "curve").is_file()


class TestResourceRelease:
    """Promptness, not leak avoidance."""

    def test_the_figure_is_cleared_by_default(self, figure, tmp_path):
        """
        Releasing the axes and the array buffers they reference.

        A job set producing thousands of figures would otherwise hold every
        buffer until the interpreter felt like collecting them.
        """
        save_figure(figure, tmp_path, "curve")
        assert figure.axes == []

    def test_clearing_can_be_declined(self, figure, tmp_path):
        """
        For a caller that intends to display the figure as well.

        A notebook writing a figure and then showing it should not get a
        blank one.
        """
        save_figure(figure, tmp_path, "curve", close=False)
        assert figure.axes != []

    def test_a_cleared_figure_is_still_reusable(self, figure, tmp_path):
        """
        ``clf`` clears, it does not destroy.

        So a caller that cleared by accident gets an empty figure rather
        than an exception from somewhere unrelated.
        """
        save_figure(figure, tmp_path, "curve")
        figure.add_subplot().plot([0, 1], [1, 0])
        assert save_figure(figure, tmp_path, "second").is_file()


class TestResolution:
    """Raster output has to be legible."""

    def test_a_higher_dpi_produces_a_larger_file(self, tmp_path):
        """
        Confirming the setting reaches matplotlib rather than being ignored.

        A dpi argument that quietly did nothing would leave every figure in
        a printed report blurry.
        """
        small = Figure(figsize=(4, 3))
        small.add_subplot().plot(np.arange(10), np.arange(10))
        low = save_figure(small, tmp_path, "low", dpi=50).stat().st_size

        large = Figure(figsize=(4, 3))
        large.add_subplot().plot(np.arange(10), np.arange(10))
        high = save_figure(large, tmp_path, "high", dpi=200).stat().st_size

        assert high > low

    def test_dpi_is_irrelevant_to_a_vector_format(self, figure, tmp_path):
        """
        Accepted and ignored, rather than rejected.

        A report writes every figure through one call, so the format and
        the dpi are set independently of each other.
        """
        assert save_figure(figure, tmp_path, "curve", figure_format="svg", dpi=600).is_file()
```

---

## 5. `tests/rade_qnet/analysis/visuals/test_visuals_figures.py`

12357 bytes · SHA-256 `715c04c3860dc524`

```python
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
```

---

## 6. `tests/rade_qnet/analysis/visuals/test_visuals_jobset.py`

9538 bytes · SHA-256 `f128864e10da8635`

```python
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
from src.rade_qnet.core.lifecycle.errors import ContractError

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
```

---

## 7. `tests/rade_qnet/analysis/visuals/test_visuals_style.py`

6592 bytes · SHA-256 `072bcbbeaf080d05`

```python
"""
Tests for the house plotting style.

The claim being protected is that importing ``rade_qnet`` does not change
anybody's matplotlib defaults. A framework that permanently restyled a user's
plots because they imported it has overstepped, and the failure is
particularly annoying because it shows up in unrelated figures much later.

The second claim is that nothing here touches ``pyplot``. That module keeps a
global figure registry, so a job set producing twelve figures across three
hundred members would accumulate thirty-six hundred live figures and exhaust
memory -- and it selects a backend at import, which in a headless worker has
historically meant a crash.
"""

from __future__ import annotations

import ast
from pathlib import Path

import matplotlib as mpl
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis import visuals
from src.rade_qnet.analysis.visuals.style import PALETTE, RC_PARAMS, figure_style


class TestTheStyleIsTemporary:
    """``rc_context``, not mutation."""

    def test_settings_are_applied_inside_the_block(self):
        """Otherwise the module would do nothing at all."""
        with figure_style():
            assert mpl.rcParams["axes.grid"] is True

    def test_settings_are_restored_on_exit(self):
        """
        The central guarantee.

        A user's own defaults must survive importing and using this
        framework.
        """
        before = mpl.rcParams["font.size"]
        with figure_style({"font.size": 42}):
            pass
        assert mpl.rcParams["font.size"] == before

    def test_settings_are_restored_after_an_exception(self):
        """
        A failed plot must not leave the style applied.

        Which is precisely what a bare ``rcParams.update`` would do, and the
        reason a context manager is used.
        """
        before = mpl.rcParams["font.size"]
        with pytest.raises(RuntimeError), figure_style({"font.size": 42}):
            raise RuntimeError("plotting failed")
        assert mpl.rcParams["font.size"] == before

    def test_importing_the_module_changes_nothing(self):
        """
        The style is opted into, never applied on import.

        A user whose unrelated notebook figures changed appearance because
        of a transitive import would be right to be annoyed.
        """
        assert mpl.rcParams["axes.spines.top"] is True


class TestOverrides:
    """Per-block customisation, scoped to the block."""

    def test_an_override_takes_effect(self):
        """A report may legitimately want a larger font."""
        with figure_style({"font.size": 14}):
            assert mpl.rcParams["font.size"] == 14

    def test_an_override_does_not_disturb_the_rest_of_the_style(self):
        """
        Overrides are merged, not substituted.

        Replacing the whole set would make a one-key override silently drop
        the grid, the spine settings and the palette.
        """
        with figure_style({"font.size": 14}):
            assert mpl.rcParams["axes.grid"] is True

    def test_an_override_does_not_mutate_the_shared_defaults(self):
        """
        The merge must copy.

        Otherwise one report's override would leak into every later figure
        in the process, which is the hardest kind of styling bug to trace.
        """
        with figure_style({"font.size": 14}):
            pass
        assert RC_PARAMS["font.size"] == 10


class TestThePalette:
    """Chosen for where figures actually end up."""

    def test_the_palette_drives_the_colour_cycle(self):
        """So two figures in one report agree on what "model" looks like."""
        with figure_style():
            cycle = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        assert tuple(cycle) == PALETTE

    def test_every_colour_is_distinct(self):
        """A repeated colour makes two series indistinguishable."""
        assert len(set(PALETTE)) == len(PALETTE)

    def test_the_first_two_colours_differ_in_lightness(self):
        """
        Figures end up in printed documents more often than anyone plans.

        The two-series case is the common one, so those two have to survive
        greyscale conversion rather than merging into one shade.
        """

        def luminance(colour):
            """
            Return a rough perceptual lightness for a hex colour.

            Parameters
            ----------
            colour
                A ``#rrggbb`` string.

            Returns
            -------
            float
                Weighted lightness in ``[0, 255]``.
            """
            red, green, blue = (int(colour[index : index + 2], 16) for index in (1, 3, 5))
            return 0.299 * red + 0.587 * green + 0.114 * blue

        assert abs(luminance(PALETTE[0]) - luminance(PALETTE[1])) > 20


class TestNoPyplot:
    """Figures are ordinary objects, not registry entries."""

    def test_a_figure_built_under_the_style_is_not_registered(self):
        """
        The property that stops a job set exhausting memory.

        A ``pyplot`` figure stays alive until explicitly closed; this one is
        collected like anything else.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415 - imported only to observe the registry

        before = plt.get_fignums()
        with figure_style():
            figure = Figure(figsize=(4, 3))
            figure.add_subplot().plot([0, 1], [1, 0])
        assert plt.get_fignums() == before

    @pytest.mark.parametrize(
        "module_path",
        sorted(Path(visuals.__file__).parent.glob("*.py")),
        ids=lambda path: path.name,
    )
    def test_no_module_in_visuals_imports_pyplot(self, module_path):
        """
        Checked across the package, because one import is enough.

        ``pyplot`` selects a backend at import time, which in a headless
        worker has historically meant an attempted GUI backend and a crash
        -- and it does so for every caller, not just the module that
        imported it.

        Read from the syntax tree rather than from the text, so a mention in
        a docstring (this file's own module docstring, for instance) is not
        mistaken for an import.
        """
        tree = ast.parse(module_path.read_text(encoding="utf-8"))
        imported = {
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for name in node.names
        } | {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any("pyplot" in (name or "") for name in imported)
```

---

## 8. `tests/rade_qnet/analysis/visuals/test_visuals_training.py`

11903 bytes · SHA-256 `df167c0059d128ce`

```python
"""
Tests for the training figures.

The gradient-norm figure is the one worth arguing for. A loss curve cannot
distinguish a converged network from a dead one: both go flat and stay flat.
The gradient norm separates them immediately -- a collapse towards zero is a
dead network, and unbounded growth is a run that is about to diverge rather
than one that already has.

Two behaviours here come from problems found while building the phase. An
absent metric produces a NaN rather than a zero, so a gap in the history
renders as a gap in the line instead of as a plunge to zero -- which in a
gradient-norm panel is a meaningful and alarming value. And a log-scaled axis
is only applied when there is something positive to plot, because
``set_yscale('log')`` over all-zero data warns and renders an empty panel.
"""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.rade_qnet.analysis.visuals.training import (
    epoch_timing_figure,
    gradient_norm_figure,
    learning_rate_figure,
    training_diagnostics_figure,
)
from src.rade_qnet.core.contract.result import EpochRecord, FitOutcome
from src.rade_qnet.core.lifecycle.errors import ContractError


def outcome(
    *,
    n_epochs: int = 5,
    with_validation: bool = True,
    metrics_per_epoch: list[dict[str, float]] | None = None,
    learning_rates: list[float] | None = None,
) -> FitOutcome:
    """
    Build a fit outcome with a plausible history.

    Parameters
    ----------
    n_epochs
        How many epoch records to produce.
    with_validation
        Whether each record carries a validation loss.
    metrics_per_epoch
        Per-epoch metric mappings, defaulting to empty ones.
    learning_rates
        Per-epoch learning rates, defaulting to a constant.

    Returns
    -------
    FitOutcome
        The outcome.
    """
    metrics = metrics_per_epoch or [{} for _ in range(n_epochs)]
    rates = learning_rates or [0.01] * n_epochs
    history = tuple(
        EpochRecord(
            epoch=index,
            train_loss=1.0 / (index + 1),
            val_loss=1.2 / (index + 1) if with_validation else None,
            metrics=metrics[index],
            learning_rate=rates[index],
            seconds=0.5 + index * 0.1,
        )
        for index in range(n_epochs)
    )
    return FitOutcome(
        history=history,
        monitor="val_loss" if with_validation else "train_loss",
        best_epoch=n_epochs - 1,
        best_monitor_value=history[-1].val_loss or history[-1].train_loss,
        stopped_early=False,
        restored_best=True,
        total_seconds=sum(record.seconds for record in history),
    )


class TestGradientNormFigure:
    """The diagnostic a loss curve cannot provide."""

    def test_both_the_mean_and_the_maximum_are_drawn(self):
        """
        Because they fail differently.

        A rising maximum against a flat mean is a handful of exploding
        samples; a collapsing mean is a dead network. One line would conflate
        the two.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[
                    {"grad_norm_mean": 1.0, "grad_norm_max": 3.0},
                    {"grad_norm_mean": 0.8, "grad_norm_max": 5.0},
                    {"grad_norm_mean": 0.6, "grad_norm_max": 9.0},
                ],
            )
        )
        assert len(figure.axes[0].lines) >= 2

    def test_the_axis_is_log_scaled_when_the_norms_are_positive(self):
        """
        Because the interesting changes are multiplicative.

        A norm going from 1 to 100 and one going from 0.01 to 1 are the same
        hundred-fold change, and on a linear axis the second is invisible.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=2,
                metrics_per_epoch=[
                    {"grad_norm_mean": 0.01, "grad_norm_max": 0.02},
                    {"grad_norm_mean": 1.0, "grad_norm_max": 100.0},
                ],
            )
        )
        assert figure.axes[0].get_yscale() == "log"

    def test_all_zero_norms_do_not_get_a_log_axis(self):
        """
        Because a log scale over zeros warns and renders an empty panel.

        And this is exactly the case worth looking at: an all-zero norm is a
        dead network, so the panel that would show it must not be the one that
        comes out blank.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=2,
                metrics_per_epoch=[
                    {"grad_norm_mean": 0.0, "grad_norm_max": 0.0},
                    {"grad_norm_mean": 0.0, "grad_norm_max": 0.0},
                ],
            )
        )
        assert figure.axes[0].get_yscale() == "linear"

    def test_an_absent_epoch_renders_as_a_gap(self):
        """
        Not as a plunge to zero.

        Zero is a meaningful and alarming value in this panel, so an epoch
        that simply did not record a norm must not look like one that recorded
        a collapse. A NaN leaves the line broken, which reads correctly.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0},
                    {},
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0},
                ],
            )
        )
        values = figure.axes[0].lines[0].get_ydata()
        assert np.isnan(values[1])

    def test_a_history_with_no_norms_at_all_is_refused(self):
        """
        Rather than drawing an empty panel.

        An empty figure in a report is read as "the run had no gradient
        problem", which is a much stronger claim than "nobody measured".
        """
        with pytest.raises(ContractError, match="no gradient norms"):
            gradient_norm_figure(outcome(n_epochs=3))

    def test_the_clipped_fraction_is_drawn_when_it_was_recorded(self):
        """
        Because a clip that binds on every step is not clipping outliers.

        It is rescaling every update, which caps the effective learning rate
        at something other than the configured one -- and nothing else in the
        output says so.
        """
        figure = gradient_norm_figure(
            outcome(
                n_epochs=2,
                metrics_per_epoch=[
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0, "grad_clipped_fraction": 1.0},
                    {"grad_norm_mean": 1.0, "grad_norm_max": 2.0, "grad_clipped_fraction": 1.0},
                ],
            )
        )
        assert len(figure.axes) > 1 or len(figure.axes[0].lines) >= 3


class TestLearningRateFigure:
    """The schedule as it actually ran."""

    def test_the_recorded_rates_are_drawn(self):
        """
        Rather than the formula recomputed from the spec.

        A recomputed curve agrees with reality only while the formula and the
        scheduler agree, and the figure would then be showing a rate the
        optimiser never used.
        """
        figure = learning_rate_figure(outcome(n_epochs=3, learning_rates=[0.1, 0.05, 0.01]))
        assert figure.axes[0].lines[0].get_ydata().tolist() == [0.1, 0.05, 0.01]

    def test_an_unrecorded_rate_is_a_gap_rather_than_a_zero(self):
        """
        Because a flat line at zero would read as a frozen optimiser.

        That is a real failure mode, so the figure must not be able to
        fabricate its appearance from missing data. An epoch with no recorded
        rate leaves the line broken instead.
        """
        history = tuple(
            EpochRecord(epoch=index, train_loss=1.0, learning_rate=None, seconds=0.1)
            for index in range(3)
        )
        figure = learning_rate_figure(
            FitOutcome(
                history=history,
                monitor="train_loss",
                stopped_early=False,
                restored_best=False,
                total_seconds=0.3,
            )
        )
        assert np.all(np.isnan(figure.axes[0].lines[0].get_ydata()))

    def test_an_empty_history_is_refused(self):
        """
        Because there is no schedule to show.

        An empty panel in a report reads as a constant rate, which is a claim
        about the run rather than an absence of data.
        """
        with pytest.raises(ContractError):
            learning_rate_figure(
                FitOutcome(
                    history=(),
                    monitor="train_loss",
                    stopped_early=False,
                    restored_best=False,
                    total_seconds=0.0,
                )
            )


class TestEpochTimingFigure:
    """Where the time went, which a total cannot show."""

    def test_each_epoch_s_duration_is_drawn(self):
        """
        Because a run that slows down over time is leaking.

        A total divided by the epoch count would average the leak away
        completely.
        """
        figure = epoch_timing_figure(outcome(n_epochs=4))
        assert len(figure.axes[0].patches or figure.axes[0].lines[0].get_ydata()) == 4

    def test_an_empty_history_is_refused(self):
        """
        Rather than drawing an empty figure.

        An empty history means the fit produced nothing, which is worth an
        error at the point it is noticed rather than a blank panel in a report
        someone has to interpret.
        """
        with pytest.raises(ContractError, match="at least one epoch"):
            epoch_timing_figure(
                FitOutcome(
                    history=(),
                    monitor="train_loss",
                    stopped_early=False,
                    restored_best=False,
                    total_seconds=0.0,
                )
            )


class TestTrainingDiagnosticsFigure:
    """The three panels on a shared epoch axis."""

    def test_three_panels_are_drawn(self):
        """
        Loss, learning rate and gradient norm, which is the diagnostic set.

        Separate figures would have to be read side by side with their x-axes
        lined up by eye, and the whole value here is in the alignment.
        """
        figure = training_diagnostics_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[{"grad_norm_mean": 1.0, "grad_norm_max": 2.0}] * 3,
            )
        )
        assert len(figure.axes) == 3

    def test_the_epoch_axis_is_shared(self):
        """
        So a spike in the gradient norm lines up with the epoch it happened in.

        Without sharing, a schedule change and a loss plateau can appear one
        epoch apart purely from independent axis autoscaling.
        """
        figure = training_diagnostics_figure(
            outcome(
                n_epochs=3,
                metrics_per_epoch=[{"grad_norm_mean": 1.0, "grad_norm_max": 2.0}] * 3,
            )
        )
        limits = {axes.get_xlim() for axes in figure.axes}
        assert len(limits) == 1

    def test_the_figure_renders_without_gradient_norms(self):
        """
        Because tracking is optional and a report must not be load-bearing.

        The panel that cannot be filled is left empty or annotated rather than
        taking the whole figure down.
        """
        assert isinstance(training_diagnostics_figure(outcome(n_epochs=3)), Figure)

    def test_the_figure_renders_without_a_validation_split(self):
        """
        Which is a legitimate configuration, not an error.

        A run with no validation fraction still has a loss curve worth
        plotting, and the missing series must not be drawn as zeros.
        """
        assert isinstance(
            training_diagnostics_figure(outcome(n_epochs=3, with_validation=False)),
            Figure,
        )
```

---

## 9. `tests/rade_qnet/analysis/visuals/test_visuals_tuning.py`

11420 bytes · SHA-256 `52554c0b2f02024a`

```python
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
from src.rade_qnet.core.lifecycle.errors import ContractError

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

        assert "sampler" not in labels(parameter_importance_figure(with_categorical, objectives))

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
```

