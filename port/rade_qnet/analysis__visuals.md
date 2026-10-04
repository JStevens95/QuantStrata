# `src/rade_qnet/analysis/visuals`

9 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 56 | 2442 | `3f467edddb1956a3` |
| 2 | `data.py` | 412 | 13659 | `e760737083820f0f` |
| 3 | `evaluation.py` | 352 | 12471 | `e2b5e6e1d13d1d12` |
| 4 | `export.py` | 95 | 3022 | `1fadb124361a60af` |
| 5 | `jobset.py` | 351 | 12065 | `56267c058caa227e` |
| 6 | `primitives.py` | 342 | 11266 | `ee99c29d23ab2900` |
| 7 | `style.py` | 105 | 3635 | `f6b3228dede4416b` |
| 8 | `training.py` | 379 | 12144 | `6449731d6368dfe8` |
| 9 | `tuning.py` | 480 | 16759 | `b0d6b5a02cec0ab7` |

---

## 1. `src/rade_qnet/analysis/visuals/__init__.py`

2442 bytes · SHA-256 `3f467edddb1956a3`

```python
"""
Figure factories -- shared across the whole framework.

Every function here has the same shape: it accepts data and styling, and
returns a figure.  It does not save, show, close or configure anything global.
That purity is what makes these functions reusable from a report writer, a
notebook, a dashboard or a unit test, and it is what makes them testable at all
-- a test can assert on axis labels, series count and data limits without
rendering to a file.

Modules
-------
``style.py``
    The house style: palettes, figure sizes, fonts and a context manager that
    applies them without mutating global state for the rest of the process.
    Also the reason this package never imports ``pyplot``.
    [Phase 1, delivered]
``primitives.py``
    Shared building blocks -- training curve, prediction scatter with a parity
    line, residual histogram, metric comparison against a baseline -- that the
    higher-level modules compose.  [Phase 1, delivered]
``export.py``
    The one place that writes a figure to disk, in a consistent format and
    resolution.  Separate from the factories by design.  [Phase 1, delivered]
``data.py``
    Data diagnostics: split layout over the scenario axis, target distribution
    per split, missingness map, feature correlation.  The split layout is the
    one artifact where a leaky split is visible at a glance.
    [Phase 2, delivered]
``jobset.py``
    Cross-job comparison: ranking, metric dispersion, status overview and
    wall time.  Forty loss curves answer nothing, because nobody reads forty
    loss curves; each of these collapses a whole set into one view.
    [Phase 4, delivered]
``training.py``
    Learning-rate trace, gradient-norm trace, per-epoch timing, and the
    stacked three-panel diagnostic that relates them.  The loss curve itself
    lives in ``primitives.py``, since every engine produces one.
    [Phase 2, delivered]

Planned modules
---------------
``graph.py``
    Graph structure: adjacency sparsity, degree distribution, neighbourhood
    weight profiles, layout of a sampled subgraph.  [Phase 3]
``evaluation.py``
    Predicted against actual, residual diagnostics, error by bucket, baseline
    comparison.  [Phase 5]
``tuning.py``
    Trial history, parameter importance, parallel coordinates.  [Phase 5]
``episodes.py``
    Reward curves, action trajectories, policy-value surfaces, hedging error
    paths.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/analysis/visuals/data.py`

13659 bytes · SHA-256 `e760737083820f0f`

```python
"""
Data diagnostics: what the splits look like and what the inputs contain.

Four figures, each answering a question that is cheap to ask and expensive to
discover later.

:func:`split_layout_figure` is the one worth arguing for. It draws the three
splits as bands along the scenario axis with the discarded boundary gaps
visible between them, and it is the only artifact in the framework where a
leaky split would be obvious at a glance. Every other check on splitting is a
number or an assertion; this is the one a person reads by accident while
looking at something else, which is how the mistakes that survive tests get
caught.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ...core.runtime.errors import ContractError
from .primitives import DEFAULT_FIGSIZE
from .style import figure_style

__all__ = [
    "feature_correlation_figure",
    "missingness_figure",
    "split_layout_figure",
    "target_distribution_figure",
]

#: Colour per split. Fixed rather than taken from the palette in rotation, so
#: that "train" is the same colour in every figure in every report -- a reader
#: should not have to check a legend twice.
_SPLIT_COLOURS = {
    "train": "#2e7d32",
    "validation": "#f9a825",
    "test": "#c62828",
}

#: Height of the split band, in axis units.
_BAND_HEIGHT = 0.6

#: A feature matrix is samples by features.
_MATRIX_RANK = 2

#: Below this many columns, correlation cell values are written in. Above it
#: the labels overlap and the colour map carries the information alone.
_ANNOTATE_LIMIT = 12


def split_layout_figure(
    splits: Mapping[str, NDArray[np.int64]],
    *,
    n_scenarios: int | None = None,
    title: str = "Split layout over the scenario axis",
) -> Figure:
    """
    Draw each split as a band along the scenario axis.

    Gaps between the bands are the scenarios the splitter discarded to keep a
    sequence window inside one split. Seeing them is the point: a
    chronological split with no visible gap and a sequence length above one is
    leaking, and this figure shows that without anyone having to compute
    anything.

    Parameters
    ----------
    splits
        Split name to scenario indices.
    n_scenarios
        Length of the full axis. ``None`` infers it from the largest index,
        which under-reports when the final scenarios were discarded -- so pass
        it when it is known.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If no split has any index, since there would be nothing to draw.
    """
    present = {name: np.asarray(indices) for name, indices in splits.items()}
    if not any(indices.size for indices in present.values()):
        raise ContractError("every split is empty; there is nothing to draw")

    axis_length = n_scenarios or int(
        max(int(indices.max()) + 1 for indices in present.values() if indices.size)
    )

    with figure_style():
        figure = Figure(figsize=(DEFAULT_FIGSIZE[0], 2.6))
        axes = figure.add_subplot()

        for row, (name, indices) in enumerate(present.items()):
            if not indices.size:
                continue
            colour = _SPLIT_COLOURS.get(name, "#546e7a")
            # Drawn as contiguous runs rather than one bar per index: a
            # thousand-scenario split would otherwise be a thousand patches,
            # and the runs are what make a gap visible as a gap.
            for start, length in _contiguous_runs(indices):
                axes.broken_barh(
                    [(start, length)], (row - _BAND_HEIGHT / 2, _BAND_HEIGHT), color=colour
                )

        axes.set_yticks(range(len(present)))
        axes.set_yticklabels([f"{name} ({indices.size})" for name, indices in present.items()])
        axes.set_xlim(0, axis_length)
        axes.set_xlabel("scenario index")
        axes.set_title(title)
        axes.invert_yaxis()
        figure.tight_layout()
        return figure


def _contiguous_runs(indices: NDArray[np.int64]) -> list[tuple[int, int]]:
    """
    Collapse sorted indices into ``(start, length)`` runs.

    Parameters
    ----------
    indices
        Scenario indices, in any order.

    Returns
    -------
    list of tuple
        One entry per contiguous run.
    """
    ordered = np.sort(np.asarray(indices, dtype=np.int64))
    if ordered.size == 0:
        return []

    # A break is any point where the index jumps by more than one.
    breaks = np.flatnonzero(np.diff(ordered) > 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [ordered.size - 1]))
    return [
        (int(ordered[start]), int(ordered[end] - ordered[start] + 1))
        for start, end in zip(starts, ends, strict=True)
    ]


def target_distribution_figure(
    targets_by_split: Mapping[str, NDArray[np.floating]],
    *,
    bins: int = 40,
    title: str = "Target distribution per split",
) -> Figure:
    """
    Overlay the target's distribution for each split.

    The figure that reveals a split whose held-out period is not comparable
    with training -- a regime change, a currency redenomination, a volatility
    collapse. A model evaluated across such a boundary can look poor while
    being correct, or good while being lucky, and no single metric
    distinguishes the two.

    Parameters
    ----------
    targets_by_split
        Split name to target values, in original units.
    bins
        Histogram bin count, shared across splits so the shapes are
        comparable.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If every split is empty.
    """
    populated = {
        name: np.asarray(values, dtype=np.float64).ravel()
        for name, values in targets_by_split.items()
        if np.asarray(values).size
    }
    if not populated:
        raise ContractError("every split's target is empty; there is nothing to draw")

    # One shared bin edge set, computed over the union.  Per-split bins would
    # make two histograms of different data look identical.
    combined = np.concatenate([values[np.isfinite(values)] for values in populated.values()])
    edges = np.histogram_bin_edges(combined, bins=bins)

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        for name, values in populated.items():
            finite = values[np.isfinite(values)]
            axes.hist(
                finite,
                bins=edges,
                density=True,
                histtype="step",
                linewidth=1.6,
                label=f"{name} (n={finite.size})",
                color=_SPLIT_COLOURS.get(name),
            )
        axes.set_xlabel("target")
        axes.set_ylabel("density")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def missingness_figure(
    features: NDArray[np.floating],
    *,
    feature_names: Sequence[str] | None = None,
    title: str = "Missing and non-finite inputs",
) -> Figure:
    """
    Show where the input matrix is missing, as a scenario-by-feature map.

    A map rather than a bar chart of per-column percentages, because the
    *pattern* is what matters. Ten per cent missing spread evenly is an
    imputation problem; the same ten per cent concentrated in one contiguous
    period is a data outage, and only one of those two is fixable by changing
    the model.

    Parameters
    ----------
    features
        Feature matrix, samples by features, before windowing.
    feature_names
        Column labels. ``None`` falls back to indices.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the matrix is not two-dimensional, or the names do not match the
        columns.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != _MATRIX_RANK:
        raise ContractError(
            f"missingness_figure expects a 2-dimensional matrix, received shape {matrix.shape}"
        )
    labels = _column_labels(feature_names, n_columns=matrix.shape[1])

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        # Transposed so features run down the vertical axis and scenarios
        # along the horizontal one, matching how a time series is read.
        axes.imshow(
            (~np.isfinite(matrix)).T,
            aspect="auto",
            interpolation="nearest",
            cmap="Greys",
            vmin=0.0,
            vmax=1.0,
        )
        axes.set_yticks(range(len(labels)))
        axes.set_yticklabels(labels)
        axes.set_xlabel("scenario index")
        missing = float((~np.isfinite(matrix)).mean()) if matrix.size else 0.0
        axes.set_title(f"{title} ({missing:.1%} missing)")
        figure.tight_layout()
        return figure


def feature_correlation_figure(
    features: NDArray[np.floating],
    *,
    feature_names: Sequence[str] | None = None,
    title: str = "Feature correlation",
) -> Figure:
    """
    Show the correlation between input columns.

    Worth having next to a basis selection: a selection that picked three
    columns correlated at 0.98 has chosen one feature three times, and the
    reported dimensionality overstates what the model actually sees.

    Parameters
    ----------
    features
        Feature matrix, samples by features.
    feature_names
        Column labels.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the matrix is not two-dimensional, has fewer than two columns, or
        the names do not match the columns.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != _MATRIX_RANK:
        raise ContractError(
            f"feature_correlation_figure expects a 2-dimensional matrix, received "
            f"shape {matrix.shape}"
        )
    if matrix.shape[1] < _MATRIX_RANK:
        raise ContractError(
            f"correlation needs at least 2 feature columns, received {matrix.shape[1]}"
        )
    labels = _column_labels(feature_names, n_columns=matrix.shape[1])

    # Rows with any missing value are dropped rather than imputed: pairwise
    # deletion can produce a correlation matrix that is not positive
    # semi-definite, which would be a figure showing something impossible.
    complete = matrix[np.isfinite(matrix).all(axis=1)]
    if complete.shape[0] < _MATRIX_RANK:
        raise ContractError(
            f"only {complete.shape[0]} row(s) have every feature present, which is "
            f"too few to correlate; check the missingness figure first"
        )

    # A constant column has zero variance, so NumPy divides by zero and warns
    # on its way to returning NaN.  Silenced because the NaN is expected and
    # handled on the next line -- a figure factory emitting numeric warnings
    # into a training log would be noise, not information.
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = np.corrcoef(complete, rowvar=False)

    # Those NaNs are shown as zero with the diagonal kept at one, which reads
    # as "no relationship" rather than leaving a blank cell a reader may take
    # for a rendering fault.
    correlation = np.nan_to_num(correlation, nan=0.0)
    np.fill_diagonal(correlation, 1.0)

    with figure_style():
        figure = Figure(figsize=(6.5, 5.5))
        axes = figure.add_subplot()
        image = axes.imshow(correlation, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        axes.set_xticks(range(len(labels)))
        axes.set_xticklabels(labels, rotation=90)
        axes.set_yticks(range(len(labels)))
        axes.set_yticklabels(labels)
        figure.colorbar(image, ax=axes, label="correlation")

        if len(labels) <= _ANNOTATE_LIMIT:
            for row in range(len(labels)):
                for column in range(len(labels)):
                    axes.text(
                        column,
                        row,
                        f"{correlation[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )

        axes.set_title(title)
        figure.tight_layout()
        return figure


def _column_labels(names: Sequence[str] | None, *, n_columns: int) -> list[str]:
    """
    Return one label per column, falling back to indices.

    Parameters
    ----------
    names
        Supplied labels, or ``None``.
    n_columns
        How many columns the matrix has.

    Returns
    -------
    list of str
        Labels.

    Raises
    ------
    ContractError
        If the supplied labels do not match the column count. Refused rather
        than truncated, because a correlation matrix with shifted labels is
        worse than one with no labels -- it is confidently wrong.
    """
    if names is None:
        return [str(index) for index in range(n_columns)]
    if len(names) != n_columns:
        raise ContractError(
            f"received {len(names)} feature name(s) for {n_columns} column(s); "
            f"mislabelled axes are worse than unlabelled ones"
        )
    return list(names)
```

---

## 3. `src/rade_qnet/analysis/visuals/evaluation.py`

12471 bytes · SHA-256 `e2b5e6e1d13d1d12`

```python
"""
Reading a scored model: where the error lives, and whether the inputs moved.

Three of the figures an evaluation wants already exist in
:mod:`~.primitives` -- predicted against observed, the residual
distribution, and metrics against a baseline -- because training needs the
same three and they were written there. They are not duplicated here. This
module adds the two views that only make sense once a model is being
*examined* rather than reported:

**Error by bucket.** A single mean absolute error is an average over a
population, and the population is rarely homogeneous. A model can be
excellent on the typical case and useless on the tail, and the headline
metric is the same as one that is mediocre everywhere. Those are completely
different models to own, and the only way to tell them apart is to cut the
error by something.

**Drift.** The one view that is about the *inputs* rather than the output.
A model whose error has not yet risen but whose inputs have moved is a model
about to be wrong, and no metric computed from predictions can say so --
because the outcomes those predictions would be scored against have not
arrived.

Plain data in, figure out
-------------------------
Every function takes arrays and mappings of plain numbers rather than a
result object. ``analysis`` may only import ``core``, so it cannot see an
:class:`~rade_qnet.core.contract.result.EvaluationResult` -- and as with
:mod:`~.jobset`, the constraint produces the better interface anyway. A
figure driven by ``{feature: value}`` can be fed from a result, a notebook,
a database query or a test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from ...core.runtime.errors import ContractError
from .primitives import DEFAULT_FIGSIZE
from .style import PALETTE, figure_style

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

__all__ = [
    "drift_figure",
    "error_by_bucket_figure",
    "residual_against_prediction_figure",
]

#: How many quantile buckets to cut the error into. Ten is enough to show a
#: tail effect and few enough that every bucket holds a usable sample at the
#: hundreds-of-rows scale a held-out split usually has.
_BUCKETS = 10

#: Where a drift measure stops being unremarkable, drawn as a reference line.
#: Matches ``analysis.metrics.drift.DRIFT_THRESHOLDS["psi"]``; duplicated as a
#: number rather than imported because a figure that imported a threshold
#: would redraw itself if the threshold moved, and a chart in a report should
#: mean the same thing a year later.
_PSI_THRESHOLD = 0.25


def error_by_bucket_figure(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    buckets: int = _BUCKETS,
    title: str = "Absolute error by observed decile",
) -> Figure:
    """
    Show how the error varies across the range of observed values.

    Cut on the *observed* value rather than the predicted one. Bucketing by
    prediction answers "when the model says a large number, is it right",
    which is a question about the model's self-knowledge. Bucketing by
    observation answers "for the cases that turned out large, was the model
    right", which is the question anybody sizing a position is actually
    asking.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values, in the same units.
    buckets
        How many quantile buckets to cut into.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        A bar chart of mean absolute error per bucket, with the overall
        mean drawn across it so that a bucket's distance from the headline
        number is readable at a glance.

    Raises
    ------
    ContractError
        If the arrays disagree in length, or hold fewer values than there
        are buckets. Refused rather than drawn: buckets holding one sample
        each produce a chart that looks like structure and is noise.
    """
    predicted, observed = _aligned(predictions, targets)

    if observed.size < buckets:
        raise ContractError(
            f"cannot cut {observed.size} value(s) into {buckets} bucket(s); the "
            f"result would be a chart of single observations, which looks like "
            f"structure and is noise"
        )

    errors = np.abs(predicted - observed)
    # Ranked rather than cut on value, so every bucket holds the same count
    # even when the observations are heavily skewed -- which, for a P&L
    # series, they always are.
    order = np.argsort(observed, kind="stable")
    groups = np.array_split(order, buckets)

    means = np.array([float(np.mean(errors[group])) for group in groups])
    edges = [f"{float(np.mean(observed[group])):.3g}" for group in groups]
    overall = float(np.mean(errors))

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot(111)
        axes.bar(range(len(means)), means, color=PALETTE[0])
        axes.axhline(
            overall,
            color=PALETTE[1],
            linestyle="--",
            label=f"overall mean absolute error ({overall:.3g})",
        )
        axes.set_xticks(range(len(means)))
        axes.set_xticklabels(edges, rotation=45, ha="right")
        axes.set_xlabel("Mean observed value in bucket")
        axes.set_ylabel("Mean absolute error")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
    return figure


def residual_against_prediction_figure(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    title: str = "Residual against prediction",
) -> Figure:
    """
    Show whether the error grows with the size of the prediction.

    The companion to the residual histogram, which shows the *shape* of the
    error and not its relationship to anything. A histogram cannot
    distinguish a model with constant error from one whose error is
    proportional to its output -- both can be symmetric and centred -- and
    those imply very different things about how far the model can be
    trusted at the extremes.

    A horizontal line at zero is drawn because the eye reads a residual
    plot by its deviation from that line, and a plot without it invites the
    reader to use the lowest point as the reference.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values, in the same units.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        A scatter of residual against prediction.

    Raises
    ------
    ContractError
        If the arrays disagree in length.
    """
    predicted, observed = _aligned(predictions, targets)
    residuals = observed - predicted

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot(111)
        axes.scatter(predicted, residuals, s=12, alpha=0.6, color=PALETTE[0])
        axes.axhline(0.0, color=PALETTE[1], linestyle="--", linewidth=1.0)
        axes.set_xlabel("Predicted")
        axes.set_ylabel("Residual (observed - predicted)")
        axes.set_title(title)
        figure.tight_layout()
    return figure


def drift_figure(
    metrics: Mapping[str, Mapping[str, float]],
    *,
    measure: str = "psi",
    threshold: float = _PSI_THRESHOLD,
    title: str = "Input drift from the training distribution",
) -> Figure:
    """
    Rank features by how far their distribution has moved.

    Horizontal bars, for the same reason the job-set ranking uses them:
    feature names are long, and a rotated x-axis label is unreadable past a
    dozen entries.

    Infinite values are drawn at the top of the finite range rather than
    dropped. An infinite measure means a feature that was constant in
    training and is not now, which is total drift and the single most
    important bar on the chart -- and matplotlib silently omits a bar of
    infinite height, so the most severe case would be the one the figure
    did not show.

    Parameters
    ----------
    metrics
        The output of
        :func:`~rade_qnet.analysis.metrics.drift.drift_metrics`: feature name
        to measure name to value.
    measure
        Which measure to rank on.
    threshold
        Where to draw the reference line.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        A horizontal bar chart, worst at the top, with features past the
        threshold drawn in the alert colour.

    Raises
    ------
    ContractError
        If there is nothing to draw, or no feature reports the measure.
    """
    if not metrics:
        raise ContractError(
            "there are no drift metrics to draw. An empty chart reads as a "
            "rendering fault rather than as an absence of data"
        )

    values = {
        name: float(measures[measure]) for name, measures in metrics.items() if measure in measures
    }
    if not values:
        raise ContractError(
            f"no feature reports a {measure!r} measure; available measures are "
            f"{sorted({key for measures in metrics.values() for key in measures})}"
        )

    ranked = sorted(values.items(), key=lambda pair: pair[1])
    names = [name for name, _ in ranked]
    heights, capped = _finite_heights([value for _, value in ranked])
    colours = [PALETTE[1] if value > threshold else PALETTE[0] for _, value in ranked]

    with figure_style():
        figure = Figure(figsize=(DEFAULT_FIGSIZE[0], max(3.0, 0.3 * len(names) + 1.5)))
        axes = figure.add_subplot(111)
        axes.barh(names, heights, color=colours)
        axes.axvline(
            threshold,
            color=PALETTE[2],
            linestyle="--",
            linewidth=1.0,
            label=f"threshold ({threshold:g})",
        )
        for index, (name, value) in enumerate(ranked):
            if np.isinf(value):
                # Labelled, because the bar is drawn at the capped height
                # and would otherwise be read as merely the largest.
                axes.text(heights[index], index, "  infinite", va="center", fontsize=8)
            del name
        axes.set_xlabel(measure)
        axes.set_title(title if not capped else f"{title} (infinite values capped)")
        axes.legend(loc="lower right")
        figure.tight_layout()
    return figure


def _finite_heights(values: list[float]) -> tuple[list[float], bool]:
    """
    Replace infinite bar heights with something matplotlib will draw.

    Parameters
    ----------
    values
        The measure per feature.

    Returns
    -------
    tuple
        The heights to draw, and whether anything was capped -- which the
        caller says in the title, so a reader is never shown a capped bar
        without being told.
    """
    finite = [value for value in values if np.isfinite(value)]
    # A fifth above the largest finite bar: clearly the biggest, without
    # compressing everything else into the axis.
    cap = (max(finite) * 1.2) if finite else 1.0
    capped = any(not np.isfinite(value) for value in values)
    return [cap if not np.isfinite(value) else value for value in values], capped


def _aligned(
    predictions: NDArray[np.floating], targets: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Flatten two arrays and check they describe the same samples.

    Parameters
    ----------
    predictions
        Model output.
    targets
        Observed values.

    Returns
    -------
    tuple
        The two flattened arrays.

    Raises
    ------
    ContractError
        If the counts disagree. Checked rather than broadcast: NumPy will
        expand an ``(n, 1)`` against an ``(n,)`` into an ``(n, n)``, and the
        figure drawn from that is a plausible-looking chart of nothing.
    """
    predicted = np.ravel(np.asarray(predictions, dtype=np.float64))
    observed = np.ravel(np.asarray(targets, dtype=np.float64))
    if predicted.shape != observed.shape:
        raise ContractError(
            f"{predicted.size} prediction(s) cannot be plotted against "
            f"{observed.size} target(s); equal counts are required"
        )
    if predicted.size == 0:
        raise ContractError("there is nothing to plot: both arrays are empty")
    return predicted, observed
```

---

## 4. `src/rade_qnet/analysis/visuals/export.py`

3022 bytes · SHA-256 `1fadb124361a60af`

```python
"""
Writing figures to disk.

The only module in ``visuals`` that touches the filesystem.

Keeping writing here, and out of the figure factories, is what lets the same
factory serve a report, a notebook and a test. See
:mod:`rade_qnet.analysis.visuals.primitives` for the other half of that rule.

Why this module closes figures
------------------------------
:func:`save_figure` calls ``figure.clf()`` after writing. A ``Figure`` built
directly is an ordinary object and will be collected eventually, but
"eventually" is doing a lot of work when a job set produces thousands of
figures holding array data -- the interpreter has no reason to collect
promptly, because the figures are small objects pointing at large buffers.
Clearing explicitly releases the buffers at a predictable point.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure

from ...core.runtime.errors import SpecError
from ...core.runtime.logging import get_logger

__all__ = ["SUPPORTED_FORMATS", "save_figure"]

_LOGGER = get_logger(__name__)

#: Formats the framework will write. Restricted deliberately: every one of
#: these is lossless or vector, because a figure that will be read carefully
#: should not be a JPEG.
SUPPORTED_FORMATS = frozenset({"png", "svg", "pdf"})


def save_figure(
    figure: Figure,
    directory: Path,
    name: str,
    *,
    figure_format: str = "png",
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """
    Write a figure and return where it went.

    Parameters
    ----------
    figure
        The figure to write.
    directory
        Destination directory, created if absent.
    name
        File stem, without an extension. Any extension present is stripped,
        so passing ``"curve.png"`` with ``figure_format="svg"`` does not
        produce ``curve.png.svg``.
    figure_format
        One of :data:`SUPPORTED_FORMATS`.
    dpi
        Resolution, for raster formats. Ignored for ``svg`` and ``pdf``.
    close
        Whether to release the figure's buffers after writing. See the module
        docstring. Pass ``False`` when the caller intends to display the
        figure as well.

    Returns
    -------
    Path
        The written file.

    Raises
    ------
    SpecError
        If the format is not supported.
    """
    if figure_format not in SUPPORTED_FORMATS:
        raise SpecError(
            f"unsupported figure format {figure_format!r}; "
            f"expected one of {sorted(SUPPORTED_FORMATS)}"
        )

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{Path(name).stem}.{figure_format}"
    figure.savefig(destination, format=figure_format, dpi=dpi, bbox_inches="tight")
    if close:
        # Releases the axes and the array buffers they reference. Figures are
        # not in a global registry here, so this is about promptness rather
        # than about avoiding a leak.
        figure.clf()
    _LOGGER.debug("wrote figure %s", destination)
    return destination
```

---

## 5. `src/rade_qnet/analysis/visuals/jobset.py`

12065 bytes · SHA-256 `56267c058caa227e`

```python
"""
Cross-job comparison: reading forty runs at once.

A single run's figures answer "is this model any good". A job set's answer a
different question: *which of these runs should I look at, and is the set as
a whole healthy?* Forty loss curves do not answer that -- nobody reads forty
loss curves -- so the figures here collapse a set into one view each.

What each figure is for
-----------------------
**Ranking** is the one anybody opens first: which clusters replicate well and
which do not. Drawn as a horizontal bar chart because cluster names are long
and a rotated x-axis label is unreadable at forty entries.

**Dispersion** answers whether the spread across jobs is the interesting
finding. A set whose scores cluster tightly is telling you the configuration
generalises; one that is bimodal is telling you there are two kinds of
cluster in the book, which is a modelling decision rather than a tuning one.

**Status** is for the sets that did not all succeed. A single failure is read
from the manifest; a pattern of failures is read here.

**Wall time** is the operational view: whether the set is dominated by a few
slow jobs, which decides whether more workers would help or whether the long
pole needs attention instead.

Plain data in, figure out
-------------------------
Every function takes mappings of plain numbers rather than a manifest.
``analysis`` may only import ``core``, so it cannot see
:class:`~rade_qnet.orchestration.jobs.manifest.JobSetManifest` -- and that
constraint turns out to be the right interface anyway. A figure that takes
``{job: value}`` can be driven from a manifest, a notebook, a database query
or a test, and the one that mattered most while writing these was the test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from ...core.runtime.errors import ContractError
from .primitives import DEFAULT_FIGSIZE
from .style import PALETTE, figure_style

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "job_status_figure",
    "metric_dispersion_figure",
    "metric_ranking_figure",
    "wall_time_figure",
]

#: Height per bar in a ranking figure. A set can have forty jobs, so the
#: figure grows with the data rather than compressing it: a fixed height
#: would overlap the labels at exactly the size where the figure stops
#: being readable and starts being decorative.
_BAR_HEIGHT_INCHES = 0.28

#: Floor and ceiling on a ranking figure's height, so a two-job set is not a
#: sliver and a four-hundred-job set does not produce an unopenable file.
_MIN_HEIGHT_INCHES = 2.5
_MAX_HEIGHT_INCHES = 20.0

#: Colours for the two outcomes. Red for failure, and the house grey-blue for
#: success: a status chart should make the failures the thing the eye finds.
_SUCCESS_COLOUR = PALETTE[0]
_FAILURE_COLOUR = PALETTE[1]


def metric_ranking_figure(
    values: Mapping[str, float],
    *,
    metric_name: str = "metric",
    ascending: bool = False,
    highlight: Sequence[str] = (),
) -> Figure:
    """
    Rank jobs by one metric, worst to best down the axis.

    Horizontal bars, because cluster identifiers are long. A vertical chart
    at forty entries needs rotated labels, and rotated labels at that count
    are not read -- which defeats the purpose of the figure that gets opened
    first.

    Parameters
    ----------
    values
        Job identifier to metric value. Jobs without the metric should be
        omitted by the caller rather than passed as zero: a zero plots, and
        is indistinguishable from a genuinely zero score.
    metric_name
        Used for the axis label.
    ascending
        Whether lower is better. Controls the sort direction only; no
        attempt is made to guess from the metric's name, because ``loss``
        and ``r2`` are both common and a wrong guess silently inverts the
        reader's conclusion.
    highlight
        Jobs to draw in the comparison colour -- a baseline cluster, or the
        ones a reader is investigating.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ContractError
        If no values are supplied. An empty ranking is a blank image that
        looks like a rendering bug, and the real cause -- no job produced
        this metric -- is worth saying.
    """
    _require(values, what=f"rank jobs by {metric_name!r}")

    # Sorted so the best entry is at the *top* of the drawn figure. Matplotlib
    # draws the first bar at the bottom, so "best last" in the list puts it
    # first on the page, which is where a reader looks.
    ordered = sorted(values.items(), key=lambda item: item[1], reverse=ascending)
    names = [name for name, _ in ordered]
    scores = [score for _, score in ordered]
    highlighted = set(highlight)

    with figure_style():
        figure = Figure(figsize=(DEFAULT_FIGSIZE[0], _height_for(len(names))))
        axes = figure.subplots()
        axes.barh(
            names,
            scores,
            color=[_FAILURE_COLOUR if name in highlighted else _SUCCESS_COLOUR for name in names],
        )
        # A reference at zero is meaningful for every metric that can be
        # negative, and harmless for the ones that cannot: r-squared below
        # zero means "worse than predicting the mean", which is the single
        # most useful threshold on the chart.
        axes.axvline(0.0, color=PALETTE[-1], linewidth=0.8)
        axes.set_xlabel(metric_name)
        axes.set_ylabel("job")
        axes.set_title(f"{metric_name} by job ({len(names)} job(s))")
        figure.tight_layout()
    return figure


def metric_dispersion_figure(
    values: Mapping[str, float],
    *,
    metric_name: str = "metric",
    bins: int = 20,
) -> Figure:
    """
    Show how one metric is distributed across a set.

    The question this answers is whether the spread *is* the finding. A
    tight cluster of scores says the configuration generalises across the
    book; a bimodal one says there are two kinds of cluster in it, which is
    a modelling decision rather than something more tuning will fix.

    Parameters
    ----------
    values
        Job identifier to metric value.
    metric_name
        Used for the axis label.
    bins
        Histogram bin count. Reduced automatically for small sets, where the
        default would draw mostly empty bins and imply a structure that is
        not there.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ContractError
        If no values are supplied.
    """
    _require(values, what=f"show the dispersion of {metric_name!r}")

    scores = np.asarray(list(values.values()), dtype=np.float64)
    # A histogram with more bins than points is noise drawn as structure.
    effective_bins = max(1, min(bins, len(scores)))

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.subplots()
        axes.hist(scores, bins=effective_bins, color=_SUCCESS_COLOUR, edgecolor="white")

        # The median rather than the mean, because a single catastrophic
        # cluster drags a mean somewhere no job actually is -- and a set
        # with one such cluster is exactly when this figure gets opened.
        median = float(np.median(scores))
        axes.axvline(
            median,
            color=_FAILURE_COLOUR,
            linestyle="--",
            linewidth=1.2,
            label=f"median {median:.4g}",
        )
        axes.set_xlabel(metric_name)
        axes.set_ylabel("jobs")
        axes.set_title(f"{metric_name} across {len(scores)} job(s)")
        axes.legend()
        figure.tight_layout()
    return figure


def job_status_figure(statuses: Mapping[str, str]) -> Figure:
    """
    Summarise which jobs succeeded and which did not.

    A single failure is read from the manifest. This is for the pattern: a
    set where a third of the jobs failed is one investigation, not thirteen.

    Parameters
    ----------
    statuses
        Job identifier to status. Anything other than ``"succeeded"`` is
        drawn as a failure, so a status vocabulary that grows later does not
        silently start rendering new states as successes.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ContractError
        If no statuses are supplied.
    """
    _require(statuses, what="summarise job status")

    succeeded = sum(1 for status in statuses.values() if status == "succeeded")
    failed = len(statuses) - succeeded

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.subplots()
        axes.bar(
            ["succeeded", "failed"],
            [succeeded, failed],
            color=[_SUCCESS_COLOUR, _FAILURE_COLOUR],
        )
        # Labelled with the counts, because the whole content of this figure
        # is two numbers and making the reader estimate them off an axis
        # would be a strange thing to do with a chart this simple.
        for index, count in enumerate((succeeded, failed)):
            axes.text(index, count, str(count), ha="center", va="bottom")
        axes.set_ylabel("jobs")
        axes.set_title(f"{succeeded} of {len(statuses)} job(s) succeeded")
        figure.tight_layout()
    return figure


def wall_time_figure(seconds: Mapping[str, float], *, top: int = 20) -> Figure:
    """
    Show where a set's time went.

    The operational question: is the set dominated by a few slow jobs? If it
    is, adding workers will not help much and the long pole is what to look
    at. If the times are even, the set is parallelising as well as it can
    and more workers is the answer.

    Parameters
    ----------
    seconds
        Job identifier to wall time.
    top
        How many of the slowest jobs to draw. A four-hundred-job set's time
        profile is carried entirely by its head, and drawing the tail makes
        the head unreadable.

    Returns
    -------
    matplotlib.figure.Figure
        The figure.

    Raises
    ------
    ContractError
        If no durations are supplied.
    """
    _require(seconds, what="show wall time")

    slowest = sorted(seconds.items(), key=lambda item: item[1])[-top:]
    names = [name for name, _ in slowest]
    durations = [duration for _, duration in slowest]
    total = sum(seconds.values())

    with figure_style():
        figure = Figure(figsize=(DEFAULT_FIGSIZE[0], _height_for(len(names))))
        axes = figure.subplots()
        axes.barh(names, durations, color=_SUCCESS_COLOUR)
        axes.set_xlabel("wall time (s)")
        axes.set_ylabel("job")
        shown = f"slowest {len(names)} of {len(seconds)}" if len(names) < len(seconds) else "by job"
        axes.set_title(f"wall time {shown} — {total:.1f}s total")
        figure.tight_layout()
    return figure


def _height_for(n_bars: int) -> float:
    """
    Choose a figure height that fits a bar chart's labels.

    Parameters
    ----------
    n_bars
        How many bars there are.

    Returns
    -------
    float
        Height in inches, clamped at both ends.
    """
    return float(np.clip(n_bars * _BAR_HEIGHT_INCHES + 1.0, _MIN_HEIGHT_INCHES, _MAX_HEIGHT_INCHES))


def _require(values: Mapping[str, object], *, what: str) -> None:
    """
    Refuse to draw an empty figure.

    An empty chart is a blank image, which reads as a rendering fault. The
    real cause -- no job produced this metric, usually because every job
    failed -- is worth saying out loud.

    Parameters
    ----------
    values
        The mapping to check.
    what
        What the caller was trying to draw, for the message.

    Raises
    ------
    ContractError
        If the mapping is empty.
    """
    if not values:
        raise ContractError(
            f"cannot {what}: no jobs were supplied. A job set whose jobs all "
            f"failed has no metrics to compare, which the manifest reports "
            f"and a figure cannot"
        )
```

---

## 6. `src/rade_qnet/analysis/visuals/primitives.py`

11266 bytes · SHA-256 `ee99c29d23ab2900`

```python
"""
Figure factories: the shared plots every model gets for free.

Every function here obeys one rule, and the rule is what makes this package
reusable: **a figure factory returns a figure and writes nothing.** No paths,
no side effects, no filesystem.

The separation that rule buys
-----------------------------
Because these functions only build figures, the same function serves a report
writer, a notebook, a dashboard and a test. A test can assert on the number of
lines and their data without a temporary directory; a notebook can display the
result without producing a file; a report writer decides where and in what
format to save it. The moment a plotting function also saved its output, each
of those callers would need its own variant.

Writing is :mod:`rade_qnet.analysis.visuals.export`'s job, and only its job.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ...core.runtime.errors import ContractError
from .style import figure_style

__all__ = [
    "metric_comparison_figure",
    "prediction_scatter_figure",
    "residual_histogram_figure",
    "training_curve_figure",
]

#: Default figure size, in inches. Wide enough for a time axis with readable
#: labels, and proportioned to sit in a document without rescaling.
DEFAULT_FIGSIZE = (8.0, 4.5)


def training_curve_figure(
    train_losses: Sequence[float | None],
    validation_losses: Sequence[float | None] = (),
    *,
    title: str = "Training curve",
    best_epoch: int | None = None,
) -> Figure:
    """
    Plot training and validation loss against epoch.

    Works for a gradient loop and for a boosted-tree fit alike, since both
    report a per-round history through
    :class:`~rade_qnet.core.contract.result.FitOutcome`.

    Parameters
    ----------
    train_losses
        One value per epoch. ``None`` entries are rendered as gaps.
    validation_losses
        One value per epoch, or empty when there is no validation split.
    title
        Figure title.
    best_epoch
        Epoch to mark with a vertical rule, normally the one a checkpoint
        restored. Marking it makes the gap between "where training stopped"
        and "which weights were kept" visible, which is otherwise invisible
        in a loss curve and is a common source of confusion when a reported
        metric does not match the final epoch.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the two series have different lengths, which would mean an epoch's
        training and validation losses were recorded against different
        indices.
    """
    if validation_losses and len(validation_losses) != len(train_losses):
        raise ContractError(
            f"training and validation curves have different lengths "
            f"({len(train_losses)} vs {len(validation_losses)}); they must be "
            f"recorded per epoch against the same index"
        )

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        epochs = np.arange(len(train_losses))

        # None is converted to NaN rather than dropped, so matplotlib leaves a
        # visible gap. Dropping the point would shift every later epoch left
        # and silently misalign the curve against the epoch axis.
        axes.plot(epochs, _as_float_array(train_losses), label="train", marker="")
        if validation_losses:
            axes.plot(epochs, _as_float_array(validation_losses), label="validation", marker="")
        if best_epoch is not None:
            axes.axvline(
                best_epoch,
                color="#808080",
                linestyle="--",
                linewidth=1.0,
                label=f"best epoch ({best_epoch})",
            )

        axes.set_xlabel("epoch")
        axes.set_ylabel("loss")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def prediction_scatter_figure(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    title: str = "Predicted against observed",
) -> Figure:
    """
    Scatter predictions against observed values, with a parity line.

    The parity line is what makes the plot diagnostic rather than decorative.
    A cloud tilted shallower than the line is the signature of a model that
    has regressed towards the mean -- it is directionally right and
    systematically under-confident, which no single error metric reveals.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the arrays have different lengths.
    """
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.size != observed.size:
        raise ContractError(
            f"predictions and targets have different lengths ({predicted.size} vs {observed.size})"
        )

    with figure_style():
        figure = Figure(figsize=(5.5, 5.5))
        axes = figure.add_subplot()
        # Small markers with transparency, because a dense scatter of
        # opaque points shows only its outline and hides where the mass is.
        axes.scatter(observed, predicted, s=8, alpha=0.4, edgecolors="none")

        if predicted.size and observed.size:
            lowest = float(min(observed.min(), predicted.min()))
            highest = float(max(observed.max(), predicted.max()))
            axes.plot(
                [lowest, highest],
                [lowest, highest],
                color="#808080",
                linestyle="--",
                linewidth=1.0,
                label="parity",
            )
            axes.legend(loc="best")
            # Equal aspect with equal limits, so a visual departure from the
            # parity line is a real departure and not an artefact of scaling.
            axes.set_xlim(lowest, highest)
            axes.set_ylim(lowest, highest)
            axes.set_aspect("equal", adjustable="box")

        axes.set_xlabel("observed")
        axes.set_ylabel("predicted")
        axes.set_title(title)
        figure.tight_layout()
        return figure


def residual_histogram_figure(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
    *,
    title: str = "Residual distribution",
    bins: int = 40,
) -> Figure:
    """
    Plot the distribution of residuals, with zero and the mean marked.

    Both markers are drawn because the distance between them *is* the bias,
    and showing it is more informative than reporting the number alone: a
    distribution centred away from zero is immediately visible as a
    systematic error rather than noise.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.
    title
        Figure title.
    bins
        Histogram bin count.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the arrays have different lengths.
    """
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    observed = np.asarray(targets, dtype=np.float64).reshape(-1)
    if predicted.size != observed.size:
        raise ContractError(
            f"predictions and targets have different lengths ({predicted.size} vs {observed.size})"
        )
    residuals = predicted - observed

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.hist(residuals, bins=bins, edgecolor="white", linewidth=0.5)
        axes.axvline(0.0, color="#808080", linestyle="--", linewidth=1.0, label="zero")
        if residuals.size:
            mean_residual = float(np.mean(residuals))
            axes.axvline(
                mean_residual,
                color="#c0504d",
                linestyle="-",
                linewidth=1.2,
                label=f"mean ({mean_residual:+.4g})",
            )
        axes.set_xlabel("residual (predicted - observed)")
        axes.set_ylabel("count")
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def metric_comparison_figure(
    metrics: Mapping[str, float],
    baseline: Mapping[str, float] | None = None,
    *,
    title: str = "Metrics against baseline",
) -> Figure:
    """
    Plot metrics as bars, beside a naive baseline where one is available.

    Only metrics present in both mappings are compared; a metric present in
    one is still shown, with no counterpart bar.

    Parameters
    ----------
    metrics
        Model metrics.
    baseline
        Naive baseline metrics, from
        :func:`~rade_qnet.analysis.metrics.regression.baseline_metrics`.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If there are no metrics to plot.
    """
    if not metrics:
        raise ContractError("cannot plot a metric comparison with no metrics")

    names = sorted(metrics)
    positions = np.arange(len(names))
    width = 0.38 if baseline else 0.6

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.bar(
            positions - (width / 2 if baseline else 0.0),
            [metrics[name] for name in names],
            width=width,
            label="model",
        )
        if baseline:
            axes.bar(
                positions + width / 2,
                # Missing baseline metrics plot as zero-height bars rather
                # than shifting the remaining bars out of alignment with
                # their labels.
                [baseline.get(name, 0.0) for name in names],
                width=width,
                label="baseline",
            )
            axes.legend(loc="best")
        axes.set_xticks(positions)
        axes.set_xticklabels(names, rotation=30, ha="right")
        axes.set_ylabel("value")
        axes.set_title(title)
        # Metrics such as bias and r2 are legitimately negative, so a zero
        # rule is drawn to keep the sign readable.
        axes.axhline(0.0, color="#808080", linewidth=0.8)
        figure.tight_layout()
        return figure


def _as_float_array(values: Sequence[float | None]) -> NDArray[np.float64]:
    """
    Convert a series that may contain ``None`` into an array with ``NaN`` gaps.

    Parameters
    ----------
    values
        One value per epoch, possibly with missing entries.

    Returns
    -------
    numpy.ndarray
        The series, with ``None`` replaced by ``NaN`` so matplotlib renders a
        gap rather than interpolating across a missing epoch.
    """
    return np.array([np.nan if value is None else float(value) for value in values])
```

---

## 7. `src/rade_qnet/analysis/visuals/style.py`

3635 bytes · SHA-256 `f6b3228dede4416b`

```python
"""
A consistent house style, applied without touching global state.

Why this module avoids ``pyplot`` entirely
------------------------------------------
Everything in ``visuals`` constructs :class:`matplotlib.figure.Figure` objects
directly rather than going through ``pyplot``, and the reason is not stylistic
preference.

``pyplot`` maintains a **global figure registry**. A figure created with
``plt.subplots()`` stays alive until explicitly closed, so a job set producing
twelve figures per member across three hundred members accumulates thirty-six
hundred live figures and exhausts memory. Worse, the warning matplotlib emits
about this is easy to suppress and easy to miss.

``pyplot`` also performs **global backend selection** at import. In a worker
process with no display, that has historically meant an attempted GUI backend
and a crash. Constructing a ``Figure`` requires no backend at all until the
figure is saved.

So: construct ``Figure`` directly, and let it be garbage collected like any
other object. :func:`figure_style` likewise uses ``rc_context``, which restores
the previous settings on exit, rather than mutating ``matplotlib.rcParams`` --
a framework that permanently changes a user's plotting defaults because they
imported it is a framework that has overstepped.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import matplotlib as mpl

__all__ = ["PALETTE", "RC_PARAMS", "figure_style"]

#: The house palette. Ordered so the first entries are the ones a two-series
#: plot will use, and chosen to stay distinguishable in greyscale -- figures
#: end up in printed documents more often than anyone plans for.
PALETTE: tuple[str, ...] = (
    "#1f4e79",  # deep blue: the primary series, usually the model
    "#c0504d",  # brick red: the comparison, usually the baseline or target
    "#4f81bd",  # mid blue
    "#9bbb59",  # olive
    "#8064a2",  # violet
    "#4bacc6",  # teal
    "#f79646",  # orange
    "#808080",  # grey: for de-emphasised or reference content
)

#: Style parameters applied inside :func:`figure_style`. Deliberately modest:
#: readable sizes, light gridlines, no chart junk.
RC_PARAMS: Mapping[str, object] = {
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "lines.linewidth": 1.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}


@contextmanager
def figure_style(overrides: Mapping[str, object] | None = None) -> Iterator[None]:
    """
    Apply the house style for the duration of a block.

    Uses ``rc_context``, so the caller's settings are restored on exit --
    including when the block raises. See the module docstring for why that
    matters.

    Parameters
    ----------
    overrides
        Additional or replacement rc parameters for this block only.

    Yields
    ------
    None
        The block runs with the style applied.

    Examples
    --------
    >>> from matplotlib.figure import Figure
    >>> with figure_style({"font.size": 14}):
    ...     figure = Figure(figsize=(4, 3))
    ...     axes = figure.add_subplot()
    ...     _ = axes.plot([0, 1], [1, 0])
    """
    parameters = dict(RC_PARAMS)
    if overrides:
        parameters.update(overrides)
    with mpl.rc_context(parameters):
        yield
```

---

## 8. `src/rade_qnet/analysis/visuals/training.py`

12144 bytes · SHA-256 `6449731d6368dfe8`

```python
"""
Training diagnostics: what happened during the fit.

:func:`training_curve_figure` already lives in :mod:`.primitives`, because the
loss curve is the one figure every model gets whatever its engine. The three
here are the traces that explain a curve rather than show it.

Why a gradient-norm trace is worth a figure
-------------------------------------------
The two most common training pathologies are both invisible in a loss curve
and both obvious here.

A gradient norm growing without bound is a run about to diverge; the loss
curve only shows it after it has already happened, by which point the
parameters are ``NaN`` and the cause is gone. A norm collapsing towards zero
is a dead network -- a saturated activation, a learning rate that overshot
once and never recovered -- and its loss curve simply goes flat, which is
indistinguishable from convergence.

Both are one glance apart in a figure that costs nothing to produce, which is
why :class:`~rade_qnet.engines.torch.callbacks.GradientNorms` records the norm
by default rather than on request.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ...core.contract.result import FitOutcome
from ...core.runtime.errors import ContractError
from .primitives import DEFAULT_FIGSIZE
from .style import figure_style

__all__ = [
    "epoch_timing_figure",
    "gradient_norm_figure",
    "learning_rate_figure",
    "training_diagnostics_figure",
]

#: Metric keys the gradient tracker writes, in the order they are drawn.
_NORM_KEYS = ("grad_norm_mean", "grad_norm_max")

#: Metric key for the clipped fraction, drawn on a second axis.
_CLIPPED_KEY = "grad_clipped_fraction"


def _series(outcome: FitOutcome, key: str) -> NDArray[np.float64]:
    """
    Extract one per-epoch metric from a fit outcome.

    Parameters
    ----------
    outcome
        The fit outcome.
    key
        Metric name, as recorded in each record's ``metrics``.

    Returns
    -------
    numpy.ndarray
        One value per epoch, with ``NaN`` where the metric was absent. ``NaN``
        rather than zero, so a gap renders as a gap: a gradient norm of zero
        and an unrecorded gradient norm mean entirely different things.
    """
    return np.array(
        [float(record.metrics.get(key, np.nan)) for record in outcome.history],
        dtype=np.float64,
    )


def _require_history(outcome: FitOutcome, *, what: str) -> None:
    """
    Raise if a fit outcome has no epochs to plot.

    Parameters
    ----------
    outcome
        The fit outcome.
    what
        Name of the figure, for the message.

    Raises
    ------
    ContractError
        If the history is empty.
    """
    if not outcome.history:
        raise ContractError(
            f"{what} needs at least one epoch of history; this fit recorded none, "
            f"which usually means it failed before its first epoch completed"
        )


def learning_rate_figure(outcome: FitOutcome, *, title: str = "Learning rate") -> Figure:
    """
    Plot the learning rate against epoch.

    Drawn on a logarithmic axis, because a schedule that matters spans orders
    of magnitude -- a cosine decay to a thousandth of its start is a straight
    line here and an indistinguishable collapse to zero on a linear one.

    Parameters
    ----------
    outcome
        The fit outcome.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the fit recorded no epochs.
    """
    _require_history(outcome, what="learning_rate_figure")
    rates = np.array(
        [
            np.nan if record.learning_rate is None else record.learning_rate
            for record in outcome.history
        ],
        dtype=np.float64,
    )

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.plot(np.arange(rates.size), rates, marker="")
        # Only when every recorded rate is positive: a log axis with a zero or
        # a NaN in the data silently drops those points.
        if np.all(np.isfinite(rates)) and np.all(rates > 0.0):
            axes.set_yscale("log")
        axes.set_xlabel("epoch")
        axes.set_ylabel("learning rate")
        axes.set_title(title)
        figure.tight_layout()
        return figure


def gradient_norm_figure(outcome: FitOutcome, *, title: str = "Gradient norm") -> Figure:
    """
    Plot the mean and maximum gradient norm, and the clipped fraction.

    Parameters
    ----------
    outcome
        The fit outcome.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the fit recorded no epochs, or recorded no gradient norms at all --
        which means the tracker was not attached, and an empty figure would
        suggest the norms were zero.
    """
    _require_history(outcome, what="gradient_norm_figure")
    series = {key: _series(outcome, key) for key in _NORM_KEYS}
    if all(np.all(np.isnan(values)) for values in series.values()):
        raise ContractError(
            f"this fit recorded no gradient norms, so there is nothing to plot. "
            f"Available metrics are "
            f"{sorted(outcome.history[0].metrics)}; attach a GradientNorms tracker "
            f"to record them"
        )

    epochs = np.arange(outcome.n_epochs)
    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        for key, values in series.items():
            axes.plot(epochs, values, marker="", label=key.replace("grad_norm_", ""))
        # Only when something is positive. A log axis given all-zero data warns
        # and then silently renders an empty panel, which reads as a missing
        # figure rather than as the dead network it actually depicts.
        if _has_positive(series.values()):
            axes.set_yscale("log")
        axes.set_xlabel("epoch")
        axes.set_ylabel("gradient norm")

        clipped = _series(outcome, _CLIPPED_KEY)
        if not np.all(np.isnan(clipped)):
            # On a twinned axis: a fraction in [0, 1] and a norm spanning
            # decades cannot share one scale, and plotting them together
            # anyway would flatten one of the two into a straight line.
            right = axes.twinx()
            right.plot(epochs, clipped, marker="", linestyle=":", color="#808080")
            right.set_ylabel("fraction of steps clipped")
            right.set_ylim(0.0, 1.0)

        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def epoch_timing_figure(outcome: FitOutcome, *, title: str = "Epoch wall time") -> Figure:
    """
    Plot the wall time of each epoch.

    Flat is the expected shape, which is what makes this useful: a rising
    trend means something is accumulating -- a growing buffer, a leak, device
    memory filling and spilling -- and a sawtooth usually means contention
    with other jobs on the same machine. Neither shows up anywhere else.

    Parameters
    ----------
    outcome
        The fit outcome.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the fit recorded no epochs.
    """
    _require_history(outcome, what="epoch_timing_figure")
    seconds = np.array([record.seconds for record in outcome.history], dtype=np.float64)

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot()
        axes.bar(np.arange(seconds.size), seconds, width=0.9)
        axes.axhline(
            float(seconds.mean()),
            color="#808080",
            linestyle="--",
            linewidth=1.0,
            label=f"mean {seconds.mean():.2f}s",
        )
        axes.set_xlabel("epoch")
        axes.set_ylabel("seconds")
        axes.set_title(f"{title} (total {seconds.sum():.1f}s)")
        axes.legend(loc="best")
        figure.tight_layout()
        return figure


def training_diagnostics_figure(
    outcome: FitOutcome, *, title: str = "Training diagnostics"
) -> Figure:
    """
    Combine the loss curve, learning rate and gradient norm in one figure.

    The panel a reader opens first. Having the three stacked on a shared epoch
    axis is what makes the relationship between them legible: a loss that
    flattens at the same epoch the gradient norm collapses is a dead network,
    while a loss that flattens as the learning rate decays to nothing has
    simply finished.

    Parameters
    ----------
    outcome
        The fit outcome.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        The figure. Nothing is written.

    Raises
    ------
    ContractError
        If the fit recorded no epochs.
    """
    _require_history(outcome, what="training_diagnostics_figure")
    epochs = np.arange(outcome.n_epochs)

    with figure_style():
        figure = Figure(figsize=(DEFAULT_FIGSIZE[0], 8.0))
        # `sharex` rather than three separate axes, so that reading a feature
        # off one panel lands on the same epoch in the others.
        loss_axes, rate_axes, norm_axes = figure.subplots(3, 1, sharex=True)

        loss_axes.plot(epochs, _losses(outcome, "train_loss"), marker="", label="train")
        validation = _losses(outcome, "val_loss")
        if not np.all(np.isnan(validation)):
            loss_axes.plot(epochs, validation, marker="", label="validation")
        if outcome.best_epoch is not None:
            loss_axes.axvline(
                outcome.best_epoch,
                color="#808080",
                linestyle="--",
                linewidth=1.0,
                label=f"best epoch ({outcome.best_epoch})",
            )
        loss_axes.set_ylabel("loss")
        loss_axes.legend(loc="best")

        rates = np.array(
            [
                np.nan if record.learning_rate is None else record.learning_rate
                for record in outcome.history
            ],
            dtype=np.float64,
        )
        rate_axes.plot(epochs, rates, marker="", color="#6a1b9a")
        if np.all(np.isfinite(rates)) and np.all(rates > 0.0):
            rate_axes.set_yscale("log")
        rate_axes.set_ylabel("learning rate")

        for key in _NORM_KEYS:
            values = _series(outcome, key)
            if not np.all(np.isnan(values)):
                norm_axes.plot(epochs, values, marker="", label=key.replace("grad_norm_", ""))
        norm_axes.set_ylabel("gradient norm")
        norm_axes.set_xlabel("epoch")
        if norm_axes.get_legend_handles_labels()[0]:
            if _has_positive(_series(outcome, key) for key in _NORM_KEYS):
                norm_axes.set_yscale("log")
            norm_axes.legend(loc="best")

        figure.suptitle(title)
        figure.tight_layout()
        return figure


def _has_positive(serieses: Iterable[NDArray[np.float64]]) -> bool:
    """
    Return whether any series holds a value a log axis can render.

    Parameters
    ----------
    serieses
        The series to inspect.

    Returns
    -------
    bool
        True if at least one finite positive value exists anywhere.
    """
    return any(bool(np.any(np.isfinite(values) & (values > 0.0))) for values in serieses)


def _losses(outcome: FitOutcome, name: str) -> NDArray[np.float64]:
    """
    Extract a loss series from a fit outcome.

    Parameters
    ----------
    outcome
        The fit outcome.
    name
        ``train_loss`` or ``val_loss``.

    Returns
    -------
    numpy.ndarray
        One value per epoch, with ``NaN`` for an absent validation loss.
    """
    return np.array(
        [
            np.nan if getattr(record, name) is None else float(getattr(record, name))
            for record in outcome.history
        ],
        dtype=np.float64,
    )
```

---

## 9. `src/rade_qnet/analysis/visuals/tuning.py`

16759 bytes · SHA-256 `b0d6b5a02cec0ab7`

```python
"""
Reading a search: what it explored, what mattered, and whether it finished.

A table of forty trials is not a thing anyone reads. The three figures here
answer the three questions a search is run to answer, and each is drawn to
make the corresponding mistake visible rather than merely to display data.

**History** -- did the search converge, or is the budget the binding
constraint? Drawn with a running best line over the per-trial points,
because the shape of that line is the answer: still descending at the last
trial means the search stopped because it ran out of budget, not because it
found the optimum, and the honest conclusion is "run more trials" rather
than "this is the best configuration".

**Importance** -- which knobs mattered? A search that varied five parameters
usually found that one of them did everything. Knowing which saves the next
search four dimensions.

**Parallel coordinates** -- what did the good trials have in common? The
only one of the three that shows *interactions*, which is where the two
summary views are blind: two parameters that each look unimportant
marginally can matter a great deal together.

Plain data in, figure out
-------------------------
As elsewhere in ``analysis``, these take sequences and mappings of plain
numbers rather than a
:class:`~rade_qnet.core.contract.result.TuningResult`. ``analysis`` may only
import ``core``, and the looser interface is better anyway.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.figure import Figure

from ...core.runtime.errors import ContractError
from .primitives import DEFAULT_FIGSIZE
from .style import PALETTE, figure_style

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    "parallel_coordinates_figure",
    "parameter_importance_figure",
    "trial_history_figure",
]

#: Fewest trials a correlation is computed from. Below this the coefficient
#: is dominated by whichever two points happened to be drawn, and an
#: importance chart built on four trials would be read with the same
#: confidence as one built on four hundred.
_MIN_FOR_IMPORTANCE = 5


def trial_history_figure(
    objectives: Sequence[float | None],
    *,
    direction: str = "minimise",
    objective: str = "objective",
    title: str = "Search history",
) -> Figure:
    """
    Show every trial's score and the best found so far.

    The running best is the point of the figure. Its shape says whether the
    search converged or merely stopped: a line still improving at the last
    trial means the budget was the binding constraint, and the honest
    conclusion is "run more trials" rather than "this is the best
    configuration".

    Failed trials are drawn as gaps rather than omitted. A search with
    eleven failures and a clean-looking curve is a different object from
    one with none, and compacting the x-axis hides the difference.

    Parameters
    ----------
    objectives
        One value per trial in trial order, with ``None`` for a trial that
        failed or produced no objective.
    direction
        ``"minimise"`` or ``"maximise"``, which decides what "best so far"
        means. Passed rather than inferred, for the reason the
        specification records it: inferring from a metric name eventually
        gets a custom metric backwards and draws the worst trial as the
        winner.
    objective
        Metric name, for the axis label.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        Scored trials as points, with the running best as a step line and
        the winner marked.

    Raises
    ------
    ContractError
        If there are no trials, or none of them scored.
    """
    if not objectives:
        raise ContractError(
            "there are no trials to draw. An empty chart reads as a rendering "
            "fault rather than as a search that produced nothing"
        )

    trials = np.arange(len(objectives))
    values = np.array(
        [np.nan if value is None else float(value) for value in objectives], dtype=np.float64
    )
    scored = np.isfinite(values)
    if not scored.any():
        raise ContractError(
            f"none of the {len(objectives)} trial(s) produced an objective, so "
            f"there is no history to draw"
        )

    running = _running_best(values, maximise=direction == "maximise")
    best_index = int(np.nanargmax(values) if direction == "maximise" else np.nanargmin(values))

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot(111)
        axes.scatter(trials[scored], values[scored], s=24, color=PALETTE[0], label="trial")
        axes.step(trials, running, where="post", color=PALETTE[1], label="best so far")
        axes.scatter(
            [best_index],
            [values[best_index]],
            s=90,
            marker="*",
            color=PALETTE[1],
            zorder=3,
            label=f"best (#{best_index})",
        )
        failures = int((~scored).sum())
        axes.set_xlabel(f"Trial ({failures} failed)" if failures else "Trial")
        axes.set_ylabel(objective)
        axes.set_title(title)
        axes.legend(loc="best")
        figure.tight_layout()
    return figure


def parameter_importance_figure(
    overrides: Sequence[Mapping[str, object]],
    objectives: Sequence[float | None],
    *,
    objective: str = "objective",
    title: str = "Parameter influence on the objective",
) -> Figure:
    """
    Rank the searched parameters by how much they moved the objective.

    Measured as the absolute rank correlation between a parameter's value
    and the objective. Rank rather than linear, because a learning rate
    sampled on a log scale has a monotonic but strongly non-linear effect,
    and a linear coefficient would report it as unimportant. Absolute,
    because the chart answers "did this matter", not "which way" -- the
    direction is read from the parallel coordinates view.

    Non-numeric parameters are skipped rather than encoded. Assigning
    arbitrary integers to a categorical and correlating against them
    produces a number that depends entirely on the order they happened to
    be listed in.

    Parameters
    ----------
    overrides
        One flat proposal per trial, dotted path to value.
    objectives
        One objective per trial, aligned with ``overrides``.
    objective
        Metric name, for the axis label.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        A horizontal bar chart, most influential at the top.

    Raises
    ------
    ContractError
        If the two sequences disagree in length, if too few trials scored,
        or if no parameter is numeric.
    """
    if len(overrides) != len(objectives):
        raise ContractError(
            f"{len(overrides)} proposal(s) cannot be paired with "
            f"{len(objectives)} objective(s); each trial needs both"
        )

    values = np.array(
        [np.nan if value is None else float(value) for value in objectives], dtype=np.float64
    )
    scored = np.isfinite(values)
    if int(scored.sum()) < _MIN_FOR_IMPORTANCE:
        raise ContractError(
            f"only {int(scored.sum())} trial(s) scored; an influence chart needs "
            f"at least {_MIN_FOR_IMPORTANCE}. Below that the coefficient is "
            f"decided by whichever two points happened to be drawn, and the "
            f"chart would be read with the confidence of a much larger search"
        )

    influence: dict[str, float] = {}
    for path in sorted({key for proposal in overrides for key in proposal}):
        column = _numeric_column(overrides, path, scored)
        if column is None:
            continue
        influence[path] = _rank_correlation(column, values[scored])

    if not influence:
        raise ContractError(
            "no searched parameter is numeric, so none can be correlated against "
            "the objective. Encoding a categorical as integers would produce a "
            "number that depends on the order its values were listed in"
        )

    ranked = sorted(influence.items(), key=lambda pair: pair[1])

    with figure_style():
        figure = Figure(figsize=(DEFAULT_FIGSIZE[0], max(3.0, 0.4 * len(ranked) + 1.5)))
        axes = figure.add_subplot(111)
        axes.barh([path for path, _ in ranked], [value for _, value in ranked], color=PALETTE[0])
        axes.set_xlim(0.0, 1.0)
        axes.set_xlabel(f"|rank correlation| with {objective}")
        axes.set_title(title)
        figure.tight_layout()
    return figure


def parallel_coordinates_figure(
    overrides: Sequence[Mapping[str, object]],
    objectives: Sequence[float | None],
    *,
    direction: str = "minimise",
    objective: str = "objective",
    title: str = "Trials across the search space",
) -> Figure:
    """
    Draw each trial as a line across the parameters it was given.

    The only one of the three views that shows interactions, which is
    exactly where the summary views are blind: two parameters that each
    look unimportant on their own can matter a great deal together, and a
    bar chart of marginal influence would report both as noise.

    Each axis is scaled to its own range, because the parameters have
    incomparable units -- a learning rate of 0.001 and a hidden size of 256
    on one axis would render the first as a flat line at zero. Lines are
    coloured by objective, so the question the figure answers -- what did
    the good trials have in common -- is read from the colour rather than
    from the geometry.

    Parameters
    ----------
    overrides
        One flat proposal per trial.
    objectives
        One objective per trial.
    direction
        Which end of the colour scale is good.
    objective
        Metric name, for the colour bar label.
    title
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
        One line per scored trial.

    Raises
    ------
    ContractError
        If the sequences disagree in length, or if fewer than two numeric
        parameters were searched -- a parallel coordinates plot of one axis
        is a strip chart, and of none is blank.
    """
    if len(overrides) != len(objectives):
        raise ContractError(
            f"{len(overrides)} proposal(s) cannot be paired with "
            f"{len(objectives)} objective(s); each trial needs both"
        )

    values = np.array(
        [np.nan if value is None else float(value) for value in objectives], dtype=np.float64
    )
    scored = np.isfinite(values)

    paths = sorted({key for proposal in overrides for key in proposal})
    columns = {
        path: column
        for path in paths
        if (column := _numeric_column(overrides, path, scored)) is not None
    }
    if len(columns) < 2:  # noqa: PLR2004
        raise ContractError(
            f"a parallel coordinates plot needs at least two numeric parameters; "
            f"this search has {len(columns)}. With one it is a strip chart, and "
            f"with none it is blank"
        )

    axes_names = list(columns)
    scaled = np.column_stack([_to_unit(columns[path]) for path in axes_names])
    scores = values[scored]

    with figure_style():
        figure = Figure(figsize=DEFAULT_FIGSIZE)
        axes = figure.add_subplot(111)
        # Reversed for a minimised objective, so that "good" is the same end
        # of the colour scale whichever way the metric runs -- a reader
        # should not have to check the direction to know which lines to look
        # at.
        colourmap = "viridis_r" if direction == "minimise" else "viridis"
        mapping = axes.scatter(np.zeros_like(scores), scaled[:, 0], c=scores, cmap=colourmap, s=0)
        for row, score in zip(scaled, scores, strict=True):
            axes.plot(
                range(len(axes_names)),
                row,
                color=mapping.to_rgba(score),
                alpha=0.8,
                linewidth=1.2,
            )

        axes.set_xticks(range(len(axes_names)))
        axes.set_xticklabels(axes_names, rotation=30, ha="right")
        axes.set_ylabel("Value, scaled to each axis's own range")
        axes.set_ylim(-0.05, 1.05)
        axes.set_title(title)
        figure.colorbar(mapping, ax=axes, label=objective)
        figure.tight_layout()
    return figure


def _running_best(values: np.ndarray, *, maximise: bool) -> np.ndarray:
    """
    Return the best value seen up to and including each position.

    Parameters
    ----------
    values
        One per trial, with ``NaN`` for a trial that did not score.
    maximise
        Which direction counts as better.

    Returns
    -------
    numpy.ndarray
        The running best, carrying the previous value across a gap so the
        line is continuous -- a failed trial did not undo the progress
        before it.
    """
    best = np.empty_like(values)
    incumbent = -np.inf if maximise else np.inf
    for index, value in enumerate(values):
        if np.isfinite(value) and ((value > incumbent) if maximise else (value < incumbent)):
            incumbent = float(value)
        best[index] = incumbent
    return best


def _numeric_column(
    overrides: Sequence[Mapping[str, object]], path: str, scored: np.ndarray
) -> np.ndarray | None:
    """
    Extract one parameter's values across the scored trials, if it is numeric.

    Parameters
    ----------
    overrides
        One flat proposal per trial.
    path
        The dotted path to extract.
    scored
        Which trials produced an objective.

    Returns
    -------
    numpy.ndarray or None
        The column, or ``None`` when the parameter is categorical, absent
        from some trial, or constant. A constant column is excluded
        because a correlation against it is undefined and a parallel axis
        for it is a horizontal line that tells the reader nothing.
    """
    raw = [proposal.get(path) for proposal, keep in zip(overrides, scored, strict=True) if keep]
    if any(value is None or isinstance(value, bool) for value in raw):
        return None
    if not all(isinstance(value, int | float) for value in raw):
        return None

    column = np.array([float(value) for value in raw], dtype=np.float64)  # type: ignore[arg-type]
    if float(np.ptp(column)) == 0.0:
        return None
    return column


def _rank_correlation(column: np.ndarray, objectives: np.ndarray) -> float:
    """
    Return the absolute Spearman correlation between two columns.

    Computed as Pearson over the ranks, which is what Spearman is, rather
    than taken from SciPy -- the framework does not otherwise depend on it,
    and a dependency added for one coefficient is a dependency every
    installation carries.

    Parameters
    ----------
    column
        The parameter's values.
    objectives
        The objective values, aligned.

    Returns
    -------
    float
        Between zero and one. Zero when either side is constant, which is
        the honest answer: a parameter that never varied cannot have
        mattered.
    """
    ranked_column = _ranks(column)
    ranked_objective = _ranks(objectives)
    if float(np.std(ranked_column)) == 0.0 or float(np.std(ranked_objective)) == 0.0:
        return 0.0
    return abs(float(np.corrcoef(ranked_column, ranked_objective)[0, 1]))


def _ranks(values: np.ndarray) -> np.ndarray:
    """
    Return the rank of each value, averaging ties.

    Ties are averaged rather than broken by position. Breaking them by
    position would make the coefficient depend on the order the trials ran
    in, so a re-run that happened to draw the same values in a different
    sequence would report a different influence.

    Parameters
    ----------
    values
        The column to rank.

    Returns
    -------
    numpy.ndarray
        One rank per value.
    """
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)

    for value in np.unique(values):
        tied = values == value
        if int(tied.sum()) > 1:
            ranks[tied] = float(np.mean(ranks[tied]))
    return ranks


def _to_unit(column: np.ndarray) -> np.ndarray:
    """
    Scale a column onto the unit interval.

    Parameters
    ----------
    column
        Values to scale. Guaranteed non-constant by
        :func:`_numeric_column`, so there is no division by zero to guard.

    Returns
    -------
    numpy.ndarray
        The scaled column.
    """
    low, high = float(np.min(column)), float(np.max(column))
    return (column - low) / (high - low)
```

