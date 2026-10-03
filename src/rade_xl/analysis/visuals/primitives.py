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

Writing is :mod:`rade_xl.analysis.visuals.export`'s job, and only its job.
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
    :class:`~rade_xl.core.contract.result.FitOutcome`.

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
        :func:`~rade_xl.analysis.metrics.regression.baseline_metrics`.
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
