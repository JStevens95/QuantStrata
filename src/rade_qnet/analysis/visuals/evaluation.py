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
