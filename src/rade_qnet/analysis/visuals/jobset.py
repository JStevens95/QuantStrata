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

from ...core.lifecycle.errors import ContractError
from .figures import DEFAULT_FIGSIZE
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
