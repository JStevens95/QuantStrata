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
why :class:`~rade_qnet.engines.torch.training.callbacks.GradientNorms` records the norm
by default rather than on request.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from ...core.contract.result import FitOutcome
from ...core.lifecycle.errors import ContractError
from .figures import DEFAULT_FIGSIZE
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
