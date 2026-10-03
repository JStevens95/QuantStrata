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
:class:`~rade_xl.core.contract.result.TuningResult`. ``analysis`` may only
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
        mapping = axes.scatter(
            np.zeros_like(scores), scaled[:, 0], c=scores, cmap=colourmap, s=0
        )
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
    raw = [
        proposal.get(path)
        for proposal, keep in zip(overrides, scored, strict=True)
        if keep
    ]
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
