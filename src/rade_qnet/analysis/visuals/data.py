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

from ...core.lifecycle.errors import ContractError
from .figures import DEFAULT_FIGSIZE
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
