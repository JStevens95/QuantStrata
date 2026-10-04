"""
Input data quality: completeness, staleness and coverage.

A model trained on forty per cent missing history should say so in its own
artifacts. These metrics exist so that the statement is a number in a bundle
rather than something a reader has to go and check.

Why this is a metric and not a validation
-----------------------------------------
Nothing here raises. A sparse or stale input is a fact about the data, not a
configuration error, and refusing to train on it would be the wrong call --
the sparse period may be exactly what the model is for.

What is wrong is training on it *silently*. So every function here returns a
number, the quality report writes those numbers next to the headline metrics,
and :func:`quality_warnings` turns the ones that cross a threshold into text a
human will read. The decision stays with the person; the information does not
stay hidden.

Staleness, and why it is measured separately from missingness
-------------------------------------------------------------
A forward-filled series has no missing values and may still carry almost no
information. A P&L series stale for thirty consecutive days reads as complete
to any completeness check, while a model consuming it learns from thirty
copies of one observation and reports thirty times the confidence it has
earned. :func:`staleness` counts consecutive repeats, which is the only way
that shows up in a number.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...core.lifecycle.errors import ContractError

__all__ = [
    "completeness",
    "entity_coverage",
    "quality_metrics",
    "quality_warnings",
    "staleness",
]

#: Completeness below this is reported as a warning. Two thirds is a
#: judgement, not a law: it is the point at which a standard deviation
#: estimated from the column starts to be noticeably noisier than the data
#: volume suggests.
COMPLETENESS_WARNING = 0.67

#: Staleness above this is reported as a warning. A tenth of observations
#: being repeats is normal for a series with holidays; a quarter is not.
STALENESS_WARNING = 0.25

#: A matrix is samples by features and nothing else.
_MATRIX_RANK = 2


def _as_matrix(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Return a two-dimensional float view of an input array.

    Parameters
    ----------
    values
        A one- or two-dimensional array.

    Returns
    -------
    numpy.ndarray
        A two-dimensional float64 array, with a one-dimensional input treated
        as a single column.

    Raises
    ------
    ContractError
        If the array has three or more dimensions. A windowed feature tensor
        reaching here would make every count below wrong by a factor of the
        window length, so it is refused rather than flattened.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != _MATRIX_RANK:
        raise ContractError(
            f"quality metrics expect a 1- or 2-dimensional array, received shape "
            f"{array.shape}. Pass the feature matrix before windowing: a window "
            f"axis would count every scenario once per window"
        )
    return array


def completeness(values: NDArray[np.floating]) -> float:
    """
    Return the fraction of entries that are present and finite.

    Infinities count as missing alongside ``NaN``. An infinity is not a
    measurement, and treating it as present would report a column as complete
    while any mean computed from it is also infinite.

    Parameters
    ----------
    values
        Feature matrix or a single series.

    Returns
    -------
    float
        Fraction in ``[0, 1]``. One for an empty input, since there is nothing
        missing from nothing -- reporting zero would make an empty split look
        like the worst possible data.
    """
    matrix = _as_matrix(values)
    if matrix.size == 0:
        return 1.0
    return float(np.isfinite(matrix).sum() / matrix.size)


def column_completeness(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Return the completeness of each column.

    Reported per column as well as overall, because the two tell different
    stories: ninety per cent overall completeness is unremarkable if it is
    spread evenly and serious if one column is entirely absent.

    Parameters
    ----------
    values
        Feature matrix.

    Returns
    -------
    numpy.ndarray
        One fraction per column.
    """
    matrix = _as_matrix(values)
    if matrix.shape[0] == 0:
        return np.ones(matrix.shape[1], dtype=np.float64)
    return np.isfinite(matrix).mean(axis=0).astype(np.float64)


def staleness(values: NDArray[np.floating]) -> float:
    """
    Return the fraction of observations identical to the one before them.

    Parameters
    ----------
    values
        Feature matrix or a single series, in scenario order. Order matters:
        this measures consecutive repeats, so a shuffled array reports
        meaningless staleness.

    Returns
    -------
    float
        Fraction in ``[0, 1]``. Zero for fewer than two rows, where
        consecutive repetition is not defined.
    """
    matrix = _as_matrix(values)
    if matrix.shape[0] < _MATRIX_RANK:
        return 0.0

    previous, current = matrix[:-1], matrix[1:]
    # A pair of NaNs is not counted as a repeat: `nan == nan` is already
    # False, which is the behaviour wanted here -- missingness is
    # `completeness`'s business and counting it twice would double-penalise
    # the same gap.
    repeats = np.isclose(current, previous, rtol=0.0, atol=0.0, equal_nan=False)
    return float(repeats.sum() / repeats.size)


def entity_coverage(
    entity_ids: tuple[str, ...] | None, *, universe: tuple[str, ...] | None = None
) -> float:
    """
    Return the fraction of the entity universe the data actually covers.

    Parameters
    ----------
    entity_ids
        Identifier per row, as carried by a prepared dataset.
    universe
        The full set of entities the model is meant to serve. ``None`` means
        the observed set is the universe, so coverage is one by definition.

    Returns
    -------
    float
        Fraction in ``[0, 1]``. One when there is nothing to compare against,
        which keeps a problem with no entity axis from reporting zero
        coverage.
    """
    if entity_ids is None or universe is None or not universe:
        return 1.0
    return float(len(set(entity_ids) & set(universe)) / len(set(universe)))


def quality_metrics(
    features: NDArray[np.floating],
    *,
    target: NDArray[np.floating] | None = None,
    entity_ids: tuple[str, ...] | None = None,
    universe: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """
    Return every quality metric for one dataset.

    Parameters
    ----------
    features
        Feature matrix, before windowing.
    target
        Target series, scored separately because a gap in the target removes a
        training example entirely while a gap in one feature does not.
    entity_ids
        Identifier per row.
    universe
        The full entity universe.

    Returns
    -------
    dict
        Metric name to value, all finite and all in ``[0, 1]``.
    """
    matrix = _as_matrix(features)
    per_column = column_completeness(matrix)

    metrics = {
        "feature_completeness": completeness(matrix),
        "worst_column_completeness": float(per_column.min()) if per_column.size else 1.0,
        "feature_staleness": staleness(matrix),
        "entity_coverage": entity_coverage(entity_ids, universe=universe),
    }
    if target is not None:
        metrics["target_completeness"] = completeness(target)
        metrics["target_staleness"] = staleness(target)
    return metrics


def quality_warnings(metrics: dict[str, float]) -> tuple[str, ...]:
    """
    Turn quality metrics that cross a threshold into readable warnings.

    Parameters
    ----------
    metrics
        The output of :func:`quality_metrics`.

    Returns
    -------
    tuple of str
        One sentence per concern, empty when there is nothing to report. The
        report writes these verbatim, so they are phrased for a person rather
        than for a log parser.
    """
    warnings: list[str] = []

    for name in ("feature_completeness", "target_completeness"):
        value = metrics.get(name)
        if value is not None and value < COMPLETENESS_WARNING:
            warnings.append(
                f"{name.replace('_', ' ')} is {value:.1%}, below the "
                f"{COMPLETENESS_WARNING:.0%} threshold; statistics estimated from "
                f"this data are noisier than the row count suggests"
            )

    worst = metrics.get("worst_column_completeness")
    if worst is not None and worst < COMPLETENESS_WARNING:
        warnings.append(
            f"the least complete feature column is {worst:.1%} present; consider "
            f"dropping it rather than letting it contribute mostly imputed values"
        )

    for name in ("feature_staleness", "target_staleness"):
        value = metrics.get(name)
        if value is not None and value > STALENESS_WARNING:
            warnings.append(
                f"{name.replace('_', ' ')} is {value:.1%}, above the "
                f"{STALENESS_WARNING:.0%} threshold; repeated observations inflate "
                f"the apparent sample size without adding information"
            )

    coverage = metrics.get("entity_coverage")
    if coverage is not None and coverage < 1.0:
        warnings.append(
            f"the data covers {coverage:.1%} of the declared entity universe; "
            f"predictions for the remainder would be extrapolation"
        )

    return tuple(warnings)
