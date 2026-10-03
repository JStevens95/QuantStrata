"""
How far today's inputs have moved from the ones a model was trained on.

A model is a function fitted to a distribution. When the distribution moves,
the model keeps producing numbers with no change in its confidence and no
error anywhere -- it is being asked questions outside the range it learned
from, and it answers them as fluently as the ones inside. Drift metrics are
the only warning available, because the usual one, a rising error, requires
the outcomes to have arrived and by then the predictions have been acted on.

Why this is a metric and not a validation
-----------------------------------------
As with :mod:`~.quality`, nothing here raises. A moved distribution is a fact
about the world rather than a configuration error, and a framework that
refused to predict through a regime change would be refusing at precisely the
moment it was most needed. What is wrong is predicting through one
*silently*, so these return numbers and the caller decides.

Choosing the measures
---------------------
Three, because each one misses something the others catch:

- **Population stability index** is the standard in credit and risk, and its
  thresholds (0.1, 0.25) are widely understood, which matters more than its
  statistical properties: a number nobody has a reference point for does not
  change anyone's behaviour.
- **Kolmogorov--Smirnov** is the largest gap between two cumulative
  distributions. It notices a shift in shape that leaves the bulk in place,
  which the index can miss when the move happens inside one bin.
- **Standardised mean shift** is the crudest and the most readable: how far
  the centre moved, in training standard deviations. A reader who ignores
  the other two will understand this one.

All three are computed per feature rather than over a summary, because drift
is almost never uniform. One input going stale while the rest hold steady is
the common case, and an average over all of them hides exactly that.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...core.runtime.errors import ContractError

__all__ = [
    "DRIFT_THRESHOLDS",
    "drift_metrics",
    "drift_warnings",
    "kolmogorov_smirnov",
    "mean_shift",
    "population_stability_index",
]

#: Default bin count for the stability index. Ten is the convention the
#: published thresholds were calibrated against, and changing it would make
#: those thresholds mean something else -- which is why it is a module
#: constant rather than a parameter with a tempting default.
_BINS = 10

#: Added to a bin's share before taking a logarithm, so an empty bin gives a
#: large contribution rather than an infinite one. Without it a single
#: unoccupied bin makes the whole index infinite, which is both useless as a
#: number and indistinguishable from a genuine catastrophe.
_FLOOR = 1e-6

#: Where each measure stops being unremarkable. The index thresholds are the
#: conventional ones; the other two are chosen to fire at roughly the same
#: severity, so a reader does not have to learn three different scales.
DRIFT_THRESHOLDS = {
    "psi": 0.25,
    "ks": 0.2,
    "mean_shift": 0.5,
}


def population_stability_index(
    baseline: NDArray[np.floating], live: NDArray[np.floating]
) -> float:
    """
    Measure how much probability mass moved between bins.

    Bins are cut on the *baseline's* quantiles, not on the pooled data.
    Pooling would let the live sample move the bin edges, which is how a
    large shift can produce a small index: the edges follow the data, each
    bin keeps roughly its share, and the measure reports that nothing
    happened.

    Parameters
    ----------
    baseline
        Values the model was trained on.
    live
        Values being predicted on now.

    Returns
    -------
    float
        Zero for identical distributions, growing without bound as they
        separate. Conventionally: below 0.1 unremarkable, 0.1 to 0.25 worth
        watching, above 0.25 worth acting on.

    Raises
    ------
    ContractError
        If either sample is empty, which has no distribution to compare.
    """
    base, current = _finite_pair(baseline, live, measure="population stability index")

    if float(np.ptp(base)) == 0.0:
        # A constant baseline has no quantiles to cut on: every edge
        # collapses onto the one value it takes. Drift is then simply
        # whether the live sample is still that constant. Checked before
        # binning rather than after, because the collapsed edge still
        # produces usable-looking bins that put both samples in the same
        # one and report no drift at all.
        return 0.0 if bool(np.all(current == base[0])) else float("inf")

    # Interior edges only: the outermost bins are left unbounded so that a
    # live value beyond anything seen in training lands in the edge bin
    # rather than being dropped. Dropping it would make the most extreme
    # evidence of drift the one thing the measure cannot see.
    edges = np.unique(np.quantile(base, np.linspace(0, 1, _BINS + 1)[1:-1]))

    base_share = _shares(base, edges)
    live_share = _shares(current, edges)
    return float(np.sum((live_share - base_share) * np.log(live_share / base_share)))


def kolmogorov_smirnov(
    baseline: NDArray[np.floating], live: NDArray[np.floating]
) -> float:
    """
    Measure the largest gap between two cumulative distributions.

    The statistic only, not the p-value. A p-value here would be misleading
    in the way it usually is with large samples: with a hundred thousand
    live rows, a difference far too small to affect any prediction is
    overwhelmingly significant, and a reader who treats significance as
    severity would act on noise every day.

    Parameters
    ----------
    baseline
        Values the model was trained on.
    live
        Values being predicted on now.

    Returns
    -------
    float
        Between zero and one. Zero for identical distributions, one when
        they do not overlap at all.

    Raises
    ------
    ContractError
        If either sample is empty.
    """
    base, current = _finite_pair(baseline, live, measure="Kolmogorov-Smirnov statistic")

    # Evaluated on the pooled support, which is where the two cumulative
    # functions can differ at all.
    grid = np.unique(np.concatenate([base, current]))
    base_cdf = np.searchsorted(np.sort(base), grid, side="right") / base.size
    live_cdf = np.searchsorted(np.sort(current), grid, side="right") / current.size
    return float(np.max(np.abs(base_cdf - live_cdf)))


def mean_shift(baseline: NDArray[np.floating], live: NDArray[np.floating]) -> float:
    """
    Measure how far the centre moved, in baseline standard deviations.

    Standardised by the baseline rather than the pooled spread, for the same
    reason the index bins on the baseline: the training distribution is the
    fixed reference the model actually learned, and letting the live sample
    influence the scale would let a drifting input flatter itself.

    Parameters
    ----------
    baseline
        Values the model was trained on.
    live
        Values being predicted on now.

    Returns
    -------
    float
        Absolute shift in baseline standard deviations. Zero when the means
        agree; infinite when the baseline was constant and the live sample
        is not, which is a real and total drift rather than a division
        error.

    Raises
    ------
    ContractError
        If either sample is empty.
    """
    base, current = _finite_pair(baseline, live, measure="mean shift")

    spread = float(np.std(base))
    difference = abs(float(np.mean(current)) - float(np.mean(base)))
    if spread == 0.0:
        return 0.0 if difference == 0.0 else float("inf")
    return difference / spread


def drift_metrics(
    baseline: NDArray[np.floating],
    live: NDArray[np.floating],
    *,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute every measure for every feature.

    Per feature rather than over a summary, because drift is almost never
    uniform: one input going stale while the rest hold steady is the common
    case, and an average over all of them hides precisely that.

    Parameters
    ----------
    baseline
        Training features, one row per sample and one column per feature.
    live
        Current features, with the same columns.
    feature_names
        Names for the columns. Defaults to positional labels, which are
        legible but not actionable -- a report that says ``feature_7``
        drifted sends the reader counting columns.

    Returns
    -------
    dict
        Feature name to a mapping of measure name to value.

    Raises
    ------
    ContractError
        If the two sets of features do not have the same number of columns,
        or if the names do not match the columns. Checked rather than
        broadcast: comparing column three of one matrix against column three
        of a differently ordered one produces a drift number for a pair of
        unrelated inputs.
    """
    # A one-dimensional input is one feature over many samples, not one
    # sample of many features. `numpy.atleast_2d` reads it the other way
    # round, which would report two hundred single-observation "features"
    # for a vector of two hundred values -- each one a measure computed
    # from a sample of size one.
    base = _as_columns(baseline)
    current = _as_columns(live)

    if base.shape[1] != current.shape[1]:
        raise ContractError(
            f"the baseline has {base.shape[1]} feature(s) and the live data has "
            f"{current.shape[1]}; drift is computed column by column, so a "
            f"mismatch would compare unrelated inputs"
        )

    names = feature_names or tuple(f"feature_{index}" for index in range(base.shape[1]))
    if len(names) != base.shape[1]:
        raise ContractError(
            f"{len(names)} feature name(s) were given for {base.shape[1]} column(s)"
        )

    return {
        name: {
            "psi": population_stability_index(base[:, index], current[:, index]),
            "ks": kolmogorov_smirnov(base[:, index], current[:, index]),
            "mean_shift": mean_shift(base[:, index], current[:, index]),
        }
        for index, name in enumerate(names)
    }


def drift_warnings(metrics: dict[str, dict[str, float]]) -> tuple[str, ...]:
    """
    Turn the numbers that crossed a threshold into sentences.

    The counterpart to :func:`~.quality.quality_warnings`, and for the same
    reason: a table of three measures across two hundred features is a
    thing nobody reads, so the handful that matter are stated in words that
    will survive being pasted into an email.

    Parameters
    ----------
    metrics
        The output of :func:`drift_metrics`.

    Returns
    -------
    tuple of str
        One sentence per feature that crossed a threshold, worst first, so
        that a truncated list keeps the part worth keeping.
    """
    offenders: list[tuple[float, str]] = []
    for name, measures in metrics.items():
        crossed = {
            measure: value
            for measure, value in measures.items()
            if value > DRIFT_THRESHOLDS.get(measure, float("inf"))
        }
        if not crossed:
            continue
        # Ranked on the index, which is the measure with conventional
        # thresholds and therefore the one a reader can calibrate against.
        severity = measures.get("psi", 0.0)
        described = ", ".join(
            f"{measure}={value:.3g}" for measure, value in sorted(crossed.items())
        )
        offenders.append(
            (
                severity,
                f"{name} has drifted from the training distribution ({described}); "
                f"predictions for it are extrapolations",
            )
        )

    return tuple(message for _, message in sorted(offenders, key=lambda pair: -pair[0]))


def _as_columns(values: NDArray[np.floating]) -> NDArray[np.float64]:
    """
    Read an input as rows of samples by columns of features.

    Parameters
    ----------
    values
        Either a matrix, or a vector holding one feature.

    Returns
    -------
    numpy.ndarray
        A two-dimensional array, with a vector turned into a single column.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _shares(values: NDArray[np.float64], edges: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Return the proportion of values falling in each bin, floored.

    Parameters
    ----------
    values
        The sample.
    edges
        Interior bin edges.

    Returns
    -------
    numpy.ndarray
        One share per bin, each at least :data:`_FLOOR`.
    """
    counts = np.bincount(np.digitize(values, edges), minlength=edges.size + 1)
    return np.maximum(counts / values.size, _FLOOR)


def _finite_pair(
    baseline: NDArray[np.floating], live: NDArray[np.floating], *, measure: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Flatten both samples to finite one-dimensional arrays.

    Non-finite values are dropped rather than propagated. A single ``NaN``
    would otherwise make every measure ``NaN``, which reads as "no drift
    could be computed" in a report that has no way to distinguish that from
    "no drift" -- and a missing value is a
    :mod:`~.quality` concern, already reported separately.

    Parameters
    ----------
    baseline
        Values the model was trained on.
    live
        Values being predicted on now.
    measure
        Name of the calling measure, for the error message.

    Returns
    -------
    tuple
        The two cleaned arrays.

    Raises
    ------
    ContractError
        If either is empty once non-finite values are removed.
    """
    base = np.ravel(np.asarray(baseline, dtype=np.float64))
    current = np.ravel(np.asarray(live, dtype=np.float64))
    base = base[np.isfinite(base)]
    current = current[np.isfinite(current)]

    if base.size == 0 or current.size == 0:
        raise ContractError(
            f"the {measure} needs finite values on both sides; the baseline has "
            f"{base.size} and the live sample has {current.size} after dropping "
            f"non-finite entries"
        )
    return base, current
