# `src/rade_qnet/analysis/metrics`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 35 | 1297 | `6bc47bdcc1b578ae` |
| 2 | `drift.py` | 401 | 14437 | `aba29d7c39dcf912` |
| 3 | `quality.py` | 291 | 9825 | `24c242fef6bec8fe` |
| 4 | `regression.py` | 342 | 11131 | `8d183bb0c63180c1` |

---

## 1. `src/rade_qnet/analysis/metrics/__init__.py`

1297 bytes · SHA-256 `6bc47bdcc1b578ae`

```python
"""
Metrics -- pure functions over arrays.

Every metric takes arrays and returns numbers.  No metric reads configuration,
touches the filesystem or holds state, which makes each one independently
testable against a hand-computed value.

One rule is enforced by the pipelines that call these: metrics are always
computed in the *original* target units.  A mean absolute error reported in
standardised space is not a quantity anyone can act on, so the evaluate
pipeline inverts target transforms before this package is reached.

Modules
-------
``regression.py``
    Error and agreement measures: mean absolute error, root mean squared
    error, bias, coefficient of determination, directional accuracy, and the
    naive baselines every headline metric is reported against.
    [Phase 1, delivered]

Planned modules
---------------
``episode.py``
    Episode return, length, terminal wealth and risk-adjusted statistics for
    interactive runs.  [Phase 7]
``drift.py``
    Distribution distance between a baseline and a live window, for monitoring
    a deployed model.  [Phase 5]
``quality.py``
    Input data quality: completeness, staleness and coverage.  A model trained
    on 40 per cent missing history should say so in its own artifacts.
    [Phase 2]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/analysis/metrics/drift.py`

14437 bytes · SHA-256 `aba29d7c39dcf912`

```python
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

from ...core.lifecycle.errors import ContractError

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


def population_stability_index(baseline: NDArray[np.floating], live: NDArray[np.floating]) -> float:
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


def kolmogorov_smirnov(baseline: NDArray[np.floating], live: NDArray[np.floating]) -> float:
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
```

---

## 3. `src/rade_qnet/analysis/metrics/quality.py`

9825 bytes · SHA-256 `24c242fef6bec8fe`

```python
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
```

---

## 4. `src/rade_qnet/analysis/metrics/regression.py`

11131 bytes · SHA-256 `8d183bb0c63180c1`

```python
"""
Regression metrics, computed on arrays in original target units.

Every function here is pure: arrays in, float out, no state and no I/O. That
makes them trivially testable and reusable by a report, a tuning objective or
a model's own custom evaluation.

Two decisions deserve stating
-----------------------------
**No silent NaN handling.** A metric over data containing ``NaN`` raises
rather than returning ``NaN`` or quietly dropping rows. Dropping rows changes
the denominator, so a model that failed to predict for half the universe can
report a *better* mean error than one that predicted for all of it -- and the
metric gives no hint that happened. Raising forces the caller to decide what
missing predictions mean.

**A reference point alongside every headline.** :func:`baseline_metrics`
computes the same metrics for a naive predictor. A mean absolute error of
0.004 is excellent or useless depending entirely on the target's scale, and
reporting it without a comparison invites the wrong conclusion. This is why
:class:`~rade_qnet.core.contract.result.EvalResult` carries ``baseline_metrics``
as a field rather than leaving it to a report.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...core.lifecycle.errors import ContractError

__all__ = [
    "baseline_metrics",
    "bias",
    "directional_accuracy",
    "mean_absolute_error",
    "r_squared",
    "regression_metrics",
    "root_mean_squared_error",
]


def _validate(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Check two arrays are comparable, and flatten them to one dimension.

    Called by every metric, so the three ways this goes wrong are rejected
    once rather than in seven places.

    Flattening means a prediction of shape ``(n, 1)`` and a target of shape
    ``(n,)`` compare correctly. Without it, numpy broadcasts them to
    ``(n, n)`` and computes the mean error against every pairing -- which
    returns a plausible number that is complete nonsense, and is one of the
    easiest mistakes to ship.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    tuple of numpy.ndarray
        The two arrays, flattened and cast to float64.

    Raises
    ------
    ContractError
        If the arrays are empty, have different lengths, or contain ``NaN``
        or infinity.
    """
    predicted = np.asarray(predictions, dtype=np.float64).reshape(-1)
    observed = np.asarray(targets, dtype=np.float64).reshape(-1)

    if predicted.size == 0:
        raise ContractError("cannot compute a metric over an empty array")
    if predicted.size != observed.size:
        raise ContractError(
            f"predictions and targets have different lengths after flattening: "
            f"{predicted.size} vs {observed.size}"
        )
    for name, values in (("predictions", predicted), ("targets", observed)):
        if not np.all(np.isfinite(values)):
            count = int(np.sum(~np.isfinite(values)))
            raise ContractError(
                f"{name} contain {count} non-finite value(s) out of {values.size}; "
                f"metrics do not drop them silently because that would change the "
                f"denominator and make the result incomparable -- decide explicitly "
                f"what a missing prediction means"
            )
    return predicted, observed


def mean_absolute_error(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> float:
    """
    Return the mean absolute error.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    float
        Mean of ``|prediction - target|``.
    """
    predicted, observed = _validate(predictions, targets)
    return float(np.mean(np.abs(predicted - observed)))


def root_mean_squared_error(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> float:
    """
    Return the root mean squared error.

    Reported alongside the mean absolute error rather than instead of it: the
    gap between the two is itself informative, since a root mean squared error
    much larger than the mean absolute error means the errors are dominated by
    a few large misses rather than spread evenly.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    float
        Square root of the mean squared error.
    """
    predicted, observed = _validate(predictions, targets)
    return float(np.sqrt(np.mean((predicted - observed) ** 2)))


def bias(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> float:
    """
    Return the mean signed error.

    Distinct from the absolute error and not redundant with it: a model can
    have a small absolute error and a large bias, meaning it is consistently
    wrong in one direction. For a financial target that is a materially
    different problem from being imprecise in both directions, and the
    absolute error alone hides it.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    float
        Mean of ``prediction - target``. Positive means over-prediction.
    """
    predicted, observed = _validate(predictions, targets)
    return float(np.mean(predicted - observed))


def r_squared(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> float:
    """
    Return the coefficient of determination.

    Computed against the variance of the targets, so a negative value is
    possible and meaningful: it says the model is worse than predicting the
    target's own mean. That is not clipped to zero, because a negative value
    is exactly the diagnosis a reader needs.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    float
        ``1 - residual_variance / total_variance``, or ``0.0`` when the
        targets are constant.

    Notes
    -----
    A constant target makes the statistic undefined -- the denominator is
    zero. Returning ``0.0`` rather than raising is the pragmatic choice, since
    a constant target legitimately occurs in a short held-out window, and
    ``0.0`` reads as "no explanatory power", which is the honest summary.
    """
    predicted, observed = _validate(predictions, targets)
    total_variance = float(np.sum((observed - np.mean(observed)) ** 2))
    if total_variance == 0.0:
        return 0.0
    residual_variance = float(np.sum((observed - predicted) ** 2))
    return 1.0 - residual_variance / total_variance


def directional_accuracy(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> float:
    """
    Return the fraction of predictions with the correct sign.

    For a target that is a return or a change, the sign often matters more
    than the magnitude: a model with a mediocre mean absolute error and good
    directional accuracy may be far more useful than the reverse.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    float
        Fraction in ``[0, 1]``. Observations with a target of exactly zero are
        excluded from both numerator and denominator, since no direction was
        available to get right; if every target is zero, returns ``0.0``.
    """
    predicted, observed = _validate(predictions, targets)
    directional = observed != 0.0
    if not np.any(directional):
        return 0.0
    agreed = np.sign(predicted[directional]) == np.sign(observed[directional])
    return float(np.mean(agreed))


def regression_metrics(
    predictions: NDArray[np.floating],
    targets: NDArray[np.floating],
) -> dict[str, float]:
    """
    Compute the standard metric set in one call.

    This is what the evaluate pipeline reports by default, so every run is
    comparable without each model choosing its own metrics.

    Parameters
    ----------
    predictions
        Model output, in original target units.
    targets
        Observed values.

    Returns
    -------
    dict
        ``mae``, ``rmse``, ``bias``, ``r2`` and ``directional_accuracy``.
    """
    return {
        "mae": mean_absolute_error(predictions, targets),
        "rmse": root_mean_squared_error(predictions, targets),
        "bias": bias(predictions, targets),
        "r2": r_squared(predictions, targets),
        "directional_accuracy": directional_accuracy(predictions, targets),
    }


def baseline_metrics(
    targets: NDArray[np.floating],
    *,
    strategy: str = "zero",
    train_targets: NDArray[np.floating] | None = None,
) -> dict[str, float]:
    """
    Compute the standard metric set for a naive predictor.

    Gives every headline number a reference point. See the module docstring
    for why that is treated as a requirement rather than a nicety.

    Parameters
    ----------
    targets
        Observed values for the split being scored.
    strategy
        The naive predictor to use:

        ``zero``
            Always predict zero. The right default for a return-like target,
            where zero is the no-information forecast.
        ``mean``
            Always predict the mean of ``train_targets``. Note that this uses
            the *training* mean, not the mean of the split being scored --
            using the latter would give the baseline information the model did
            not have, making it an unbeatable and meaningless reference.
    train_targets
        Training targets, required for the ``mean`` strategy.

    Returns
    -------
    dict
        The same keys as :func:`regression_metrics`.

    Raises
    ------
    ContractError
        If the strategy is unknown, or ``mean`` is requested without
        ``train_targets``.
    """
    observed = np.asarray(targets, dtype=np.float64).reshape(-1)

    if strategy == "zero":
        predicted = np.zeros_like(observed)
    elif strategy == "mean":
        if train_targets is None:
            raise ContractError(
                "the 'mean' baseline needs train_targets: the mean must come from "
                "the training split, since using the scored split's own mean would "
                "give the baseline information the model never had"
            )
        training = np.asarray(train_targets, dtype=np.float64).reshape(-1)
        if training.size == 0:
            raise ContractError("the 'mean' baseline needs at least one training target")
        predicted = np.full_like(observed, float(np.mean(training)))
    else:
        raise ContractError(f"unknown baseline strategy {strategy!r}; expected 'zero' or 'mean'")

    return regression_metrics(predicted, observed)
```

