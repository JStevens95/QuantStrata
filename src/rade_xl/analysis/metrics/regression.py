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
:class:`~rade_xl.core.contract.result.EvalResult` carries ``baseline_metrics``
as a field rather than leaving it to a report.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ...core.runtime.errors import ContractError

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
