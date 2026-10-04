"""
Turning a fitted model and a data bundle into scored metrics.

Extracted from :class:`~rade_qnet.orchestration.pipelines.train.TrainPipeline`
when evaluation arrived, because of a requirement that reads as a one-line
test and is actually a design constraint:

    Re-evaluating a saved bundle must reproduce the metrics recorded in it.

Two implementations of scoring cannot satisfy that for long. They would agree
on the day they were written and then drift -- one gains a guard, the other
gains a different one, and the first anybody hears of it is a re-evaluation
that disagrees with a bundle by a fraction nobody can account for. So there
is one implementation, and both pipelines call it.

What makes scoring harder than it looks
---------------------------------------
Every helper here exists because of a failure that produces a *number* rather
than an error:

- **Two passes must line up.** Scoring iterates a source twice, once for the
  forward pass and once for the targets. A shuffled source reorders between
  them, so each prediction is compared against an unrelated target. The counts
  still match and every metric still computes; the only symptom is a model
  that appears to score at random on the split it was trained on.
  :func:`scoring_source` routes through
  :class:`~rade_qnet.core.contract.source.OrderedSource`, and verifies rather
  than assumes.

- **The source decides which rows exist.** A sequence model discards the first
  ``length - 1`` rows of each split, because no complete window ends there.
  Targets read from the dataset would include them, misaligning every metric
  by a few rows -- degrading a score rather than breaking it.
  :func:`collect_targets` reads through the source for that reason.

- **Shapes that broadcast.** NumPy will expand an ``(n, 1)`` against an
  ``(n,)`` into an ``(n, n)``, and the mean of that is a plausible number
  with no meaning. :func:`align` checks instead.

- **Units.** Metrics are computed after
  :meth:`~rade_qnet.core.contract.state.FittedState.inverse_transform_targets`,
  so they are in the target's original units. An error in standardised space
  is not a quantity anyone can act on, and two runs' standardised errors --
  each scaled by its own training mean -- are not comparable to each other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...analysis.metrics.regression import baseline_metrics, regression_metrics
from ...core.contract.data import TensorBatchData
from ...core.contract.result import EvalResult
from ...core.contract.source import BatchSource, OrderedSource
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from numpy.typing import NDArray

    from ...core.contract.data import DataBundle

__all__ = [
    "EVALUATED_SPLITS",
    "FEATURE_MATRIX_RANK",
    "TARGET_KEY",
    "TRAIN_SPLIT",
    "align",
    "collect_targets",
    "feature_matrix",
    "order_is_stable",
    "score_splits",
    "scoring_source",
    "source_for",
    "static_inputs",
]

_LOGGER = get_logger(__name__)

#: The split a model learns from, and the one whose statistics define the
#: baseline. Named rather than written inline because three functions need it
#: to mean the same thing.
TRAIN_SPLIT = "train"

#: The splits worth scoring, in report order. Training is included on purpose:
#: a model that scores well on train and badly on validation has overfitted,
#: and that is a different problem from one that scores badly on both. Without
#: the training score the two are indistinguishable in the saved bundle.
EVALUATED_SPLITS: tuple[str, ...] = ("train", "validation", "test")

#: The batch key holding the observed values.
TARGET_KEY = "target"

#: The rank a feature array must have for the flat quality metrics to mean
#: anything: one row per sample, one column per feature.
FEATURE_MATRIX_RANK = 2


def score_splits(
    data: DataBundle[object],
    predict: Callable[[BatchSource], NDArray[np.floating]],
    *,
    splits: tuple[str, ...] = EVALUATED_SPLITS,
) -> dict[str, EvalResult]:
    """
    Score a model on every available split, in the target's original units.

    The one implementation of scoring, called both at the end of training and
    when a saved bundle is re-evaluated. Takes the forward pass as a callable
    rather than an engine and a handle, so that the caller decides how
    predictions are produced -- training already holds a prepared handle,
    while evaluation has just rebuilt one -- without this function needing to
    know about either.

    Parameters
    ----------
    data
        The data bundle, supplying the sources and the fitted state that
        inverts the target transform.
    predict
        Runs a forward pass over one source and returns its raw output, in
        the model's own output space. Inverting is done here, so a caller
        that inverted too would invert twice.
    splits
        Which splits to score, in report order. Any that the bundle does not
        carry are skipped rather than raising: a run configured without a
        validation split is a legitimate configuration, not an error.

    Returns
    -------
    dict
        Split name to result, each with its metrics and a naive baseline.
    """
    # The training targets, in original units, for the `mean` baseline.
    # Deliberately the training mean rather than the scored split's: using
    # the latter would give the baseline information the model did not have,
    # making it unbeatable and therefore useless as a reference.
    train_source = scoring_source(data, TRAIN_SPLIT)
    train_targets = collect_targets(data, train_source, split=TRAIN_SPLIT)

    evaluations: dict[str, EvalResult] = {}
    for name in splits:
        if name not in data.splits:
            continue
        source = scoring_source(data, name)
        predictions = data.state.inverse_transform_targets(
            np.asarray(predict(source), dtype=np.float64)
        )
        targets = collect_targets(data, source, split=name)
        predictions, targets = align(predictions, targets, split=name)

        evaluations[name] = EvalResult(
            split=name,
            metrics=regression_metrics(predictions, targets),
            n_samples=int(targets.shape[0]),
            in_original_units=True,
            baseline_metrics=baseline_metrics(
                targets, strategy="mean", train_targets=train_targets
            ),
        )
        _LOGGER.info(
            "scored %s on %d sample(s): %s",
            name,
            evaluations[name].n_samples,
            {key: round(value, 6) for key, value in evaluations[name].metrics.items()},
        )
    return evaluations


def static_inputs(data: DataBundle[object]) -> Mapping[str, object]:
    """
    Return the training split's static inputs.

    Parameters
    ----------
    data
        The data bundle.

    Returns
    -------
    Mapping
        Input name to tensor. Empty for a model with no static inputs, which
        is most of them.
    """
    return source_for(data, TRAIN_SPLIT).static


def source_for(data: DataBundle[object], split: str) -> BatchSource:
    """
    Return one split's batch source.

    Parameters
    ----------
    data
        The data bundle.
    split
        Split name.

    Returns
    -------
    BatchSource
        The split's source.

    Raises
    ------
    ContractError
        If the split's payload is not a batch source. The tensor engines
        consume :class:`~rade_qnet.core.contract.data.TensorBatchData`, whose
        ``loader`` is the source; anything else means the model's data
        build produced a payload for a different engine.
    """
    payload = data.split(split)
    if isinstance(payload, TensorBatchData):
        loader = payload.loader
        if isinstance(loader, BatchSource):
            return loader
        raise ContractError(
            f"the {split!r} split's loader is a {type(loader).__name__}, which "
            f"does not satisfy BatchSource; a gradient engine needs a source "
            f"it can re-iterate and ask for a batch count"
        )
    if isinstance(payload, BatchSource):
        return payload
    raise ContractError(
        f"the {split!r} split carries a {type(payload).__name__}, which is "
        f"neither TensorBatchData nor a BatchSource. A gradient engine cannot "
        f"consume it; check that the model's data build targets this engine"
    )


def scoring_source(data: DataBundle[object], split: str) -> BatchSource:
    """
    Return one split's source in a form safe to traverse twice.

    Scoring takes two passes over a split -- one for the forward pass and
    one to collect the targets -- and pairs the results row for row. A
    training source reshuffles between passes, so pairing them directly
    compares each prediction against an unrelated target. The sample
    counts still match and every metric still computes, so the only
    symptom is a model that appears to score at random on the split it was
    trained on.

    Routes through
    :class:`~rade_qnet.core.contract.source.OrderedSource` where the source
    offers it, and otherwise verifies that the source is already stable
    rather than assuming it.

    Parameters
    ----------
    data
        The data bundle.
    split
        Split name.

    Returns
    -------
    BatchSource
        A source whose passes line up.

    Raises
    ------
    ContractError
        If the source's order varies between passes and it offers no
        ordered view. Raised rather than scored, because a plausible wrong
        number is worse than no number.
    """
    source = source_for(data, split)
    if isinstance(source, OrderedSource):
        source = source.ordered()

    if not order_is_stable(source):
        raise ContractError(
            f"the {split!r} source yields its samples in a different order on "
            f"each pass, and does not implement OrderedSource.ordered(). "
            f"Scoring needs two passes to line up, so every metric for this "
            f"split would be computed against mismatched targets. Implement "
            f"ordered() to return a stable-order view"
        )
    return source


def collect_targets(
    data: DataBundle[object], source: BatchSource, *, split: str
) -> NDArray[np.float64]:
    """
    Collect one split's targets, in the target's original units.

    Read by iterating the source rather than from the dataset directly,
    for a reason worth stating: the source decides which rows are usable.
    A sequence model discards the first ``length - 1`` rows of every split
    because no complete window ends there, and targets taken from the
    dataset would include them -- silently misaligning every metric by a
    few rows in a way that degrades a score rather than breaking it.

    Parameters
    ----------
    data
        The data bundle, for the fitted state that inverts the target.
    source
        The split's source. Taken as a parameter rather than re-derived
        from ``data``, because the caller has already resolved it to a
        stable-order view and re-deriving would hand back the shuffled
        training source -- which is exactly the misalignment
        :func:`scoring_source` exists to prevent.
    split
        Split name, for messages.

    Returns
    -------
    numpy.ndarray
        Targets, one row per sample, inverse-transformed.

    Raises
    ------
    ContractError
        If a batch has no target, or the source yielded no batches.
    """
    collected: list[NDArray[np.float64]] = []
    for batch in source.batches():
        if TARGET_KEY not in batch:
            raise ContractError(
                f"a batch from the {split!r} source has keys {sorted(batch)} and "
                f"no {TARGET_KEY!r}; a split cannot be scored without targets"
            )
        collected.append(np.asarray(batch[TARGET_KEY], dtype=np.float64))

    if not collected:
        raise ContractError(
            f"the {split!r} source yielded no batches, so it cannot be scored. "
            f"A batch size larger than the split with drop_last set produces "
            f"this"
        )
    return data.state.inverse_transform_targets(np.concatenate(collected, axis=0))


def feature_matrix(data: DataBundle[object]) -> NDArray[np.float64] | None:
    """
    Collect the training split's features as one matrix, if it is flat.

    Returns ``None`` rather than raising for a source whose features are
    not a flat sample-by-feature matrix -- a sequence model's windows, a
    graph model's node table. Quality metrics for those are the model's own
    business, and a framework that guessed would report a completeness
    figure over the wrong axis.

    Read through the ordered view, because one of the quality metrics is
    order-dependent: staleness counts rows that repeat the row before them,
    and over a shuffled pass that is a measure of nothing. Completeness and
    coverage would survive the shuffle, which is what makes this the kind
    of mistake that produces three plausible numbers and one meaningless
    one.

    Parameters
    ----------
    data
        The data bundle.

    Returns
    -------
    numpy.ndarray or None
        The feature matrix, or ``None`` if the shape is not flat.
    """
    source = scoring_source(data, TRAIN_SPLIT)
    blocks: list[NDArray[np.float64]] = []
    for batch in source.batches():
        for name in sorted(batch):
            if name == TARGET_KEY:
                continue
            array = np.asarray(batch[name], dtype=np.float64)
            if array.ndim != FEATURE_MATRIX_RANK:
                return None
            blocks.append(array)
    if not blocks:
        return None
    return np.concatenate(blocks, axis=0)


def order_is_stable(source: BatchSource) -> bool:
    """
    Return whether two passes over a source yield targets in the same order.

    Compares the targets rather than the features because they are the smaller
    array by a wide margin -- one column against a window of many -- and any
    reordering that would misalign a metric reorders both.

    Parameters
    ----------
    source
        The source to check.

    Returns
    -------
    bool
        True if two passes agree. True for an unbounded source, which cannot
        be checked this way and is never scored by this pipeline anyway.
    """
    if source.steps_per_epoch is None:
        return True
    passes = [
        np.concatenate(
            [np.ravel(np.asarray(batch[TARGET_KEY])) for batch in source.batches()],
            axis=0,
        )
        for _ in range(2)
    ]
    first, second = passes
    return first.shape == second.shape and bool(np.array_equal(first, second))


def align(
    predictions: NDArray[np.float64], targets: NDArray[np.float64], *, split: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Reduce predictions and targets to matching one-dimensional arrays.

    Parameters
    ----------
    predictions
        Model output, in original units.
    targets
        Observed values, in original units.
    split
        Split name, for the message.

    Returns
    -------
    tuple
        The two arrays, flattened.

    Raises
    ------
    ContractError
        If the sample counts disagree. Checked rather than broadcast, because
        NumPy will happily broadcast a ``(n, 1)`` against an ``(n,)`` into an
        ``(n, n)`` and the resulting metric is a number that looks plausible.
    """
    flat_predictions = np.ravel(predictions)
    flat_targets = np.ravel(targets)
    if flat_predictions.shape != flat_targets.shape:
        raise ContractError(
            f"on the {split!r} split the model produced {flat_predictions.size} "
            f"prediction(s) for {flat_targets.size} target(s). Equal counts are "
            f"required; a mismatch usually means the source dropped rows the "
            f"target collection kept, or the model's output has an extra axis"
        )
    return flat_predictions, flat_targets
