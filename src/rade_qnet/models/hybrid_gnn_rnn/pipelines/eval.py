"""
The hybrid network's evaluation pipeline.

It adds a per-target breakdown to the result, and changes nothing else.

Why an aggregate metric is not enough for this model
-----------------------------------------------------
A replication model predicts several target instruments at once, and the
framework's metrics pool every target into one number. For most models that
is the right summary. For this one it hides the failure the model is most
likely to have.

A book of thirty targets where twenty-nine replicate well and one does not
has a good mean absolute error. The one that does not is typically the one
with the fewest neighbours in the similarity graph -- a new product type, an
unusual underlying, a maturity at the edge of the book -- and it is exactly
the position a desk would want flagged, because it is the position the model
is least entitled to an opinion about. The aggregate reports it as noise.

So this file computes the same error metric per target and attaches the
worst few to the result's notes, alongside the spread. Notes rather than
metrics, deliberately: metrics are compared across runs and across models,
and a per-target key is meaningless the moment the universe changes. A note
is read by a person looking at one result.

Why it extends ``score`` rather than adding a report
------------------------------------------------------
Reports write figures to a run directory. An evaluation of a saved bundle
often happens in a notebook or a service, where the useful place for a
finding is the returned object. The report path remains available to anyone
who wants the figures; this is the part that survives being returned over
an API.

Why the breakdown can be absent
--------------------------------
It needs two things the framework does not guarantee: identifiers for the
entity axis, and a prediction count that divides by them. Both hold for this
model's own data module and neither holds if the pipeline is pointed at a
bundle with a flattened target axis. When they do not hold the notes are
simply omitted -- an evaluation that silently loses its per-target detail is
a worse outcome than one that reports it is unavailable, so the omission is
itself recorded as a note.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ....orchestration.pipelines.evaluate import EvaluatePipeline
from ....orchestration.pipelines.scoring import collect_targets, scoring_source

if TYPE_CHECKING:
    from ....core.contract.data import DataBundle
    from ....core.contract.result import EvalResult, EvaluationResult
    from ....core.runtime.handles import ModelHandle
    from ....orchestration.pipelines.reload import LoadedBundle

__all__ = ["HybridEvalPipeline", "breakdown_notes", "per_target_errors"]

_LOGGER = logging.getLogger(__name__)

#: How many of the worst targets to name. Enough to show whether the tail is
#: one instrument or a cluster of them, short enough to read at a glance.
_WORST_REPORTED = 5

#: Which split the breakdown is computed for. The held-out one: a per-target
#: breakdown of training error measures how well each target was memorised,
#: which is not the question anyone is asking.
_BREAKDOWN_SPLIT = "test"


def per_target_errors(
    predictions: np.ndarray, targets: np.ndarray, *, n_targets: int
) -> np.ndarray:
    """
    Return each target's mean absolute error across scenarios.

    Parameters
    ----------
    predictions, targets
        Flat arrays of equal length, laid out scenario-major: all targets
        for the first scenario, then all targets for the second. That is
        the order :func:`numpy.ravel` produces from a ``(scenario, target)``
        matrix, which is the shape this model predicts in.
    n_targets
        How many columns to fold the flat arrays back into.

    Returns
    -------
    numpy.ndarray
        One error per target, in the entity order the data declared.
    """
    errors = np.abs(np.asarray(predictions, dtype=float) - np.asarray(targets, dtype=float))
    return errors.reshape(-1, n_targets).mean(axis=0)


class HybridEvalPipeline(EvaluatePipeline):
    """
    The framework's evaluation pipeline, plus a per-target breakdown.

    A tier-2 override: it extends two hooks and leaves the stage sequence
    untouched.
    """

    def score(
        self,
        loaded: LoadedBundle,
        handle: ModelHandle,
        data: DataBundle[object],
    ) -> dict[str, EvalResult]:
        """
        Score as the framework does, then record the per-target breakdown.

        The breakdown is stashed on the instance rather than returned,
        because the stage's return type is the framework's and widening it
        here would make this model's evaluation incompatible with every
        caller that expects the base contract. A pipeline instance runs
        once, so there is nothing for the stash to go stale against.

        Parameters
        ----------
        loaded
            The opened bundle.
        handle
            The restored model.
        data
            The rebuilt data.

        Returns
        -------
        dict
            Exactly what the framework's scoring produced.
        """
        evaluations = super().score(loaded, handle, data)
        self._breakdown = self._compute_breakdown(loaded, handle, data)
        return evaluations

    def report(
        self,
        loaded: LoadedBundle,
        data: DataBundle[object],
        evaluations: dict[str, EvalResult],
    ) -> EvaluationResult:
        """
        Assemble the framework's result with the breakdown attached.

        Parameters
        ----------
        loaded
            The opened bundle.
        data
            The rebuilt data.
        evaluations
            The scored splits.

        Returns
        -------
        EvaluationResult
            The framework's result, with per-target notes.
        """
        result = super().report(loaded, data, evaluations)
        notes = getattr(self, "_breakdown", None)
        if not notes:
            return result
        return result.model_copy(update={"notes": {**result.notes, **notes}})

    def _compute_breakdown(
        self,
        loaded: LoadedBundle,
        handle: ModelHandle,
        data: DataBundle[object],
    ) -> dict[str, str]:
        """
        Run the held-out split and hand the numbers to :func:`breakdown_notes`.

        Everything that needs an engine lives here and everything that
        needs only arithmetic lives in the function below, which is what
        lets the interesting cases -- a count that does not fold, a model
        with no entity axis -- be tested without restoring a model.

        Parameters
        ----------
        loaded
            The opened bundle, for the engine the weights were trained with.
        handle
            The restored model.
        data
            The rebuilt data, for the entity identifiers and the split.

        Returns
        -------
        dict
            Notes to merge into the result.
        """
        source = scoring_source(data, _BREAKDOWN_SPLIT)
        if source is None:
            return {"per_target": f"unavailable: no {_BREAKDOWN_SPLIT!r} split"}

        # Inverted before comparing, exactly as the framework's scoring
        # does. A raw forward pass returns scaled values and
        # `collect_targets` returns original units, so comparing them
        # directly produces an error an order of magnitude too large --
        # large enough to look like a broken model rather than a broken
        # comparison, which is what makes it worth a comment.
        predictions = data.state.inverse_transform_targets(
            np.asarray(self._engine(loaded).predict(handle, source), dtype=np.float64)
        )
        targets = collect_targets(data, source, split=_BREAKDOWN_SPLIT)
        return breakdown_notes(
            tuple(data.entity_ids or ()), np.ravel(predictions), np.ravel(targets)
        )


def breakdown_notes(
    entities: tuple[str, ...], predictions: np.ndarray, targets: np.ndarray
) -> dict[str, str]:
    """
    Describe each target's error, or say why that could not be done.

    Parameters
    ----------
    entities
        The target identifiers, in the order the model predicts them.
    predictions, targets
        Flat arrays, scenario-major.

    Returns
    -------
    dict
        Notes to merge into the result. Never empty: an unavailable
        breakdown is reported rather than dropped, because a result with
        no per-target notes reads exactly like a book whose targets all
        replicated perfectly.
    """
    if not entities:
        return {"per_target": "unavailable: the data declares no entity identifiers"}
    if predictions.size != targets.size or predictions.size % len(entities):
        return {
            "per_target": (
                f"unavailable: {predictions.size} prediction(s) do not fold "
                f"into {len(entities)} target(s)"
            )
        }

    errors = per_target_errors(predictions, targets, n_targets=len(entities))
    # Sorted worst-first: the question this breakdown answers is always
    # "which target is the model failing on", never "which is it best at".
    order = np.argsort(errors)[::-1][:_WORST_REPORTED]
    worst = ", ".join(f"{entities[i]} {errors[i]:.4g}" for i in order)
    _LOGGER.info("worst-replicated target(s) on %r: %s", _BREAKDOWN_SPLIT, worst)
    return {
        "per_target_worst": worst,
        "per_target_spread": (
            f"best {errors.min():.4g}, median {float(np.median(errors)):.4g}, "
            f"worst {errors.max():.4g} across {len(entities)} target(s)"
        ),
    }
