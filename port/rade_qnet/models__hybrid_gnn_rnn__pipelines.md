# `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 54 | 2553 | `2ee06142a0e2a995` |
| 2 | `eval.py` | 260 | 9716 | `ec3525c753752a11` |
| 3 | `train.py` | 79 | 3264 | `5e713339832a8ff7` |
| 4 | `tune.py` | 218 | 9384 | `c6982aae644e43c9` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/__init__.py`

2553 bytes · SHA-256 `2ee06142a0e2a995`

```python
"""
Pipeline overrides for the hybrid graph-temporal network.

A model may override any of the four lifecycle pipelines independently,
which is why there is a file per pipeline rather than one file for all of
them. A model that needs a custom evaluation but standard training should
not have to open a file that also contains training code.

Every override here subclasses the framework base and extends a hook. None
of them replaces a stage, and that is the measure of whether the framework's
step granularity is right: an override that reimplements a stage is how a
general framework acquires its first special case.

Modules
-------
``train.py``
    Adds the graph diagnostics report. The report is contributed regardless
    of what the spec asks for, because a user narrowing the report set
    should not thereby switch off the only diagnostic that can tell them
    the graph is broken.
``eval.py``
    Adds a per-target breakdown to the result. A replication model's
    aggregate error hides the one target it cannot replicate, which is
    usually the one a desk most wants flagged.
``tune.py``
    Discards proposals whose block width does not divide by its head count,
    so no trial is spent discovering a configuration that cannot build.

Why there is no ``infer.py``
-----------------------------
It was planned, to handle target instruments unseen during training by
transferring from their nearest neighbours in the learned output space.
That is still the right feature and it is not implementable yet.

An entity absent from training is also absent from the fitted state a
reloaded model applies. For this model that means no graph node, no entry
in ``target_indices`` and no row in the encoded attribute table -- all three
come from the saved ``HybridState``, not from the inference source. There is
nothing to transfer *to*, and inventing a node at inference time would make
the transfer a function of a graph the model was never trained against.

The framework side of this is recorded as deviation 8.3 of the Phase 5
charter, which also explains why ``Inductive`` deliberately stayed a
declaration rather than growing a hook nothing could implement. The
inference pipeline is correct as it stands: it refuses an unseen entity
rather than returning a confident default, which is the behaviour that
matters until the mechanism exists.
"""

from .eval import HybridEvalPipeline
from .train import HybridTrainPipeline
from .tune import HybridTunePipeline

__all__ = ["HybridEvalPipeline", "HybridTrainPipeline", "HybridTunePipeline"]
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/eval.py`

9716 bytes · SHA-256 `ec3525c753752a11`

```python
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
from ....orchestration.stages.scoring import collect_targets, scoring_source

if TYPE_CHECKING:
    from ....core.contract.data import DataBundle
    from ....core.contract.result import EvalResult, EvaluationResult
    from ....core.lifecycle.handles import ModelHandle
    from ....orchestration.stages.reload import LoadedBundle

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
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/train.py`

3264 bytes · SHA-256 `5e713339832a8ff7`

```python
"""
The hybrid network's training pipeline.

It adds one report and changes nothing else.

Why that is the whole file
---------------------------
The framework's training sequence -- resolve, build data, declare
signature, build model, materialise, prepare hardware, fit, evaluate,
persist, report -- is the sequence this model needs. Every stage it would
have wanted to customise turned out to be customisable from the spec or
from the model definition instead, which is the outcome the framework was
designed for and is worth stating plainly rather than leaving as an
absence.

In particular, three things that *look* like they need a custom pipeline
do not:

- fitting scalers on training rows only, which the data module's split
  handling already enforces along the scenario axis while leaving the
  entity axis free;
- building the graph over the full universe, which is the same mechanism
  seen from the other side -- the entity axis is not a leakage axis;
- caching the graph embedding across an evaluation pass, which the
  ``Precomputable`` capability handles without the pipeline knowing.

If this file ever has to replace a stage rather than extend a hook, that
is a finding about the framework's step granularity and belongs in Phase 2
rather than here. A pipeline override that reimplements a stage is how a
general framework acquires its first special case.
"""

from __future__ import annotations

from ....orchestration.pipelines.train import TrainPipeline
from ..reports import HybridGraphReport

__all__ = ["HYBRID_REPORTS", "HybridTrainPipeline"]

#: Reports this model contributes regardless of what the spec asks for.
#:
#: Not merely defaulted, because a default can be overwritten by a spec that
#: lists its own reports -- and a user narrowing the report set to save time
#: should not thereby switch off the only diagnostic that can tell them the
#: graph is broken. The graph is the model's largest assumption and the one
#: least visible in any metric.
#:
#: Taken from the report class rather than written as a string, which also
#: makes this import load-bearing: enabling the pipeline is what guarantees
#: the report is registered by the time the pipeline resolves it.
HYBRID_REPORTS: tuple[str, ...] = (HybridGraphReport.component_name,)


class HybridTrainPipeline(TrainPipeline):
    """
    The framework's training pipeline, plus the graph diagnostics report.

    A tier-2 override: it extends a hook and leaves the stage sequence
    untouched.
    """

    def report_names(self) -> tuple[str, ...]:
        """
        Return the spec's reports followed by this model's own.

        Appended rather than prepended so the spec's ordering is preserved
        and a summary report that lists the figures produced still runs
        where the user put it. Duplicates are dropped rather than rejected:
        a user who names ``hybrid_graph`` explicitly has asked for
        something already guaranteed, which is a harmless redundancy and
        not an error worth failing a run over.

        Returns
        -------
        tuple of str
            Registered report names, in order, without repeats.
        """
        ordered = dict.fromkeys((*super().report_names(), *HYBRID_REPORTS))
        return tuple(ordered)
```

---

## 4. `tranql/models/rade/rade_qnet/rade_qnet/models/hybrid_gnn_rnn/pipelines/tune.py`

9384 bytes · SHA-256 `c6982aae644e43c9`

```python
"""
The hybrid network's tuning pipeline.

It discards proposals the architecture cannot build, and changes nothing
else.

Why that is worth a file
------------------------
Two of this model's layers reshape a width into heads::

    fusion:    width -> (fusion_heads,    width // fusion_heads)
    attention: width -> (attention_heads, width // attention_heads)

Neither reshape is meaningful unless the width divides exactly, so both
blocks raise :class:`~rade_qnet.core.lifecycle.errors.ContractError` at
construction when it does not. That is the right behaviour -- the
alternative is dropping ``width % heads`` features per head silently -- but
it makes a joint search over ``units`` and ``*_heads`` wasteful in a way the
framework cannot see.

Consider ``units in {16, 24, 32}`` crossed with ``fusion_heads in {1, 3, 4}``.
Four of the nine cells cannot build. A grid search spends 44% of its budget
discovering that, and each discovery costs a resolve, a model construction
and a traceback rather than a cheap arithmetic check. A random search over a
wider range does proportionally worse and, unlike the grid, does not even
fail the same way twice.

The framework cannot prevent this on its own. :class:`TuneSpec` validates
that a proposal *parses* -- that every path exists on the spec and every
value has the right type and range, which is Phase 5's cure for defect 7 --
and that is as far as a general validator can go. Whether two
independently-valid numbers are compatible *with each other* is a statement
about this architecture, and it lives with this architecture.

Why these are dropped rather than rejected
-------------------------------------------
An infeasible proposal is not a user error. A user who writes
``units: [16, 24, 32]`` and ``fusion_heads: [1, 3, 4]`` has asked a perfectly
reasonable question; it simply has fewer than nine answers. Failing the
search would force them to enumerate the feasible cells by hand, which is
both tedious and exactly the arithmetic this file already does.

So the proposals are filtered and the count is logged. What is *not* allowed
is a search that silently becomes empty: if the filter removes everything,
the space and the architecture genuinely contradict each other, and that is
a user error worth stopping for.

Why a pipeline override rather than a spec validator
------------------------------------------------------
``HybridModelSpec`` could reject the combination in a ``model_validator``,
and then every trial would fail fast instead of failing late. That is a
worse outcome, not a better one: the trial would still be *spent*. The
budget is the scarce thing in a search, not the seconds.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import ValidationError

from ....core.lifecycle.errors import ContractError, SpecError
from ....orchestration.pipelines.tune import TunePipeline
from ....orchestration.stages.search import expand
from ..spec import HybridModelSpec

if TYPE_CHECKING:
    from typing import Any

    from ....core.spec.run import RunSpec

__all__ = ["MODEL_NAME", "HybridTunePipeline", "is_buildable"]

_LOGGER = logging.getLogger(__name__)

#: The blocks whose width is reshaped into heads, as ``(block, head field)``.
#:
#: Written as data rather than as two ``if`` statements so that a third
#: multi-head block added to the network is a one-line change here, and so
#: that forgetting to make it is visible as an omission from a list rather
#: than as an absence of code.
_HEADED_BLOCKS: tuple[tuple[str, str], ...] = (
    ("fusion", "fusion_heads"),
    ("attention", "attention_heads"),
)

#: The registered name this pipeline belongs to, used to recognise a run
#: spec it is entitled to have an opinion about.
MODEL_NAME = "hybrid_gnn_rnn"


def is_buildable(spec: RunSpec) -> bool:
    """
    Report whether this architecture can be constructed as specified.

    Mirrors the checks the layers themselves perform, deliberately: the
    layer is the authority and this is a cheap preview of its answer. The
    duplication is bounded -- two divisibility tests -- and the cost of
    them disagreeing is a trial that this function admitted and the model
    then refused, which surfaces as an ordinary failed trial rather than
    as a wrong number. The reverse duplication, where this function is the
    only check, is the one that would be dangerous.

    The model's parameters are parsed here rather than read off the spec,
    because a run spec carries a :class:`ComponentRef` -- a name and a bag
    of raw values -- and the typed spec that knows how a block's width is
    resolved does not exist until the pipeline's ``resolve`` stage. Reading
    ``params`` directly would mean reimplementing the defaulting rules that
    turn ``units`` into a per-block width, which is precisely the kind of
    second opinion this function is trying not to be.

    Parameters
    ----------
    spec
        A candidate run specification, already parsed and validated.

    Returns
    -------
    bool
        True if every multi-head block's width divides by its head count.

    Raises
    ------
    SpecError
        If the model parameters do not parse at all, which is a different
        question with a different answer: the trial is not infeasible, the
        request is malformed, and silently dropping it would turn a typo
        into a smaller search rather than an error.
    """
    reference = spec.model
    # Guards against being handed another model's run spec, which happens
    # if this pipeline is attached to the wrong definition. Answering True
    # is correct: this function knows of no reason that spec cannot build.
    if getattr(reference, "name", None) != MODEL_NAME:
        return True

    try:
        model = HybridModelSpec.model_validate(dict(reference.params))
    except ValidationError as error:
        raise SpecError(f"the proposed model parameters do not parse: {error}") from error

    return all(model.width(block) % getattr(model, field) == 0 for block, field in _HEADED_BLOCKS)


class HybridTunePipeline(TunePipeline):
    """
    The framework's tuning pipeline, minus the trials that cannot build.

    ``propose`` is a stage rather than a hook, so this is a tier-3
    override -- but an additive one: it calls the framework's stage and
    filters the result, never replacing it. The stage sequence itself is
    untouched, and the framework's own validation still runs first, which
    is what keeps a mistyped search path a user error rather than a
    silently smaller search.
    """

    def propose(self) -> list[dict[str, Any]]:
        """
        Return the framework's proposals with the infeasible ones removed.

        Called after ``super()``, never instead of it, so the framework's
        own validation still runs first: a proposal that does not parse is
        rejected as a user error before this file sees it, and only
        proposals that are individually valid reach the compatibility test.

        Each surviving proposal is resolved to a run spec a second time --
        the base resolved it to validate it and kept only the flat form.
        Re-resolving costs a spec construction per trial and saves passing
        a parallel list of specs through a hook signature that every other
        model would then have to understand. The budget this file protects
        is measured in trials, not in parses.

        Returns
        -------
        list of dict
            One flat proposal per surviving trial.

        Raises
        ------
        SpecError
            If a proposal's model parameters do not parse. The framework's
            own check cannot see this: a run spec carries the model as a
            name and an untyped bag of values, and the bag is not validated
            against this model's schema until the ``resolve`` stage -- by
            which point the trial has been spent. Extending defect 7's cure
            to reach model parameters is a side-benefit of having to parse
            them here anyway.
        ContractError
            If no proposal in the whole space can be built, which means the
            search space and the architecture contradict each other.
        """
        proposals = super().propose()
        feasible = [
            proposal
            for index, proposal in enumerate(proposals)
            if is_buildable(self.spec.run_spec_for(expand(proposal), trial=index))
        ]

        discarded = len(proposals) - len(feasible)
        if not feasible:
            raise ContractError(
                f"none of the {len(proposals)} proposed trial(s) can be built: in "
                f"every one, a block's width is not divisible by its head count. "
                f"The search space and the architecture contradict each other, so "
                f"running it would produce a search with no trials rather than a "
                f"search with no winner. Check 'units' against "
                f"{[field for _, field in _HEADED_BLOCKS]}"
            )
        if discarded:
            _LOGGER.info(
                "discarded %d of %d proposed trial(s) whose width does not divide "
                "by its head count; %d remain",
                discarded,
                len(proposals),
                len(feasible),
            )
        return feasible
```

