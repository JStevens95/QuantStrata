"""
Searching a space of configurations for the one that scores best.

A search is forty training runs that differ in a handful of values, and
almost everything that makes it useful or useless is about the bookkeeping
around those runs rather than the runs themselves.

What this pipeline is careful about
-----------------------------------
**Building the data once.** Forty trials over one dataset should read it
once. The outline asked for a general step cache to achieve that; this holds
the prepared dataset for the search instead, for the reasons recorded in
[Phase 5 §8.1](../../docs/phases/PHASE_5_EVALUATE_INFER_TUNE.md). Where the
trials vary the *source*, there is nothing to reuse and the pipeline says so
rather than quietly reusing the wrong thing.

**Failing trials are recorded, not swallowed.** A search that drops a failing
trial and reports the best of the rest looks exactly like a search in which
every trial succeeded. If eleven of forty failed because an override was
nonsense, the conclusion drawn from the remaining twenty-nine is probably
wrong, and nothing in a swallowed-failure report would suggest it. Same
decision as a job set, for the same reason.

**The winner is picked once, by a stated rule.** Direction comes from the
specification rather than from guessing at the metric's name, and ties go to
the earlier trial so a re-run picks the same winner.

**Selection is on validation.** The default objective split is validation,
because selecting on test turns the held-out split into part of the training
procedure -- and the test metric then reported for the winner overstates it
by however much the search exploited that split.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ...core.contract.result import TrialRecord, TuningResult
from ...core.runtime.components import get_model
from ...core.runtime.context import RunContext
from ...core.runtime.errors import ComponentError, ContractError, SpecError
from ...core.runtime.logging import get_logger
from ...core.runtime.pipeline import Pipeline
from ...core.spec.run import SupervisedRunSpec
from .search import expand, propose
from .train import TrainPipeline

if TYPE_CHECKING:
    from ...core.capability.definition import PredictorDefinition
    from ...core.contract.data import DataBundle
    from ...core.spec.tune import TuneSpec

__all__ = ["TunePipeline"]

_LOGGER = get_logger(__name__)


class TunePipeline(Pipeline[TuningResult]):
    """
    Run a search and return its trials, with the best one identified.

    Parameters
    ----------
    context
        Ambient state for the run.
    spec
        The search specification.
    definition
        The model definition. Defaults to ``None``, meaning resolve it from
        the base specification's model name. Passed in rather than always
        resolved so a test can supply a definition without registering it
        globally, exactly as :class:`~.train.TrainPipeline` allows.

    Attributes
    ----------
    shared_data
        The dataset reused across trials, available after ``build_data``.
        ``None`` when the search varies the source, in which case each trial
        builds its own.
    """

    stages = (
        "resolve",
        "propose",
        "build_data",
        "run_trials",
        "select",
        "refit",
    )

    def __init__(
        self,
        context: RunContext,
        spec: TuneSpec,
        definition: PredictorDefinition | None = None,
    ) -> None:
        """
        Store the search and the model it searches over.

        Parameters
        ----------
        context
            Ambient state for the run.
        spec
            The search specification.
        definition
            The model definition, or ``None`` to resolve it by name.
        """
        super().__init__(context)
        self.spec = spec
        self.definition = definition
        self.shared_data: DataBundle[object] | None = None
        self._builds = 0

    @property
    def data_builds(self) -> int:
        """
        Return how many times the dataset was actually built.

        Exposed because "built once per search" is a claim worth testing,
        and a test that could only time the search would be measuring the
        machine rather than the pipeline.

        Returns
        -------
        int
            The count.
        """
        return self._builds

    def run(self) -> TuningResult:
        """
        Execute the stage sequence.

        Returns
        -------
        TuningResult
            Every trial, and which one won.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """
        definition = self.step("resolve", self.resolve)
        proposals = self.step("propose", self.propose)
        self.step("build_data", lambda: self.build_data(definition))
        trials = self.step("run_trials", lambda: self.run_trials(definition, proposals))
        result = self.step("select", lambda: self.select(trials))
        return self.step("refit", lambda: self.refit(definition, result))

    def resolve(self) -> PredictorDefinition:
        """
        Resolve the model, and validate every proposal's *shape* before any run.

        Resolution first, for the same reason training resolves before
        building: a misspelled model name should cost milliseconds rather
        than a data build.

        Parameters
        ----------
        None

        Returns
        -------
        PredictorDefinition
            The definition every trial trains.

        Raises
        ------
        ContractError
            If the base specification names no model and none was supplied.
        """
        if self.definition is not None:
            return self.definition

        name = self.spec.base.get("model")
        if isinstance(name, dict):
            name = name.get("name")
        if not isinstance(name, str):
            raise ContractError(
                "the search's base specification names no model, and no "
                "definition was supplied; a search has to know what it is "
                "searching over"
            )
        try:
            definition = get_model(name)()
        except ComponentError as error:
            raise ContractError(
                f"the search names the model {name!r}, which is not registered "
                f"in this process; import the package that defines it first"
            ) from error
        self.definition = definition
        return definition

    def propose(self) -> list[dict[str, Any]]:
        """
        Produce every trial's proposal, and validate each one now.

        Validating the whole set before the first trial runs is the cure for
        defect 7. A typo in a dotted path is a mistake nobody typed -- the
        values are generated -- and left unchecked it would be explored,
        measured and reported as a dimension that makes no difference, which
        reads as a finding rather than a bug.

        Returns
        -------
        list of dict
            One flat proposal per trial.

        Raises
        ------
        SpecError
            If any proposal does not merge into a valid run specification.
        """
        proposals = propose(self.spec)
        failures: list[str] = []
        for index, proposal in enumerate(proposals):
            try:
                self.spec.run_spec_for(expand(proposal), trial=index)
            except SpecError as error:
                failures.append(f"  trial {index} ({proposal}): {error}")

        if failures:
            raise SpecError(
                f"{len(failures)} of {len(proposals)} proposed trial(s) do not "
                f"produce a valid run specification. The search space names a "
                f"path the specification does not have, so the search would "
                f"explore it and report that it makes no difference:\n"
                + "\n".join(failures[: self._REPORTED_FAILURES])
            )
        return proposals

    #: How many invalid proposals to spell out. A broken path breaks every
    #: trial, so printing forty identical messages buries the one that
    #: matters.
    _REPORTED_FAILURES = 3

    def build_data(self, definition: PredictorDefinition) -> DataBundle[object] | None:
        """
        Build the dataset once, when every trial shares one.

        Skipped when the space varies anything under ``source``: the trials
        then have genuinely different datasets, and handing them a shared
        one would mean the search measured something other than what it
        proposed -- the quietest possible way for a search to be wrong.

        Parameters
        ----------
        definition
            The model definition.

        Returns
        -------
        DataBundle or None
            The shared dataset, or ``None`` when each trial builds its own.
        """
        if self._varies_the_source():
            _LOGGER.info(
                "the search varies the source, so each trial builds its own "
                "dataset; there is nothing shared to reuse"
            )
            return None

        spec = self.spec.run_spec_for(expand({}), trial=0)
        if not isinstance(spec, SupervisedRunSpec):
            raise ContractError(
                f"tuning builds data for a supervised run; the base "
                f"specification resolves to a {type(spec).__name__}"
            )

        data = definition.build_data(spec)
        self._builds += 1
        self.shared_data = data
        _LOGGER.info(
            "built the dataset once for %d trial(s): %s",
            self.spec.trials,
            {name: data.splits[name] is not None for name in data.split_names},
        )
        return data

    def run_trials(
        self, definition: PredictorDefinition, proposals: list[dict[str, Any]]
    ) -> list[TrialRecord]:
        """
        Train and score one model per proposal.

        Parameters
        ----------
        definition
            The model definition.
        proposals
            The flat proposals.

        Returns
        -------
        list of TrialRecord
            One record per trial, in trial order, successes and failures
            alike.
        """
        records: list[TrialRecord] = []
        for index, proposal in enumerate(proposals):
            records.append(self._trial(definition, proposal, index=index))
        succeeded = sum(1 for record in records if record.succeeded)
        _LOGGER.info("%d of %d trial(s) succeeded", succeeded, len(records))
        return records

    def select(self, trials: list[TrialRecord]) -> TuningResult:
        """
        Pick the winner, by the direction the specification states.

        Parameters
        ----------
        trials
            Every trial.

        Returns
        -------
        TuningResult
            The trials and the winning index.

        Raises
        ------
        ContractError
            If no trial produced the objective. Raised rather than returning
            a result with no winner, because a search that selected nothing
            and said so quietly would be read as a search that found nothing
            good.
        """
        scored = [record for record in trials if record.objective is not None]
        if not scored:
            raise ContractError(
                f"no trial produced the objective {self.spec.objective!r} on the "
                f"{self.spec.objective_split!r} split, so there is nothing to "
                f"select between. {len(trials)} trial(s) ran and "
                f"{sum(1 for record in trials if not record.succeeded)} failed"
            )

        best = scored[0]
        for record in scored[1:]:
            # Ties go to the incumbent, which makes the winner the earliest
            # best trial and therefore stable across a re-run.
            if self.spec.is_better(record.objective, best.objective):
                best = record

        result = TuningResult(
            trials=tuple(trials),
            best_trial=best.trial,
            objective=self.spec.objective,
            direction=self.spec.direction,
            objective_split=self.spec.objective_split,
            name=self.spec.name,
        )
        _LOGGER.info("%s", result.describe())
        return result

    def refit(
        self, definition: PredictorDefinition, result: TuningResult
    ) -> TuningResult:
        """
        Optionally retrain the winner on train and validation combined.

        Off unless asked for. The extra data usually helps, but the
        resulting model's reported objective was measured on rows it has now
        trained on, so the number beside it is no longer a held-out estimate
        -- and that trade should be a choice someone made rather than a
        default they inherited.

        Not yet implemented beyond recording the intent: combining two
        splits means re-deriving a split, which this phase spent its effort
        making impossible to do by accident. It is Phase 6 work, and the
        specification field exists so the search's shape is settled now.

        Parameters
        ----------
        definition
            The model definition.
        result
            The selected result.

        Returns
        -------
        TuningResult
            The result, unchanged when no refit was asked for.

        Raises
        ------
        ContractError
            If a refit was asked for. Refused rather than silently skipped:
            a user who set ``refit: true`` and received a model trained on
            the training split alone would have no way to tell.
        """
        del definition
        if not self.spec.refit:
            return result
        raise ContractError(
            "refit is not implemented yet. Combining the training and "
            "validation splits means deriving a new split, which this phase "
            "deliberately made hard to do by accident; it is Phase 6 work. "
            "Set refit: false and retrain the winning configuration directly "
            "if you need the combined fit now"
        )

    def _trial(
        self,
        definition: PredictorDefinition,
        proposal: dict[str, Any],
        *,
        index: int,
    ) -> TrialRecord:
        """
        Run one trial, recording its outcome whether it worked or not.

        Parameters
        ----------
        definition
            The model definition.
        proposal
            The flat proposal.
        index
            Trial number.

        Returns
        -------
        TrialRecord
            The outcome.
        """
        started = time.perf_counter()
        spec = self.spec.run_spec_for(expand(proposal), trial=index)

        try:
            pipeline = _SharedDataTrainPipeline(
                context=self._trial_context(index),
                spec=spec,
                definition=definition,
                shared=self.shared_data,
            )
            outcome = pipeline.execute()
            objective = self._objective_of(outcome)
            record = TrialRecord(
                trial=index,
                overrides=dict(proposal),
                status="succeeded",
                objective=objective,
                metrics=dict(
                    outcome.evaluations[self.spec.objective_split].metrics
                    if self.spec.objective_split in outcome.evaluations
                    else {}
                ),
                bundle_directory=(
                    None if pipeline.saved is None else str(pipeline.saved.directory)
                ),
                wall_seconds=time.perf_counter() - started,
            )
        except Exception as error:
            # Recorded rather than raised, for the reason in the module
            # docstring: a search that stopped at the first bad trial would
            # lose thirty-nine good ones, and one that dropped it silently
            # would report a conclusion drawn from a partial sweep.
            record = TrialRecord(
                trial=index,
                overrides=dict(proposal),
                status="failed",
                wall_seconds=time.perf_counter() - started,
                failure_kind=type(error).__name__,
                failure_message=str(error),
            )
            _LOGGER.warning("trial %d failed: %s: %s", index, type(error).__name__, error)

        return record

    def _trial_context(self, index: int) -> RunContext:
        """
        Give one trial its own directory beneath the search's.

        Every trial trains a model, and a trained model is persisted -- so
        without this, forty trials would all claim version 1 of the same
        bundle path and thirty-nine would fail on a refusal to overwrite.
        The refusal is correct; sharing the directory was the mistake.

        A directory per trial also keeps the winner's bundle on disk, which
        is the difference between acting on a search and repeating it. The
        cost is that a long search leaves forty bundles behind, which is the
        honest price of being able to use the one that won.

        Parameters
        ----------
        index
            Trial number.

        Returns
        -------
        RunContext
            The search's context, redirected at the trial's directory and
            carrying the trial number as its job identifier so log lines
            from concurrent-looking output can be told apart.
        """
        return replace(
            self.context,
            output_directory=self.context.output_directory / "trials" / f"trial-{index}",
            job_id=f"trial-{index}",
        )

    def _objective_of(self, outcome: object) -> float | None:
        """
        Read the objective metric from a trial's result.

        Parameters
        ----------
        outcome
            The trial's training result.

        Returns
        -------
        float or None
            The value, or ``None`` if the split or the metric is absent --
            which :meth:`select` then reports rather than treating as a
            score of zero.
        """
        evaluations = getattr(outcome, "evaluations", {})
        evaluation = evaluations.get(self.spec.objective_split)
        if evaluation is None:
            return None
        return evaluation.metrics.get(self.spec.objective)

    def _varies_the_source(self) -> bool:
        """
        Return whether any search axis changes how the data is built.

        Checked on the path prefix rather than by building twice and
        comparing: the point is to decide *before* building anything.

        Returns
        -------
        bool
            True if a shared dataset would be wrong.
        """
        return any(
            path == "source" or path.startswith("source.")
            for path in self.spec.space.paths
        )


class _SharedDataTrainPipeline(TrainPipeline):
    """
    A training pipeline that reuses a dataset instead of building one.

    A subclass overriding one stage rather than a flag on
    :class:`~.train.TrainPipeline`, so that the ordinary training path has
    no branch in it at all. Training is the pipeline every other phase
    depends on, and a conditional there would be reachable from every run
    rather than only from a search.

    Parameters
    ----------
    context
        Ambient state for the run.
    spec
        The trial's validated specification.
    definition
        The model definition.
    shared
        The dataset to reuse, or ``None`` to build one normally.
    """

    def __init__(
        self,
        context: RunContext,
        spec: SupervisedRunSpec,
        definition: PredictorDefinition,
        *,
        shared: DataBundle[object] | None,
    ) -> None:
        """
        Store the shared dataset alongside the usual arguments.

        Parameters
        ----------
        context
            Ambient state for the run.
        spec
            The trial's specification.
        definition
            The model definition.
        shared
            The dataset to reuse, or ``None``.
        """
        super().__init__(context=context, spec=spec, definition=definition)
        self._shared = shared

    def build_data(self, spec: SupervisedRunSpec) -> DataBundle[object]:
        """
        Return the shared dataset, or build one when there is none.

        Parameters
        ----------
        spec
            The trial's specification.

        Returns
        -------
        DataBundle
            The dataset.
        """
        if self._shared is not None:
            return self._shared
        return super().build_data(spec)
