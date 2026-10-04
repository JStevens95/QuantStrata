"""
The template-method base every pipeline derives from.

A pipeline is a fixed sequence of small, typed, individually overridable
stages. :meth:`Pipeline.run` composes them; :meth:`Pipeline.step` wraps each
one. That split is what makes the four customisation tiers possible:

1. **Spec only.** Write no pipeline code at all.
2. **Add observation.** Attach hooks and reports.
3. **Override one stage.** Subclass, replace one method, inherit the rest --
   including its timing, logging, error wrapping and caching.
4. **Override** ``run``. Change the sequence itself. Rare, and the only tier
   that gives up the instrumentation.

Tier 3 is the one that justifies this design, and it only works because a
stage is small enough to be replaced in isolation. A monolithic ``train()``
offers tiers 1, 2 and 4 and nothing between them -- which in practice means
every non-standard model copies the whole function and diverges from it.

What the wrapper buys
---------------------
:meth:`step` is the single place that times a stage, logs its start and end,
notifies hooks, reports errors and wraps failures with the stage name. Putting
it in one place has a concrete payoff: an override written by a user gets all
of it for free and cannot forget any of it. The alternative -- each stage
instrumenting itself -- guarantees that the one stage someone adds later is
the one with no timing and an unhelpful traceback.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from ..provenance.logging import get_logger
from .errors import StageError

if TYPE_CHECKING:
    from pathlib import Path

    from .context import RunContext

__all__ = ["Pipeline"]

_LOGGER = get_logger(__name__)


class Pipeline[ResultT](ABC):
    """
    Base class for the train, evaluate, infer and tune pipelines.

    Generic over its result type, so a subclass declares what it produces and
    a caller knows without inspecting the implementation.

    Parameters
    ----------
    context
        Ambient state for the run.

    Notes
    -----
    A pipeline instance is used once. It is not reset between runs, and it
    holds no state that two runs could share, which is what makes it safe to
    construct one per job in a parallel job set.
    """

    #: Stage names, in the order :meth:`run` invokes them. Declared rather than
    #: inferred so the sequence is documentation, can be rendered in a report,
    #: and can be compared against what actually ran.
    stages: tuple[str, ...] = ()

    def __init__(self, context: RunContext) -> None:
        """
        Store the run context.

        Parameters
        ----------
        context
            Ambient state for the run.
        """
        self.context = context
        self._completed: list[str] = []
        self._timings: dict[str, float] = {}

    @abstractmethod
    def run(self) -> ResultT:
        """
        Execute the pipeline.

        Implementations compose their stages through :meth:`step` so each one
        is timed, logged and reported. A subclass that overrides this method
        entirely takes on that responsibility itself.

        Returns
        -------
        ResultT
            Whatever this pipeline produces.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised, naming the stage.
        """

    def execute(self) -> ResultT:
        """
        Run the pipeline with run-level instrumentation.

        The entry point a caller should use. :meth:`run` is the method a
        subclass writes; this is the method that surrounds it with the
        ``on_run_start`` and ``on_run_end`` notifications, the logging
        context, and the guarantee that ``on_run_end`` fires even on failure.

        Keeping the two apart means a subclass overriding ``run`` -- tier 4 --
        still gets the run-level bookkeeping, and that an overriding author
        cannot forget to report that the run ended.

        Returns
        -------
        ResultT
            The pipeline's result.

        Raises
        ------
        StageError
            Wrapping whatever a stage raised.
        """
        context = self.context
        succeeded = False
        with context.activate():
            _LOGGER.info("starting %s", type(self).__name__)
            for line in context.describe():
                _LOGGER.debug("  %s", line)
            context.notify(
                lambda hook: hook.on_run_start(context.run_id, context.spec_digest),
                description="run start",
            )
            try:
                result = self.run()
                succeeded = True
            finally:
                # Fires on the failure path too: a hook that opened a file or a
                # tracker run must be told the run is over, and told the truth
                # about whether it worked.
                context.notify(
                    lambda hook: hook.on_run_end(context.run_id, succeeded=succeeded),
                    description="run end",
                )
                if context.tracker is not None:
                    try:
                        context.tracker.finish(succeeded=succeeded)
                    except Exception:  # Broad by design: tracking is never load-bearing.
                        _LOGGER.warning("tracker failed to finish; continuing", exc_info=True)
            _LOGGER.info("completed %s in %.2fs", type(self).__name__, sum(self._timings.values()))
            return result

    def step[StepResultT](self, name: str, operation: Callable[[], StepResultT]) -> StepResultT:
        """
        Run one stage, instrumented.

        Takes a zero-argument callable rather than a method name so a stage's
        own arguments stay explicit and type checked at the call site:
        ``bundle = self.step("build_data", lambda: self.build_data(spec))``.

        Parameters
        ----------
        name
            Stage name. Should appear in :attr:`stages`.
        operation
            The work to perform.

        Returns
        -------
        StepResultT
            Whatever the operation returned.

        Raises
        ------
        StageError
            Wrapping whatever the operation raised. The original exception is
            chained, so no detail is lost, but the message names the stage --
            which turns "``KeyError: 'features'``" into something that
            identifies where to look.
        """
        context = self.context
        with context.activate(stage=name):
            context.notify(lambda hook: hook.on_stage_start(name), description=f"{name} start")
            _LOGGER.info("stage %s starting", name)
            started = time.perf_counter()
            try:
                result = operation()
            except Exception as error:
                # Rebound because Python unbinds an `except ... as` name at the
                # end of the block, which would leave the closure below empty.
                failure = error
                elapsed = time.perf_counter() - started
                self._timings[name] = elapsed
                _LOGGER.error("stage %s failed after %.2fs", name, elapsed)
                context.notify(
                    lambda hook: hook.on_stage_error(name, failure),
                    description=f"{name} error",
                )
                # A StageError is already attributed, so re-wrapping it would
                # produce "stage fit failed [StageError] stage fit failed ..."
                # for a nested pipeline. Let the inner attribution stand.
                if isinstance(failure, StageError):
                    raise
                raise StageError(name, failure, run_id=context.run_id) from failure
            elapsed = time.perf_counter() - started
            self._timings[name] = elapsed
            self._completed.append(name)
            _LOGGER.info("stage %s completed in %.2fs", name, elapsed)
            context.notify(
                lambda hook: hook.on_stage_end(name, seconds=elapsed),
                description=f"{name} end",
            )
            return result

    def report_metrics(self, stage: str, metrics: Mapping[str, float]) -> None:
        """
        Publish metrics to the hooks and the tracker.

        Parameters
        ----------
        stage
            Stage that produced them.
        metrics
            Metric name to value, in original target units. Metrics in a
            model's internal space must be inverted before reaching here --
            see :meth:`~rade_qnet.core.contract.state.FittedState.inverse_transform_targets`
            -- because everything downstream of this call treats them as
            comparable across runs.
        """
        self.context.notify(
            lambda hook: hook.on_metrics(stage, metrics),
            description=f"{stage} metrics",
        )
        self.context.track_metrics(metrics)

    def report_artifact(self, name: str, path: Path) -> None:
        """
        Publish a written file to the hooks and the tracker.

        Parameters
        ----------
        name
            Logical name of the artifact.
        path
            Where it was written.
        """
        self.context.notify(
            lambda hook: hook.on_artifact(name, str(path)),
            description=f"artifact {name}",
        )
        self.context.track_artifact(path, name=name)

    @property
    def completed_stages(self) -> tuple[str, ...]:
        """Stages that finished, in the order they finished."""
        return tuple(self._completed)

    @property
    def timings(self) -> Mapping[str, float]:
        """Wall time per stage, including a stage that failed."""
        return dict(self._timings)
