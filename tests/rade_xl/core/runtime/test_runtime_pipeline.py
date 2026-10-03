"""
Tests for the pipeline base and its instrumented step runner.

``step()`` is the single choke point every stage passes through, and this module
pins down everything a user's stage override gets for free by going through it:
timing, lifecycle events, logging context, and a failure that always names its
stage. The point of centralising those is that an override written later cannot
forget any of them -- so these tests are really testing that the customisation
tiers hold.
"""

from __future__ import annotations

import logging

import pytest

from src.rade_xl.core.runtime.context import RunContext
from src.rade_xl.core.runtime.errors import StageError
from src.rade_xl.core.runtime.hooks import PipelineHook
from src.rade_xl.core.runtime.logging import configure_logging, current_context
from src.rade_xl.core.runtime.pipeline import Pipeline
from src.rade_xl.testkit.fixtures import RecordingHook


class DemoPipeline(Pipeline[str]):
    """
    A three-stage pipeline that can be made to fail at a chosen stage.

    Parameters
    ----------
    context
        Ambient state for the run.
    fail_at
        Stage that should raise, or ``None`` for a clean run.
    """

    stages = ("first", "second", "third")

    def __init__(self, context: RunContext, *, fail_at: str | None = None) -> None:
        """Store the failure point."""
        super().__init__(context)
        self.fail_at = fail_at
        self.context_during_stage: dict[str, str] | None = None

    def run(self) -> str:
        """Run three stages and return a marker."""
        for stage in self.stages:
            self.step(stage, lambda stage=stage: self._work(stage))
        return "done"

    def _work(self, stage: str) -> str:
        """Do a stage's work, recording the bound logging identifiers."""
        self.context_during_stage = current_context()
        if self.fail_at == stage:
            message = f"{stage} could not complete"
            raise KeyError(message)
        return stage


@pytest.fixture
def context(run_directory):
    """
    Provide a context with a recording hook.

    Parameters
    ----------
    run_directory
        Per-test working directory.

    Returns
    -------
    RunContext
        A context whose hook records every event.
    """
    configure_logging(level=logging.CRITICAL, force=True)
    return RunContext(
        run_id="r-001",
        spec_digest="d" * 64,
        output_directory=run_directory,
        hooks=(RecordingHook(),),
    )


class TestSuccessfulRun:
    """A clean run reports the full lifecycle in order."""

    def test_the_result_is_returned(self, context):
        """``execute`` returns what ``run`` produced."""
        assert DemoPipeline(context).execute() == "done"

    def test_every_stage_is_recorded_as_completed(self, context):
        """Completion order is recorded, not just a count."""
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert pipeline.completed_stages == ("first", "second", "third")

    def test_every_stage_is_timed(self, context):
        """
        Timing is automatic.

        A stage override gets it without doing anything, which is the point
        of routing everything through one runner.
        """
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert sorted(pipeline.timings) == ["first", "second", "third"]
        assert all(duration >= 0.0 for duration in pipeline.timings.values())

    def test_the_event_sequence_is_correct(self, context):
        """
        Run start, then paired stage events, then run end.

        The pairing is what a progress display depends on.
        """
        hook = context.hooks[0]
        DemoPipeline(context).execute()
        assert hook.names() == (
            "run_start",
            "stage_start",
            "stage_end",
            "stage_start",
            "stage_end",
            "stage_start",
            "stage_end",
            "run_end",
        )

    def test_the_run_is_reported_as_succeeded(self, context):
        """``on_run_end`` tells the truth about the outcome."""
        hook = context.hooks[0]
        DemoPipeline(context).execute()
        assert hook.events[-1] == ("run_end", "r-001", True)


class TestStageContext:
    """Each stage runs with its own logging identifiers bound."""

    def test_the_stage_name_is_bound_during_the_stage(self, context):
        """
        A log line from inside a stage is attributed to that stage.

        Including a line emitted by a user's model code, which knows nothing
        about the pipeline.
        """
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert pipeline.context_during_stage == {"run_id": "r-001", "stage": "third"}

    def test_identifiers_are_released_after_the_run(self, context):
        """Nothing leaks into whatever runs next in the process."""
        DemoPipeline(context).execute()
        assert current_context() == {}


class TestFailure:
    """A failure is attributed, reported, and not swallowed."""

    def test_the_failure_names_the_stage(self, context):
        """
        The whole reason for wrapping.

        A bare ``KeyError: 'features'`` six frames into a forward pass does
        not say which stage produced it.
        """
        with pytest.raises(StageError, match="stage 'second' failed"):
            DemoPipeline(context, fail_at="second").execute()

    def test_the_original_exception_is_chained(self, context):
        """Attribution adds information; it must not discard any."""
        with pytest.raises(StageError) as caught:
            DemoPipeline(context, fail_at="second").execute()
        assert isinstance(caught.value.__cause__, KeyError)

    def test_the_run_identifier_is_included(self, context):
        """A job set's failure must say which run it came from."""
        with pytest.raises(StageError, match="r-001"):
            DemoPipeline(context, fail_at="first").execute()

    def test_later_stages_do_not_run(self, context):
        """A pipeline stops at the first failure."""
        pipeline = DemoPipeline(context, fail_at="second")
        with pytest.raises(StageError):
            pipeline.execute()
        assert pipeline.completed_stages == ("first",)

    def test_the_failing_stage_is_still_timed(self, context):
        """
        How long a stage ran before failing is diagnostic.

        A stage that failed after two seconds and one that failed after two
        hours are different problems.
        """
        pipeline = DemoPipeline(context, fail_at="second")
        with pytest.raises(StageError):
            pipeline.execute()
        assert "second" in pipeline.timings

    def test_the_error_is_reported_to_hooks(self, context):
        """An observer learns which stage failed and how."""
        hook = context.hooks[0]
        with pytest.raises(StageError):
            DemoPipeline(context, fail_at="second").execute()
        assert ("stage_error", "second", "KeyError") in hook.events

    def test_run_end_fires_on_failure(self, context):
        """
        An observer is always told the run is over.

        A hook holding an open file or a tracker run must be able to close it
        whatever happened.
        """
        hook = context.hooks[0]
        with pytest.raises(StageError):
            DemoPipeline(context, fail_at="second").execute()
        assert hook.events[-1] == ("run_end", "r-001", False)

    def test_a_nested_stage_error_is_not_wrapped_twice(self, context):
        """
        Attribution happens once.

        A sub-pipeline's ``StageError`` already names its stage; re-wrapping
        would produce "stage fit failed [StageError] stage fit failed ...".
        """

        class Outer(Pipeline[str]):
            """Runs a stage that raises an already-attributed error."""

            stages = ("outer",)

            def run(self) -> str:
                """Raise a pre-attributed StageError from inside a step."""
                return self.step("outer", self._inner)

            def _inner(self) -> str:
                """Raise as though a nested pipeline had failed."""
                raise StageError("inner", ValueError("deep failure"), run_id="r-001")

        with pytest.raises(StageError) as caught:
            Outer(context).execute()
        assert str(caught.value).count("failed") == 1
        assert "inner" in str(caught.value)


class TestReporting:
    """Metrics and artifacts reach hooks and the tracker."""

    def test_metrics_are_published(self, context):
        """A dashboard sees metrics without the pipeline knowing about it."""

        class Reporting(Pipeline[None]):
            """Publishes one metric set."""

            stages = ()

            def run(self) -> None:
                """Publish metrics."""
                self.report_metrics("evaluate", {"mae": 0.25})

        hook = context.hooks[0]
        Reporting(context).execute()
        assert ("metrics", "evaluate", {"mae": 0.25}) in hook.events

    def test_artifacts_are_published(self, context):
        """A written file is announced as a path, not as contents."""

        class Writing(Pipeline[None]):
            """Publishes one artifact."""

            stages = ()

            def run(self) -> None:
                """Publish an artifact."""
                self.report_artifact("summary", self.context.reports_directory / "summary.md")

        hook = context.hooks[0]
        Writing(context).execute()
        assert ("artifact", "summary") in hook.events


class TestHookTolerance:
    """A broken observer cannot break a run, even at run level."""

    def test_a_hook_failing_at_run_start_does_not_fail_the_run(self, run_directory):
        """Consistent with the rule everywhere else that a hook may not."""

        class Broken(PipelineHook):
            """Raises on every hook point."""

            def on_run_start(self, run_id: str, spec_digest: str) -> None:
                """Raise."""
                message = "hook is broken"
                raise RuntimeError(message)

            def on_stage_start(self, stage: str) -> None:
                """Raise."""
                message = "hook is broken"
                raise RuntimeError(message)

        context = RunContext(
            run_id="r", spec_digest="d", output_directory=run_directory, hooks=(Broken(),)
        )
        assert DemoPipeline(context).execute() == "done"


class TestStageDeclaration:
    """The declared sequence is part of the pipeline's interface."""

    def test_stages_are_declared(self, context):
        """
        Declared rather than inferred.

        So the sequence can be rendered in a report and compared against what
        actually ran.
        """
        assert DemoPipeline(context).stages == ("first", "second", "third")

    def test_completed_stages_match_the_declaration_on_a_clean_run(self, context):
        """
        What ran is what was declared.

        A stage that runs without being declared would be invisible to any
        progress display driven by the declaration.
        """
        pipeline = DemoPipeline(context)
        pipeline.execute()
        assert pipeline.completed_stages == pipeline.stages
