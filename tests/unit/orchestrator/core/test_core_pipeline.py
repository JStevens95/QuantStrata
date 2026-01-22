from __future__ import annotations

from dataclasses import dataclass

import logging
from pathlib import Path

import pytest

from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline, PipelineRunner
from src.orchestrator.core.step import Step
from src.orchestrator.core.errors import StepError


@dataclass(frozen=True, slots=True)
class AppendStep(Step):
    """
    Simple step that appends its name into ctx.state["trace"].

    This is intentionally tiny so we can test runner selection logic
    without touching any domain modules.
    """

    def run(self, ctx: Context) -> Context:
        trace = ctx.state.setdefault("trace", [])
        trace.append(self.name)
        return ctx


@dataclass(frozen=True, slots=True)
class FailingStep(Step):
    """Step that always raises (used to test safe_run wrapping)."""

    def run(self, ctx: Context) -> Context:
        raise ValueError("boom")


def _ctx() -> Context:
    # Minimal context for unit tests.
    logger = logging.getLogger("test.orchestrator")
    logger.handlers = []  # ensure clean logger in pytest
    logger.addHandler(logging.NullHandler())

    # artifact_store/cfg are not used in these tests, so stub objects are fine.
    return Context(run_id="run_test", cfg=object(), logger=logger, artifact_store=object())


def test_pipeline_runner_runs_all_steps_in_order() -> None:
    pipeline = Pipeline(
        name="demo",
        steps=[AppendStep("a"), AppendStep("b"), AppendStep("c")],
    )
    ctx = PipelineRunner().run(pipeline, _ctx())
    assert ctx.state["trace"] == ["a", "b", "c"]


def test_pipeline_runner_only_filter() -> None:
    pipeline = Pipeline(
        name="demo",
        steps=[AppendStep("a"), AppendStep("b"), AppendStep("c")],
    )
    runner = PipelineRunner(only={"b", "c"})
    ctx = runner.run(pipeline, _ctx())
    assert ctx.state["trace"] == ["b", "c"]


def test_pipeline_runner_skip_filter() -> None:
    pipeline = Pipeline(
        name="demo",
        steps=[AppendStep("a"), AppendStep("b"), AppendStep("c")],
    )
    runner = PipelineRunner(skip={"b"})
    ctx = runner.run(pipeline, _ctx())
    assert ctx.state["trace"] == ["a", "c"]


def test_pipeline_runner_resume_from() -> None:
    pipeline = Pipeline(
        name="demo",
        steps=[AppendStep("a"), AppendStep("b"), AppendStep("c")],
    )
    runner = PipelineRunner(resume_from="b")
    ctx = runner.run(pipeline, _ctx())
    assert ctx.state["trace"] == ["b", "c"]


def test_step_safe_run_wraps_exceptions_as_step_error() -> None:
    step = FailingStep("explode")
    with pytest.raises(StepError) as err:
        step.safe_run(_ctx())
    # Helpful signal: StepError includes step name.
    assert "explode" in str(err.value)