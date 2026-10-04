"""
Tests for the pipeline hook base.

The property being pinned down is that :class:`PipelineHook` is usable by
*partial* implementation. Every method is a no-op, so a subclass overrides only
what it cares about and keeps working when the framework adds a new hook point.
If any method were abstract, adding one would break every hook in existence --
including hooks written outside this repository.
"""

from __future__ import annotations

import inspect

import pytest

from src.rade_qnet.core.runtime.hooks import PipelineHook

#: Every hook point, with arguments that satisfy its signature. Kept as data so
#: a newly added hook point is a one-line change here rather than a new test.
HOOK_CALLS = (
    ("on_run_start", ("r-1", "digest"), {}),
    ("on_run_end", ("r-1",), {"succeeded": True}),
    ("on_stage_start", ("fit",), {}),
    ("on_stage_end", ("fit",), {"seconds": 1.5}),
    ("on_stage_error", ("fit", ValueError("boom")), {}),
    ("on_epoch_end", (0, {"train_loss": 0.5}), {}),
    ("on_metrics", ("evaluate", {"mae": 0.1}), {}),
    ("on_artifact", ("summary", "/tmp/summary.md"), {}),
)


class TestDefaultImplementation:
    """The base is a working "observe nothing" implementation."""

    @pytest.mark.parametrize(("method", "args", "kwargs"), HOOK_CALLS)
    def test_every_hook_point_is_callable_and_returns_none(self, method, args, kwargs):
        """
        The base can be used directly, and every hook point is a no-op.

        Returning ``None`` is part of the contract: a hook observes and must
        not be able to alter what the pipeline does, because then two runs
        with identical specs would differ according to who was watching.
        """
        assert getattr(PipelineHook(), method)(*args, **kwargs) is None

    @pytest.mark.parametrize(("method", "args", "kwargs"), HOOK_CALLS)
    def test_no_hook_point_is_abstract(self, method, args, kwargs):
        """
        A partial implementation is valid.

        If any method were abstract, adding a hook point would be a breaking
        change for every existing hook.
        """
        del args, kwargs
        assert not getattr(getattr(PipelineHook, method), "__isabstractmethod__", False)

    def test_the_base_is_instantiable(self):
        """Used as a null object wherever a hook is optional."""
        assert isinstance(PipelineHook(), PipelineHook)


class TestPartialSubclass:
    """A subclass overriding one method inherits the rest."""

    def test_overriding_one_method_leaves_the_others_working(self):
        """The property that keeps hooks cheap to write."""

        class OnlyStages(PipelineHook):
            """Observes stage starts and nothing else."""

            def __init__(self) -> None:
                self.stages: list[str] = []

            def on_stage_start(self, stage: str) -> None:
                """Record the stage."""
                self.stages.append(stage)

        hook = OnlyStages()
        hook.on_stage_start("fit")
        # The inherited methods still work, so a pipeline can call all of them
        # without checking which the subclass implemented.
        hook.on_run_start("r-1", "digest")
        hook.on_epoch_end(0, {"train_loss": 0.1})
        hook.on_artifact("summary", "/tmp/x")
        assert hook.stages == ["fit"]


class TestSignatures:
    """Keyword-only arguments guard against positional misreading."""

    @pytest.mark.parametrize(
        ("method", "parameter"),
        [("on_run_end", "succeeded"), ("on_stage_end", "seconds")],
    )
    def test_ambiguous_arguments_are_keyword_only(self, method, parameter):
        """
        A bare ``True`` or ``1.5`` at a call site says nothing.

        Making these keyword-only means a hook implementation cannot silently
        bind them in the wrong order, and a reader of the call site can tell
        what the value means.
        """
        signature = inspect.signature(getattr(PipelineHook, method))
        assert signature.parameters[parameter].kind is inspect.Parameter.KEYWORD_ONLY

    def test_the_error_hook_receives_the_exception_itself(self):
        """
        Not a formatted string.

        A hook that wants to classify failures needs the exception's type and
        its traceback, neither of which survives being rendered to text.
        """
        signature = inspect.signature(PipelineHook.on_stage_error)
        assert signature.parameters["error"].annotation == "BaseException"
