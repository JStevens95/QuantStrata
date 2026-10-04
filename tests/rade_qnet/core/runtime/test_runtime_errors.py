"""
Tests for the error hierarchy.

The property that matters most here is unusual enough to be worth naming: that
``SpecError`` is a ``ValueError``. That dual inheritance is what lets pydantic
collect a cross-field validator's error and attach a field path to it. Without
it, a field-level failure and a cross-field failure report through two
different mechanisms with two different shapes, and a caller cannot handle both.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.runtime.errors import (
    BundleError,
    CapabilityError,
    ComponentError,
    ContractError,
    RadeQNetError,
    SpecError,
    StageError,
)


class TestHierarchy:
    """Every error is catchable as one framework base."""

    @pytest.mark.parametrize(
        "error_type",
        [SpecError, ContractError, CapabilityError, ComponentError, BundleError],
    )
    def test_every_error_derives_from_the_framework_base(self, error_type):
        """A caller can catch anything the framework raises with one except."""
        assert issubclass(error_type, RadeQNetError)

    def test_stage_error_derives_from_the_framework_base(self):
        """StageError is constructed differently but belongs to the hierarchy."""
        assert issubclass(StageError, RadeQNetError)

    def test_spec_error_is_also_a_value_error(self):
        """
        Assert the dual inheritance pydantic depends on.

        Pydantic only collects ``ValueError`` from a validator. If this
        inheritance is removed, a cross-field validator's error escapes raw
        and loses the field path that a field-level error keeps -- so the two
        failure modes stop reporting the same way.
        """
        assert issubclass(SpecError, ValueError)

    def test_contract_error_is_not_a_value_error(self):
        """
        A contract failure is a framework bug, not bad user input.

        Keeping it off ``ValueError`` means a pydantic validator cannot
        swallow it, which is correct: a contract violation should surface as
        itself rather than as a field-validation message.
        """
        assert not issubclass(ContractError, ValueError)


class TestStageError:
    """StageError attributes a failure to the stage that produced it."""

    def test_message_names_the_stage_and_the_cause(self):
        """The message identifies where to look, not just what broke."""
        error = StageError("fit", KeyError("features"))
        assert "stage 'fit' failed" in str(error)
        assert "KeyError" in str(error)
        assert "features" in str(error)

    def test_message_includes_the_run_when_one_is_known(self):
        """A job-set failure must say which run it came from."""
        error = StageError("build_data", ValueError("bad"), run_id="r-007")
        assert "r-007" in str(error)

    def test_message_omits_the_run_when_none_is_known(self):
        """A run-less failure does not render an empty prefix."""
        error = StageError("build_data", ValueError("bad"))
        assert "run" not in str(error).split("stage")[0]

    def test_the_cause_is_retained(self):
        """
        The original exception stays reachable.

        The wrapper adds attribution; it must not discard the detail needed to
        diagnose the underlying fault.
        """
        cause = KeyError("features")
        error = StageError("fit", cause)
        assert error.cause is cause
        assert error.stage == "fit"


class TestMessages:
    """Errors carry their message, since every message is a diagnosis."""

    @pytest.mark.parametrize(
        "error_type",
        [SpecError, ContractError, CapabilityError, ComponentError, BundleError],
    )
    def test_message_round_trips(self, error_type):
        """str() returns what was passed, with no decoration."""
        assert str(error_type("something specific went wrong")) == "something specific went wrong"
