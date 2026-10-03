"""
Tests for the hardware specification.

The design under test is that ``determinism`` is three-valued rather than a
boolean. Forcing deterministic kernels costs performance, and some operations
have no deterministic implementation at all, so "on or off" cannot express the
real choice. ``warn`` is the setting that says "prefer determinism, tell me
where it was not available" -- which is the honest default for research work.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.rade_xl.core.spec.hardware import HardwareSpec


class TestDefaults:
    """The spec is constructible bare, because other specs default it."""

    def test_it_constructs_with_no_arguments(self):
        """
        A nested spec reached through a ``default_factory`` must be bare.

        If this raised, a ``RunSpec`` could not be built without naming every
        hardware field -- the second of the ten diagnosed defects.
        """
        assert HardwareSpec() is not None

    def test_the_defaults_are_the_portable_choices(self):
        """
        Out of the box, a spec runs anywhere.

        Automatic device selection, full precision, no compilation, no
        distribution: nothing that requires particular hardware to be present.
        """
        spec = HardwareSpec()
        assert spec.device == "auto"
        assert spec.precision == "fp32"
        assert spec.compile_model is False
        assert spec.distributed == "none"
        assert spec.determinism == "off"


class TestDeterminism:
    """Determinism is a three-valued choice, not a flag."""

    @pytest.mark.parametrize("level", ["off", "warn", "strict"])
    def test_every_level_is_accepted(self, level):
        """All three are valid settings."""
        assert HardwareSpec(determinism=level).determinism == level

    def test_a_boolean_is_rejected(self):
        """
        ``determinism: true`` is not a valid setting.

        The rejection is the point: a boolean cannot distinguish "try, and
        warn me" from "fail if you cannot", and conflating them is how a run
        ends up claiming determinism it does not have.
        """
        with pytest.raises(ValidationError):
            HardwareSpec(determinism=True)

    def test_an_unknown_level_is_rejected(self):
        """A misspelled level fails at load, not at the first kernel."""
        with pytest.raises(ValidationError):
            HardwareSpec(determinism="deterministic")


class TestImpossibleCombinations:
    """Combinations that cannot work are rejected at load time."""

    def test_half_precision_on_cpu_is_rejected(self):
        """
        fp16 on a CPU is not merely slow.

        Most CPU kernels have no fp16 implementation, so the run fails part
        way through -- after the data build. Rejecting it at load turns hours
        into milliseconds.
        """
        with pytest.raises(ValidationError, match="fp16"):
            HardwareSpec(device="cpu", precision="fp16")

    def test_a_device_index_without_a_device_is_rejected(self):
        """
        Asking for device 3 of an unspecified device is meaningless.

        Silently ignoring the index would be worse: a user who asked for a
        particular GPU would get an arbitrary one and no warning.
        """
        with pytest.raises(ValidationError, match="device_index"):
            HardwareSpec(device="auto", device_index=3)

    def test_a_device_index_with_a_named_device_is_accepted(self):
        """The legitimate form of the same request."""
        assert HardwareSpec(device="cuda", device_index=3).device_index == 3

    def test_bfloat16_on_cpu_is_accepted(self):
        """
        bf16 has CPU support, unlike fp16.

        Rejecting both because they are "half precision" would block a
        configuration that works.
        """
        assert HardwareSpec(device="cpu", precision="bf16").precision == "bf16"


class TestStrictness:
    """Typos fail at load, and a spec cannot be mutated afterwards."""

    def test_an_unknown_key_is_rejected(self):
        """
        ``extra="forbid"`` in action.

        A misspelled key that was ignored would mean the run trains with a
        default, reports plausible numbers, and nobody finds out.
        """
        with pytest.raises(ValidationError):
            HardwareSpec(devise="cuda")

    def test_the_spec_is_frozen(self):
        """A stage must not be able to change the hardware mid-run."""
        spec = HardwareSpec()
        with pytest.raises(ValidationError):
            spec.device = "cuda"


class TestRoundTrip:
    """A spec survives serialisation exactly."""

    def test_dump_and_reload_is_exact(self):
        """
        The first of the ten diagnosed defects.

        A spec that does not round-trip cannot be recorded in a bundle,
        because the recorded configuration would not reproduce the run.
        """
        spec = HardwareSpec(
            device="cuda",
            device_index=1,
            precision="bf16",
            compile_model=True,
            distributed="ddp",
            determinism="strict",
            threads_per_worker=4,
        )
        assert HardwareSpec.model_validate(spec.model_dump()) == spec

    def test_json_round_trip_is_exact(self):
        """The form a bundle actually stores."""
        spec = HardwareSpec(device="mps", precision="fp32", determinism="warn")
        assert HardwareSpec.model_validate_json(spec.model_dump_json()) == spec
