"""
Tests for hardware resolution.

The design decision these tests pin down is *when to degrade and when to
refuse*. An absent accelerator is a property of the machine, not a mistake in
the job: the same configuration that runs on a GPU box should still run on a
laptop, slower, so a missing device warns and falls back to the CPU. An
unrecognised precision name is a mistake in the job, so it raises.

The subtle case is that degrading can recreate a combination the spec already
refused. ``HardwareSpec`` rejects fp16 on the CPU at load time, but requesting
``device='cuda', precision='fp16'`` on a machine with no CUDA lands on the CPU
still asking for fp16. If that were honoured the spec's guarantee would depend
on which machine ran the job, so the same rule is reapplied after resolution.

These tests cannot assume an accelerator is present, so they assert on
*consistency* -- that the resolved precision is legal for the resolved device,
whatever the machine -- rather than on a particular device being chosen.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.rade_qnet.core.runtime.errors import EngineError
from src.rade_qnet.core.spec.hardware import HardwareSpec
from src.rade_qnet.engines.torch.hardware.devices import (
    apply_thread_budget,
    autocast_for,
    available_accelerators,
    compile_if_requested,
    resolve_device,
    resolve_hardware,
)

CPU = torch.device("cpu")


class TestAvailableAccelerators:
    """What this machine can actually use."""

    def test_the_cpu_is_always_last_and_always_present(self):
        """
        So there is always a fallback and it is always the least preferred.

        An empty list would make ``accelerators[0]`` an index error on exactly
        the machines with no accelerator -- the ones that most need to work.
        """
        accelerators = available_accelerators()
        assert accelerators[-1] == "cpu"
        assert accelerators.count("cpu") == 1


class TestResolveDevice:
    """Choosing a device, and falling back rather than failing."""

    def test_auto_picks_the_best_available(self):
        """
        Which is the first entry, since the list is ordered best first.

        Asserted against the list rather than against ``cuda`` so the test
        means the same thing on a laptop and on a GPU box.
        """
        device = resolve_device(HardwareSpec(device="auto"))
        assert device.type == available_accelerators()[0]

    def test_an_explicit_cpu_request_is_honoured(self):
        """
        Even on a machine that has a GPU.

        Reproducing a result sometimes means reproducing the device it was
        produced on, so an explicit request is never second-guessed.
        """
        assert resolve_device(HardwareSpec(device="cpu")).type == "cpu"

    def test_an_unavailable_device_degrades_to_the_cpu(self):
        """
        Rather than failing, so one config runs everywhere.

        A GPU config that refuses to start on a laptop means the laptop needs
        its own config, and then the two drift apart.
        """
        device = resolve_device(HardwareSpec(device="cuda"))
        assert device.type in {"cuda", "cpu"}

    def test_the_fallback_is_logged_as_a_warning(self, caplog):
        """
        Because a silent fallback has no visible cause.

        A run that is a hundred times slower for no stated reason sends the
        investigation to the model first, and the device last.
        """
        if "cuda" in available_accelerators():
            pytest.skip("this machine has CUDA, so nothing degrades")
        with caplog.at_level("WARNING"):
            resolve_device(HardwareSpec(device="cuda"))
        assert "cpu" in caplog.text

    def test_a_device_index_is_carried_through(self):
        """
        So a multi-GPU box can pin a job to one card.

        Without it every job on the machine lands on device zero and they
        contend for its memory.
        """
        accelerator = available_accelerators()[0]
        if accelerator == "cpu":
            pytest.skip("indices are meaningless on the cpu")
        device = resolve_device(HardwareSpec(device=accelerator, device_index=0))
        assert device.index == 0


class TestAutocast:
    """Reduced precision, where it is available."""

    def test_full_precision_returns_a_null_context(self):
        """
        Because fp32 means "do not autocast".

        That is a different decision from "autocast to float32", and only the
        first is what the precision name asks for.
        """
        with autocast_for(CPU, "fp32"):
            assert torch.zeros(1).dtype == torch.float32

    def test_an_unknown_precision_raises(self):
        """
        Unlike an absent accelerator, which degrades.

        A precision name is written by hand into a config; a missing GPU is a
        fact about the machine. Only the first is a mistake to be fixed.
        """
        with pytest.raises(EngineError, match="precision"):
            autocast_for(CPU, "fp8")

    def test_an_unsupported_precision_degrades_on_that_device(self):
        """
        Falling back rather than failing.

        Autocast is not implemented on MPS, which is a fact about the machine
        rather than a mistake in the job -- the same reasoning as the device
        fallback.
        """
        with autocast_for(torch.device("mps"), "bf16"):
            pass


class TestResolveHardware:
    """The combination, and the rule that survives degradation."""

    def test_the_resolved_precision_is_legal_for_the_resolved_device(self):
        """
        The invariant that has to hold however the pairing arose.

        ``HardwareSpec`` refuses fp16 on the CPU at load time, but a cuda
        request on a CUDA-less machine lands on the CPU still asking for fp16.
        If that were honoured, the spec's guarantee would depend on which
        machine happened to run the job.
        """
        resolved = resolve_hardware(HardwareSpec(device="cuda", precision="fp16"))
        assert not (resolved.precision == "fp16" and resolved.device.type == "cpu")

    def test_a_scaler_is_built_only_for_fp16(self):
        """
        Because bf16 cannot underflow its gradients.

        It has the same exponent range as fp32, so a scaler for it is pure
        overhead on every backward pass.
        """
        resolved = resolve_hardware(HardwareSpec(device="cpu", precision="bf16"))
        assert resolved.scaler is None

    def test_degradation_is_reported_as_such(self):
        """
        So a report can say the run did not get the hardware it asked for.

        Without it, a throughput regression between two runs of the same
        config has no recorded explanation.
        """
        resolved = resolve_hardware(HardwareSpec(device="cuda", precision="fp32"))
        assert resolved.degraded == (resolved.device.type != "cuda")

    def test_the_description_is_loggable(self):
        """
        Naming both what was asked for and what was obtained.

        One without the other does not show that a fallback happened.
        """
        described = resolve_hardware(HardwareSpec(device="auto")).describe()
        assert {"device", "precision", "requested_device"} <= set(described)


class TestCompile:
    """Opt-in, and a no-op when it is off."""

    def test_compilation_is_off_by_default(self):
        """
        Because it is a surprise a user should opt into.

        Compilation costs a warm-up of tens of seconds, and the object it
        returns no longer looks like the module that went in.
        """
        model = nn.Linear(4, 1)
        assert compile_if_requested(model, spec=HardwareSpec()) is model

    def test_a_compiled_model_still_predicts(self):
        """
        Which is the only property worth asserting about the compiler.

        Compilation is a performance feature; if it changed the numbers it
        would be a correctness bug in torch, not something this framework can
        meaningfully check.
        """
        model = nn.Linear(4, 1)
        inputs = torch.ones(2, 4)
        expected = model(inputs)
        compiled = compile_if_requested(model, spec=HardwareSpec(compile_model=True))
        assert torch.allclose(compiled(inputs), expected, atol=1e-5)


class TestThreadBudget:
    """
    Applying the thread budget from the specification, in this process.

    The budget decides the order in which a reduction accumulates, so it
    decides the last few significant figures of every metric. Setting it by
    environment variable works only for a freshly spawned worker; the
    process that launched it has already imported Torch. Leaving it there
    meant the same job scored differently sequentially than pooled, which is
    defect 13 and the reason this is a specification field.
    """

    def test_a_pinned_budget_is_applied(self):
        """The requested count is in effect afterwards."""
        previous = torch.get_num_threads()
        try:
            assert apply_thread_budget(HardwareSpec(threads_per_worker=1)) == 1
            assert torch.get_num_threads() == 1
        finally:
            torch.set_num_threads(previous)

    def test_no_budget_leaves_the_machine_default_alone(self):
        """
        ``None`` means "whatever this machine chose".

        Not "one thread": defaulting to a single thread would make every
        unconfigured run slow in exchange for a reproducibility guarantee
        the user did not ask for.
        """
        previous = torch.get_num_threads()

        assert apply_thread_budget(HardwareSpec()) == previous
        assert torch.get_num_threads() == previous

    def test_applying_twice_is_safe(self):
        """
        Called once per job, so repetition has to be harmless.

        Torch is resizing a pool here rather than creating one.
        """
        previous = torch.get_num_threads()
        try:
            apply_thread_budget(HardwareSpec(threads_per_worker=1))
            assert apply_thread_budget(HardwareSpec(threads_per_worker=1)) == 1
        finally:
            torch.set_num_threads(previous)

    def test_resolving_hardware_applies_the_budget(self):
        """
        The budget is applied by the same call that resolves the device.

        Nothing else in a fit is guaranteed to run in every path, so hanging
        it off hardware resolution is what makes "the spec says one thread"
        true wherever the job lands.
        """
        previous = torch.get_num_threads()
        try:
            resolve_hardware(HardwareSpec(device="cpu", threads_per_worker=1))
            assert torch.get_num_threads() == 1
        finally:
            torch.set_num_threads(previous)
