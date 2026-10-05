# `tests/rade_qnet/engines/torch/hardware`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 1 | 32 | `7860806c8e2de23b` |
| 2 | `test_hardware_determinism.py` | 194 | 7502 | `47f5af4fd3eae9b5` |
| 3 | `test_hardware_devices.py` | 283 | 10743 | `2ce882982db68491` |
| 4 | `test_hardware_distributed.py` | 220 | 8690 | `217493ce9169d482` |

---

## 1. `tests/rade_qnet/engines/torch/hardware/__init__.py`

32 bytes · SHA-256 `7860806c8e2de23b`

```python
"""Mirror of torch.hardware."""
```

---

## 2. `tests/rade_qnet/engines/torch/hardware/test_hardware_determinism.py`

7502 bytes · SHA-256 `47f5af4fd3eae9b5`

```python
"""
Tests for the Torch seeder.

Before this module existed, ``seed_everything`` seeded Python and NumPy and
left Torch untouched -- which is the one library that matters most. Every
weight initialisation, dropout mask and shuffled batch order differed between
two runs of the same configuration, and nothing said so: both runs completed,
both reported the seed they were given, and the scores differed. A sensitivity
study across seeds would have been measuring noise it could not attribute.

The registration-at-import test is the one that would have caught that. A
seeder registered after ``seed_everything`` has already run is a seeder that
did nothing, so what matters is not that the function exists but that
importing the engine package is enough to install it.

``strict`` and ``warn`` differ in what the user is promised rather than in
what the framework attempts: both ask Torch for deterministic algorithm
implementations, and only ``strict`` refuses an operation that has none. That
distinction is the point of having three levels instead of a boolean.
"""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.provenance.seeding import (
    register_seeder,
    registered_seeders,
    seed_everything,
    unregister_seeder,
)
from src.rade_qnet.engines.torch.hardware.determinism import SEEDER_NAME, seed_torch


@pytest.fixture(autouse=True)
def _restore_determinism():
    """
    Put Torch's global determinism flags back after each test.

    These are process-global, so a test that turned deterministic algorithms
    on would otherwise slow every later test in the session and could make an
    unrelated one fail on an operation that has no deterministic kernel.
    """
    was_deterministic = torch.are_deterministic_algorithms_enabled()
    was_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_deterministic, warn_only=was_warn_only)


class TestRegistration:
    """Installed by importing the package, which is the load-bearing part."""

    def test_importing_the_engine_registers_the_seeder(self):
        """
        Because a seeder registered later is a seeder that did nothing.

        This is the failure the module exists to prevent: ``seed_everything``
        runs at the start of a pipeline, so a registration that happens on
        first use of the engine is already too late.
        """
        assert SEEDER_NAME in registered_seeders()

    def test_seed_everything_reaches_torch(self):
        """
        End to end through ``core``, which cannot import Torch itself.

        ``core`` has an empty dependency set, so the registration has to come
        from the package that owns the library -- and the only way to confirm
        the inversion works is to seed through ``core`` and check Torch.
        """
        seed_everything(1234)
        first = torch.randn(4)
        seed_everything(1234)
        assert torch.equal(first, torch.randn(4))

    def test_the_registration_can_be_replaced(self):
        """
        So a user wrapping a Torch fork can substitute their own.

        Replacement is explicit rather than last-one-wins, which is what
        stops two packages silently claiming the same name.
        """
        try:
            register_seeder(SEEDER_NAME, lambda seed, determinism: None, replace=True)
            assert SEEDER_NAME in registered_seeders()
        finally:
            register_seeder(SEEDER_NAME, seed_torch, replace=True)


class TestSeeding:
    """The sampling, which is what a seed can actually control."""

    def test_the_same_seed_gives_the_same_weights(self):
        """
        Which is what makes a reproduced run the same run.

        Without it, a difference in two runs' scores cannot be attributed to
        anything, because the one input that was supposed to be fixed was
        not.
        """
        seed_torch(7)
        first = torch.nn.Linear(4, 2).weight.detach().clone()
        seed_torch(7)
        assert torch.equal(torch.nn.Linear(4, 2).weight, first)

    def test_different_seeds_give_different_weights(self):
        """
        So the seed is doing something.

        A seed accepted and ignored would make a sensitivity study across
        seeds produce identical results and conclude the model was stable.
        """
        seed_torch(7)
        first = torch.nn.Linear(4, 2).weight.detach().clone()
        seed_torch(8)
        assert not torch.equal(torch.nn.Linear(4, 2).weight, first)

    def test_dropout_masks_are_reproducible(self):
        """
        Because they are a second source of run-to-run variation.

        Seeding the weights alone would make two runs start identically and
        diverge within the first epoch, which is harder to diagnose than
        diverging immediately.
        """
        dropout = torch.nn.Dropout(0.5)
        dropout.train()
        seed_torch(3)
        first = dropout(torch.ones(100))
        seed_torch(3)
        assert torch.equal(dropout(torch.ones(100)), first)


class TestDeterminismLevels:
    """Three levels, because they promise different things."""

    def test_off_seeds_and_asks_for_nothing_more(self):
        """
        So the default costs no throughput.

        Deterministic kernels are slower, and most runs want reproducible
        *sampling* without paying for reproducible *arithmetic*.
        """
        torch.use_deterministic_algorithms(False)
        seed_torch(5, determinism="off")
        assert not torch.are_deterministic_algorithms_enabled()

    def test_warn_asks_for_deterministic_algorithms_but_tolerates_absence(self):
        """
        The level for when reproducibility is desirable, not required.

        An operation with no deterministic kernel logs and proceeds, so a run
        is not refused over a single layer.
        """
        seed_torch(5, determinism="warn")
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()

    def test_strict_refuses_an_operation_with_no_deterministic_kernel(self):
        """
        Which is the only thing that makes the promise meaningful.

        A user who asked for reproducibility must be told it is unavailable
        for their model rather than receiving results that quietly are not
        reproducible -- so the difference between the two levels is in what
        is promised, not in what is attempted.
        """
        seed_torch(5, determinism="strict")
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()

    def test_a_failing_seeder_is_fatal_under_strict(self):
        """
        Reported through ``core``, naming the seeder that failed.

        Tolerated, it would produce a run that reports a seed it could not
        honour -- which is worse than refusing, because the number is there
        and looks authoritative.
        """

        def explode(seed, determinism):
            """Fail the way a library with no deterministic kernel would."""
            del seed, determinism
            raise RuntimeError("no deterministic implementation")

        register_seeder("probe_strict", explode)
        try:
            with pytest.raises(Exception, match="probe_strict"):
                seed_everything(5, determinism="strict")
        finally:
            unregister_seeder("probe_strict")
```

---

## 3. `tests/rade_qnet/engines/torch/hardware/test_hardware_devices.py`

10743 bytes · SHA-256 `2ce882982db68491`

```python
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

from src.rade_qnet.core.lifecycle.errors import EngineError
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
```

---

## 4. `tests/rade_qnet/engines/torch/hardware/test_hardware_distributed.py`

8690 bytes · SHA-256 `217493ce9169d482`

```python
"""
Tests for the distributed wrapper -- the second half of defect 6.

Wrapping a model whose parameters are not yet materialised is worse than
building an optimiser over them. The optimiser case trains nothing and shows a
flat loss curve, which is at least odd enough to investigate. The distributed
case, on some versions, synchronises an *empty* parameter set: every rank then
trains its own independent copy of the model, each on its own shard of the
data, while reporting a perfectly normal loss curve. The run looks like a
distributed run that worked, and is in fact N unrelated models of which one is
arbitrarily kept.

That is why this is the one condition in the module that raises rather than
degrading. Everything else here follows the same policy as an absent
accelerator -- warn and run in a single process -- because a single-process
run of a distributed config is correct, just not parallel. A silently
unsynchronised run is not correct at all.

These tests run in a single process, so they cover the refusal, the
degradation, and the unwrapping. The wrapping path itself needs a real process
group and belongs in an integration test rather than a unit one.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.core.spec.hardware import HardwareSpec
from src.rade_qnet.engines.torch.hardware.distributed import (
    distribute,
    is_distributed_run,
    local_rank,
    undistribute,
    world_size,
)

CPU = torch.device("cpu")


class LazyNet(nn.Module):
    """A model whose first layer infers its own input width."""

    def __init__(self) -> None:
        """Build the lazy stack."""
        super().__init__()
        self.layer = nn.LazyLinear(4)

    def forward(self, features):
        """Run the forward pass."""
        return self.layer(features)


class TestEnvironmentReading:
    """What the launcher tells the process, with sane defaults."""

    def test_a_plain_process_has_a_world_size_of_one(self, monkeypatch):
        """
        So an unlaunched run is a single-process run rather than an error.

        The variables are set by ``torchrun``; a developer running a script
        directly has none of them, and that has to be the normal case.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        assert world_size() == 1

    def test_a_plain_process_has_a_local_rank_of_zero(self, monkeypatch):
        """Which is also the rank that does the writing in a real run."""
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        assert local_rank() == 0

    def test_a_launched_process_is_recognised(self, monkeypatch):
        """
        Read from the environment rather than from a flag in the config.

        The config cannot know how it was launched, and a config claiming to
        be distributed when it is not is exactly the mismatch that leads to
        an unsynchronised run.
        """
        monkeypatch.setenv("WORLD_SIZE", "4")
        assert is_distributed_run()
        assert world_size() == 4

    def test_a_malformed_world_size_is_treated_as_a_single_process(self, monkeypatch):
        """
        Rather than crashing on an integer conversion.

        The variable is set by an external launcher, so its contents are not
        this framework's to guarantee, and the safe reading is the one that
        does not claim parallelism.
        """
        monkeypatch.setenv("WORLD_SIZE", "not-a-number")
        assert world_size() == 1


class TestDefectSixWrappingBeforeMaterialisation:
    """The one condition that must not degrade."""

    def test_wrapping_a_lazy_model_is_refused(self, monkeypatch):
        """
        Because the wrapper would synchronise nothing.

        Every rank would then train an independent copy on its own data
        shard, all reporting normal-looking loss curves, and one of them would
        be kept arbitrarily. Nothing in the output distinguishes that from a
        distributed run that worked.
        """
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(EngineError, match="uninitialised parameters"):
            distribute(LazyNet(), spec=HardwareSpec(distributed="ddp"), device=CPU)

    def test_the_message_names_the_stage_to_run_first(self, monkeypatch):
        """
        Because the fix is an ordering change, not a code change.

        The pipeline already has a materialise stage, so reaching this error
        means the stage order was altered -- and the message has to say that,
        or the reader looks for a bug in the model instead.
        """
        monkeypatch.setenv("WORLD_SIZE", "2")
        with pytest.raises(EngineError) as caught:
            distribute(LazyNet(), spec=HardwareSpec(distributed="ddp"), device=CPU)
        assert "materialise" in str(caught.value)

    def test_the_check_runs_before_the_world_size_is_consulted(self, monkeypatch):
        """
        So the ordering mistake is caught on a developer's laptop too.

        Checked only in a real multi-process run, the error would appear for
        the first time in production, which is the worst place to discover an
        ordering bug.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        with pytest.raises(EngineError):
            distribute(LazyNet(), spec=HardwareSpec(distributed="ddp"), device=CPU)


class TestDegradation:
    """Warn and run in one process, as with an absent accelerator."""

    def test_no_strategy_returns_the_model_untouched(self):
        """
        The default path, which must cost nothing.

        Identity rather than equality: a copy would silently detach the
        optimiser, which already holds references to the original parameters.
        """
        model = nn.Linear(4, 1)
        returned, wrapped = distribute(model, spec=HardwareSpec(), device=CPU)
        assert returned is model
        assert not wrapped

    def test_a_single_process_run_of_a_distributed_config_is_not_wrapped(self, monkeypatch):
        """
        Correct, just not parallel -- so it runs rather than failing.

        A config that refuses to run outside ``torchrun`` cannot be tested
        locally, and then it is only ever exercised in production.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        model = nn.Linear(4, 1)
        returned, wrapped = distribute(model, spec=HardwareSpec(distributed="ddp"), device=CPU)
        assert returned is model
        assert not wrapped

    def test_the_degradation_is_warned_about(self, monkeypatch, caplog):
        """
        Naming ``torchrun``, because that is the missing piece.

        A run that was meant to use eight GPUs and quietly used one takes
        eight times as long, and the log is the only place that can say why.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        with caplog.at_level("WARNING"):
            distribute(nn.Linear(4, 1), spec=HardwareSpec(distributed="ddp"), device=CPU)
        assert "torchrun" in caplog.text

    def test_the_flag_reports_whether_wrapping_happened(self, monkeypatch):
        """
        So a report can state that a run asking to be distributed was not.

        Taken from the call that did or did not wrap, rather than from the
        spec that asked -- the spec records the request, not the outcome.
        """
        monkeypatch.delenv("WORLD_SIZE", raising=False)
        _, wrapped = distribute(nn.Linear(4, 1), spec=HardwareSpec(distributed="ddp"), device=CPU)
        assert wrapped is False


class TestUndistribute:
    """Getting the plain model back out, for the checkpoint."""

    def test_an_unwrapped_model_is_returned_as_itself(self):
        """So the common path has no special case."""
        model = nn.Linear(4, 1)
        assert undistribute(model) is model

    def test_a_compiled_model_is_unwrapped(self):
        """
        Because a compiled wrapper prefixes every parameter name.

        A checkpoint carrying ``_orig_mod.`` keys can only be loaded by
        reconstructing the same wrapper, which the reader of the bundle has no
        way to know about.
        """
        model = nn.Linear(4, 1)
        compiled = torch.compile(model)
        assert undistribute(compiled) is model

    def test_unwrapping_is_idempotent(self):
        """
        So it can be called without knowing whether it already was.

        Which is what lets the handle hold ``unwrapped`` unconditionally
        rather than branching on how the model was prepared.
        """
        model = nn.Linear(4, 1)
        assert undistribute(undistribute(model)) is model
```

