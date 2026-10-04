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

from src.rade_qnet.core.runtime.errors import EngineError
from src.rade_qnet.core.spec.hardware import HardwareSpec
from src.rade_qnet.engines.torch.distributed import (
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
