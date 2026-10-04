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

from src.rade_qnet.core.runtime.seeding import (
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
