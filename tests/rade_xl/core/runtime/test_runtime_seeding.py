"""
Tests for seeding and determinism.

Two designs are under test here, and both were chosen over an obvious
alternative.

``determinism`` is three-valued (``off``, ``warn``, ``strict``) rather than a
boolean, because forcing deterministic kernels has a real performance cost and
some operations have no deterministic implementation at all. A boolean hides
that trade-off; the implementation this replaces set it globally inside a
swallowed ``try``/``except``, so a run could silently be non-deterministic
while claiming otherwise.

Per-job seeds are *derived by hashing*, not by adding an index. That gives
order independence -- a single failed job re-run alone reproduces -- and avoids
the correlation that consecutive integer seeds can produce.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.rade_xl.core.runtime.errors import SpecError
from src.rade_xl.core.runtime.seeding import (
    derive_seed,
    register_seeder,
    registered_seeders,
    seed_everything,
    unregister_seeder,
)


@pytest.fixture(autouse=True)
def _clear_seeders():
    """
    Remove any seeder a test registered.

    The registry is module-level, so a leftover seeder would be invoked by
    every later test -- and the failure would appear in whichever test ran
    next, not the one at fault.
    """
    before = set(registered_seeders())
    yield
    for name in set(registered_seeders()) - before:
        unregister_seeder(name)


class TestSeedEverything:
    """Seeding is reproducible and reports what it applied."""

    def test_the_applied_seed_is_returned(self):
        """
        The caller learns what was actually used.

        Recorded in the training result, so a run can be repeated without
        re-deriving the value.
        """
        assert seed_everything(1234) == 1234

    def test_seeding_makes_the_standard_library_reproducible(self):
        """Two runs with one seed draw the same numbers."""
        seed_everything(7)
        first = [random.random() for _ in range(5)]
        seed_everything(7)
        assert [random.random() for _ in range(5)] == first

    def test_seeding_makes_numpy_reproducible(self):
        """The legacy global numpy state is seeded too."""
        seed_everything(7)
        first = np.random.rand(5)  # noqa: NPY002 - legacy API is what is seeded.
        seed_everything(7)
        assert np.allclose(np.random.rand(5), first)  # noqa: NPY002

    def test_different_seeds_give_different_draws(self):
        """Confirms the seed is actually in use."""
        seed_everything(1)
        first = [random.random() for _ in range(5)]
        seed_everything(2)
        assert [random.random() for _ in range(5)] != first


class TestSeederRegistry:
    """Engines register their own seeding, because core cannot import them."""

    def test_a_registered_seeder_is_invoked(self):
        """
        The indirection that lets ``core`` seed PyTorch without importing it.

        ``core`` may not import a training library, so an engine registers a
        callback at import and ``core`` calls it.
        """
        seen: list[tuple[int, str]] = []
        register_seeder("probe", lambda seed, determinism: seen.append((seed, determinism)))
        seed_everything(11, determinism="warn")
        assert seen == [(11, "warn")]

    def test_registering_twice_is_rejected(self):
        """
        A duplicate name is an error, not a silent replacement.

        Two engines claiming one name means one of them is not being seeded,
        and a silent overwrite makes that undetectable.
        """
        register_seeder("probe", lambda seed, determinism: None)
        with pytest.raises(Exception, match="probe"):
            register_seeder("probe", lambda seed, determinism: None)

    def test_replacing_is_possible_when_asked_for_explicitly(self):
        """A test double can take over, but has to say so."""
        register_seeder("probe", lambda seed, determinism: None)
        calls: list[int] = []
        register_seeder("probe", lambda seed, determinism: calls.append(seed), replace=True)
        seed_everything(3)
        assert calls == [3]

    def test_unregistering_removes_the_seeder(self):
        """Needed so a test can clean up after itself."""
        register_seeder("probe", lambda seed, determinism: None)
        unregister_seeder("probe")
        assert "probe" not in registered_seeders()


class TestDeterminismLevels:
    """A failing seeder is tolerated or fatal according to the level chosen."""

    def test_a_failing_seeder_is_tolerated_when_determinism_is_off(self):
        """
        Best effort means best effort.

        With determinism off the caller has not asked for a guarantee, so a
        seeder that cannot comply should not end the run.
        """

        def explode(seed, determinism):
            message = "no deterministic kernel for this operation"
            raise RuntimeError(message)

        register_seeder("explosive", explode)
        assert seed_everything(5, determinism="off") == 5

    def test_a_failing_seeder_is_fatal_when_determinism_is_strict(self):
        """
        Strict means the guarantee is load-bearing.

        This is the defect being fixed: the previous implementation forced
        determinism inside a swallowed ``try``/``except``, so a run could
        report itself as deterministic when it was not. Under ``strict`` the
        failure must surface.
        """

        def explode(seed, determinism):
            message = "no deterministic kernel for this operation"
            raise RuntimeError(message)

        register_seeder("explosive", explode)
        with pytest.raises(SpecError, match="explosive"):
            seed_everything(5, determinism="strict")

    @pytest.mark.parametrize("level", ["off", "warn", "strict"])
    def test_every_level_is_accepted(self, level):
        """All three levels are valid, and none of them is a boolean."""
        assert seed_everything(5, determinism=level) == 5


class TestDeriveSeed:
    """Per-job seeds are derived by hashing, not by adding an index."""

    def test_the_same_labels_derive_the_same_seed(self):
        """
        The reproducibility property.

        Re-running one failed job alone must reproduce the result it would
        have had inside the full job set.
        """
        assert derive_seed(100, "EURUSD") == derive_seed(100, "EURUSD")

    def test_different_labels_derive_different_seeds(self):
        """Two job-set members must not share a seed."""
        assert derive_seed(100, "EURUSD") != derive_seed(100, "USDJPY")

    def test_different_base_seeds_derive_different_seeds(self):
        """Changing the run's seed changes every member's seed."""
        assert derive_seed(100, "EURUSD") != derive_seed(101, "EURUSD")

    def test_derivation_is_order_independent(self):
        """
        A job's seed does not depend on its position in the job list.

        Which is exactly what ``base + index`` would not give: reordering the
        jobs, or re-running a subset, would change every seed.
        """
        first = [derive_seed(1, job) for job in ("a", "b", "c")]
        second = [derive_seed(1, job) for job in ("c", "b", "a")]
        assert first == list(reversed(second))

    def test_multiple_labels_compose(self):
        """A seed can be derived per job and per fold without collision."""
        assert derive_seed(1, "EURUSD", "fold-0") != derive_seed(1, "EURUSD", "fold-1")

    def test_labels_are_not_merely_concatenated(self):
        """
        Label boundaries are significant.

        If labels were joined without a separator, ``("ab", "c")`` and
        ``("a", "bc")`` would derive the same seed -- so two different jobs
        could silently share one.
        """
        assert derive_seed(1, "ab", "c") != derive_seed(1, "a", "bc")

    def test_the_derived_seed_is_in_range(self):
        """
        The result fits the range generators accept.

        Numpy rejects a seed outside ``[0, 2**32)``, so a raw digest would
        fail at the point of use rather than here.
        """
        for job in ("EURUSD", "USDJPY", "a very long job identifier indeed"):
            seed = derive_seed(12345, job)
            assert 0 <= seed < 2**32
