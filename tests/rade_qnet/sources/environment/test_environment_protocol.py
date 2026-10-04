"""
Tests for the environment protocol.

Two things are worth testing about a protocol, and only two. That an object
with the right members satisfies it, so a third-party environment needs no
import of ours; and that an object missing one does not, so the framework can
name the fault instead of failing on the first ``step``.

The third group is about :class:`StepOutcome`, and it is there because of the
specific defect the class exists to prevent: ``terminated`` and ``truncated``
are two booleans sitting next to each other, and transposing them produces an
agent that treats a time-limited episode as a genuine failure. The tests
assert that they are named, independent, and both reachable through
``done`` -- which is the only place the two are deliberately conflated.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.rade_qnet.core.contract.signature import SpaceSpec
from src.rade_qnet.sources.environment import Environment, StepOutcome

OBSERVATION = SpaceSpec(kind="box", shape=(2,), dtype="float32")
ACTION = SpaceSpec(kind="discrete", n=3)


class Complete:
    """An environment with every member the protocol asks for."""

    observation_space = OBSERVATION
    action_space = ACTION

    def reset(self, *, seed: int | None = None):
        """
        Start an episode.

        Parameters
        ----------
        seed
            Ignored by this stub.

        Returns
        -------
        numpy.ndarray
            A zero observation.
        """
        del seed
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        """
        Take one step.

        Parameters
        ----------
        action
            Ignored by this stub.

        Returns
        -------
        StepOutcome
            A constant outcome.
        """
        del action
        return StepOutcome(np.ones(2, dtype=np.float32), reward=1.0)


class TestWhatSatisfiesTheProtocol:
    """An environment is anything with the four members."""

    def test_a_plain_object_with_the_members_satisfies_it(self):
        """No base class and no import of ours is needed."""
        assert isinstance(Complete(), Environment)

    def test_an_object_missing_a_member_does_not(self):
        """The whole point: the fault is detectable before anything is stepped."""

        class MissingStep:
            observation_space = OBSERVATION
            action_space = ACTION

            def reset(self, *, seed=None):
                """Return nothing of interest."""
                del seed

        assert not isinstance(MissingStep(), Environment)

    def test_the_spaces_are_readable_without_resetting(self):
        """
        Construction alone is enough to describe an environment.

        This is what lets a policy be built before anything is stepped, and
        rebuilt later from a bundle with no environment at all.
        """
        environment = Complete()
        assert environment.observation_space.shape == (2,)
        assert environment.action_space.n == 3


class TestStepOutcome:
    """The two episode-end flags, kept apart on purpose."""

    def test_an_ongoing_step_is_neither_terminated_nor_truncated(self):
        """The common case needs no arguments beyond the reward."""
        outcome = StepOutcome(np.zeros(2), reward=0.5)
        assert not outcome.terminated
        assert not outcome.truncated
        assert not outcome.done

    def test_terminated_and_truncated_are_independent(self):
        """
        Setting one does not set the other.

        A test of something obvious, kept because the defect it guards is a
        future refactor collapsing the two into one field -- at which point
        this fails rather than the bias appearing silently in a value
        estimate.
        """
        ended = StepOutcome(np.zeros(2), reward=0.0, terminated=True)
        cut_short = StepOutcome(np.zeros(2), reward=0.0, truncated=True)

        assert ended.terminated and not ended.truncated
        assert cut_short.truncated and not cut_short.terminated

    def test_done_is_true_for_either_reason(self):
        """A collector only needs to know when to reset, so it reads one flag."""
        assert StepOutcome(np.zeros(2), reward=0.0, terminated=True).done
        assert StepOutcome(np.zeros(2), reward=0.0, truncated=True).done

    def test_the_outcome_cannot_be_edited_after_the_fact(self):
        """
        Frozen, so a wrapper rewrites rewards by building a new outcome.

        A mutable outcome invites a reward wrapper to edit the environment's
        answer in place, after which the bundle's lineage records a reward
        the agent never received.
        """
        outcome = StepOutcome(np.zeros(2), reward=1.0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            outcome.reward = 2.0

    def test_info_defaults_to_empty_and_is_never_shared(self):
        """
        Two outcomes do not share one info mapping.

        A mutable default would give every outcome in a run the same
        dictionary, so one wrapper's diagnostics would appear on every step.
        """
        first = StepOutcome(np.zeros(2), reward=0.0)
        second = StepOutcome(np.zeros(2), reward=0.0)
        assert first.info == {}
        assert first.info is not second.info
