"""
Tests for the rollout batch source.

The claim this module has to earn is that an environment being acted in is
indistinguishable, from a loop's point of view, from a dataset. So the first
group tests that it satisfies ``BatchSource``, and the rest test the parts
where "just a source" is doing real work:

**Unboundedness is declared, not implied.** ``steps_per_epoch`` of ``None``
is the single value that selects the step driver over the epoch driver. A
source that reported its batch size there instead would be driven by the
wrong loop, and every metric denominator and schedule period would be wrong
with it -- silently, because both numbers are plausible.

**Episode boundaries land inside batches.** The hard part of a collector is
that an episode ends partway through a batch: the next transition belongs to
a fresh episode, the flags have to mark where, and the statistics have to
count the episode that finished rather than the batch that contained it.

**Termination and truncation stay apart.** A step limit is not a terminal
state. Conflating them biases every value estimate that bootstraps past the
boundary, which is a wrong answer rather than an error, so it is tested
directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import TARGET_KEY
from src.rade_qnet.core.contract.signature import SpaceSpec
from src.rade_qnet.core.contract.source import BatchSource
from src.rade_qnet.core.lifecycle.errors import ContractError
from src.rade_qnet.sources.batching.rollout import (
    ACTION_KEY,
    NEXT_OBSERVATION_KEY,
    OBSERVATION_KEY,
    TERMINATED_KEY,
    TRUNCATED_KEY,
    RolloutSource,
)
from src.rade_qnet.sources.environment import StepOutcome


class Counter:
    """
    An environment that counts up and terminates on a fixed step.

    The observation is the step index, so a test can read back exactly which
    transitions a batch contains and where an episode restarted.

    Parameters
    ----------
    episode_length
        The step on which the episode terminates.
    """

    observation_space = SpaceSpec(kind="box", shape=(1,), dtype="float32")
    action_space = SpaceSpec(kind="discrete", n=2)

    def __init__(self, episode_length: int = 3) -> None:
        self.episode_length = episode_length
        self.step_index = 0
        self.seeds: list[int | None] = []
        self.resets = 0

    def reset(self, *, seed: int | None = None):
        """
        Start a new episode, recording the seed it was given.

        Parameters
        ----------
        seed
            Recorded so a test can assert the seeding policy.

        Returns
        -------
        numpy.ndarray
            The zero observation.
        """
        self.seeds.append(seed)
        self.resets += 1
        self.step_index = 0
        return np.zeros(1, dtype=np.float32)

    def step(self, action):
        """
        Advance one step.

        Parameters
        ----------
        action
            Recorded in the reward, so a test can tell which action produced
            which transition.

        Returns
        -------
        StepOutcome
            The next observation, with the reward carrying the action.
        """
        self.step_index += 1
        return StepOutcome(
            np.full(1, self.step_index, dtype=np.float32),
            reward=float(action),
            terminated=self.step_index >= self.episode_length,
        )


class Endless(Counter):
    """An environment with no terminal state, to exercise truncation alone."""

    def __init__(self) -> None:
        super().__init__(episode_length=10**9)


def always(action: int):
    """
    Build an action selector that always chooses one action.

    Parameters
    ----------
    action
        The action to return.

    Returns
    -------
    callable
        A selector ignoring its observation.
    """
    return lambda observation: np.int64(action)


class TestItIsJustASource:
    """The framework's central claim, tested rather than asserted."""

    def test_it_satisfies_batch_source(self):
        """A loop written against ``BatchSource`` can consume an environment."""
        source = RolloutSource(Counter(), act=always(1), batch_size=2)
        assert isinstance(source, BatchSource)

    def test_an_object_that_is_not_an_environment_is_refused_by_name(self):
        """
        The fault is named at construction, not at the first step.

        A collector that discovered this on its first ``step`` would report an
        ``AttributeError`` from inside a loop, which points at the loop rather
        than at the model's ``build_environment``.
        """
        with pytest.raises(ContractError, match="does not satisfy Environment"):
            RolloutSource(object(), act=always(0), batch_size=2)

    def test_the_signature_describes_the_batches_it_actually_yields(self):
        """
        Every declared key is present, with the declared shape.

        The signature is what a bundle records, so a signature that disagreed
        with the batches would misdescribe the run permanently.
        """
        source = RolloutSource(Counter(), act=always(1), batch_size=4)
        batch = source.collect()

        for name, spec in source.signature.dynamic.items():
            assert name in batch, f"{name} is declared but not produced"
            # The leading dimension is declared as varying, so it is the one
            # position compared loosely.
            assert np.asarray(batch[name]).shape[1:] == spec.shape[1:]

    def test_the_reward_is_the_target(self):
        """
        An interactive batch has the same shape as a supervised one.

        Keeping the reward under the shared target key is what makes the two
        paths comparable rather than merely adjacent.
        """
        source = RolloutSource(Counter(), act=always(1), batch_size=4)
        batch = source.collect()
        assert np.asarray(batch[TARGET_KEY]).tolist() == [1.0, 1.0, 1.0, 1.0]

    def test_it_has_no_static_inputs(self):
        """An environment delivers everything through the stream."""
        assert RolloutSource(Counter(), act=always(0), batch_size=2).static == {}


class TestUnboundedness:
    """The one value that selects the loop driver."""

    def test_steps_per_epoch_is_none(self):
        """
        Declared unbounded, so the step driver is selected.

        Reporting the batch size here instead would be plausible and wrong:
        the epoch driver would accept the source and treat one batch as a
        complete pass.
        """
        assert RolloutSource(Counter(), act=always(0), batch_size=8).steps_per_epoch is None

    def test_n_samples_is_none(self):
        """There is no pass whose samples could be counted."""
        assert RolloutSource(Counter(), act=always(0), batch_size=8).n_samples is None

    def test_batches_never_run_out(self):
        """
        The generator keeps collecting, which is what "unbounded" means.

        Drained with an explicit bound rather than a ``for`` over the whole
        thing, because the whole thing does not end.
        """
        source = RolloutSource(Counter(), act=always(1), batch_size=2)
        stream = source.batches()
        collected = [next(stream) for _ in range(5)]
        assert len(collected) == 5
        assert all(np.asarray(batch[TARGET_KEY]).shape == (2,) for batch in collected)


class TestEpisodeBoundaries:
    """What happens when an episode ends partway through a batch."""

    def test_a_batch_spans_the_end_of_an_episode(self):
        """
        One batch holds the end of one episode and the start of the next.

        With a three-step episode and a five-transition batch, the
        observations acted on are steps 0, 1, 2 of the first episode and then
        0, 1 of the second -- so the counter restarts inside the batch.
        """
        environment = Counter(episode_length=3)
        source = RolloutSource(environment, act=always(1), batch_size=5)

        batch = source.collect()
        acted_on = np.asarray(batch[OBSERVATION_KEY]).reshape(-1).tolist()
        assert acted_on == [0.0, 1.0, 2.0, 0.0, 1.0]

    def test_the_terminal_flag_marks_the_boundary(self):
        """
        Exactly the terminating transition is flagged.

        This is what a bootstrapping learner reads, so a flag one position out
        would make it bootstrap past a terminal state.
        """
        source = RolloutSource(Counter(episode_length=3), act=always(1), batch_size=5)
        batch = source.collect()
        assert np.asarray(batch[TERMINATED_KEY]).tolist() == [False, False, True, False, False]

    def test_the_terminal_observation_is_kept(self):
        """
        The last ``next_observation`` of an episode is the terminal one.

        Not the reset observation of the next episode. A learner needs the
        state the agent arrived in, and overwriting it with a fresh reset
        would make every episode's final transition describe the wrong
        outcome.
        """
        source = RolloutSource(Counter(episode_length=3), act=always(1), batch_size=4)
        batch = source.collect()
        arrived_in = np.asarray(batch[NEXT_OBSERVATION_KEY]).reshape(-1).tolist()
        assert arrived_in == [1.0, 2.0, 3.0, 1.0]

    def test_episode_statistics_count_episodes_not_batches(self):
        """
        Two three-step episodes are reported, from one six-transition batch.

        The mean return is the number anybody actually asks for, and it is
        per episode -- a per-batch mean would change meaning whenever the
        batch size changed.
        """
        source = RolloutSource(Counter(episode_length=3), act=always(1), batch_size=6)
        source.collect()

        summary = source.episode_summary()
        assert summary == {"episode_return": 3.0, "episode_length": 3.0}
        assert source.describe()["episodes_finished"] == 2

    def test_no_statistics_before_the_first_episode_finishes(self):
        """
        Empty rather than zero.

        A run whose first episode is still going has no return, and reporting
        ``0.0`` would be indistinguishable from an agent earning nothing.
        """
        source = RolloutSource(Counter(episode_length=100), act=always(1), batch_size=4)
        source.collect()
        assert source.episode_summary() == {}


class TestTruncation:
    """A step limit is not a terminal state."""

    def test_the_step_limit_truncates_without_terminating(self):
        """
        The two flags disagree, which is the whole point.

        An environment with no terminal state, cut off at three steps: the
        transition is ``truncated`` and not ``terminated``, so a value
        estimate bootstraps past it rather than treating it as a failure.
        """
        source = RolloutSource(Endless(), act=always(1), batch_size=3, max_episode_steps=3)
        batch = source.collect()

        assert np.asarray(batch[TRUNCATED_KEY]).tolist() == [False, False, True]
        assert np.asarray(batch[TERMINATED_KEY]).tolist() == [False, False, False]

    def test_the_limit_restarts_the_episode(self):
        """A truncated episode is still over, so the next step is a fresh one."""
        environment = Endless()
        source = RolloutSource(environment, act=always(1), batch_size=4, max_episode_steps=2)
        batch = source.collect()

        acted_on = np.asarray(batch[OBSERVATION_KEY]).reshape(-1).tolist()
        assert acted_on == [0.0, 1.0, 0.0, 1.0]
        assert environment.resets == 2

    def test_a_truncated_episode_is_counted(self):
        """Its return is reported, or a run with no terminal state reports nothing."""
        source = RolloutSource(Endless(), act=always(1), batch_size=4, max_episode_steps=2)
        source.collect()
        assert source.episode_summary() == {"episode_return": 2.0, "episode_length": 2.0}


class TestSeeding:
    """Only the first reset is seeded."""

    def test_the_first_episode_gets_the_seed(self):
        """So a run is reproducible from its spec."""
        environment = Counter(episode_length=2)
        RolloutSource(environment, act=always(1), batch_size=1, seed=11).collect()
        assert environment.seeds == [11]

    def test_later_episodes_are_not_re_seeded(self):
        """
        Otherwise every episode would be the same episode.

        Re-seeding on each reset is an easy mistake that looks like extra
        rigour and destroys the variety the agent needs to learn from.
        """
        environment = Counter(episode_length=2)
        RolloutSource(environment, act=always(1), batch_size=5, seed=11).collect()
        assert environment.seeds == [11, None, None]

    def test_nothing_is_touched_until_the_first_collection(self):
        """
        Constructing a source does not reset the environment.

        A pipeline builds its sources before it decides what to drive them
        with, and a source that stepped at construction would make a failed
        run leave the environment half-used.
        """
        environment = Counter()
        RolloutSource(environment, act=always(1), batch_size=2)
        assert environment.resets == 0


class TestTheActionSelector:
    """The policy arrives as a callable, so that ``sources`` stays free of Torch."""

    def test_the_selector_sees_the_observation_it_acts_on(self):
        """
        Not the one that follows it.

        Off by one here would train every policy on the consequence of the
        action rather than on the state that prompted it.
        """
        seen: list[float] = []

        def record(observation):
            seen.append(float(np.asarray(observation).reshape(-1)[0]))
            return np.int64(1)

        source = RolloutSource(Counter(episode_length=100), act=record, batch_size=3)
        batch = source.collect()

        assert seen == np.asarray(batch[OBSERVATION_KEY]).reshape(-1).tolist()

    def test_the_chosen_action_is_the_one_recorded(self):
        """A batch's actions are what the selector returned, unmodified."""
        source = RolloutSource(Counter(episode_length=100), act=always(1), batch_size=3)
        batch = source.collect()
        assert np.asarray(batch[ACTION_KEY]).tolist() == [1, 1, 1]
