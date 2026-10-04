"""
Fresh on-policy experience, presented as an ordinary batch source.

This is the module that makes the framework's central claim concrete. A
training loop consumes :class:`~rade_qnet.core.contract.source.BatchSource`
and nothing else; a dataset satisfies it, and so does an environment being
acted in. The loop therefore does not branch on whether it is doing
supervised or interactive learning -- it asks for batches.

Unbounded is the whole point
----------------------------
:attr:`RolloutSource.steps_per_epoch` is ``None``, and that single value is
what selects the loop driver: ``fit_epochs`` for a bounded dataset,
``fit_steps`` for this. The rule is stated in
:mod:`rade_qnet.core.contract.source` and it is the source that declares
which it is, so the engine never guesses. There is no meaningful notion of a
pass over an environment, and a source that invented one -- reporting, say,
the batch size as a step count -- would silently corrupt every callback,
progress figure and learning-rate schedule computed from it.

Why the policy arrives as a callable
------------------------------------
``sources`` may import ``core`` and nothing else, which means this module
cannot touch a policy: a policy is an engine-native object -- a
``torch.nn.Module`` here -- and importing one would break the layering that
lets a spec be parsed on a host with no training library installed.

So the caller passes ``act``, a plain callable from one observation to one
action. The engine owns it, because selecting an action is where the engine's
library genuinely appears: it is a forward pass, under ``no_grad``, with
whatever exploration the learner wants. What is left here is the part that is
identical for every engine -- stepping, resetting, episode bookkeeping and
stacking -- written once against NumPy, exactly as the package charter
requires.

Why the reward is the batch's target
------------------------------------
:class:`~rade_qnet.core.contract.signature.InputSignature` requires a target,
and for a rollout the reward is the honest answer: it is the quantity the
batch exists to explain. Naming it under
:data:`~rade_qnet.core.contract.data.TARGET_KEY` rather than inventing a
``"reward"`` key means an interactive batch has the same shape as a
supervised one, which is what keeps the two paths comparable rather than
merely adjacent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import TARGET_KEY
from ...core.contract.signature import InputSignature, PolicySignature, TensorSpec
from ...core.lifecycle.errors import ContractError
from ...core.provenance.logging import get_logger
from ..environment.protocol import Environment

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from ...core.contract.data import Batch, TensorLike
    from ...core.contract.signature import SpaceSpec

__all__ = [
    "ACTION_KEY",
    "NEXT_OBSERVATION_KEY",
    "OBSERVATION_KEY",
    "TERMINATED_KEY",
    "TRUNCATED_KEY",
    "RolloutSource",
]

_LOGGER = get_logger(__name__)

#: What the policy saw.
OBSERVATION_KEY = "observation"

#: What it did.
ACTION_KEY = "action"

#: What it saw next. Carried because a bootstrapping learner needs the value
#: of the state it arrived in, and recomputing it by shifting the observation
#: column is wrong at every episode boundary -- which is precisely where a
#: value estimate matters most.
NEXT_OBSERVATION_KEY = "next_observation"

#: Whether the task itself ended.
TERMINATED_KEY = "terminated"

#: Whether the episode was cut short from outside the task. Separate from
#: :data:`TERMINATED_KEY` for the reason given on
#: :class:`~rade_qnet.sources.environment.protocol.StepOutcome`.
TRUNCATED_KEY = "truncated"

#: Episode statistics a collector accumulates, reported through
#: :meth:`RolloutSource.episode_summary` so a run has something interpretable
#: to show. A mean loss says nothing about an agent; a mean episode return is
#: the number anybody actually asks for.
_RETURN_KEY = "episode_return"
_LENGTH_KEY = "episode_length"


class RolloutSource:
    """
    An unbounded batch source that collects experience by acting.

    Satisfies :class:`~rade_qnet.core.contract.source.BatchSource`.

    Parameters
    ----------
    environment
        The environment to act in.
    act
        Chooses an action for one observation. Supplied by the engine, for
        the layering reason in the module docstring.

        May be ``None`` at construction and supplied later through
        :meth:`bind`, because the two cannot always be built in that order: a
        learner is constructed from this source's own
        :attr:`policy_signature`, so a pipeline that had to pass a selector
        up front would need the learner before the source it is built from.
        A source with no selector refuses to collect rather than acting
        arbitrarily.
    batch_size
        Transitions per batch.
    seed
        Seeds the environment's first reset. Subsequent resets do not
        re-seed, so episodes differ from one another -- re-seeding every
        episode would make them all the same episode.
    max_episode_steps
        Truncate an episode after this many steps, or ``None`` for no limit.
        Worth having because an environment with no terminal state and no
        limit yields one infinite episode, from which no episode statistic is
        ever reported and no bootstrapping learner ever sees a boundary.

    Raises
    ------
    ContractError
        If ``environment`` does not satisfy
        :class:`~rade_qnet.sources.environment.protocol.Environment`, named
        here rather than failing on the first ``step``.
    """

    def __init__(
        self,
        environment: Environment,
        *,
        act: Callable[[TensorLike], TensorLike] | None = None,
        batch_size: int,
        seed: int = 0,
        max_episode_steps: int | None = None,
    ) -> None:
        self.environment = self._checked(environment)
        self.act = act
        self.batch_size = batch_size
        self.seed = seed
        self.max_episode_steps = max_episode_steps

        self._observation: TensorLike | None = None
        self._episode_steps = 0
        self._episode_reward = 0.0
        self._returns: list[float] = []
        self._lengths: list[int] = []

    @staticmethod
    def _checked(environment: Environment) -> Environment:
        """
        Refuse an object that is not an environment.

        Parameters
        ----------
        environment
            The candidate.

        Returns
        -------
        Environment
            The same object.

        Raises
        ------
        ContractError
            If it lacks any member of the protocol.
        """
        if not isinstance(environment, Environment):
            raise ContractError(
                f"{type(environment).__name__} does not satisfy Environment; it "
                f"needs observation_space, action_space, reset() and step(). "
                f"A model's build_environment() must return one"
            )
        return environment

    @property
    def signature(self) -> InputSignature:
        """
        The declared interface of the batches this source yields.

        Returns
        -------
        InputSignature
            Observations, actions, next observations and the two episode-end
            flags as dynamic inputs, with the reward as the target.
        """
        observation = _spec_for(self.environment.observation_space)
        action = _spec_for(self.environment.action_space)
        flag = TensorSpec(shape=(None,), dtype="bool", description="episode-end flag")
        return InputSignature(
            dynamic={
                OBSERVATION_KEY: observation,
                ACTION_KEY: action,
                NEXT_OBSERVATION_KEY: observation,
                TERMINATED_KEY: flag,
                TRUNCATED_KEY: flag,
            },
            target=TensorSpec(shape=(None,), dtype="float32", description="step reward"),
        )

    def bind(self, act: Callable[[TensorLike], TensorLike]) -> None:
        """
        Supply the action selector, replacing any already set.

        Called by the engine once it has resolved the learner that acts.
        Separate from the constructor because of the ordering described under
        the ``act`` parameter, and a method rather than a bare attribute
        assignment so that the hand-off is a named, searchable event rather
        than something a reader has to notice.

        Parameters
        ----------
        act
            Chooses an action for one observation.
        """
        self.act = act
        _LOGGER.debug("bound an action selector to the rollout source")

    @property
    def policy_signature(self) -> PolicySignature:
        """
        The declared interface of the *policy*, as distinct from the batches.

        Two signatures, because the source sits between two things that need
        different descriptions. :attr:`signature` describes the experience it
        yields, in tensor terms, which is what a loop and a bundle read.
        This describes the spaces, which is what a learner needs to turn a
        network's output into an action the environment will accept -- and a
        tensor description cannot serve: it loses the number of discrete
        actions and the bounds of a continuous space.

        Returns
        -------
        PolicySignature
            The environment's two declared spaces.
        """
        return PolicySignature(
            observation=self.environment.observation_space,
            action=self.environment.action_space,
        )

    @property
    def static(self) -> Mapping[str, TensorLike]:
        """
        Inputs constant across every batch.

        Returns
        -------
        Mapping
            Always empty: an environment has no static inputs, and a policy
            that needs one gets it from its own construction rather than
            from the experience stream.
        """
        return {}

    @property
    def steps_per_epoch(self) -> None:
        """
        Always ``None``, because the source is unbounded.

        Returns
        -------
        None
            Declares this source as the kind ``fit_steps`` drives. See the
            module docstring.
        """
        return None

    @property
    def n_samples(self) -> None:
        """
        Always ``None``, because the source is unbounded.

        Returns
        -------
        None
            There is no pass whose samples could be counted.
        """
        return None

    def batches(self) -> Iterator[Batch]:
        """
        Yield batches of freshly collected experience, indefinitely.

        Re-iterable in the sense the protocol requires -- calling this again
        continues collecting -- but deliberately *not* reproducible batch for
        batch across calls, because the policy improves between them and
        re-collecting identical experience would mean the source had ignored
        it. The reproducibility that matters for an interactive run is the
        seeded environment plus the seeded policy, not a fixed batch
        sequence.

        Yields
        ------
        Batch
            ``batch_size`` transitions, stacked along the leading axis.
        """
        while True:
            yield self.collect()

    def collect(self) -> Batch:
        """
        Collect exactly one batch of transitions.

        Separate from :meth:`batches` so a driver that wants one update's
        worth of experience can ask for it directly, which is what
        ``fit_steps`` does. The generator is then a thin loop over this
        rather than the other way around.

        Returns
        -------
        Batch
            One batch, keyed as the signature declares.

        Raises
        ------
        ContractError
            If no action selector has been bound.
        """
        observations: list[TensorLike] = []
        actions: list[TensorLike] = []
        next_observations: list[TensorLike] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated: list[bool] = []

        if self.act is None:
            raise ContractError(
                "this rollout source has no action selector, so it cannot "
                "collect: something must call bind() with the learner that "
                "acts on the policy. An interactive engine does this when it "
                "resolves the learner"
            )

        for _ in range(self.batch_size):
            observation = self._current_observation()
            action = self.act(observation)
            outcome = self.environment.step(action)

            self._episode_steps += 1
            self._episode_reward += float(outcome.reward)
            cut_short = outcome.truncated or self._reached_the_step_limit()

            observations.append(observation)
            actions.append(action)
            next_observations.append(outcome.observation)
            rewards.append(float(outcome.reward))
            terminated.append(bool(outcome.terminated))
            truncated.append(bool(cut_short))

            if outcome.terminated or cut_short:
                self._close_episode()
            else:
                self._observation = outcome.observation

        return {
            OBSERVATION_KEY: np.stack([np.asarray(value) for value in observations]),
            ACTION_KEY: np.stack([np.asarray(value) for value in actions]),
            NEXT_OBSERVATION_KEY: np.stack([np.asarray(value) for value in next_observations]),
            TARGET_KEY: np.asarray(rewards, dtype=np.float32),
            TERMINATED_KEY: np.asarray(terminated, dtype=np.bool_),
            TRUNCATED_KEY: np.asarray(truncated, dtype=np.bool_),
        }

    def episode_summary(self) -> dict[str, float]:
        """
        Return the mean return and length of the episodes finished so far.

        The interpretable half of an interactive run. A learner's loss is not
        comparable between algorithms and barely comparable between runs; a
        mean episode return is the number that answers "is this agent any
        good". Reported per update by the driver and folded into the run's
        metrics.

        Returns
        -------
        dict
            Mean return and mean length, or empty when no episode has
            finished yet -- empty rather than zero, because a run whose first
            episode is still going has no return, and reporting ``0.0`` would
            be indistinguishable from an agent earning nothing.
        """
        if not self._returns:
            return {}
        return {
            _RETURN_KEY: float(np.mean(self._returns)),
            _LENGTH_KEY: float(np.mean(self._lengths)),
        }

    def _current_observation(self) -> TensorLike:
        """
        Return the observation to act on, resetting if no episode is running.

        Returns
        -------
        TensorLike
            The current observation.
        """
        if self._observation is None:
            # Seeded only on the very first reset. See the ``seed`` parameter.
            first = self.seed if not self._returns and self._episode_steps == 0 else None
            self._observation = self.environment.reset(seed=first)
        return self._observation

    def _reached_the_step_limit(self) -> bool:
        """
        Whether the running episode has hit its step limit.

        Returns
        -------
        bool
            True when a limit is configured and has been reached.
        """
        return self.max_episode_steps is not None and self._episode_steps >= self.max_episode_steps

    def _close_episode(self) -> None:
        """Record the finished episode's statistics and arm the next reset."""
        self._returns.append(self._episode_reward)
        self._lengths.append(self._episode_steps)
        _LOGGER.debug(
            "episode finished: return=%.4f length=%d", self._episode_reward, self._episode_steps
        )
        self._episode_reward = 0.0
        self._episode_steps = 0
        # Cleared rather than reset here, so the next collect() resets lazily
        # and a source that is never iterated never touches the environment.
        self._observation = None

    def describe(self) -> dict[str, object]:
        """
        Return a short description, for logs and reports.

        Returns
        -------
        dict
            Batch size, the step limit, and how many episodes have finished.
        """
        return {
            "kind": "rollout",
            "batch_size": self.batch_size,
            "max_episode_steps": self.max_episode_steps,
            "episodes_finished": len(self._returns),
        }


def _spec_for(space: SpaceSpec) -> TensorSpec:
    """
    Describe one space as the tensor a batch of it will be.

    Parameters
    ----------
    space
        An observation or action space.

    Returns
    -------
    TensorSpec
        The batched shape, with a leading ``None`` for the batch dimension.

    Raises
    ------
    ContractError
        If the space names a kind this function has not been taught, which
        would otherwise produce a signature that silently misdescribes the
        batches.
    """
    if space.kind == "box":
        return TensorSpec(shape=(None, *space.shape), dtype=space.dtype)
    if space.kind == "discrete":
        # One index per sample, not a one-hot row: an index is what an
        # environment accepts and what a discrete policy emits, and widening
        # it here would make every learner undo the widening.
        return TensorSpec(shape=(None,), dtype="int64")
    raise ContractError(
        f"space kind {space.kind!r} has no tensor description; expected 'box' or 'discrete'"
    )
