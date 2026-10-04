"""
Tests for the no-op interactive learner.

A learner that updates nothing has exactly one interesting property, and it
is the one that is easy to lose: it must update *nothing*. The central test
compares every parameter before and after a batch of experience, because a
no-op learner that quietly trained would destroy its own purpose -- it is
the control a real algorithm is compared against, and a control that moves
makes the comparison meaningless.

The rest is about :meth:`RandomLearner.act`, which does three jobs at once:
it really forward-passes the policy, so a lazily initialised network cannot
go unmaterialised; it samples rather than taking an argmax, so episodes vary
and the boundary handling gets exercised; and it produces something the
environment will actually accept, which is where the action space has to be
respected rather than assumed.
"""

from __future__ import annotations

import pytest
import torch

from src.rade_qnet.core.contract.data import TARGET_KEY
from src.rade_qnet.core.contract.signature import PolicySignature, SpaceSpec
from src.rade_qnet.core.lifecycle.components import get_learner
from src.rade_qnet.core.lifecycle.errors import EngineError
from src.rade_qnet.engines.torch.learners.random import RandomLearner

DISCRETE = PolicySignature(
    observation=SpaceSpec(kind="box", shape=(3,), dtype="float32"),
    action=SpaceSpec(kind="discrete", n=4),
)
BOXED = PolicySignature(
    observation=SpaceSpec(kind="box", shape=(3,), dtype="float32"),
    action=SpaceSpec(kind="box", shape=(2,), dtype="float32", low=-1.0, high=1.0),
)


class Head(torch.nn.Module):
    """
    A one-layer policy sized from a signature.

    Parameters
    ----------
    signature
        Supplies the observation width and the output width.
    width
        Output width. Defaults to the signature's action size.
    """

    def __init__(self, signature: PolicySignature, width: int | None = None) -> None:
        super().__init__()
        if width is None:
            action = signature.action
            width = action.n if action.kind == "discrete" else action.shape[0]
        self.layer = torch.nn.Linear(signature.observation.shape[0], width)
        self.calls = 0

    def forward(self, *, observation: torch.Tensor) -> torch.Tensor:
        """
        Score one batch of observations.

        Parameters
        ----------
        observation
            A batch of observations.

        Returns
        -------
        torch.Tensor
            One row of scores per observation.
        """
        self.calls += 1
        return self.layer(observation)


def experience(rewards: list[float]) -> dict[str, torch.Tensor]:
    """
    Build a minimal batch of experience.

    Parameters
    ----------
    rewards
        The rewards the batch carries.

    Returns
    -------
    dict
        A batch with the reward under the shared target key.
    """
    return {TARGET_KEY: torch.tensor(rewards, dtype=torch.float32)}


class TestItLearnsNothing:
    """The property the whole module exists for."""

    def test_no_parameter_changes(self):
        """
        Every weight is bit-for-bit identical after an update.

        If this ever fails, the control has become a competitor and no
        comparison against it means anything.
        """
        policy = Head(DISCRETE)
        learner = RandomLearner(signature=DISCRETE)
        before = {name: value.detach().clone() for name, value in policy.named_parameters()}

        learner.update(policy, experience([1.0, 0.0, 1.0]))

        for name, value in policy.named_parameters():
            assert torch.equal(before[name], value), f"{name} changed"

    def test_no_gradient_is_even_accumulated(self):
        """
        Not merely un-stepped: no graph is built at all.

        A learner that computed gradients and declined to apply them would
        pass the test above while paying the cost and holding the graph.
        """
        policy = Head(DISCRETE)
        RandomLearner(signature=DISCRETE).update(policy, experience([1.0]))
        assert all(value.grad is None for value in policy.parameters())

    def test_the_loss_is_flat_whatever_the_rewards(self):
        """
        Reported as zero, every time.

        The honest answer: this learner has no objective. A loss that moved
        with the reward would look like learning.
        """
        learner = RandomLearner(signature=DISCRETE)
        policy = Head(DISCRETE)

        assert learner.update(policy, experience([0.0, 0.0]))["loss"] == 0.0
        assert learner.update(policy, experience([9.0, -3.0]))["loss"] == 0.0

    def test_the_mean_reward_is_reported_beside_it(self):
        """
        So the history is not a column of zeros.

        Without it, a working no-op run and a run that never collected
        anything produce identical records.
        """
        learner = RandomLearner(signature=DISCRETE)
        scalars = learner.update(Head(DISCRETE), experience([2.0, 4.0]))
        assert scalars["mean_reward"] == pytest.approx(3.0)

    def test_a_batch_with_no_reward_is_refused(self):
        """A batch with no reward is not experience, whatever else it holds."""
        learner = RandomLearner(signature=DISCRETE)
        with pytest.raises(EngineError, match="not experience"):
            learner.update(Head(DISCRETE), {"observation": torch.zeros(2, 3)})


class TestChoosingAnAction:
    """``act`` really drives the policy, and respects the action space."""

    def test_the_policy_is_forward_passed(self):
        """
        Which is what proves a lazily shaped network was materialised.

        A network that is never forward-passed has no parameters, and the
        materialise stage exists to catch exactly that -- but only if
        something afterwards actually calls it.
        """
        policy = Head(DISCRETE)
        RandomLearner(signature=DISCRETE).act(policy, torch.zeros(3))
        assert policy.calls == 1

    def test_a_discrete_action_is_a_single_valid_index(self):
        """In range, and one value -- which is what the environment accepts."""
        learner = RandomLearner(signature=DISCRETE)
        policy = Head(DISCRETE)

        for _ in range(20):
            action = learner.act(policy, torch.zeros(3))
            assert action.numel() == 1
            assert 0 <= int(action) < DISCRETE.action.n

    def test_discrete_actions_vary(self):
        """
        Sampled rather than argmaxed.

        An untrained network's argmax is very nearly constant, and a constant
        action rarely reaches a terminal state -- leaving the episode
        boundary handling, which is the part most worth exercising,
        unexercised.
        """
        learner = RandomLearner(signature=DISCRETE, seed=0)
        policy = Head(DISCRETE)
        chosen = {int(learner.act(policy, torch.zeros(3))) for _ in range(40)}
        assert len(chosen) > 1

    def test_the_same_seed_gives_the_same_actions(self):
        """
        So an interactive run is reproducible end to end.

        Seeded in a dedicated generator rather than from the global stream,
        so two learners in one process do not interleave.
        """
        policy = Head(DISCRETE)
        first = [int(RandomLearner(signature=DISCRETE, seed=5).act(policy, torch.zeros(3)))]
        second = [int(RandomLearner(signature=DISCRETE, seed=5).act(policy, torch.zeros(3)))]
        assert first == second

    def test_a_box_action_is_shaped_and_within_bounds(self):
        """
        Clamped rather than trusted.

        An untrained head emits whatever its initialisation gives, which for
        a bounded space is routinely outside the bounds -- and an
        out-of-range action fails inside the environment, where the cause is
        invisible.
        """
        learner = RandomLearner(signature=BOXED)
        policy = Head(BOXED)

        action = learner.act(policy, torch.zeros(3))
        assert tuple(action.shape) == BOXED.action.shape
        assert bool((action >= BOXED.action.low).all())
        assert bool((action <= BOXED.action.high).all())

    def test_a_policy_of_the_wrong_width_is_refused(self):
        """
        Rather than reshaped.

        An output of the wrong width means the policy was built against a
        different space than the environment declares. Sampling from it
        anyway would produce actions the environment rejects at some
        unrelated later moment.
        """
        learner = RandomLearner(signature=DISCRETE)
        with pytest.raises(EngineError, match="discrete action space"):
            learner.act(Head(DISCRETE, width=DISCRETE.action.n + 1), torch.zeros(3))

    def test_an_action_never_carries_a_gradient(self):
        """
        Chosen under ``no_grad``.

        An action that retained its graph would keep the whole forward pass
        alive until the update -- for a replay buffer, for the whole run.
        """
        learner = RandomLearner(signature=BOXED)
        action = learner.act(Head(BOXED), torch.zeros(3))
        assert not action.requires_grad


class TestItIsResolvedLikeAnyOtherLearner:
    """Registered by name, so the registry has a reader beyond its own tests."""

    def test_the_registry_returns_it(self):
        """
        Registered under the name the spec's ``learner`` field accepts.

        Registration happens when the module is imported, which the import of
        ``RandomLearner`` at the top of this file is enough to do.
        """
        assert get_learner("random") is RandomLearner
