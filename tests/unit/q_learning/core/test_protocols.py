"""
Unit tests for src.q_learning.core.protocols.

Tests that RLEnvironment and RLAgent are runtime checkable and that concrete
implementations satisfy the protocols.
"""

import pytest

from src.q_learning.core.protocols import RLAgent, RLEnvironment
from src.q_learning.environments.base import BaseEnv


class TestRLEnvironmentProtocol:
    def test_base_env_is_rl_environment(self):
        env = BaseEnv(state_dim=1, n_actions=2, max_steps=5)
        assert isinstance(env, RLEnvironment)

    def test_base_env_reset_step(self):
        env = BaseEnv(state_dim=2, n_actions=3, max_steps=10, seed=42)
        state, info = env.reset()
        assert state.shape == (2,)
        assert "step" in info
        state2, reward, term, trunc, info2 = env.step(1)
        assert state2.shape == (2,)
        assert isinstance(reward, (int, float))
        assert term is False
        assert "step" in info2


class MinimalAgent:
    """Minimal RLAgent implementation for tests."""

    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions
        self._params = {"n_actions": n_actions}

    def select_action(self, state, *, training=False, explore=True):
        import random
        return random.randint(0, self.n_actions - 1)

    def update(self, transitions=None, batch=None):
        return None

    def get_parameters(self):
        return dict(self._params)

    def set_parameters(self, params):
        self._params = dict(params)
        self.n_actions = self._params.get("n_actions", self.n_actions)


class TestRLAgentProtocol:
    def test_minimal_agent_is_rl_agent(self):
        agent = MinimalAgent(n_actions=3)
        assert isinstance(agent, RLAgent)

    def test_select_action_returns_action(self):
        agent = MinimalAgent(n_actions=2)
        action = agent.select_action([0.0, 0.0], training=False, explore=True)
        assert action in (0, 1)

    def test_get_set_parameters(self):
        agent = MinimalAgent(n_actions=5)
        params = agent.get_parameters()
        assert params["n_actions"] == 5
        agent.set_parameters({"n_actions": 7})
        assert agent.n_actions == 7
