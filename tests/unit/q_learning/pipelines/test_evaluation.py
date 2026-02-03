"""
Unit tests for src.q_learning.pipelines.evaluation.

Tests evaluate_agent() with minimal agent and BaseEnv.
"""

import pytest

from src.q_learning.core import RLAgent
from src.q_learning.pipelines.evaluation import evaluate_agent
from src.q_learning.environments import BaseEnv
from src.q_learning.core.types import RLEvaluationResult


class DeterministicAgent:
    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions

    def select_action(self, state, *, training=False, explore=True):
        return 1

    def update(self, transitions=None, batch=None):
        return None

    def get_parameters(self):
        return {"n_actions": self.n_actions}

    def set_parameters(self, params):
        self.n_actions = params.get("n_actions", self.n_actions)


@pytest.fixture
def env():
    return BaseEnv(state_dim=1, n_actions=3, max_steps=20, seed=123)


@pytest.fixture
def agent():
    return DeterministicAgent(n_actions=3)


class TestEvaluateAgent:
    def test_returns_evaluation_result(self, agent, env):
        result = evaluate_agent(agent, env, n_episodes=5, max_steps_per_episode=20)
        assert isinstance(result, RLEvaluationResult)
        assert len(result.returns) == 5
        assert len(result.lengths) == 5
        assert result.mean_length > 0

    def test_metrics_included(self, agent, env):
        result = evaluate_agent(
            agent, env, n_episodes=5, metrics=["sharpe", "max_drawdown", "win_rate"]
        )
        assert "sharpe" in result.metrics
        assert "max_drawdown" in result.metrics
        assert "win_rate" in result.metrics

    def test_no_exploration_by_default(self, agent, env):
        result = evaluate_agent(agent, env, n_episodes=3, explore=False)
        assert len(result.returns) == 3
