"""
Unit tests for src.q_learning.pipelines.training.

Tests run_training() and RLTrainingLoop with a minimal agent and BaseEnv.
"""

import pytest
import tempfile
from pathlib import Path

from src.q_learning.core import RLAgent, RLTrainingConfig
from src.q_learning.core.types import Transition
from src.q_learning.pipelines.training import run_training, RLTrainingLoop
from src.q_learning.environments import BaseEnv


class SimpleAgent:
    """Agent that returns fixed action and accepts updates."""

    def __init__(self, n_actions: int = 3):
        self.n_actions = n_actions
        self._params = {"n_actions": n_actions}
        self.updates_received = 0

    def select_action(self, state, *, training=False, explore=True):
        return 1  # fixed

    def update(self, transitions=None, batch=None):
        self.updates_received += 1
        return {"dummy": 0.0}

    def get_parameters(self):
        return dict(self._params)

    def set_parameters(self, params):
        self._params = dict(params)
        self.n_actions = self._params.get("n_actions", self.n_actions)


@pytest.fixture
def env():
    return BaseEnv(state_dim=2, n_actions=3, max_steps=10, seed=42)


@pytest.fixture
def agent():
    return SimpleAgent(n_actions=3)


@pytest.fixture
def config():
    return RLTrainingConfig(
        n_episodes=5,
        max_steps_per_episode=10,
        log_every=2,
        verbose=0,
        checkpoint_dir=None,
    )


class TestRLTrainingLoop:
    def test_initialization(self, agent, env, config):
        loop = RLTrainingLoop(agent, env, config)
        assert loop.agent is agent
        assert loop.env is env
        assert loop.config is config

    def test_run_returns_result(self, agent, env, config):
        loop = RLTrainingLoop(agent, env, config)
        result = loop.run()
        assert result.episode_returns is not None
        assert len(result.episode_returns) == 5
        assert len(result.episode_lengths) == 5
        assert result.best_episode_return > float("-inf")
        assert result.training_time_seconds >= 0
        assert agent.updates_received == 5

    def test_run_with_checkpoint_dir(self, agent, env):
        with tempfile.TemporaryDirectory() as tmp:
            config = RLTrainingConfig(
                n_episodes=3,
                max_steps_per_episode=5,
                verbose=0,
                checkpoint_dir=tmp,
                save_best_only=True,
            )
            result = run_training(agent, env, config)
            assert len(result.episode_returns) == 3
            files = list(Path(tmp).glob("*.json"))
            assert len(files) >= 1


class TestRunTraining:
    def test_run_training_returns_result(self, agent, env, config):
        result = run_training(agent, env, config)
        assert result is not None
        assert len(result.episode_returns) == config.n_episodes
        assert result.config is config
