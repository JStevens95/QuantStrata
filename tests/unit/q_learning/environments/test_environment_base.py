"""
Unit tests for src.q_learning.environments.base.

Tests BaseEnv reset/step and protocol compliance.
"""

import pytest
import numpy as np

from src.q_learning.core.protocols import RLEnvironment
from src.q_learning.environments.base import BaseEnv


class TestBaseEnv:
    def test_reset_returns_state_and_info(self):
        env = BaseEnv(state_dim=3, n_actions=2, max_steps=10, seed=42)
        state, info = env.reset()
        assert isinstance(state, np.ndarray)
        assert state.shape == (3,)
        assert state.dtype == np.float32
        assert "step" in info
        assert info["step"] == 0

    def test_step_returns_five_tuple(self):
        env = BaseEnv(state_dim=1, n_actions=3, max_steps=5, seed=0)
        env.reset()
        state, reward, terminated, truncated, info = env.step(1)
        assert state.shape == (1,)
        assert isinstance(reward, (int, float))
        assert terminated is False
        assert "step" in info

    def test_episode_terminates_at_max_steps(self):
        env = BaseEnv(state_dim=1, n_actions=2, max_steps=3, seed=0)
        env.reset()
        for _ in range(3):
            state, reward, terminated, truncated, info = env.step(0)
        assert truncated is True

    def test_reset_with_seed_reproducible(self):
        env = BaseEnv(state_dim=2, n_actions=2, seed=99)
        s1, _ = env.reset(seed=99)
        env.step(0)
        env.reset(seed=99)
        s2, _ = env.reset(seed=99)
        np.testing.assert_array_almost_equal(s1, s2)

    def test_step_without_reset_raises(self):
        env = BaseEnv(state_dim=1, n_actions=2)
        with pytest.raises(RuntimeError):
            env.step(0)

    def test_is_rl_environment(self):
        env = BaseEnv(state_dim=1, n_actions=2)
        assert isinstance(env, RLEnvironment)
