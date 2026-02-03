"""
Unit tests for src.q_learning.core.types.

Tests Transition, RLTrainingConfig, RLTrainingResult, RLEvaluationResult.
"""

import json
import pytest
import tempfile
from pathlib import Path

from src.q_learning.core.types import (
    Transition,
    RLTrainingConfig,
    RLTrainingResult,
    RLEvaluationResult,
)


# =============================================================================
# Transition
# =============================================================================


class TestTransition:
    def test_to_dict(self):
        t = Transition(
            state=[1.0, 2.0],
            action=1,
            reward=0.5,
            next_state=[1.1, 2.1],
            terminated=False,
            truncated=False,
            info={"step": 1},
        )
        d = t.to_dict()
        assert d["state"] == [1.0, 2.0]
        assert d["action"] == 1
        assert d["reward"] == 0.5
        assert d["terminated"] is False
        assert d["info"]["step"] == 1

    def test_default_info(self):
        t = Transition(state=0, action=0, reward=0.0, next_state=0, terminated=False, truncated=False)
        assert t.info == {}


# =============================================================================
# RLTrainingConfig
# =============================================================================


class TestRLTrainingConfig:
    def test_default_values(self):
        config = RLTrainingConfig()
        assert config.n_episodes == 1000
        assert config.max_steps_per_episode == 0
        assert config.gamma == 0.99
        assert config.verbose == 1

    def test_custom_values(self):
        config = RLTrainingConfig(
            n_episodes=100,
            max_steps_per_episode=50,
            checkpoint_dir="/tmp/ckpt",
            log_every=10,
        )
        assert config.n_episodes == 100
        assert config.max_steps_per_episode == 50
        assert config.checkpoint_dir == "/tmp/ckpt"
        assert config.log_every == 10

    def test_to_dict(self):
        config = RLTrainingConfig(n_episodes=5, gamma=0.95)
        d = config.to_dict()
        assert d["n_episodes"] == 5
        assert d["gamma"] == 0.95


# =============================================================================
# RLTrainingResult
# =============================================================================


class TestRLTrainingResult:
    def test_to_dict_includes_config(self):
        config = RLTrainingConfig(n_episodes=10)
        result = RLTrainingResult(
            episode_returns=[1.0, 2.0],
            episode_lengths=[10, 20],
            best_episode_return=2.0,
            best_episode=2,
            config=config,
        )
        d = result.to_dict()
        assert d["episode_returns"] == [1.0, 2.0]
        assert d["best_episode_return"] == 2.0
        assert "config" in d
        assert d["config"]["n_episodes"] == 10

    def test_to_json_from_json_roundtrip(self):
        config = RLTrainingConfig(n_episodes=3, gamma=0.9)
        result = RLTrainingResult(
            episode_returns=[0.1, 0.2, 0.15],
            episode_lengths=[5, 6, 5],
            best_episode_return=0.2,
            best_episode=2,
            config=config,
            training_time_seconds=1.5,
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            result.to_json(path)
            loaded = RLTrainingResult.from_json(path)
            assert loaded.episode_returns == result.episode_returns
            assert loaded.best_episode_return == result.best_episode_return
            assert loaded.config is not None
            assert loaded.config.n_episodes == 3
        finally:
            Path(path).unlink(missing_ok=True)


# =============================================================================
# RLEvaluationResult
# =============================================================================


class TestRLEvaluationResult:
    def test_to_dict(self):
        r = RLEvaluationResult(
            mean_return=1.0,
            std_return=0.2,
            mean_length=10.0,
            returns=[1.0, 1.2, 0.8],
            lengths=[10, 11, 9],
            metrics={"sharpe": 0.5, "win_rate": 0.66},
        )
        d = r.to_dict()
        assert d["mean_return"] == 1.0
        assert d["metrics"]["sharpe"] == 0.5
        assert d["returns"] == [1.0, 1.2, 0.8]
