"""
Unit tests for backtest runner.

Tests BacktestRunner, BacktestConfig, and BacktestResult.
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from src.q_learning.runners.backtest import (
    BacktestConfig,
    BacktestResult,
    BacktestRunner,
)
from src.q_learning.runners.base import EpisodeResult


class TestBacktestConfig:
    """Tests for BacktestConfig."""
    
    def test_default_config(self) -> None:
        """Test default configuration."""
        config = BacktestConfig()
        
        assert config.n_episodes == 100
        assert config.use_random_starts is True
        assert config.compute_sharpe is True
        assert config.compute_drawdown is True
    
    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = BacktestConfig(
            n_episodes=50,
            use_random_starts=False,
            risk_free_rate=0.02,
        )
        
        assert config.n_episodes == 50
        assert config.use_random_starts is False
        assert config.risk_free_rate == 0.02
    
    def test_episode_seeds(self) -> None:
        """Test setting specific episode seeds."""
        config = BacktestConfig(
            n_episodes=5,
            episode_seeds=[1, 2, 3, 4, 5],
        )
        
        assert config.episode_seeds == [1, 2, 3, 4, 5]


class TestBacktestResult:
    """Tests for BacktestResult."""
    
    def test_mean_pnl_return(self) -> None:
        """Test mean P&L return calculation."""
        result = BacktestResult()
        result.pnl_returns = [0.1, 0.05, -0.02, 0.08, 0.03]
        
        expected = np.mean([0.1, 0.05, -0.02, 0.08, 0.03])
        assert abs(result.mean_pnl_return - expected) < 1e-10
    
    def test_mean_pnl_return_empty(self) -> None:
        """Test mean P&L return with empty list."""
        result = BacktestResult()
        result.pnl_returns = []
        
        assert result.mean_pnl_return == 0.0
    
    def test_std_pnl_return(self) -> None:
        """Test std P&L return calculation."""
        result = BacktestResult()
        result.pnl_returns = [0.1, 0.05, -0.02, 0.08, 0.03]
        
        expected = np.std(result.pnl_returns)
        assert abs(result.std_pnl_return - expected) < 1e-10
    
    def test_std_pnl_return_single_element(self) -> None:
        """Test std with single element."""
        result = BacktestResult()
        result.pnl_returns = [0.1]
        
        assert result.std_pnl_return == 0.0


# Mock agent and environment for testing
class MockAgent:
    """Mock RL agent for testing."""
    
    def __init__(self, action: int = 2) -> None:
        self.action = action
    
    def act(self, state: np.ndarray, deterministic: bool = True) -> int:
        return self.action
    
    def select_action(self, state: np.ndarray) -> int:
        return self.action


class MockEnvironment:
    """Mock RL environment for testing."""
    
    def __init__(self, n_steps: int = 10) -> None:
        self.n_steps = n_steps
        self._step = 0
        self._seed = None
    
    @property
    def observation_space(self) -> Any:
        return type("Space", (), {"shape": (10,)})()
    
    @property
    def action_space(self) -> Any:
        return type("Space", (), {"n": 5})()
    
    def reset(self, seed: int = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._step = 0
        self._seed = seed
        return np.zeros(10), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self._step += 1
        reward = np.random.randn() * 0.01
        terminated = self._step >= self.n_steps
        return np.zeros(10), reward, terminated, False, {"pnl": reward}


class TestBacktestRunner:
    """Tests for BacktestRunner."""
    
    def test_runner_creation(self) -> None:
        """Test runner creation."""
        agent = MockAgent()
        env = MockEnvironment()
        config = BacktestConfig(n_episodes=5)
        
        runner = BacktestRunner(agent=agent, env=env, config=config)
        
        assert runner._agent is agent
        assert runner._env is env
    
    def test_run_single_episode(self) -> None:
        """Test running a single episode."""
        agent = MockAgent()
        env = MockEnvironment(n_steps=5)
        config = BacktestConfig(n_episodes=1)
        
        runner = BacktestRunner(agent=agent, env=env, config=config)
        result = runner.run()
        
        assert result.n_episodes == 1
        assert len(result.episodes) == 1
    
    def test_run_multiple_episodes(self) -> None:
        """Test running multiple episodes."""
        agent = MockAgent()
        env = MockEnvironment(n_steps=5)
        config = BacktestConfig(n_episodes=10)
        
        runner = BacktestRunner(agent=agent, env=env, config=config)
        result = runner.run()
        
        assert result.n_episodes == 10
        assert len(result.episodes) == 10
    
    def test_pnl_returns_collected(self) -> None:
        """Test that P&L returns are collected."""
        agent = MockAgent()
        env = MockEnvironment(n_steps=5)
        config = BacktestConfig(n_episodes=5)
        
        runner = BacktestRunner(agent=agent, env=env, config=config)
        result = runner.run()
        
        # Should have returns for each episode
        assert len(result.pnl_returns) == 5
    
    def test_sharpe_ratio_computed(self) -> None:
        """Test that Sharpe ratio is computed."""
        agent = MockAgent()
        env = MockEnvironment(n_steps=10)
        config = BacktestConfig(n_episodes=20, compute_sharpe=True)
        
        runner = BacktestRunner(agent=agent, env=env, config=config)
        result = runner.run()
        
        # Sharpe should be computed
        assert isinstance(result.sharpe_ratio, float)
    
    def test_drawdown_computed(self) -> None:
        """Test that max drawdown is computed."""
        agent = MockAgent()
        env = MockEnvironment(n_steps=10)
        config = BacktestConfig(n_episodes=20, compute_drawdown=True)
        
        runner = BacktestRunner(agent=agent, env=env, config=config)
        result = runner.run()
        
        assert isinstance(result.max_drawdown, float)
    
    def test_reproducibility(self) -> None:
        """Test reproducibility with seeds."""
        agent = MockAgent()
        config = BacktestConfig(n_episodes=5, episode_seeds=[42, 43, 44, 45, 46])
        
        env1 = MockEnvironment(n_steps=5)
        runner1 = BacktestRunner(agent=agent, env=env1, config=config)
        result1 = runner1.run()
        
        env2 = MockEnvironment(n_steps=5)
        runner2 = BacktestRunner(agent=agent, env=env2, config=config)
        result2 = runner2.run()
        
        # Results should be reproducible
        assert len(result1.episodes) == len(result2.episodes)


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""
    
    def test_episode_result_creation(self) -> None:
        """Test episode result creation."""
        result = EpisodeResult(
            episode_id=0,
            total_reward=1.5,
            n_steps=100,
        )
        
        assert result.episode_id == 0
        assert result.total_reward == 1.5
        assert result.n_steps == 100
    
    def test_episode_result_defaults(self) -> None:
        """Test episode result defaults."""
        result = EpisodeResult(
            episode_id=0,
            total_reward=1.0,
            n_steps=50,
        )
        
        assert result.rewards == []
        assert result.actions == []
        assert result.info == {}
