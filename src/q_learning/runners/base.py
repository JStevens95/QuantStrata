"""
Base runner protocol and configuration for RL agent execution.

Defines the interface that all runners (backtest, live) must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

from src.q_learning.core.protocols import RLAgent, RLEnvironment


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class RunnerConfig:
    """Base configuration for runners."""
    
    # Execution settings
    explore: bool = False  # Use exploration during execution
    render: bool = False  # Render environment (if supported)
    
    # Logging
    log_interval: int = 100  # Log every N steps
    verbose: bool = True
    
    # Callbacks
    on_step: Optional[callable] = None  # Called after each step
    on_episode_end: Optional[callable] = None  # Called after each episode


# =============================================================================
# Results
# =============================================================================


@dataclass
class EpisodeResult:
    """Result from a single episode."""
    
    episode_id: int
    total_reward: float
    n_steps: int
    final_info: Dict[str, Any]
    rewards: List[float] = field(default_factory=list)
    actions: List[Any] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get episode duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0


@dataclass
class RunResult:
    """Result from a complete run (multiple episodes)."""
    
    episodes: List[EpisodeResult]
    total_steps: int
    total_time_seconds: float
    config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_episodes(self) -> int:
        """Number of episodes."""
        return len(self.episodes)
    
    @property
    def mean_reward(self) -> float:
        """Mean total reward across episodes."""
        if not self.episodes:
            return 0.0
        return np.mean([e.total_reward for e in self.episodes])
    
    @property
    def std_reward(self) -> float:
        """Standard deviation of rewards."""
        if len(self.episodes) < 2:
            return 0.0
        return np.std([e.total_reward for e in self.episodes])
    
    @property
    def mean_steps(self) -> float:
        """Mean steps per episode."""
        if not self.episodes:
            return 0.0
        return np.mean([e.n_steps for e in self.episodes])
    
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        rewards = [e.total_reward for e in self.episodes]
        steps = [e.n_steps for e in self.episodes]
        
        return {
            "n_episodes": self.n_episodes,
            "total_steps": self.total_steps,
            "total_time_seconds": self.total_time_seconds,
            "mean_reward": self.mean_reward,
            "std_reward": self.std_reward,
            "min_reward": min(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "mean_steps": self.mean_steps,
            "min_steps": min(steps) if steps else 0,
            "max_steps": max(steps) if steps else 0,
        }


# =============================================================================
# Base Runner
# =============================================================================


class BaseRunner(ABC):
    """
    Abstract base class for RL agent runners.
    
    Defines the interface for running agents in different contexts
    (backtesting, live trading, etc.).
    
    Subclasses must implement:
    - run(): Execute the agent
    - _run_episode(): Run a single episode
    """
    
    def __init__(
        self,
        agent: RLAgent,
        env: RLEnvironment,
        config: Optional[RunnerConfig] = None,
    ) -> None:
        """
        Initialize runner.
        
        Parameters
        ----------
        agent : RLAgent
            Trained agent to execute.
        env : RLEnvironment
            Environment to run in.
        config : RunnerConfig, optional
            Runner configuration.
        """
        self.agent = agent
        self.env = env
        self.config = config or RunnerConfig()
        
        self._step_count: int = 0
        self._episode_count: int = 0
    
    @abstractmethod
    def run(self, **kwargs: Any) -> RunResult:
        """
        Execute the agent.
        
        Returns
        -------
        RunResult
            Execution results.
        """
        ...
    
    def _run_episode(
        self,
        episode_id: int,
        seed: Optional[int] = None,
        record_history: bool = True,
    ) -> EpisodeResult:
        """
        Run a single episode.
        
        Parameters
        ----------
        episode_id : int
            Episode identifier.
        seed : int, optional
            Random seed for environment reset.
        record_history : bool
            Whether to record action/reward history.
            
        Returns
        -------
        EpisodeResult
            Episode result.
        """
        start_time = datetime.now()
        
        # Reset environment
        state, info = self.env.reset(seed=seed)
        
        # Initialize tracking
        total_reward = 0.0
        rewards: List[float] = []
        actions: List[Any] = []
        n_steps = 0
        
        # Run episode
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Select action
            action = self.agent.select_action(
                state,
                training=False,
                explore=self.config.explore,
            )
            
            # Execute step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            # Update tracking
            total_reward += reward
            n_steps += 1
            self._step_count += 1
            
            if record_history:
                rewards.append(reward)
                actions.append(action)
            
            # Callback
            if self.config.on_step:
                self.config.on_step(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    info=info,
                )
            
            # Logging
            if self.config.verbose and n_steps % self.config.log_interval == 0:
                self._log_step(episode_id, n_steps, total_reward, info)
            
            state = next_state
        
        end_time = datetime.now()
        
        # Episode end callback
        if self.config.on_episode_end:
            self.config.on_episode_end(
                episode_id=episode_id,
                total_reward=total_reward,
                n_steps=n_steps,
                info=info,
            )
        
        return EpisodeResult(
            episode_id=episode_id,
            total_reward=total_reward,
            n_steps=n_steps,
            final_info=info,
            rewards=rewards,
            actions=actions,
            start_time=start_time,
            end_time=end_time,
        )
    
    def _log_step(
        self,
        episode_id: int,
        step: int,
        total_reward: float,
        info: Dict[str, Any],
    ) -> None:
        """Log step information."""
        print(f"Episode {episode_id}, Step {step}: reward={total_reward:.4f}")


__all__ = [
    "BaseRunner",
    "RunnerConfig",
    "RunResult",
    "EpisodeResult",
]
