"""
Data types for Q-Learning / RL: config, results, transitions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import json


@dataclass
class Transition:
    """
    Single transition (s, a, r, s', terminated, truncated).

    Used for replay buffers and batch updates.
    """

    state: Any
    action: Any
    reward: float
    next_state: Any
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLTrainingConfig:
    """
    Configuration for an RL training run.

    Parameters
    ----------
    n_episodes : int
        Number of episodes to run.
    max_steps_per_episode : int
        Maximum steps per episode (0 = no limit).
    learning_rate : float
        Learning rate for the agent optimizer (if applicable).
    gamma : float
        Discount factor for returns.
    checkpoint_dir : str, optional
        Directory to save checkpoints; None = no checkpointing.
    checkpoint_frequency : int
        Save checkpoint every N episodes (0 = only save best/last).
    save_best_only : bool
        If True, only save when episode return improves.
    log_every : int
        Log metrics every N episodes.
    eval_episodes : int
        Number of evaluation episodes (no exploration) for logging.
    verbose : int
        Verbosity (0 = silent, 1 = progress, 2 = detailed).
    """

    n_episodes: int = 1000
    max_steps_per_episode: int = 0
    learning_rate: float = 0.001
    gamma: float = 0.99
    checkpoint_dir: Optional[str] = None
    checkpoint_frequency: int = 0
    save_best_only: bool = True
    log_every: int = 1
    eval_episodes: int = 5
    verbose: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RLTrainingResult:
    """
    Output of an RL training run.

    Parameters
    ----------
    episode_returns : list of float
        Total return per episode.
    episode_lengths : list of int
        Steps per episode.
    history : dict
        Additional history (e.g. loss, epsilon).
    best_episode_return : float
        Best episode return achieved.
    best_episode : int
        Episode index of best return.
    config : RLTrainingConfig, optional
        Configuration used.
    training_time_seconds : float
        Total training time.
    metadata : dict
        Additional metadata.
    """

    episode_returns: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    history: Dict[str, List[float]] = field(default_factory=dict)
    best_episode_return: float = float("-inf")
    best_episode: int = 0
    config: Optional[RLTrainingConfig] = None
    training_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.config is not None:
            d["config"] = self.config.to_dict()
        return d

    def to_json(self, path: str) -> None:
        """Save to JSON (episode_returns/history as lists; state/action in metadata not serialised)."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "RLTrainingResult":
        with open(path) as f:
            d = json.load(f)
        d["config"] = RLTrainingConfig(**d["config"]) if d.get("config") else None
        return cls(**d)


@dataclass
class RLEvaluationResult:
    """
    Standardised evaluation output for an RL agent.

    Parameters
    ----------
    mean_return : float
        Mean episode return over eval episodes.
    std_return : float
        Standard deviation of episode returns.
    mean_length : float
        Mean episode length.
    returns : list of float
        Per-episode returns.
    lengths : list of int
        Per-episode lengths.
    metrics : dict
        Additional metrics (e.g. Sharpe, drawdown, win rate).
    """

    mean_return: float = 0.0
    std_return: float = 0.0
    mean_length: float = 0.0
    returns: List[float] = field(default_factory=list)
    lengths: List[int] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
