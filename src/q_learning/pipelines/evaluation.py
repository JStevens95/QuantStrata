"""
Standardised RL evaluation for QuantStrata.

Provides evaluate_agent() that computes common metrics (mean return, Sharpe, drawdown,
episode stats) and returns RLEvaluationResult for comparison and serialisation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.q_learning.core.protocols import RLAgent, RLEnvironment
from src.q_learning.core.types import RLEvaluationResult

logger = logging.getLogger(__name__)


def _sharpe_ratio(returns: List[float], risk_free: float = 0.0) -> float:
    """Sharpe ratio (annualisation not applied; use for relative comparison)."""
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = variance ** 0.5 if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean - risk_free) / std


def _max_drawdown(cumulative_returns: List[float]) -> float:
    """Max drawdown from a list of cumulative returns (e.g. running sum of rewards)."""
    if not cumulative_returns:
        return 0.0
    peak = cumulative_returns[0]
    max_dd = 0.0
    for r in cumulative_returns:
        peak = max(peak, r)
        dd = peak - r
        if dd > max_dd:
            max_dd = dd
    return max_dd


def _win_rate(returns: List[float]) -> float:
    """Fraction of episodes with positive return."""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)


def evaluate_agent(
    agent: RLAgent,
    env: RLEnvironment,
    n_episodes: int = 10,
    max_steps_per_episode: int = 0,
    explore: bool = False,
    metrics: Optional[List[str]] = None,
) -> RLEvaluationResult:
    """
    Evaluate an RL agent over a fixed number of episodes (no exploration by default).

    Parameters
    ----------
    agent : RLAgent
        Agent to evaluate.
    env : RLEnvironment
        Environment (reset/step interface).
    n_episodes : int
        Number of evaluation episodes.
    max_steps_per_episode : int
        Max steps per episode (0 = no limit).
    explore : bool
        If True, agent uses exploration (e.g. epsilon-greedy); else greedy.
    metrics : list of str, optional
        Extra metrics to compute: "sharpe", "max_drawdown", "win_rate".
        Defaults to ["sharpe", "max_drawdown", "win_rate"].

    Returns
    -------
    RLEvaluationResult
        Mean/std return, mean length, per-episode returns/lengths, and metrics dict.

    Example
    -------
    >>> from src.q_learning.pipelines import evaluate_agent
    >>> result = evaluate_agent(agent, env, n_episodes=20)
    >>> print(result.mean_return, result.metrics["sharpe"])
    """
    if metrics is None:
        metrics = ["sharpe", "max_drawdown", "win_rate"]

    returns: List[float] = []
    lengths: List[int] = []

    for _ in range(n_episodes):
        state, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action = agent.select_action(state, training=False, explore=explore)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
            if terminated or truncated:
                break
            if max_steps_per_episode > 0 and steps >= max_steps_per_episode:
                break
        returns.append(total_reward)
        lengths.append(steps)

    mean_return = sum(returns) / len(returns) if returns else 0.0
    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns) if returns else 0.0
    std_return = variance ** 0.5
    mean_length = sum(lengths) / len(lengths) if lengths else 0.0

    computed: Dict[str, float] = {}
    if "sharpe" in metrics:
        computed["sharpe"] = _sharpe_ratio(returns)
    if "max_drawdown" in metrics:
        cum = []
        s = 0.0
        for r in returns:
            s += r
            cum.append(s)
        computed["max_drawdown"] = _max_drawdown(cum)
    if "win_rate" in metrics:
        computed["win_rate"] = _win_rate(returns)

    return RLEvaluationResult(
        mean_return=mean_return,
        std_return=std_return,
        mean_length=mean_length,
        returns=returns,
        lengths=lengths,
        metrics=computed,
    )


__all__ = ["evaluate_agent"]
