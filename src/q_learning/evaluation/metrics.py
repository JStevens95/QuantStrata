"""
Common RL evaluation metrics: returns, Sharpe, drawdown, win rate.

Used by pipelines.evaluation and available for custom evaluation/reporting.
"""

from __future__ import annotations

from typing import List


def sharpe_ratio(returns: List[float], risk_free: float = 0.0) -> float:
    """
    Sharpe ratio of a list of (e.g. episode) returns.

    No annualisation; use for relative comparison across runs.
    """
    if not returns:
        return 0.0
    mean = sum(returns) / len(returns)
    variance = sum((r - mean) ** 2 for r in returns) / len(returns)
    std = variance ** 0.5 if variance > 0 else 0.0
    if std == 0:
        return 0.0
    return (mean - risk_free) / std


def max_drawdown(cumulative_returns: List[float]) -> float:
    """
    Maximum drawdown from a sequence of cumulative returns (e.g. running sum of rewards).
    """
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


def win_rate(returns: List[float]) -> float:
    """Fraction of episodes (or periods) with positive return."""
    if not returns:
        return 0.0
    return sum(1 for r in returns if r > 0) / len(returns)
