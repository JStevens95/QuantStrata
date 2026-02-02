"""
Performance Metrics for Backtesting.

This module provides standard performance metrics for evaluating trading strategies:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Win Rate
- Profit Factor

All metrics follow industry-standard definitions and can be annualized.

Example
-------
>>> from src.backtesting.core.metrics import PerformanceMetrics, compute_sharpe_ratio
>>> import numpy as np
>>>
>>> returns = np.array([0.01, -0.005, 0.02, 0.015, -0.01, 0.008])
>>> sharpe = compute_sharpe_ratio(returns, risk_free_rate=0.02, periods_per_year=252)
>>> print(f"Sharpe: {sharpe:.2f}")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


# =============================================================================
# Performance Metrics Container
# =============================================================================

@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """
    Container for performance metrics computed from a backtest.
    
    Attributes
    ----------
    total_return : float
        Total cumulative return over the period.
    annualized_return : float
        Annualized return (geometric).
    annualized_volatility : float
        Annualized volatility of returns.
    sharpe_ratio : float
        Annualized Sharpe ratio.
    sortino_ratio : float
        Annualized Sortino ratio (downside deviation).
    max_drawdown : float
        Maximum peak-to-trough drawdown.
    max_drawdown_duration : int
        Duration of longest drawdown (in periods).
    calmar_ratio : float
        Annualized return / max drawdown.
    win_rate : float
        Fraction of positive return periods.
    profit_factor : float
        Gross profits / gross losses.
    num_trades : int
        Number of trading periods.
    best_return : float
        Best single-period return.
    worst_return : float
        Worst single-period return.
    avg_return : float
        Average return per period.
    avg_win : float
        Average winning return.
    avg_loss : float
        Average losing return.
    risk_free_rate : float
        Risk-free rate used for calculations.
    periods_per_year : int
        Number of periods per year (for annualization).
    """
    
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    best_return: float
    worst_return: float
    avg_return: float
    avg_win: float
    avg_loss: float
    risk_free_rate: float
    periods_per_year: int
    
    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"PerformanceMetrics\n"
            f"  Total Return:     {self.total_return:+.2%}\n"
            f"  Ann. Return:      {self.annualized_return:+.2%}\n"
            f"  Ann. Volatility:  {self.annualized_volatility:.2%}\n"
            f"  Sharpe Ratio:     {self.sharpe_ratio:.2f}\n"
            f"  Sortino Ratio:    {self.sortino_ratio:.2f}\n"
            f"  Max Drawdown:     {self.max_drawdown:.2%}\n"
            f"  Calmar Ratio:     {self.calmar_ratio:.2f}\n"
            f"  Win Rate:         {self.win_rate:.1%}\n"
            f"  Profit Factor:    {self.profit_factor:.2f}\n"
            f"  Best Return:      {self.best_return:+.2%}\n"
            f"  Worst Return:     {self.worst_return:+.2%}\n"
            f"  Num Periods:      {self.num_trades}"
        )


# =============================================================================
# Individual Metric Functions
# =============================================================================

def compute_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Sharpe = (E[R] - Rf) / σ(R) × √(periods_per_year)
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns (not prices).
    risk_free_rate : float
        Annual risk-free rate (e.g., 0.02 for 2%).
    periods_per_year : int
        Number of periods per year (252 for daily, 12 for monthly).
    
    Returns
    -------
    float
        Annualized Sharpe ratio.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    
    # Convert annual risk-free to per-period
    rf_per_period = risk_free_rate / periods_per_year
    
    excess_returns = returns - rf_per_period
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess < 1e-12:
        return 0.0
    
    return float(mean_excess / std_excess * np.sqrt(periods_per_year))


def compute_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Compute annualized Sortino ratio.
    
    Sortino = (E[R] - Rf) / σ_downside(R) × √(periods_per_year)
    
    Uses downside deviation (std of returns below target) instead of total std.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        Number of periods per year.
    target_return : float
        Target return for downside calculation (default 0).
    
    Returns
    -------
    float
        Annualized Sortino ratio.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    
    rf_per_period = risk_free_rate / periods_per_year
    target_per_period = target_return / periods_per_year
    
    excess_returns = returns - rf_per_period
    mean_excess = np.mean(excess_returns)
    
    # Downside deviation: std of returns below target
    downside = np.minimum(returns - target_per_period, 0)
    downside_std = np.sqrt(np.mean(downside ** 2))
    
    if downside_std < 1e-12:
        return float("inf") if mean_excess > 0 else 0.0
    
    return float(mean_excess / downside_std * np.sqrt(periods_per_year))


def compute_max_drawdown(
    returns: np.ndarray,
) -> tuple[float, int, int, int]:
    """
    Compute maximum drawdown and its duration.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    
    Returns
    -------
    tuple
        (max_drawdown, duration, peak_idx, trough_idx)
        - max_drawdown: Maximum peak-to-trough decline (positive number)
        - duration: Number of periods from peak to recovery (or end)
        - peak_idx: Index of the peak before max drawdown
        - trough_idx: Index of the trough
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 1:
        return 0.0, 0, 0, 0
    
    # Compute cumulative returns (wealth index)
    wealth = np.cumprod(1 + returns)
    
    # Running maximum
    running_max = np.maximum.accumulate(wealth)
    
    # Drawdown at each point
    drawdowns = (running_max - wealth) / running_max
    
    # Maximum drawdown
    max_dd = float(np.max(drawdowns))
    trough_idx = int(np.argmax(drawdowns))
    
    # Find peak before trough
    peak_idx = int(np.argmax(wealth[:trough_idx + 1])) if trough_idx > 0 else 0
    
    # Find recovery (if any)
    recovery_mask = wealth[trough_idx:] >= wealth[peak_idx]
    if np.any(recovery_mask):
        recovery_idx = trough_idx + int(np.argmax(recovery_mask))
        duration = recovery_idx - peak_idx
    else:
        duration = len(returns) - peak_idx
    
    return max_dd, duration, peak_idx, trough_idx


def compute_calmar_ratio(
    returns: np.ndarray,
    periods_per_year: int = 252,
) -> float:
    """
    Compute Calmar ratio: annualized return / max drawdown.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    periods_per_year : int
        Number of periods per year.
    
    Returns
    -------
    float
        Calmar ratio.
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 1:
        return 0.0
    
    # Annualized return (geometric)
    total_return = np.prod(1 + returns) - 1
    n_years = len(returns) / periods_per_year
    if n_years <= 0:
        return 0.0
    
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    
    # Max drawdown
    max_dd, _, _, _ = compute_max_drawdown(returns)
    
    if max_dd < 1e-12:
        return float("inf") if ann_return > 0 else 0.0
    
    return float(ann_return / max_dd)


def compute_win_rate(returns: np.ndarray) -> float:
    """
    Compute win rate: fraction of positive returns.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    
    Returns
    -------
    float
        Win rate (0 to 1).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return 0.0
    return float(np.mean(returns > 0))


def compute_profit_factor(returns: np.ndarray) -> float:
    """
    Compute profit factor: sum(wins) / |sum(losses)|.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns.
    
    Returns
    -------
    float
        Profit factor (> 1 is profitable).
    """
    returns = np.asarray(returns, dtype=float)
    
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    total_gains = float(np.sum(gains)) if gains.size > 0 else 0.0
    total_losses = float(np.abs(np.sum(losses))) if losses.size > 0 else 0.0
    
    if total_losses < 1e-12:
        return float("inf") if total_gains > 0 else 0.0
    
    return total_gains / total_losses


# =============================================================================
# Compute All Metrics
# =============================================================================

def compute_all_metrics(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> PerformanceMetrics:
    """
    Compute all performance metrics from a return series.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of periodic returns (not prices).
    risk_free_rate : float
        Annual risk-free rate.
    periods_per_year : int
        Number of periods per year.
    
    Returns
    -------
    PerformanceMetrics
        All computed metrics.
    
    Examples
    --------
    >>> returns = np.random.randn(252) * 0.01 + 0.0003  # Simulated daily returns
    >>> metrics = compute_all_metrics(returns, risk_free_rate=0.02, periods_per_year=252)
    >>> print(metrics)
    """
    returns = np.asarray(returns, dtype=float)
    
    if returns.size == 0:
        return PerformanceMetrics(
            total_return=0.0,
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration=0,
            calmar_ratio=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            num_trades=0,
            best_return=0.0,
            worst_return=0.0,
            avg_return=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )
    
    # Total and annualized returns
    total_return = float(np.prod(1 + returns) - 1)
    n_years = len(returns) / periods_per_year
    if n_years > 0:
        ann_return = float((1 + total_return) ** (1 / n_years) - 1)
    else:
        ann_return = 0.0
    
    # Volatility
    ann_vol = float(np.std(returns, ddof=1) * np.sqrt(periods_per_year))
    
    # Ratios
    sharpe = compute_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = compute_sortino_ratio(returns, risk_free_rate, periods_per_year)
    max_dd, dd_duration, _, _ = compute_max_drawdown(returns)
    calmar = compute_calmar_ratio(returns, periods_per_year)
    
    # Trade statistics
    win_rate = compute_win_rate(returns)
    profit_factor = compute_profit_factor(returns)
    
    # Return statistics
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration=dd_duration,
        calmar_ratio=calmar,
        win_rate=win_rate,
        profit_factor=profit_factor,
        num_trades=len(returns),
        best_return=float(np.max(returns)),
        worst_return=float(np.min(returns)),
        avg_return=float(np.mean(returns)),
        avg_win=float(np.mean(wins)) if wins.size > 0 else 0.0,
        avg_loss=float(np.mean(losses)) if losses.size > 0 else 0.0,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )
