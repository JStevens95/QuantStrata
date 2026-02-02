"""
Backtesting Core Module.

This module provides the core backtesting infrastructure:
- BacktestEngine: Main engine for running backtests
- BacktestResult: Results container with metrics
- BacktestConfig: Configuration for backtests
"""

from src.backtesting.core.engine import (
    BacktestEngine,
    BacktestResult,
    BacktestConfig,
)
from src.backtesting.core.metrics import (
    PerformanceMetrics,
    compute_sharpe_ratio,
    compute_max_drawdown,
    compute_sortino_ratio,
    compute_calmar_ratio,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "BacktestConfig",
    "PerformanceMetrics",
    "compute_sharpe_ratio",
    "compute_max_drawdown",
    "compute_sortino_ratio",
    "compute_calmar_ratio",
]
