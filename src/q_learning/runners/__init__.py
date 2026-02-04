"""
Agent runners for executing trained RL agents.

Provides utilities for deploying RL agents in:
- Backtesting: Historical data replay
- Live/Paper: Real-time execution via streaming

Usage:
    from src.q_learning.runners import BacktestRunner, LiveRunner
    
    # Run backtesting
    runner = BacktestRunner(
        agent=trained_agent,
        env=trading_env,
    )
    results = runner.run(n_episodes=10)
    
    # Run live
    live_runner = LiveRunner(
        agent=trained_agent,
        env=streaming_env,
    )
    live_runner.start()
"""

from src.q_learning.runners.base import BaseRunner, RunnerConfig, RunResult
from src.q_learning.runners.backtest import BacktestRunner, BacktestConfig, BacktestResult
from src.q_learning.runners.live import LiveRunner, LiveConfig

__all__ = [
    # Base
    "BaseRunner",
    "RunnerConfig",
    "RunResult",
    # Backtest
    "BacktestRunner",
    "BacktestConfig",
    "BacktestResult",
    # Live
    "LiveRunner",
    "LiveConfig",
]
