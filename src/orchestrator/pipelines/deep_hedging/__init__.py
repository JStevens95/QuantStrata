"""
Deep Hedging Pipelines.

- backtest_agent: Run a trained deep hedging agent in backtest mode
  (synthetic or historical data).
"""

from src.orchestrator.pipelines.deep_hedging.backtest_agent import (
    build_pipeline as backtest_hedging_agent,
)

__all__ = ["backtest_hedging_agent"]
