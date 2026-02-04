"""
Adapters for integrating deep hedging with other library components.

Provides:
- BacktestEngineAdapter: Run hedging agents through backtesting framework
- HistoricalDataAdapter: Prepare historical data for hedging environments

Usage:
    from src.deep_hedging.adapters import BacktestEngineAdapter, HistoricalDataAdapter
    
    # Run hedging agent in backtest
    adapter = BacktestEngineAdapter(agent=trained_agent)
    result = adapter.run_backtest(data_provider=historical_data)
"""

from src.deep_hedging.adapters.backtesting import (
    BacktestEngineAdapter,
    HedgingStrategy,
    BacktestConfig,
)
from src.deep_hedging.adapters.historical_data import (
    HistoricalDataAdapter,
    HistoricalMarketData,
)

__all__ = [
    "BacktestEngineAdapter",
    "HedgingStrategy",
    "BacktestConfig",
    "HistoricalDataAdapter",
    "HistoricalMarketData",
]
