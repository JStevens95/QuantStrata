"""
QuantStrata Backtesting Module.

This module provides infrastructure for backtesting trading strategies:

Core Components:
- BacktestEngine: Main engine for running backtests
- BacktestResult: Results container with performance metrics
- PerformanceMetrics: Sharpe, Sortino, max drawdown, etc.

Data Providers:
- HistoricalDataProvider: Base class for data providers
- CsvDataProvider: Load data from CSV files
- DictDataProvider: In-memory data for testing

P&L Attribution:
- PnLAttribution: Time series of P&L breakdowns
- attribute_pnl_to_greeks: Decompose P&L into Greek factors

Example
-------
>>> from src.backtesting import BacktestEngine, BacktestConfig, DictDataProvider
>>> from datetime import date
>>>
>>> # Create data
>>> data = {
...     date(2024, 1, 1): {"AAPL": 150.0},
...     date(2024, 1, 2): {"AAPL": 152.0},
...     date(2024, 1, 3): {"AAPL": 151.0},
... }
>>> provider = DictDataProvider(data)
>>>
>>> # Define strategy
>>> def buy_and_hold(market, portfolio, context):
...     if context.step == 0:
...         # Buy on first day
...         return [SimpleOrder("AAPL", 100)]
...     return []
>>>
>>> # Run backtest
>>> engine = BacktestEngine(config=BacktestConfig())
>>> result = engine.run(
...     strategy=buy_and_hold,
...     data_provider=provider,
...     initial_capital=100_000,
... )
>>> print(result.metrics)
"""

# Note: Imports are deliberately minimal to avoid circular imports.
# Use explicit imports from submodules for full functionality.
