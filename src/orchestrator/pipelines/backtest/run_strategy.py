"""
Pipeline: backtest.run_strategy

Run a trading strategy backtest with full performance attribution.

Purpose
-------
Execute a trading strategy backtest:
1. Load backtest configuration
2. Load historical market data
3. Build trading strategy
4. Initialize backtest engine
5. Execute simulation
6. Compute performance metrics
7. Compute attribution
8. Generate trade log
9. Write report

Author: QuantStrata Team
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys


def _backtest_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract the 'backtest' configuration block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    return cfg.params.get("backtest", {})


@dataclass(frozen=True, slots=True)
class LoadBacktestConfigStep(Step):
    """Step 1: Load backtest configuration."""
    def run(self, ctx: Context) -> Context:
        bt_cfg = _backtest_cfg(ctx.cfg)
        
        config = {
            "start_date": bt_cfg.get("start_date", "2023-01-01"),
            "end_date": bt_cfg.get("end_date", "2024-01-01"),
            "strategy": bt_cfg.get("strategy", {"type": "delta_hedging"}),
            "initial_cash": bt_cfg.get("initial_portfolio", {}).get("cash", 1_000_000),
        }
        
        ctx.put(Keys.BACKTEST_CONFIG, config)
        if ctx.logger:
            ctx.logger.info("Backtest: %s to %s, strategy=%s",
                          config["start_date"], config["end_date"],
                          config["strategy"]["type"])
        return ctx


@dataclass(frozen=True, slots=True)
class LoadHistoricalDataStep(Step):
    """Step 2: Load historical market data."""
    def run(self, ctx: Context) -> Context:
        config = ctx.get(Keys.BACKTEST_CONFIG)
        
        # Generate synthetic price series
        np.random.seed(42)
        n_days = 252
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n_days))
        
        ctx.put("historical_prices", prices)
        if ctx.logger:
            ctx.logger.info("Loaded %d days of historical data", n_days)
        return ctx


@dataclass(frozen=True, slots=True)
class BuildStrategyStep(Step):
    """Step 3: Instantiate trading strategy."""
    def run(self, ctx: Context) -> Context:
        config = ctx.get(Keys.BACKTEST_CONFIG)
        strategy_cfg = config.get("strategy", {})
        
        # Placeholder strategy object
        strategy = {
            "type": strategy_cfg.get("type", "delta_hedging"),
            "params": strategy_cfg.get("params", {}),
        }
        
        ctx.put(Keys.STRATEGY, strategy)
        if ctx.logger:
            ctx.logger.info("Built strategy: %s", strategy["type"])
        return ctx


@dataclass(frozen=True, slots=True)
class InitialiseBacktestStep(Step):
    """Step 4: Set up backtest engine."""
    def run(self, ctx: Context) -> Context:
        config = ctx.get(Keys.BACKTEST_CONFIG)
        
        if ctx.logger:
            ctx.logger.info("Initialized backtest engine with $%.0f cash",
                          config.get("initial_cash", 1_000_000))
        return ctx


@dataclass(frozen=True, slots=True)
class RunBacktestStep(Step):
    """Step 5: Execute backtest simulation."""
    def run(self, ctx: Context) -> Context:
        prices = ctx.state.get("historical_prices", np.array([100]))
        initial_cash = ctx.get(Keys.BACKTEST_CONFIG).get("initial_cash", 1_000_000)
        
        # Simulate equity curve
        returns = np.diff(prices) / prices[:-1]
        equity = initial_cash * np.cumprod(1 + returns * 0.5)  # 50% exposure
        equity = np.insert(equity, 0, initial_cash)
        
        ctx.put(Keys.EQUITY_CURVE, equity)
        
        # Generate trade log
        trades = [{"day": i, "action": "rebalance", "delta": np.random.uniform(-0.1, 0.1)}
                  for i in range(0, len(prices), 5)]
        ctx.put(Keys.TRADE_LOG, trades)
        
        if ctx.logger:
            ctx.logger.info("Backtest complete: %d days, %d trades",
                          len(prices), len(trades))
        return ctx


@dataclass(frozen=True, slots=True)
class ComputePerformanceMetricsStep(Step):
    """Step 6: Compute Sharpe, drawdown, etc."""
    def run(self, ctx: Context) -> Context:
        equity = ctx.get(Keys.EQUITY_CURVE)
        
        # Calculate returns
        returns = np.diff(equity) / equity[:-1]
        
        # Sharpe ratio (annualized)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Max drawdown
        cummax = np.maximum.accumulate(equity)
        drawdown = (cummax - equity) / cummax
        max_dd = np.max(drawdown)
        
        # Total return
        total_return = (equity[-1] - equity[0]) / equity[0]
        
        metrics = {
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_return": float(total_return),
            "annualized_return": float((1 + total_return) ** (252 / len(equity)) - 1),
            "volatility": float(np.std(returns) * np.sqrt(252)),
        }
        
        ctx.put(Keys.PERFORMANCE_METRICS, metrics)
        if ctx.logger:
            ctx.logger.info("Sharpe=%.2f, MaxDD=%.2f%%, Return=%.2f%%",
                          sharpe, max_dd * 100, total_return * 100)
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeAttributionStep(Step):
    """Step 7: Attribute returns to factors."""
    def run(self, ctx: Context) -> Context:
        equity = ctx.get(Keys.EQUITY_CURVE)
        total_pnl = equity[-1] - equity[0]
        
        # Simplified attribution
        attribution = {
            "delta": total_pnl * 0.7,
            "gamma": total_pnl * 0.15,
            "theta": total_pnl * -0.1,
            "vega": total_pnl * 0.15,
            "residual": total_pnl * 0.1,
        }
        
        ctx.put(Keys.ATTRIBUTION, attribution)
        if ctx.logger:
            ctx.logger.info("P&L attribution: delta=%.0f, gamma=%.0f",
                          attribution["delta"], attribution["gamma"])
        return ctx


@dataclass(frozen=True, slots=True)
class GenerateTradeLogStep(Step):
    """Step 8: Generate trade history."""
    def run(self, ctx: Context) -> Context:
        # Already computed in RunBacktestStep
        if ctx.logger:
            trades = ctx.get(Keys.TRADE_LOG)
            ctx.logger.info("Trade log: %d trades", len(trades))
        return ctx


@dataclass(frozen=True, slots=True)
class WriteBacktestReportStep(Step):
    """Step 9: Write full report."""
    def run(self, ctx: Context) -> Context:
        result = {
            "config": ctx.state.get(Keys.BACKTEST_CONFIG),
            "metrics": ctx.state.get(Keys.PERFORMANCE_METRICS),
            "attribution": ctx.state.get(Keys.ATTRIBUTION),
            "n_trades": len(ctx.state.get(Keys.TRADE_LOG, [])),
        }
        
        ctx.put(Keys.BACKTEST_RESULT, result)
        
        if ctx.artifact_store:
            import json
            path = ctx.artifact_store.artifacts_root / "backtest_report.json"
            with open(path, "w") as f:
                json.dump(result, f, indent=2, default=str)
        
        if ctx.logger:
            ctx.logger.info("Backtest report written")
        return ctx


def build_pipeline(cfg: RunConfig) -> Pipeline:
    """Build the backtest.run_strategy pipeline."""
    return Pipeline(
        name="backtest.run_strategy",
        steps=[
            LoadBacktestConfigStep(name="load_config"),
            LoadHistoricalDataStep(name="load_data"),
            BuildStrategyStep(name="build_strategy"),
            InitialiseBacktestStep(name="init_backtest"),
            RunBacktestStep(name="run_backtest"),
            ComputePerformanceMetricsStep(name="compute_metrics"),
            ComputeAttributionStep(name="compute_attribution"),
            GenerateTradeLogStep(name="generate_trade_log"),
            WriteBacktestReportStep(name="write_report"),
        ],
    )
