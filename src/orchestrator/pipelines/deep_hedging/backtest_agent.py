"""
Pipeline: deep_hedging.backtest_agent

Run a trained (or benchmark) deep hedging agent in backtest mode using
synthetic or historical price/volatility paths.

Purpose
-------
1. Load or create hedging agent (from state or use delta-hedge benchmark)
2. Build synthetic or load historical price/vol data
3. Run BacktestEngineAdapter
4. Store HedgingBacktestResult in state and optionally as artifact
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, Optional

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step
from src.orchestrator.core.state_keys import StateKeys as Keys


def _deep_hedging_backtest_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract deep_hedging.backtest config block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    dh = cfg.params.get("deep_hedging", {})
    return dh.get("backtest", {})


@dataclass(slots=True)
class LoadAgentOrBenchmarkStep(Step):
    """Step 1: Load trained agent from state/path or use delta-hedge benchmark."""
    def run(self, ctx: Context) -> Context:
        agent = ctx.state.get(Keys.DEEP_AGENT)
        if agent is None:
            try:
                from src.deep_hedging.agents.delta import DeltaHedgingAgent
                agent = DeltaHedgingAgent()
                if ctx.logger:
                    ctx.logger.info("Using DeltaHedgingAgent as benchmark (no trained agent in state)")
            except ImportError:
                if ctx.logger:
                    ctx.logger.warning("deep_hedging not available; agent set to None")
        ctx.put("hedging_agent", agent)
        return ctx


@dataclass(slots=True)
class BuildBacktestDataStep(Step):
    """Step 2: Build synthetic price/vol paths for backtest (or load from provider)."""
    def run(self, ctx: Context) -> Context:
        cfg = _deep_hedging_backtest_cfg(ctx.cfg)
        n_days = int(cfg.get("n_days", 63))
        spot_0 = float(cfg.get("spot_initial", 100.0))
        vol_0 = float(cfg.get("volatility", 0.20))
        mu = float(cfg.get("drift", 0.05))
        seed = int(cfg.get("seed", 42))
        rng = np.random.default_rng(seed)
        dt = 1.0 / 252.0
        z = rng.standard_normal(n_days)
        log_returns = (mu - 0.5 * vol_0**2) * dt + vol_0 * np.sqrt(dt) * z
        prices = spot_0 * np.exp(np.cumsum(np.concatenate([[0], log_returns])))
        volatilities = np.full(n_days + 1, vol_0)
        base = date.today()
        dates = [base + timedelta(days=i) for i in range(n_days + 1)]
        ctx.put("backtest_prices", prices)
        ctx.put("backtest_volatilities", volatilities)
        ctx.put("backtest_dates", dates)
        if ctx.logger:
            ctx.logger.info("Built synthetic backtest data: n_days=%s, spot_0=%s", n_days, spot_0)
        return ctx


@dataclass(slots=True)
class RunBacktestStep(Step):
    """Step 3: Run BacktestEngineAdapter and store result."""
    def run(self, ctx: Context) -> Context:
        agent = ctx.state.get("hedging_agent")
        if agent is None:
            if ctx.logger:
                ctx.logger.warning("No hedging agent; skipping backtest")
            ctx.put(Keys.BACKTEST_RESULT, None)
            return ctx
        prices = ctx.state.get("backtest_prices")
        volatilities = ctx.state.get("backtest_volatilities")
        dates = ctx.state.get("backtest_dates")
        if prices is None or volatilities is None:
            if ctx.logger:
                ctx.logger.warning("Missing backtest data; skipping backtest")
            ctx.put(Keys.BACKTEST_RESULT, None)
            return ctx
        cfg = _deep_hedging_backtest_cfg(ctx.cfg)
        try:
            from src.deep_hedging.adapters.backtesting import (
                BacktestConfig,
                BacktestEngineAdapter,
                OptionParams,
            )
            option_params = OptionParams(
                strike=float(prices[0]),
                maturity=dates[min(int(cfg.get("maturity_days", 30)), len(dates) - 1)],
                option_type=str(cfg.get("option_type", "call")),
                notional=1.0,
            )
            adapter_cfg = BacktestConfig(
                transaction_cost=float(cfg.get("transaction_cost", 0.001)),
                maturity_days=int(cfg.get("maturity_days", 30)),
                option_type=option_params.option_type,
            )
            adapter = BacktestEngineAdapter(agent=agent, config=adapter_cfg)
            result = adapter.run_backtest(
                prices=prices,
                volatilities=volatilities,
                dates=dates,
                option_params=option_params,
                risk_free_rate=float(cfg.get("risk_free_rate", 0.05)),
                run_benchmark=True,
            )
            ctx.put(Keys.BACKTEST_RESULT, result)
            if ctx.logger:
                ctx.logger.info(
                    "Backtest complete: total_pnl=%.2f, sharpe=%.2f, outperformance=%s",
                    result.total_pnl, result.sharpe_ratio, result.outperformance,
                )
            if ctx.artifact_store and hasattr(result, "summary"):
                summary = result.summary()
                path = ctx.artifact_store.artifacts_root / "hedging_backtest_summary.json"
                import json
                with open(path, "w") as f:
                    json.dump(summary, f, indent=2)
        except ImportError as e:
            if ctx.logger:
                ctx.logger.warning("Could not run backtest: %s", e)
            ctx.put(Keys.BACKTEST_RESULT, None)
        return ctx


def build_pipeline() -> Pipeline:
    """Build the deep_hedging.backtest_agent pipeline."""
    return Pipeline(
        name="deep_hedging.backtest_agent",
        steps=[
            LoadAgentOrBenchmarkStep(name="load_agent_or_benchmark"),
            BuildBacktestDataStep(name="build_backtest_data"),
            RunBacktestStep(name="run_backtest"),
        ],
    )
