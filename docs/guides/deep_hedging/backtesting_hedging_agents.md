# Backtesting Hedging Agents

This guide explains how to run a trained (or benchmark) deep hedging agent in backtest mode using historical or synthetic price paths.

---

## Overview

The library provides:

- **BacktestEngineAdapter** (`src.deep_hedging.adapters.backtesting`): Wraps a hedging agent so it can be driven by a price/volatility time series. It computes P&L, costs, and optional delta-hedge benchmark.
- **Pipeline** `deep_hedging.backtest_agent`: Loads an agent, builds or loads backtest data, runs the adapter, and stores the result in context/artifacts.
- **Example script** `examples/pipelines/run_backtest_hedging_agent.py`: Runs the pipeline with a default config (synthetic data, delta-hedge agent).

---

## 1. Using the pipeline

From code:

```python
from src.orchestrator.pipelines.deep_hedging.backtest_agent import build_pipeline
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.context import Context
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.artifacts.store import ArtifactStore

config = RunConfig(
    pipeline="deep_hedging.backtest_agent",
    params={
        "deep_hedging": {
            "backtest": {
                "n_days": 63,
                "spot_initial": 100.0,
                "volatility": 0.20,
                "maturity_days": 30,
                "option_type": "call",
                "transaction_cost": 0.001,
                "risk_free_rate": 0.05,
                "seed": 42,
            },
        },
    },
)

ctx = Context(
    run_id="backtest-1",
    cfg=config,
    logger=None,
    artifact_store=ArtifactStore(artifacts_root=Path("artifacts")),
)
pipeline = build_pipeline()
ctx = PipelineRunner().run(pipeline, ctx)

result = ctx.state.get("backtest_result")
if result:
    print(f"Total P&L: {result.total_pnl:.2f}, Sharpe: {result.sharpe_ratio:.2f}")
```

If no trained agent is in `ctx.state["deep_agent"]`, the pipeline uses **DeltaHedgingAgent** as a benchmark so the backtest still runs.

---

## 2. Using the adapter directly

For full control (e.g. your own data provider), use the adapter:

```python
from datetime import date
import numpy as np
from src.deep_hedging.adapters.backtesting import (
    BacktestEngineAdapter,
    BacktestConfig,
    OptionParams,
)
from src.deep_hedging.agents.delta import DeltaHedgingAgent

# Your price/vol series (e.g. from historical provider)
prices = np.array([100.0, 101.2, 99.5, ...])  # length N
volatilities = np.array([0.20, 0.21, 0.19, ...])  # length N
dates = [date(2025, 1, 1), date(2025, 1, 2), ...]

option_params = OptionParams(
    strike=100.0,
    maturity=dates[-1],
    option_type="call",
    notional=1.0,
)

config = BacktestConfig(
    transaction_cost=0.001,
    maturity_days=30,
    option_type="call",
)

agent = DeltaHedgingAgent()  # or your trained DeepHedgingAgent
adapter = BacktestEngineAdapter(agent=agent, config=config)

result = adapter.run_backtest(
    prices=prices,
    volatilities=volatilities,
    dates=dates,
    option_params=option_params,
    risk_free_rate=0.05,
    run_benchmark=True,
)

print(result.summary())
```

---

## 3. Config parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_days` | 63 | Number of days for synthetic path |
| `spot_initial` | 100.0 | Initial spot |
| `volatility` | 0.20 | Constant vol for synthetic path |
| `maturity_days` | 30 | Option maturity in days |
| `option_type` | "call" | "call" or "put" |
| `transaction_cost` | 0.001 | Proportional cost per trade |
| `risk_free_rate` | 0.05 | Risk-free rate |
| `seed` | 42 | RNG seed for synthetic data |

---

## 4. Result fields

`HedgingBacktestResult` includes:

- **total_pnl**, **hedging_pnl**, **option_pnl**, **total_cost**
- **sharpe_ratio**, **max_drawdown**
- **benchmark_pnl**, **outperformance** (vs delta hedge when `run_benchmark=True`)
- **pnl_history**, **position_history**

---

*Reference: [Deep hedging theory](../../reference/deep_hedging/theory.md).*
