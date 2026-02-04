# Q-Learning Runners Reference

Technical reference for running trained RL agents: backtest and live. See also [RL Framework](rl_framework.md) and [Environments](environments.md).

---

## Overview

Runners execute a trained **RLAgent** in an **RLEnvironment** and collect results. They do not train the agent; they evaluate or deploy it.

| Runner | Module | Purpose |
|--------|--------|---------|
| **BacktestRunner** | `q_learning.runners.backtest` | Run agent over multiple episodes on historical/simulated data; compute metrics |
| **LiveRunner** | `q_learning.runners.live` | Run agent with streaming engine for live or paper trading |
| **BaseRunner** | `q_learning.runners.base` | Base class and protocol for runners |

---

## BacktestRunner

**Module:** `src.q_learning.runners.backtest`

- **Constructor:** `BacktestRunner(agent, env, config=None)` with `BacktestConfig`.
- **Config:** `BacktestConfig`: `n_episodes`, `episode_seeds`, `use_random_starts`, `start_indices`, `parallel`, `n_jobs`, `compute_sharpe`, `compute_drawdown`, `risk_free_rate`, `benchmark_agent`.
- **Method:** `run(**kwargs) -> BacktestResult`.
- **Result:** `BacktestResult`: `episodes`, `pnl_returns`, `sharpe_ratio`, `max_drawdown`, `benchmark_results`, `outperformance`, plus `mean_pnl_return`, `std_pnl_return`, `summary()`.

Example:

```python
from src.q_learning.runners.backtest import BacktestRunner, BacktestConfig
from src.q_learning.environments import TradingEnvironment

env = TradingEnvironment(data_provider=provider, config=TradingEnvConfig())
agent = load_agent("path/to/agent", agent_factory=MyAgent)
runner = BacktestRunner(agent=agent, env=env, config=BacktestConfig(n_episodes=100))
result = runner.run()
print(result.mean_pnl_return, result.sharpe_ratio)
```

---

## LiveRunner

**Module:** `src.q_learning.runners.live`

- **Constructor:** `LiveRunner(agent, env, config=None)` with `LiveConfig`.
- **Purpose:** Run agent in a streaming (live or paper) environment; typically env is a `StreamingEnvironment` that consumes market updates and returns state/reward from execution.
- **Method:** `run(**kwargs)` — runs until stopped or end of stream.

Use for deployment after validating the agent in backtest.

---

## BaseRunner

**Module:** `src.q_learning.runners.base`

- **Protocol:** Runners hold `agent` and `env`, and implement `run()` returning a `RunResult`-like object.
- **RunResult:** Base result type with `episodes`, `n_episodes`, and optional `summary()`.
- **EpisodeResult:** Per-episode `episode_id`, `total_reward`, `n_steps`, `rewards`, `actions`, `info`.

---

## Orchestrator integration

- **Pipeline `rl.backtest_agent`:** Loads agent (from state or artifact path), builds or loads env (e.g. `BaseEnv` from config), runs `BacktestRunner`, and stores `BacktestResult` in context state under `RL_BACKTEST_RESULT`.
- **Pipeline `rl.deploy_agent`:** Loads agent from artifact path and puts it in state (`RL_AGENT`) for use by downstream pipelines (e.g. backtest or live).

See [Deploying RL Agents](../../guides/q_learning/deploying_rl_agents.md) for step-by-step usage.
