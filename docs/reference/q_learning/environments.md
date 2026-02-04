# Q-Learning Environments Reference

Technical reference for RL environments: Trading, Hedging, and Streaming. See also [RL Framework](rl_framework.md) for protocols and training.

---

## Overview

Environments implement the `RLEnvironment` protocol: `reset(*, seed=None, options=None) -> (state, info)` and `step(action) -> (state, reward, terminated, truncated, info)`. They provide the bridge between library infrastructure (backtesting, pricers, streaming) and the generic RL training/deployment pipelines.

| Environment | Module | Purpose |
|-------------|--------|---------|
| **TradingEnvironment** | `q_learning.environments.trading` | Wrap historical price data for trading-strategy RL |
| **HedgingEnvironment** | `q_learning.environments.hedging` | Wrap pricers and market for delta-hedging RL |
| **StreamingEnvironment** | `q_learning.environments.streaming` | Wrap streaming engine for live execution |
| **BaseEnv** | `q_learning.environments.base` | Minimal env for tests and demos |

---

## TradingEnvironment

**Module:** `src.q_learning.environments.trading`

- **Config:** `TradingEnvConfig`: `initial_capital`, `transaction_cost`, `slippage`, `max_steps`, `lookback_window`, `action_type` (discrete/continuous), `n_discrete_actions`, etc.
- **Data:** Requires a data provider (e.g. `SimpleDataProvider(prices)` with shape `(n_steps,)` or `(n_steps, n_assets)`).
- **State:** Typically observation dict or array (prices, position, PnL, cash, etc. depending on config).
- **Action:** Discrete (e.g. position target buckets) or continuous (e.g. weight or delta).
- **Reward:** e.g. period PnL or risk-adjusted return.

Use with `BacktestRunner` for evaluation and with `run_training()` for training.

---

## HedgingEnvironment

**Module:** `src.q_learning.environments.hedging`

- Wraps pricers and market simulation for learning hedge ratios (e.g. delta-hedging agents).
- State typically includes underlying price, time to expiry, position, Greeks, PnL.
- Action: hedge ratio or hedge quantity.
- Reward: risk-adjusted PnL minus transaction costs.

Use with deep hedging agents or generic RL agents that learn hedging policies.

---

## StreamingEnvironment

**Module:** `src.q_learning.environments.streaming`

- Wraps the streaming engine for **live or paper** execution.
- Same agent interface as backtesting; environment feeds live (or replayed) market updates and returns rewards/state from executed trades.

Use with `LiveRunner` for deployment.

---

## BaseEnv

**Module:** `src.q_learning.environments.base`

- **Purpose:** Minimal `RLEnvironment` for unit tests and pipeline demos.
- **State:** 1D array of length `state_dim`.
- **Action:** Integer in `[0, n_actions - 1]`.
- **Transition:** `next_state = state + delta + noise`; reward configurable (e.g. negative distance from target).
- **Episode end:** When `step >= max_steps` or optional done condition.

Use when no market data or pricer is needed (e.g. `rl.backtest_agent` pipeline with no agent path uses `BaseEnv` + stub agent).
