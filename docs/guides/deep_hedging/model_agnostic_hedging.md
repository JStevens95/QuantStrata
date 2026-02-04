# Model-Agnostic Hedging (Historical Data)

This guide describes how to train and evaluate hedging agents using **historical price paths** instead of a parametric model (GBM, Heston). That makes the policy **model-agnostic**: it learns from real (or replay) data without assuming specific dynamics.

---

## Overview

- **HistoricalHedgingEnv** (`src.deep_hedging.environments.historical`): RL environment that consumes pre-computed price (and optionally volatility) paths. No internal simulation model.
- **HistoricalDataAdapter** (`src.deep_hedging.adapters.historical_data`): Prepares historical data (e.g. from a market data provider or CSV) into the format expected by the historical environment.

Use model-agnostic hedging when:

- You want the agent to learn from actual market history.
- You do not want to commit to GBM/Heston for simulation.
- You are backtesting or doing out-of-sample evaluation on real data.

---

## 1. Data preparation

Prepare at least:

- **Prices:** array of shape `(n_paths, n_steps+1)` or `(n_steps+1,)` for a single path. Each row is one scenario (path).
- **Volatilities** (optional): same shape; if missing, a constant or simple estimator can be used.
- **Dates** (optional): for time-to-expiry and day count.

The **HistoricalDataAdapter** can wrap a provider that returns time series and produce the arrays the env needs.

```python
from src.deep_hedging.adapters.historical_data import HistoricalDataAdapter

# Example: wrap a list of (date, price) or a provider
adapter = HistoricalDataAdapter(
    provider=my_historical_provider,
    symbol="SPY",
    start_date=date(2024, 1, 1),
    end_date=date(2024, 12, 31),
)
paths, vols, dates = adapter.get_paths_and_vols()
```

---

## 2. Historical environment

```python
from src.deep_hedging.environments.historical import HistoricalHedgingEnv

env = HistoricalHedgingEnv(
    price_paths=paths,       # (n_paths, n_steps+1)
    volatility_paths=vols,   # optional
    strike=100.0,
    option_type="call",
    risk_free_rate=0.05,
    transaction_cost=0.001,
)

state = env.reset()
done = False
while not done:
    action = agent.select_action(state, training=False)
    state, reward, done, info = env.step(action)
```

The env steps through one path at a time (or minibatches of paths), so the agent sees many scenarios over training.

---

## 3. Training and evaluation

- **Training:** Use the same `HedgingTrainer` / `train_deep_hedging()` pattern with `HistoricalHedgingEnv` instead of `GBMHedgingEnv`. The agent learns from the empirical distribution of the provided paths.
- **Evaluation:** Run the trained agent on held-out historical paths or a separate backtest window; compare P&L and risk metrics to delta hedging.

---

## 4. Caveats

- **Limited paths:** With a single historical path you have one scenario; use bootstrap or multiple instruments to get more paths.
- **Regime change:** Past data may not reflect future regimes; combine with stress scenarios if needed.
- **Volatility:** If you do not have historical vol series, you must supply an estimate (e.g. rolling realised vol) so the agent state is consistent.

---

*Reference: [Deep hedging theory](../../reference/deep_hedging/theory.md).*
