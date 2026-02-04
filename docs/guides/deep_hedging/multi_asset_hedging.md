# Multi-Asset Deep Hedging

This guide describes how to hedge a portfolio of options on **multiple underlyings** using the multi-asset hedging environment.

---

## Overview

**MultiAssetHedgingEnv** (`src.deep_hedging.environments.multi_asset`) extends the single-asset GBM hedging environment to:

- Simulate several correlated underlyings (e.g. basket or multi-name options).
- Let the agent choose hedge ratios for each underlying.
- Account for cross-gamma and correlation in the reward.

Use it when the option payoff depends on more than one asset (e.g. spread, basket, worst-of).

---

## 1. When to use multi-asset

- **Basket options:** Payoff on weighted sum or average of assets.
- **Spread options:** Payoff on difference of two underlyings.
- **Worst-of / best-of:** Payoff on min/max of several assets.
- **Correlation-sensitive books:** You want one policy that hedges multiple deltas and respects correlation.

---

## 2. Environment API

```python
from src.deep_hedging.environments.multi_asset import MultiAssetHedgingEnv

# Build env with n_underlyings, correlation matrix, strikes, etc.
env = MultiAssetHedgingEnv(
    n_underlyings=2,
    spot_initial=[100.0, 50.0],
    volatilities=[0.20, 0.25],
    correlation_matrix=[[1.0, 0.5], [0.5, 1.0]],
    risk_free_rate=0.05,
    maturity=0.25,
    n_steps=63,
    option_type="basket",  # or "spread", "worst_of", etc.
    # ... other config
)

state = env.reset()
done = False
while not done:
    action = agent.select_action(state, training=False)  # vector of hedge ratios
    state, reward, done, info = env.step(action)
```

State typically includes spot levels, time to expiry, current positions, and optionally deltas/gammas per asset. Action is a vector of hedge ratios (one per underlying).

---

## 3. Training

Use the same training loop as single-asset deep hedging: `HedgingTrainer` or `train_deep_hedging()` with `MultiAssetHedgingEnv` and a policy that outputs a vector of actions. Cost and risk measures apply to the combined P&L across underlyings.

---

## 4. Backtesting

To backtest a multi-asset agent on historical data, you need:

- Historical paths for all underlyings (and, if used, vol surfaces).
- **HistoricalDataAdapter** or a custom data loader that returns arrays of shape `(n_days, n_underlyings)` for prices and optionally vols.

The current **BacktestEngineAdapter** is written for a single underlying. Extending it to multi-asset would require passing multiple price/vol series and option params per asset; that is a natural future extension.

---

*Reference: [Deep hedging theory](../../reference/deep_hedging/theory.md).*
