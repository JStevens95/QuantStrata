# Q-Learning & RL Framework Guide

User guide for the QuantStrata Q-Learning / RL integration (Phase 7.2): how to implement an agent and environment, train, evaluate, and deploy.

---

## When to use what

| Goal | Entry point | Docs |
|------|-------------|------|
| Train an RL agent with a generic loop | `run_training(agent, env, config)` | Below, Reference |
| Evaluate an agent (returns, Sharpe, drawdown) | `evaluate_agent(agent, env, n_episodes)` | Reference: rl_framework.md |
| Save/load agent and run inference | `save_agent()`, `load_agent()`, `select_action()` | Reference: rl_framework.md |
| Use a minimal env for tests or as template | `BaseEnv` from `q_learning.environments` | Reference: rl_framework.md |

---

## Quick start: minimal agent + BaseEnv

```python
from src.q_learning.core import RLAgent, RLEnvironment, RLTrainingConfig, Transition
from src.q_learning.pipelines import run_training, evaluate_agent
from src.q_learning.environments import BaseEnv

# Minimal agent: random action, no learning (for structure check)
class RandomAgent:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions
    def select_action(self, state, *, training=False, explore=True):
        return __import__("random").randint(0, self.n_actions - 1)
    def update(self, transitions=None, batch=None):
        return None
    def get_parameters(self):
        return {"n_actions": self.n_actions}
    def set_parameters(self, params):
        self.n_actions = params.get("n_actions", self.n_actions)

env = BaseEnv(state_dim=2, n_actions=3, max_steps=50)
agent = RandomAgent(n_actions=3)
config = RLTrainingConfig(n_episodes=20, max_steps_per_episode=50, log_every=5)
result = run_training(agent, env, config)
print(result.best_episode_return, result.episode_returns)

eval_result = evaluate_agent(agent, env, n_episodes=5)
print(eval_result.mean_return, eval_result.metrics)
```

---

## Implementing an agent

Your agent must implement `RLAgent`:

- `select_action(state, training=False, explore=True)` — return action (e.g. int or array).
- `update(transitions=None, batch=None)` — optional; return a dict of metrics (e.g. loss).
- `get_parameters()` / `set_parameters(params)` — for checkpointing and deployment.

Use the same pattern as Phase 7.1 ML: any compliant agent can be trained via `run_training()` and evaluated via `evaluate_agent()`.

---

## Implementing an environment

Your environment must implement `RLEnvironment`:

- `reset(seed=None, options=None)` → (state, info).
- `step(action)` → (state, reward, terminated, truncated, info).

For trading/hedging, wrap the backtesting engine, pricers, and market data; expose state (e.g. positions, Greeks, mid) and actions (e.g. hedge ratio, order size) and reward (e.g. PnL or risk-adjusted return).

---

## Tutorials and reference

- **Technical reference:** [RL Framework Reference](../../reference/q_learning/rl_framework.md).
- **Progress:** [Phase 7.2 Q-Learning](../../development/progress/phase_7_2_q_learning.md).
