# Deploying RL Agents

This guide shows how to backtest and deploy trained Q-Learning / RL agents using the library’s runners and orchestrator pipelines.

---

## Prerequisites

- A trained agent saved with `save_agent(agent, artifact_dir, ...)` (see [RL Framework](rl_framework.md)).
- For backtest: an environment (e.g. `TradingEnvironment` with data provider, or `BaseEnv` for demos).
- For live: a `StreamingEnvironment` and streaming engine.

---

## 1. Backtest an agent (standalone)

Use `BacktestRunner` with your agent and environment:

```python
from src.q_learning.runners.backtest import BacktestRunner, BacktestConfig
from src.q_learning.pipelines.inference import load_agent
from src.q_learning.environments.trading import TradingEnvironment, SimpleDataProvider, TradingEnvConfig

# Load agent (you must provide the agent class/factory used when saving)
agent = load_agent("path/to/artifact", agent_factory=MyAgentClass, factory_kwargs={})

# Build environment (e.g. historical prices)
prices = ...  # shape (n_steps,) or (n_steps, n_assets)
provider = SimpleDataProvider(prices)
env = TradingEnvironment(data_provider=provider, config=TradingEnvConfig())

# Run backtest
runner = BacktestRunner(
    agent=agent,
    env=env,
    config=BacktestConfig(n_episodes=100, compute_sharpe=True),
)
result = runner.run()

print("Mean return:", result.mean_pnl_return)
print("Sharpe:", result.sharpe_ratio)
print("Max drawdown:", result.max_drawdown)
```

---

## 2. Backtest via orchestrator pipeline

Use the `rl.backtest_agent` pipeline so the runner and config are driven by `RunConfig`:

```python
from src.orchestrator.runtime.entrypoints import run_pipeline_from_config
from src.orchestrator.config.schemas import RunConfig

config = RunConfig(
    pipeline="rl.backtest_agent",
    params={
        "rl": {
            "agent_path": "saved_agent",   # relative to artifacts root, or put agent in state
            "agent_factory": MyAgentClass,  # required if agent_path set
            "backtest": {
                "n_episodes": 50,
                "state_dim": 1,
                "n_actions": 3,
                "max_steps": 50,
                "seed": 42,
            },
        },
    },
)
ctx = run_pipeline_from_config(config)
result = ctx.state.get("rl_backtest_result")
```

If you do not set `agent_path` or `agent_factory`, the pipeline uses a **stub agent** and **BaseEnv** so you can run a demo backtest without a trained agent.

---

## 3. Load agent for deployment (rl.deploy_agent)

To load a saved agent into context for use by another pipeline (e.g. backtest or a live runner):

```python
config = RunConfig(
    pipeline="rl.deploy_agent",
    params={
        "rl": {
            "agent_path": "prod_agent_v1",
            "agent_factory": MyAgentClass,
            "agent_factory_kwargs": {},
        },
    },
)
ctx = run_pipeline_from_config(config)
agent = ctx.state.get("rl_agent")
# Use agent with BacktestRunner or LiveRunner, or pass context to rl.backtest_agent
```

---

## 4. Run live (conceptual)

After backtesting, deploy with `LiveRunner` and a `StreamingEnvironment`:

```python
from src.q_learning.runners.live import LiveRunner, LiveConfig
from src.q_learning.environments.streaming import StreamingEnvironment

env = StreamingEnvironment(streaming_engine=engine, ...)
agent = load_agent("path/to/agent", agent_factory=MyAgentClass)
runner = LiveRunner(agent=agent, env=env, config=LiveConfig())
runner.run()
```

Details depend on your streaming engine and brokerage adapter; see [Streaming and Live Data](../../reference/streaming_live_data.md) and [Environments](../../reference/q_learning/environments.md).

---

## 5. Example script

See `examples/pipelines/run_deploy_rl_agent.py` for a runnable example that either runs `rl.deploy_agent` or `rl.backtest_agent` and prints the result.
