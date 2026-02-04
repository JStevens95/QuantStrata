#!/usr/bin/env python3
"""
Example: RL Agent Backtest / Deploy Pipelines

Demonstrates rl.backtest_agent (and optionally rl.deploy_agent).

Usage:
  python examples/pipelines/run_deploy_rl_agent.py

Without a saved agent, the backtest pipeline uses a stub agent and BaseEnv
so you can run a demo. To backtest a real agent, save it with
q_learning.pipelines.inference.save_agent(), then set rl.agent_path and
rl.agent_factory in params (or run rl.deploy_agent first and chain).
"""

from pathlib import Path

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.orchestrator.artifacts.store import ArtifactStore
from src.orchestrator.pipelines.rl.backtest_agent import build_pipeline


def main() -> None:
    print("=" * 60)
    print("RL Backtest Agent Pipeline")
    print("=" * 60)

    config = RunConfig(
        pipeline="rl.backtest_agent",
        params={
            "rl": {
                "backtest": {
                    "n_episodes": 10,
                    "state_dim": 1,
                    "n_actions": 3,
                    "max_steps": 50,
                    "seed": 42,
                    "compute_sharpe": True,
                    "compute_drawdown": True,
                },
            },
        },
    )

    artifacts_root = Path(__file__).resolve().parents[1] / "artifacts" / "rl_backtest"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(artifacts_root=artifacts_root)
    ctx = Context(
        run_id="rl-backtest-example",
        cfg=config,
        logger=None,
        artifact_store=store,
    )

    pipeline = build_pipeline(config)
    runner = PipelineRunner()
    ctx = runner.run(pipeline, ctx)

    result = ctx.state.get(Keys.RL_BACKTEST_RESULT)
    if result is not None:
        print("\nRL Backtest result:")
        print(f"  n_episodes:    {result.n_episodes}")
        print(f"  mean_return:   {result.mean_pnl_return:.4f}")
        print(f"  sharpe_ratio:  {result.sharpe_ratio:.2f}")
        print(f"  max_drawdown:  {result.max_drawdown:.4f}")
    else:
        print("\nNo RL backtest result (q_learning or env/agent missing).")

    print("\nDone.")


if __name__ == "__main__":
    main()
