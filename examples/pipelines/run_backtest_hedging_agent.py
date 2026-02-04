#!/usr/bin/env python3
"""
Example: Deep Hedging Backtest Agent Pipeline

Runs the deep_hedging.backtest_agent pipeline to evaluate a hedging agent
(delta-hedge benchmark or a trained deep agent) on synthetic price paths.

Usage:
  python examples/pipelines/run_backtest_hedging_agent.py

Without a pre-trained agent in state, the pipeline uses DeltaHedgingAgent
as benchmark. To backtest a trained agent, run ml.train_deep_hedging first
and pass the resulting agent path, or run this pipeline in a workflow that
loads the agent into context.
"""

from pathlib import Path

from src.orchestrator.pipelines.deep_hedging.backtest_agent import build_pipeline
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.context import Context
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.state_keys import StateKeys as Keys
from src.orchestrator.artifacts.store import ArtifactStore


def main() -> None:
    print("=" * 60)
    print("Deep Hedging Backtest Agent Pipeline")
    print("=" * 60)

    # Default config: synthetic 63-day backtest, delta-hedge agent
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

    artifacts_root = Path(__file__).resolve().parents[1] / "artifacts" / "hedging_backtest"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(artifacts_root=artifacts_root)
    ctx = Context(
        run_id="run-backtest-hedging-1",
        cfg=config,
        logger=None,
        artifact_store=store,
    )

    pipeline = build_pipeline()
    runner = PipelineRunner()
    ctx = runner.run(pipeline, ctx)

    result = ctx.state.get(Keys.BACKTEST_RESULT)
    if result is not None:
        print("\nBacktest result:")
        print(f"  Total P&L:     {result.total_pnl:,.2f}")
        print(f"  Hedging P&L:   {result.hedging_pnl:,.2f}")
        print(f"  Option P&L:   {result.option_pnl:,.2f}")
        print(f"  Total cost:   {result.total_cost:,.2f}")
        print(f"  Sharpe:       {result.sharpe_ratio:.2f}")
        print(f"  Max drawdown: {result.max_drawdown:.2f}")
        if result.outperformance is not None:
            print(f"  vs Delta:     {result.outperformance:,.2f}")
    else:
        print("\nNo backtest result (agent or deep_hedging module unavailable).")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
