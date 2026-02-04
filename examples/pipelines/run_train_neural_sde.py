#!/usr/bin/env python3
"""
Example: Train Neural SDE Pipeline

Runs the ml.train_neural_sde pipeline to build and train a Neural SDE
on synthetic GBM paths (or pre-loaded paths in context).

Usage:
  python examples/pipelines/run_train_neural_sde.py

Config is passed via RunConfig.params["ml"]["neural_sde"]. After the run,
the trained model and training result are in context state; optionally
saved to the artifact store.
"""

from pathlib import Path

from src.orchestrator.pipelines.ml.train_neural_sde import build_pipeline
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.context import Context
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.artifacts.store import ArtifactStore


def main() -> None:
    print("=" * 60)
    print("Train Neural SDE Pipeline")
    print("=" * 60)

    config = RunConfig(
        pipeline="ml.train_neural_sde",
        params={
            "ml": {
                "neural_sde": {
                    "n_paths": 2000,
                    "n_steps": 50,
                    "S0": 100.0,
                    "drift": 0.05,
                    "volatility": 0.20,
                    "seed": 42,
                    "hidden_dims": [64, 64],
                    "n_epochs": 50,
                    "learning_rate": 1e-3,
                    "batch_size": 32,
                    "n_sim_paths": 500,
                    "n_sim_steps": 50,
                    "patience": 10,
                    "verbose": True,
                },
            },
        },
    )

    artifacts_root = Path(__file__).resolve().parents[1] / "artifacts" / "neural_sde_example"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(artifacts_root=artifacts_root)
    ctx = Context(
        run_id="neural_sde_example",
        cfg=config,
        logger=None,
        artifact_store=store,
    )

    pipeline = build_pipeline()
    runner = PipelineRunner()

    ctx = runner.run(pipeline, ctx)

    result = ctx.state.get("neural_sde_training_result")
    if result is not None:
        print("\nTraining result:")
        print(f"  final_loss: {result.final_loss:.6f}")
        print(f"  converged:  {result.converged}")
        print(f"  epoch:      {result.epoch}")
        if hasattr(result, "summary"):
            print("  summary:   ", result.summary())
    else:
        print("\nNo training result (model or paths missing, or Neural SDE not installed).")

    print("\nDone.")


if __name__ == "__main__":
    main()
