#!/usr/bin/env python3
"""
Example: Hyperparameter Tuning Pipeline

Demonstrates hyperparameter tuning with experiment tracking
and model registry integration.
"""

import logging
import numpy as np
from pathlib import Path

from src.orchestrator.artifacts.store import ArtifactStore
from src.orchestrator.config.loader import RunConfig
from src.orchestrator.config.schemas import IOConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.errors import StepError
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.pipelines.ml.hyperparameter_tuning import (
    HyperparameterTuningConfig,
    create_hyperparameter_tuning_pipeline,
)


def create_dummy_objective():
    """
    Create a dummy objective function for demonstration.
    
    In practice, this would train and evaluate a model.
    """
    def objective(params, trial):
        # Simulate model training
        lr = params["learning_rate"]
        hidden = params["hidden_units"]
        
        # Dummy loss (lower is better)
        # Optimal around lr=0.001, hidden=128
        loss = (
            0.1 * (np.log10(lr) + 3) ** 2 +  # Optimal at lr=0.001
            0.001 * (hidden - 128) ** 2 / 1000 +  # Optimal at 128
            0.05 * np.random.randn()  # Noise
        )
        
        # Report intermediate values for pruning
        for epoch in range(10):
            intermediate = loss * (1 - epoch / 20) + 0.02 * np.random.randn()
            trial.report(intermediate, epoch)
            
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        
        return loss
    
    return objective


def main():
    """Run hyperparameter tuning example."""
    print("=" * 60)
    print("Hyperparameter Tuning Pipeline Example")
    print("=" * 60)
    
    # Configuration
    config = HyperparameterTuningConfig(
        search_space_config={
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-1,
                "log": True,
            },
            "hidden_units": {
                "type": "int",
                "low": 32,
                "high": 256,
            },
            "activation": {
                "type": "categorical",
                "choices": ["relu", "tanh", "gelu"],
            },
            "use_dropout": {
                "type": "bool",
            },
        },
        n_trials=20,  # Reduced for demo
        direction="minimize",
        use_pruning=True,
        experiment_name="demo_tuning",
        tracking_backend="memory",
        register_best=False,  # Skip registration for demo
        output_dir="./output/tuning_demo",
    )
    
    # Create objective
    objective_fn = create_dummy_objective()
    
    # Create pipeline (no config/objective on pipeline; put in context)
    pipeline = create_hyperparameter_tuning_pipeline(
        config=config,
        objective_fn=objective_fn,
    )
    
    artifacts_root = Path(config.output_dir).resolve().parent
    artifacts_root.mkdir(parents=True, exist_ok=True)
    store = ArtifactStore(workdir=artifacts_root, run_id="tuning_demo")
    cfg = RunConfig(
        pipeline="hyperparameter_tuning",
        io=IOConfig(workdir=str(artifacts_root)),
        params={},
    )
    logger = logging.getLogger("hyperparameter_tuning")
    ctx = Context(run_id="tuning_demo", cfg=cfg, logger=logger, artifact_store=store)
    ctx.state["tuning_config"] = config
    ctx.state["objective_fn"] = objective_fn
    
    print("\nRunning hyperparameter tuning...")
    print(f"  Search space: {list(config.search_space_config.keys())}")
    print(f"  Trials: {config.n_trials}")
    print()
    
    try:
        PipelineRunner().run(pipeline, ctx)
    except ImportError as e:
        print(f"Note: Required dependency not available ({e})")
        print("Install with: pip install optuna tensorflow")
        return
    except StepError as e:
        cause = e.__cause__
        if isinstance(cause, ImportError) or (
            cause and ("TensorFlow" in str(cause) or "optuna" in str(cause).lower())
        ):
            print(f"Note: ML dependencies not available ({cause})")
            print("Install with: pip install optuna tensorflow")
            return
        raise
    
    # Get results from context
    result = ctx.state.get("tuning_result")
    
    if result:
        print("\n" + "=" * 60)
        print("Tuning Results")
        print("=" * 60)
        print(f"Best score: {result.best_score:.6f}")
        print(f"Best config:")
        for k, v in result.best_config.items():
            print(f"  {k}: {v}")
        print(f"\nTrials completed: {result.n_completed}")
        print(f"Trials pruned: {result.n_pruned}")
        
        # Save results
        output_dir = Path(config.output_dir)
        if output_dir.exists():
            print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
