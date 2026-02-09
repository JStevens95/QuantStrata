"""
Hyperparameter Tuning Pipeline.

Orchestrated pipeline for running hyperparameter optimization
with experiment tracking and model registry integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.orchestrator.core.context import Context, PipelineContext
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step


@dataclass
class HyperparameterTuningConfig:
    """Configuration for hyperparameter tuning pipeline."""
    
    # Search space configuration
    search_space_config: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Tuning settings
    n_trials: int = 100
    direction: str = "minimize"
    sampler: str = "tpe"
    
    # Pruning
    use_pruning: bool = True
    n_startup_trials: int = 5
    n_warmup_steps: int = 10
    
    # Tracking
    experiment_name: str = "tuning_experiment"
    tracking_backend: str = "memory"
    tracking_uri: Optional[str] = None
    
    # Registry
    register_best: bool = True
    registry_path: str = "./model_registry"
    model_name: str = "tuned_model"
    
    # Output
    output_dir: str = "./tuning_results"


class CreateSearchSpaceStep(Step):
    """Step to create search space from configuration."""
    
    name = "create_search_space"
    
    def run(self, ctx: PipelineContext) -> Context:
        from src.machine_learning.tuning import SearchSpace
        
        config = ctx.get("tuning_config")
        space_config = config.search_space_config
        
        space = SearchSpace()
        for param_name, param_config in space_config.items():
            param_type = param_config.get("type", "float")
            
            if param_type == "float":
                space.add_float(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "int":
                space.add_int(
                    param_name,
                    param_config["low"],
                    param_config["high"],
                    log=param_config.get("log", False),
                )
            elif param_type == "categorical":
                space.add_categorical(param_name, param_config["choices"])
            elif param_type == "bool":
                space.add_bool(param_name)
        
        ctx.set("search_space", space)
        if ctx.logger:
            ctx.logger.info("Created search space with %s parameters", len(space))
        return ctx


class SetupTrackingStep(Step):
    """Step to setup experiment tracking."""
    
    name = "setup_tracking"
    
    def run(self, ctx: PipelineContext) -> Context:
        from src.machine_learning.core.tracking import create_tracker
        
        config = ctx.get("tuning_config")
        
        tracker = create_tracker(
            backend=config.tracking_backend,
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri,
        )
        
        ctx.set("tracker", tracker)
        if ctx.logger:
            ctx.logger.info("Setup %s tracking", config.tracking_backend)
        return ctx


class RunTuningStep(Step):
    """Step to run hyperparameter tuning."""
    
    name = "run_tuning"
    
    def run(self, ctx: PipelineContext) -> Context:
        from src.machine_learning.tuning import run_optuna_tuning, MedianPruner
        
        config = ctx.get("tuning_config")
        space = ctx.get("search_space")
        objective_fn = ctx.get("objective_fn")
        tracker = ctx.get("tracker")
        
        # Setup pruner
        pruner = None
        if config.use_pruning:
            pruner = MedianPruner(
                n_startup_trials=config.n_startup_trials,
                n_warmup_steps=config.n_warmup_steps,
            )
        
        # Wrap objective with tracking
        def tracked_objective(params, trial):
            with tracker.start_run(f"trial_{trial.number}"):
                tracker.log_params(params)
                result = objective_fn(params, trial)
                tracker.log_metrics({"score": result})
                return result
        
        # Run tuning
        result = run_optuna_tuning(
            objective_fn=tracked_objective,
            search_space=space,
            n_trials=config.n_trials,
            direction=config.direction,
            pruner=pruner,
            sampler=config.sampler,
        )
        
        ctx.set("tuning_result", result)
        if ctx.logger:
            ctx.logger.info(
                "Tuning complete: best score = %s, completed = %s, pruned = %s",
                result.best_score, result.n_completed, result.n_pruned,
            )
        return ctx


class SaveResultsStep(Step):
    """Step to save tuning results."""
    
    name = "save_results"
    
    def run(self, ctx: PipelineContext) -> Context:
        config = ctx.get("tuning_config")
        result = ctx.get("tuning_result")
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result.save(output_dir / "tuning_result.json")
        
        if ctx.logger:
            ctx.logger.info("Saved results to %s", output_dir)
        return ctx


class RegisterBestModelStep(Step):
    """Step to register best model to registry."""
    
    name = "register_best_model"
    
    def run(self, ctx: PipelineContext) -> Context:
        from src.machine_learning.registry import ModelRegistry
        
        config = ctx.get("tuning_config")
        
        if not config.register_best:
            if ctx.logger:
                ctx.logger.info("Skipping model registration")
            return ctx
        
        result = ctx.get("tuning_result")
        best_model_path = ctx.state.get("best_model_path")
        
        if best_model_path is None:
            if ctx.logger:
                ctx.logger.warning("No best model path provided, skipping registration")
            return ctx
        
        registry = ModelRegistry(config.registry_path)
        
        version = registry.register_model(
            name=config.model_name,
            model_path=best_model_path,
            metrics={"best_score": result.best_score},
            params=result.best_config,
            tags={"source": "hyperparameter_tuning"},
        )
        
        ctx.set("registered_version", version)
        if ctx.logger:
            ctx.logger.info("Registered model %s version %s", config.model_name, version.version)
        return ctx


def create_hyperparameter_tuning_pipeline(
    config: HyperparameterTuningConfig,
    objective_fn: Callable[[Dict[str, Any], Any], float],
) -> Pipeline:
    """
    Create hyperparameter tuning pipeline.
    
    Parameters
    ----------
    config : HyperparameterTuningConfig
        Pipeline configuration.
    objective_fn : callable
        Objective function (params, trial) -> score.
        
    Returns
    -------
    Pipeline
        Configured pipeline.
    """
    pipeline = Pipeline(
        name="hyperparameter_tuning",
        steps=[
            CreateSearchSpaceStep(name="create_search_space"),
            SetupTrackingStep(name="setup_tracking"),
            RunTuningStep(name="run_tuning"),
            SaveResultsStep(name="save_results"),
            RegisterBestModelStep(name="register_best_model"),
        ],
    )
    # Caller must put tuning_config and objective_fn in ctx.state before running.
    return pipeline


__all__ = [
    "HyperparameterTuningConfig",
    "create_hyperparameter_tuning_pipeline",
]
