"""
Pipeline: ml.train_neural_sde

Train a Neural SDE (drift/diffusion networks) on historical or synthetic paths.

Purpose
-------
1. Load or generate training paths (synthetic GBM for demo)
2. Build NeuralSDEDynamics (networks + solver)
3. Train via NeuralSDETrainer
4. Store trained model and TrainingResult in state; optionally save artifact
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline
from src.orchestrator.core.step import Step


def _neural_sde_cfg(cfg: RunConfig) -> Dict[str, Any]:
    """Extract ml.neural_sde config block."""
    if not isinstance(cfg.params, dict):
        raise TypeError("RunConfig.params must be a dict")
    ml = cfg.params.get("ml", {})
    return ml.get("neural_sde", {})


@dataclass(frozen=True, slots=True)
class BuildTrainingDataStep(Step):
    """Step 1: Generate synthetic GBM paths for training (or load from state)."""
    def run(self, ctx: Context) -> Context:
        paths = ctx.state.get("neural_sde_training_paths")
        if paths is not None:
            if ctx.logger:
                ctx.logger.info("Using pre-loaded training paths")
            return ctx
        cfg = _neural_sde_cfg(ctx.cfg)
        n_paths = int(cfg.get("n_paths", 2000))
        n_steps = int(cfg.get("n_steps", 50))
        S0 = float(cfg.get("S0", 100.0))
        mu = float(cfg.get("drift", 0.05))
        sigma = float(cfg.get("volatility", 0.20))
        seed = int(cfg.get("seed", 42))
        rng = np.random.default_rng(seed)
        dt = 1.0 / n_steps
        paths = np.zeros((n_paths, n_steps + 1))
        paths[:, 0] = S0
        for i in range(1, n_steps + 1):
            z = rng.standard_normal(n_paths)
            paths[:, i] = paths[:, i - 1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
            )
        ctx.put("neural_sde_training_paths", paths)
        if ctx.logger:
            ctx.logger.info("Generated synthetic training paths: %s x %s", n_paths, n_steps + 1)
        return ctx


@dataclass(frozen=True, slots=True)
class BuildModelStep(Step):
    """Step 2: Build NeuralSDEDynamics."""
    def run(self, ctx: Context) -> Context:
        model = ctx.state.get("neural_sde_model")
        if model is not None:
            return ctx
        cfg = _neural_sde_cfg(ctx.cfg)
        try:
            from src.models.neural_sde import (
                NeuralSDEDynamics,
                NeuralDriftNetwork,
                NeuralDiffusionNetwork,
                EulerMaruyamaSolver,
            )
            hidden = list(cfg.get("hidden_dims", [64, 64]))
            drift_net = NeuralDriftNetwork(hidden_dims=hidden)
            diffusion_net = NeuralDiffusionNetwork(hidden_dims=hidden)
            solver = EulerMaruyamaSolver()
            model = NeuralSDEDynamics(
                drift_network=drift_net,
                diffusion_network=diffusion_net,
                solver=solver,
            )
            ctx.put("neural_sde_model", model)
            if ctx.logger:
                ctx.logger.info("Built NeuralSDEDynamics")
        except ImportError as e:
            if ctx.logger:
                ctx.logger.warning("Neural SDE not available: %s", e)
            ctx.put("neural_sde_model", None)
        return ctx


@dataclass(frozen=True, slots=True)
class TrainStep(Step):
    """Step 3: Train Neural SDE and store result."""
    def run(self, ctx: Context) -> Context:
        model = ctx.state.get("neural_sde_model")
        paths = ctx.state.get("neural_sde_training_paths")
        if model is None or paths is None:
            if ctx.logger:
                ctx.logger.warning("Missing model or paths; skipping training")
            ctx.put("neural_sde_training_result", None)
            return ctx
        cfg = _neural_sde_cfg(ctx.cfg)
        try:
            from src.models.neural_sde.training import NeuralSDETrainer, TrainingConfig

            training_cfg = TrainingConfig(
                n_epochs=int(cfg.get("n_epochs", 50)),
                learning_rate=float(cfg.get("learning_rate", 1e-3)),
                batch_size=int(cfg.get("batch_size", 32)),
                n_sim_paths=int(cfg.get("n_sim_paths", 500)),
                n_sim_steps=int(cfg.get("n_sim_steps", 50)),
                patience=int(cfg.get("patience", 10)),
                verbose=bool(cfg.get("verbose", True)),
            )
            trainer = NeuralSDETrainer(config=training_cfg, seed=int(cfg.get("seed", 42)))
            result = trainer.fit(model, historical_paths=paths)
            ctx.put("neural_sde_training_result", result)
            if ctx.logger:
                ctx.logger.info(
                    "Neural SDE training complete: final_loss=%.6f, converged=%s",
                    result.final_loss, result.converged,
                )
            if ctx.artifact_store and hasattr(result, "summary"):
                summary = result.summary()
                path = ctx.artifact_store.artifacts_root / "neural_sde_training_summary.json"
                import json
                with open(path, "w") as f:
                    json.dump(summary, f, indent=2)
        except Exception as e:
            if ctx.logger:
                ctx.logger.warning("Training failed: %s", e)
            ctx.put("neural_sde_training_result", None)
        return ctx


def build_pipeline(cfg: Optional[RunConfig] = None) -> Pipeline:
    """Build the ml.train_neural_sde pipeline."""
    return Pipeline(
        name="ml.train_neural_sde",
        steps=[
            BuildTrainingDataStep(name="build_training_data"),
            BuildModelStep(name="build_model"),
            TrainStep(name="train"),
        ],
    )
