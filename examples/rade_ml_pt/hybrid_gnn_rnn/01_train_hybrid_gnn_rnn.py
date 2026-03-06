#!/usr/bin/env python3
"""
End-to-End Training Example: Hybrid GNN-RNN Model (PyTorch)
============================================================

This script walks through every component of the Hybrid GNN-RNN training
pipeline using synthetic data.  It is designed to:

  1. Validate that every stage of the framework is wired correctly.
  2. Serve as living documentation of what each component expects and produces.
  3. Be runnable out-of-the-box:
     ``python examples/rade_ml_pt/hybrid_gnn_rnn/01_train_hybrid_gnn_rnn.py``

Architecture overview
---------------------
The Hybrid GNN-RNN model predicts **target trade PnL** from a portfolio of
elementary trades.  It combines:

  - **GNN**  -- learns structural relationships between trades via a k-NN graph
    built from encoded trade attributes (moneyness, delta, vega, product type ...).
  - **RNN**  -- captures temporal dynamics from historical PnL sequences.
  - **Fusion + Attention + Projection** -- merges GNN and RNN embeddings,
    attends to target trades, and projects to PnL predictions.

Pipeline stages
---------------
  0. **Synthetic data generation** -- creates realistic trade PnL and attributes.
  1. **Data build** -- PnL standardisation, dimensionality reduction, attribute
     encoding, graph construction, DataLoader creation.
  2. **Model instantiation** -- HybridGnnRnn PyTorch model.
  3. **Training** -- compile (optimizer, loss, metrics) and fit via Trainer.
  4. **Training results** -- loss history, best epoch, model summary.
  5. **Evaluation** -- Evaluator with RMSE, MAE, R^2, MAPE and residual statistics.
  6. **Model registration** -- persist model + metadata to ModelRegistry.
  7. **Inference** -- load from registry via InferenceRunner, predict on test batch.
  8. **Training plots** -- 4-panel figure displayed on screen (not saved).
"""
from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# ---------------------------------------------------------------------------
# Reproducibility & PyTorch environment setup
# ---------------------------------------------------------------------------
import os
os.environ["PYTHONHASHSEED"] = "42"

import torch
import numpy as np
import pandas as pd

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from src.rade_ml_pt.data.io import CacheLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.hybrid_gnn_rnn_pt")


# ======================================================================
# 0.  Synthetic Data Generation
# ======================================================================

def make_synthetic_data(
    workdir: Path,
    n_scenarios: int = 200,
    n_elementary: int = 20,
    n_target: int = 4,
    seed: int = 42,
) -> dict:
    """
    Generate synthetic PnL and attribute files mimicking a real trade universe.

    Returns a ``job`` dict with ``cluster_info`` paths -- the same structure
    that a production job scheduler would produce.
    """
    rng = np.random.RandomState(seed)

    underlyings = ["EURUSD", "GBPUSD"]
    product_types = ["vanilla_option", "forward"]

    elem_ids = []
    idx = 1
    for und in underlyings:
        for prod in product_types:
            n_per_group = n_elementary // (len(underlyings) * len(product_types))
            for i in range(n_per_group):
                elem_ids.append(f"{und}|{prod}|{idx}")
                idx += 1
    while len(elem_ids) < n_elementary:
        elem_ids.append(f"EURUSD|vanilla_option|{idx}")
        idx += 1
    elem_ids = elem_ids[:n_elementary]

    tgt_ids = [f"EURUSD|vanilla_option|tgt_{i+1}" for i in range(n_target)]

    logger.info(f"Trade universe: {n_elementary} elementary + {n_target} target = {n_elementary + n_target} total")

    elem_pnl_arr = rng.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01
    mix_weights = rng.randn(n_elementary, n_target).astype(np.float32) * 0.3
    tgt_pnl_arr = elem_pnl_arr @ mix_weights + rng.randn(n_scenarios, n_target).astype(np.float32) * 0.002

    elem_pnl = pd.DataFrame(elem_pnl_arr, columns=elem_ids)
    tgt_pnl = pd.DataFrame(tgt_pnl_arr, columns=tgt_ids)

    logger.info(f"PnL shapes: elementary {elem_pnl.shape}, target {tgt_pnl.shape}")

    def _make_attrs(trade_ids: list, trade_type: str = "option") -> dict:
        n = len(trade_ids)
        return {
            "trade_id": trade_ids,
            "moneyness": rng.uniform(0.8, 1.2, n).tolist(),
            "yrs_to_maturity": rng.uniform(0.1, 2.0, n).tolist(),
            "delta": rng.uniform(-1.0, 1.0, n).tolist(),
            "vega": rng.uniform(0.0, 0.5, n).tolist(),
            "product_type": [tid.split("|")[1] for tid in trade_ids],
            "product_subtype": ["european"] * n,
            "trade_type": [trade_type] * n,
            "underlying_risk_factors": [["FX"]] * n,
        }

    elem_attrs = _make_attrs(elem_ids, trade_type="elementary")
    tgt_attrs = _make_attrs(tgt_ids, trade_type="target")

    workdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "elementary_pnl_path": str(workdir / "elem_pnl.pkl"),
        "target_pnl_path": str(workdir / "tgt_pnl.pkl"),
        "elementary_attribs_path": str(workdir / "elem_attrs.pkl"),
        "target_attribs_path": str(workdir / "tgt_attrs.pkl"),
    }
    for key, path in paths.items():
        data = {
            "elementary_pnl_path": elem_pnl,
            "target_pnl_path": tgt_pnl,
            "elementary_attribs_path": elem_attrs,
            "target_attribs_path": tgt_attrs,
        }[key]
        CacheLoader.save_data(data, path)

    logger.info(f"Data written to {workdir}")
    return {"cluster_info": paths}


# ======================================================================
# 1.  Configuration
# ======================================================================

def build_configs(workdir: Path, job: dict) -> "PipelineConfig":
    """Build the full pipeline configuration."""
    from src.rade_ml_pt.pipelines.config import PipelineConfig
    from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        BasisSelectionConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml_pt.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig, ReduceLrConfig

    artifacts_dir = workdir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_config = HybridGnnRnnDataConfig(
        folders=FolderEnvironmentConfig(root_folder=str(workdir)),
        validation_split=0.10,
        test_split=0.05,
        seq_length=1,
        batch_size=16,
        shuffle=True,
        cache=False,
        drop_remainder=False,
        transform_type="standard",
        dimensionality=DimensionalityConfig(
            reduction_mode="basis_selection",
            basis_selection=BasisSelectionConfig(
                var_threshold=0.999999,
                method="pca",
                max_components=10,
            ),
        ),
        graph_builder=GraphBuilderConfig(
            k=3,
            distance_metric="euclidean",
            include_quota=False,
            alpha_moneyness=1.0,
            alpha_maturity=1.0,
            alpha_delta=1.0,
            alpha_vega=1.0,
            alpha_prod_type=1.0,
            alpha_prod_subtype=0.5,
            alpha_underlying=1.0,
            alpha_underlying_rf=0.5,
        ),
        attribute_encoder=AttributeEncoderConfig(
            numeric_keys=["moneyness", "yrs_to_maturity", "delta", "vega"],
            categorical_keys=["product_type", "product_subtype", "trade_type"],
            multi_label_keys=["underlying_risk_factors"],
            num_decay_terms=3,
        ),
        plot_trade_graph=False,
        plot_pnl_distribution=False,
        save_intermediate_files=False,
        seed=42,
    )

    training_config = TrainingConfig(
        epochs=500,
        loss="mse",
        metrics=["mse", "mae"],
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1e-3,
            beta_1=0.9,
            beta_2=0.999,
        ),
        early_stopping=EarlyStoppingConfig(
            patience=30,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),
        lr_reduction=ReduceLrConfig(
            monitor="val_loss",
            mode="min",
            initial_lr=1e-3,
            patience=10,
            factor=0.8,
            min_lr=1e-6,
        ),
        strategy="auto",
        mixed_precision=False,
        xla_compile=False,
        verbose=True,
    )

    config = PipelineConfig(
        training_config=training_config.to_dict(),
        data_config=data_config,
        model_config=None,
        registry_dir=None,
        tracking_dir=None,
        artifacts_dir=str(artifacts_dir),
        metadata={
            "job": job,
            "run_name": "hybrid_gnn_rnn_pt_example",
            "generate_training_report": False,
        },
    )

    logger.info("Configuration built")
    return config


# ======================================================================
# 2.  Step-by-Step Pipeline Execution
# ======================================================================

def run_step_by_step(config: "PipelineConfig") -> None:
    """Execute each pipeline stage individually with detailed logging."""
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
    from src.rade_ml_pt.training.trainer import Trainer, setup_training_environment

    pipeline = HybridGnnRnnTrainPipeline(config)
    training_config = pipeline._resolve_training_config()
    seed = pipeline._resolve_seed()
    setup_training_environment(training_config, seed)

    # ---- Stage 1: Build Data ----
    print("\n" + "=" * 70)
    print("STAGE 1: BUILD DATA")
    print("=" * 70)

    data_result = pipeline.build_data(config)

    print(f"\n  Data result type: {type(data_result).__name__}")
    print(f"  Elementary PnL shape: {data_result.elementary_pnl.shape}")
    print(f"  Target PnL shape:     {data_result.target_pnl.shape}")
    print(f"\n  Metadata keys: {sorted(data_result.metadata.keys())}")

    # Inspect one batch from the DataLoader
    print("\n  --- DataLoader batch inspection ---")
    for batch in data_result.train_ds:
        if isinstance(batch, (tuple, list)):
            inputs_batch, targets_batch = batch[0], batch[1]
        else:
            inputs_batch = batch
            targets_batch = None
        if isinstance(inputs_batch, dict):
            print(f"  Input keys:     {sorted(inputs_batch.keys())}")
            for k, v in sorted(inputs_batch.items()):
                shape = v.shape if hasattr(v, "shape") else "N/A"
                print(f"    {k:25s} -> {shape}")
        if targets_batch is not None:
            print(f"  Targets shape:  {targets_batch.shape}")
        break

    # ---- Stage 2: Build Model ----
    print("\n" + "=" * 70)
    print("STAGE 2: BUILD MODEL")
    print("=" * 70)

    model = pipeline.build_model(config, data_result)

    print(f"\n  Model class: {type(model).__name__}")
    print(f"  Model name:  {model.model_name}")

    # ---- Stage 3: Trainer (Compile + Fit) ----
    print("\n" + "=" * 70)
    print("STAGE 3: COMPILE + TRAIN")
    print("=" * 70)

    trainer = Trainer(model=model, config=training_config, seed=seed)

    print(f"\n  Training config:")
    print(f"    Epochs:          {training_config.epochs}")
    print(f"    Loss:            {training_config.loss}")

    print("\n  Starting training...")
    result = trainer.fit(
        train_data=data_result.train_ds,
        val_data=data_result.val_ds,
    )

    # ---- Stage 4: Results ----
    print("\n" + "=" * 70)
    print("STAGE 4: TRAINING RESULTS")
    print("=" * 70)

    print(f"\n  Training time:   {result.training_time_seconds:.1f}s")
    print(f"  Final epoch:     {result.final_epoch}")
    print(f"  Best epoch:      {result.best_epoch}")
    print(f"  Best train loss: {result.best_train_loss:.6f}")
    print(f"  Best val loss:   {result.best_val_loss:.6f}")
    print(f"  Stopped early:   {result.stopped_early}")

    # ---- Stage 5: Evaluation ----
    print("\n" + "=" * 70)
    print("STAGE 5: EVALUATION")
    print("=" * 70)

    from src.rade_ml_pt.evaluation.evaluator import Evaluator
    from src.rade_ml_pt.evaluation.metrics import rmse, mae, r_squared, mape

    eval_ds = data_result.test_ds if data_result.test_ds is not None else data_result.val_ds
    evaluator = Evaluator(model, loss_fn=torch.nn.MSELoss())
    eval_result = evaluator.run(
        eval_ds,
        additional_metrics={"rmse": rmse, "mae": mae, "r_squared": r_squared, "mape": mape},
    )

    print(f"\n  {eval_result.summary()}")

    # ---- Stage 6: Model Registration ----
    print("\n" + "=" * 70)
    print("STAGE 6: MODEL REGISTRATION")
    print("=" * 70)

    from src.rade_ml_pt.registry.store import ModelRegistry

    registry_dir = Path(config.artifacts_dir) / "registry"
    registry = ModelRegistry(str(registry_dir))

    entry = registry.register(
        model=model,
        training_result=result,
        tags=["hybrid_gnn_rnn", "latest"],
        description="Hybrid GNN-RNN PyTorch example model (synthetic data)",
    )

    print(f"\n  Registry dir:  {registry_dir}")
    print(f"  Version:       {entry.version}")
    print(f"  Tags:          {entry.tags}")

    # ---- Stage 7: Inference ----
    print("\n" + "=" * 70)
    print("STAGE 7: INFERENCE (via InferenceRunner)")
    print("=" * 70)

    from src.rade_ml_pt.inference.runner import InferenceRunner

    runner = InferenceRunner.from_registry(registry, version_or_tag="latest")
    print(f"\n  Loaded model version: {runner.model_version}")

    # grab one batch of inputs from the eval DataLoader
    for batch in eval_ds:
        if isinstance(batch, (tuple, list)):
            test_inputs = batch[0]
        else:
            test_inputs = batch
        break

    infer_result = runner.predict(
        inputs=test_inputs,
        sample_ids=[f"trade_{i}" for i in range(len(data_result.target_pnl.columns))],
        metadata={"source": "example_script", "dataset": "synthetic_test"},
    )

    print(f"\n  InferenceResult:")
    print(f"    n_samples:       {infer_result.n_samples}")
    print(f"    latency:         {infer_result.latency_seconds:.4f}s")
    preds = infer_result.predictions
    print(f"    prediction shape: {preds.shape}")

    # ---- Stage 8: Training Plots ----
    print("\n" + "=" * 70)
    print("STAGE 8: TRAINING PLOTS (displaying on screen)")
    print("=" * 70)

    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    from src.rade_ml_pt.training.plots import show_training_plots
    show_training_plots(result)

    return result


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="rade_ml_pt_hybrid_gnn_rnn_"))
    logger.info(f"Working directory: {workdir}")

    job = make_synthetic_data(workdir, n_scenarios=200, n_elementary=20, n_target=4)
    config = build_configs(workdir, job)
    result = run_step_by_step(config)

    logger.info(f"All artifacts in: {workdir}")


if __name__ == "__main__":
    main()
