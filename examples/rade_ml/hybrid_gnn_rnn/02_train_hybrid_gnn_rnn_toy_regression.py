#!/usr/bin/env python3
"""
Toy Regression Example: Hybrid GNN-RNN Model
============================================

Identical to 01_train_hybrid_gnn_rnn.py except for the **input data**.

This script uses toy regression data where target PnL has a trivial, learnable
relationship to elementary PnL:

    target_pnl = mean(elementary_pnl) + negligible_noise

If the model is working correctly, it should achieve:
  - Very low train/val loss (e.g. < 1e-3)
  - R² close to 1.0 on the test set
  - RMSE << 1 (in standardized space)

If these metrics are good → model is fine; poor performance in example 01
is due to data, not architecture.

If these metrics are poor → investigate model logic/math.

Generalization analysis (when train R² > 0.5 but test R² < 0):
The hybrid GNN-RNN is designed for graph-structured, attribute-driven aggregation. The toy
task (target = mean(elementary)) requires uniform aggregation over all elementaries, with
no role for attribute similarity. The graph and attention are built from attribute-space
k-NN, which does not encode "all trades contribute equally". The model must learn a
uniform mean through many nonlinear layers; with moderate data this can overfit
train/val without capturing the true operation. See IMPROVEMENTS.md for discussion.

Run: ``python examples/rade_ml/hybrid_gnn_rnn/02_train_hybrid_gnn_rnn_toy_regression.py``
"""
from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import os
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["PYTHONHASHSEED"] = "42"

import tensorflow as tf
tf.config.experimental.enable_op_determinism()
tf.keras.utils.set_random_seed(42)

import numpy as np
import pandas as pd

from src.rade_ml.data.io import CacheLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.hybrid_gnn_rnn.toy_regression")


# ======================================================================
# 0.  Toy Regression Data Generation
# ======================================================================
#
# Target PnL = mean(elementary_pnl) + negligible noise.
# The model receives elementary PnL in pnl_history; predicting the mean
# is a trivial linear aggregation the model should easily learn.
# ======================================================================

def make_toy_regression_data(
    workdir: Path,
    n_scenarios: int = 1000,
    n_elementary: int = 20,
    n_target: int = 4,
    seed: int = 42,
    noise_std: float = 1e-6,
) -> dict:
    """
    Generate toy data where target PnL = mean(elementary_pnl) + tiny noise.

    This is trivially learnable. Used to verify model correctness.
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

    logger.info(f"[TOY REGRESSION] Trade universe: {n_elementary} elementary + {n_target} target")
    logger.info(f"[TOY REGRESSION] Target PnL = mean(elementary_pnl) + N(0, {noise_std})")

    # Elementary PnL: small Gaussian (same scale as example 01)
    elem_pnl_arr = rng.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01

    # Target = mean across elementary + negligible noise (trivially learnable)
    mean_pnl = np.mean(elem_pnl_arr, axis=1, keepdims=True)
    tgt_pnl_arr = np.broadcast_to(mean_pnl, (n_scenarios, n_target)).astype(np.float32)
    tgt_pnl_arr += rng.randn(n_scenarios, n_target).astype(np.float32) * noise_std

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
    # Targets with centroid-like attributes so they sit among elementaries in graph.
    tgt_attrs = {
        "trade_id": tgt_ids,
        "moneyness": [1.0] * n_target,
        "yrs_to_maturity": [1.0] * n_target,
        "delta": [0.0] * n_target,
        "vega": [0.25] * n_target,
        "product_type": ["vanilla_option"] * n_target,
        "product_subtype": ["european"] * n_target,
        "trade_type": ["target"] * n_target,
        "underlying_risk_factors": [["FX"]] * n_target,
    }

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

    return {"cluster_info": paths}


# ======================================================================
# 1.  Configuration (same as 01)
# ======================================================================

def build_configs(workdir: Path, job: dict) -> "PipelineConfig":
    from src.rade_ml.pipelines.config import PipelineConfig
    from src.rade_ml.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig, ReduceLrConfig

    artifacts_dir = workdir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    data_config = HybridGnnRnnDataConfig(
        folders=FolderEnvironmentConfig(root_folder=str(workdir)),
        validation_split=0.10,
        test_split=0.10,
        seq_length=1,
        batch_size=32,
        shuffle=True,
        cache=False,
        drop_remainder=False,
        # "none" preserves target=mean(elementary). StandardScaler uses separate per-column
        # scalers for elementary and target, which breaks that linear relationship.
        transform_type="none",
        # No dimensionality reduction: keep ALL elementary trades so target = mean(elementary) holds.
        dimensionality=DimensionalityConfig(reduction_mode="none"),
        # k=n_elementary so targets can aggregate from all elementaries via graph.
        graph_builder=GraphBuilderConfig(
            k=20,
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
        loss="mae",
        metrics=["mse", "mae"],
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
        early_stopping=EarlyStoppingConfig(
            patience=50,
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
        mixed_precision=True,
        xla_compile=True,
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
            "run_name": "hybrid_gnn_rnn_toy_regression",
            "generate_training_report": False,
        },
    )

    return config


# ======================================================================
# 2.  Step-by-Step Pipeline (same as 01)
# ======================================================================

def run_step_by_step(config: "PipelineConfig"):
    import tensorflow as tf
    from src.rade_ml.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
    from src.rade_ml.training.trainer import Trainer, setup_training_environment
    from src.rade_ml.evaluation.evaluator import Evaluator
    from src.rade_ml.evaluation.metrics import rmse, mae, r_squared, mape
    from src.rade_ml.registry.store import ModelRegistry
    from src.rade_ml.inference.runner import InferenceRunner
    import src.rade_ml.models.hybrid_gnn_rnn.model  # noqa: F401

    pipeline = HybridGnnRnnTrainPipeline(config)
    training_config = pipeline._resolve_training_config()
    seed = pipeline._resolve_seed()
    setup_training_environment(training_config, seed)

    print("\n" + "=" * 70)
    print("STAGE 1: BUILD DATA (Toy Regression: target = mean(elementary))")
    print("=" * 70)
    data_result = pipeline.build_data(config)
    print(f"\n  Elementary PnL shape: {data_result.elementary_pnl.shape}")
    print(f"  Target PnL shape:     {data_result.target_pnl.shape}")
    print(f"  Test scenarios:       {data_result.metadata['test_size']:.0%}")

    print("\n" + "=" * 70)
    print("STAGE 2: BUILD MODEL")
    print("=" * 70)
    model = pipeline.build_model(config, data_result)
    print(f"  Model: {model.name}")

    print("\n" + "=" * 70)
    print("STAGE 3: COMPILE + TRAIN")
    print("=" * 70)
    trainer = Trainer(model=model, config=training_config, seed=seed)
    trainer.compile()
    result = trainer.fit(
        train_data=data_result.train_ds,
        val_data=data_result.val_ds,
    )

    print("\n" + "=" * 70)
    print("STAGE 4: TRAINING RESULTS")
    print("=" * 70)
    print(f"  Best epoch:    {result.best_epoch}")
    print(f"  Best val loss: {result.best_val_loss:.6f}")
    print(f"  Best train:    {result.best_train_loss:.6f}")

    print("\n" + "=" * 70)
    print("STAGE 5: EVALUATION (Toy regression sanity check)")
    print("=" * 70)
    eval_ds = data_result.test_ds if data_result.test_ds is not None else data_result.val_ds
    # Baseline: target = mean(elementary) — trivial R² should be ~1.0 if data relationship holds
    y_true_all, y_baseline_all, y_pred_all = [], [], []
    for x, y in eval_ds:
        pnl = x["pnl_history"].numpy()
        baseline = np.mean(pnl, axis=(1, 2))[:, np.newaxis]  # [B, 1]
        baseline = np.broadcast_to(baseline, (pnl.shape[0], y.shape[1]))
        y_true_all.append(y.numpy())
        y_baseline_all.append(baseline)
        pred = model(x, training=False)
        y_pred_all.append(pred.numpy() if hasattr(pred, "numpy") else pred)
    y_true = np.concatenate(y_true_all, axis=0)
    y_baseline = np.concatenate(y_baseline_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    r2_baseline = r_squared(y_true, y_baseline)
    print(f"\n  Baseline (predict mean): R² = {r2_baseline:.4f}")
    if r2_baseline < 0.99:
        print("  WARNING: Baseline R² < 0.99 — target=mean(elementary) may not hold in scaled data")
    evaluator = Evaluator(model)
    eval_result = evaluator.run(
        eval_ds,
        additional_metrics={"rmse": rmse, "mae": mae, "r_squared": r_squared, "mape": mape},
    )
    print(f"\n  {eval_result.summary()}")
    r2_test = eval_result.metrics.get("r_squared")
    # Train-set evaluation: if model fits train, R²_train should be high; low R²_test = generalization issue.
    train_eval = evaluator.run(
        data_result.train_ds,
        additional_metrics={"r_squared": r_squared},
        return_predictions=False,
    )
    r2_train = train_eval.metrics.get("r_squared")
    print(f"\n  Train R² = {r2_train:.4f} | Test R² = {r2_test:.4f} | Baseline R² = {r2_baseline:.4f}")
    if r2_train is not None and r2_test is not None:
        if r2_train > 0.5 and r2_test < 0:
            print("  → Model fits train but not test (generalization gap; see docstring for analysis)")
        elif r2_train < 0.5:
            print("  → Model does not fit train (investigate architecture / pipeline)")
        else:
            print("  → PASS (model learned and generalizes)")

    print("\n" + "=" * 70)
    print("STAGE 6: MODEL REGISTRATION")
    print("=" * 70)
    registry_dir = Path(config.artifacts_dir) / "registry"
    registry = ModelRegistry(str(registry_dir))
    entry = registry.register(
        model=model,
        training_result=result,
        tags=["hybrid_gnn_rnn", "toy_regression", "latest"],
        description="Hybrid GNN-RNN toy regression (target=mean elementary)",
    )
    print(f"  Version: {entry.version}")

    print("\n" + "=" * 70)
    print("STAGE 7: INFERENCE")
    print("=" * 70)
    runner = InferenceRunner.from_registry(registry, version_or_tag="latest")
    infer_ds = eval_ds.take(1).map(lambda x, _y: x)
    infer_result = runner.predict(
        inputs=infer_ds,
        sample_ids=[f"tgt_{i}" for i in range(4)],
        metadata={"source": "toy_regression"},
    )
    print(f"  Predictions shape: {infer_result.predictions.shape}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    from src.rade_ml.training.plots import show_training_plots
    show_training_plots(result)

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    workdir = Path(tempfile.mkdtemp(prefix="rade_ml_toy_regression_"))
    logger.info(f"Working directory: {workdir}")

    job = make_toy_regression_data(workdir, n_scenarios=1000, n_elementary=20, n_target=4, noise_std=1e-6)
    config = build_configs(workdir, job)
    run_step_by_step(config)

    logger.info(f"Artifacts in: {workdir}")


if __name__ == "__main__":
    main()
