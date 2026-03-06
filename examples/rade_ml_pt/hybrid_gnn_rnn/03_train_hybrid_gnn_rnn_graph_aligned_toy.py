#!/usr/bin/env python3
"""
Graph-Aligned Toy Regression: Hybrid GNN-RNN Model (PyTorch)
=============================================================

This example uses toy data where target PnL is a **graph-weighted mean** of
elementary neighbors' PnL -- matching the model's inductive bias.

    target_pnl[i] = sum over j in neighbors(i) of (w_ij * elementary_pnl[j])

The graph is the same k-NN graph the model uses (built from attributes). This
aligns with real exotics: target PnL depends on hedging basket (elementary)
PnL via attribute-similarity relationships.

Run: ``python examples/rade_ml_pt/hybrid_gnn_rnn/03_train_hybrid_gnn_rnn_graph_aligned_toy.py``
"""
from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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
from src.rade_ml_pt.utilities.attribute_encoder import TradeAttributeEncoder
from src.rade_ml_pt.utilities.graph_builder import TradeGraphBuilder
from src.rade_ml_pt.data.hybrid_gnn_rnn.build import encode_trade_attributes, build_trade_graph
from src.rade_ml_pt.data.hybrid_gnn_rnn.config import GraphBuilderConfig, AttributeEncoderConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.hybrid_gnn_rnn_pt.graph_aligned_toy")


# ======================================================================
# 0.  Graph-Aligned Toy Data Generation
# ======================================================================

def make_graph_aligned_toy_data(
    workdir: Path,
    n_scenarios: int = 1000,
    n_elementary: int = 20,
    n_target: int = 4,
    seed: int = 42,
    noise_std: float = 1e-6,
) -> dict:
    """
    Generate toy data where target PnL = graph-weighted mean of elementary PnL.
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

    logger.info(f"[GRAPH-ALIGNED TOY] Trade universe: {n_elementary} elementary + {n_target} target")
    logger.info("[GRAPH-ALIGNED TOY] Target PnL = graph-weighted mean of neighbors' elementary PnL")

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

    enc_config = AttributeEncoderConfig(
        numeric_keys=["moneyness", "yrs_to_maturity", "delta", "vega"],
        categorical_keys=["product_type", "product_subtype", "trade_type"],
        multi_label_keys=["underlying_risk_factors"],
        num_decay_terms=3,
    )
    encoder, encoder_results = encode_trade_attributes(
        config=enc_config,
        elementary_attrs=elem_attrs,
        target_attrs=tgt_attrs,
    )

    graph_config = GraphBuilderConfig(
        k=20, distance_metric="euclidean", include_quota=False,
        alpha_moneyness=1.0, alpha_maturity=1.0, alpha_delta=1.0, alpha_vega=1.0,
        alpha_prod_type=1.0, alpha_prod_subtype=0.5, alpha_underlying=1.0, alpha_underlying_rf=0.5,
    )
    builder, graph_result = build_trade_graph(
        config=graph_config,
        encoded_features=encoder_results["combined_encoded"],
    )
    adj = builder.adjacency_dense

    n_elem, n_tgt = n_elementary, n_target
    adj_tgt_elem = adj[n_elem : n_elem + n_tgt, :n_elem]

    elem_pnl_arr = rng.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01
    tgt_pnl_arr = (adj_tgt_elem @ elem_pnl_arr.T).T.astype(np.float32)
    tgt_pnl_arr += rng.randn(n_scenarios, n_target).astype(np.float32) * noise_std

    elem_pnl = pd.DataFrame(elem_pnl_arr, columns=elem_ids)
    tgt_pnl = pd.DataFrame(tgt_pnl_arr, columns=tgt_ids)

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
# 1.  Configuration
# ======================================================================

def build_configs(workdir: Path, job: dict) -> "PipelineConfig":
    from src.rade_ml_pt.pipelines.config import PipelineConfig
    from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml_pt.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig, ReduceLrConfig

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
        transform_type="none",
        dimensionality=DimensionalityConfig(reduction_mode="none"),
        graph_builder=GraphBuilderConfig(
            k=20, distance_metric="euclidean", include_quota=False,
            alpha_moneyness=1.0, alpha_maturity=1.0, alpha_delta=1.0, alpha_vega=1.0,
            alpha_prod_type=1.0, alpha_prod_subtype=0.5, alpha_underlying=1.0, alpha_underlying_rf=0.5,
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

    model_config = {
        "general": {},
        "gnn_layer": {"general": {"architecture": "default", "layers": 2, "layer_type": "mixed_graph_sage",
                                 "dropout_rate": 0.05, "use_bias": True, "use_residual": True, "batch_norm": True,
                                 "aggregator_op": "mean"},
                     "parameters": {"units": 64, "activation": "relu",
                                    "kernel_initializer": "glorot_uniform", "bias_initializer": "zeros"}},
        "rnn_layer": {"general": {"architecture": "default", "layers": 2, "layer_type": "lstm",
                                  "dropout_rate": 0.05, "use_bias": True},
                     "parameters": {"units": 64, "activation": "relu", "recurrent_activation": "sigmoid",
                                    "kernel_initializer": "glorot_uniform", "recurrent_initializer": "orthogonal",
                                    "bias_initializer": "zeros"}},
        "fusion_layer": {"general": {"fusion_mode": "gate", "dropout_rate": 0.05, "num_heads": 1, "k_nbrs": 50},
                        "parameters": {"units": 48, "activation": "sigmoid",
                                        "kernel_initializer": "he_uniform", "bias_initializer": "zeros"}},
        "attention_layer": {"general": {"layer_type": "standard", "use_residual": True, "use_layer_norm": True,
                                        "attention_mode": True, "num_heads": 1, "dropout_rate": 0.05, "k_nbrs": 50},
                           "parameters": {"units": 24, "activation": "tanh",
                                          "kernel_initializer": "he_uniform", "bias_initializer": "zeros"}},
        "projection_layer": {"general": {"dropout_rate": 0.05, "baseline_new_mode": "output_mix",
                                         "use_baseline_norm": True, "use_attn_scale_new": False,
                                         "use_attn_bias_new": False, "knn_k": 5, "knn_mode": "cosine_softmax",
                                         "knn_temperature": 5.0, "knn_power": 2.0, "residual_new_damp": 1.0},
                            "parameters": {"units": 24, "activation": "gelu",
                                           "kernel_initializer": "glorot_uniform", "bias_initializer": "zeros"}},
    }

    training_config = TrainingConfig(
        epochs=500,
        loss="mae",
        metrics=["mse", "mae"],
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
        early_stopping=EarlyStoppingConfig(
            patience=80,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),
        lr_reduction=ReduceLrConfig(
            monitor="val_loss", mode="min",
            initial_lr=1e-3, patience=15, factor=0.8, min_lr=1e-6,
        ),
        strategy="auto",
        mixed_precision=False,
        xla_compile=False,
        verbose=True,
    )

    config = PipelineConfig(
        training_config=training_config.to_dict(),
        data_config=data_config,
        model_config=model_config,
        registry_dir=None,
        tracking_dir=None,
        artifacts_dir=str(artifacts_dir),
        metadata={"job": job, "run_name": "hybrid_gnn_rnn_pt_graph_aligned_toy", "generate_training_report": False},
    )
    return config


# ======================================================================
# 2.  Step-by-Step Pipeline
# ======================================================================

def run_step_by_step(config: "PipelineConfig"):
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.train import HybridGnnRnnTrainPipeline
    from src.rade_ml_pt.training.trainer import Trainer, setup_training_environment
    from src.rade_ml_pt.evaluation.evaluator import Evaluator
    from src.rade_ml_pt.evaluation.metrics import rmse, mae, r_squared, mape
    from src.rade_ml_pt.registry.store import ModelRegistry

    pipeline = HybridGnnRnnTrainPipeline(config)
    training_config = pipeline._resolve_training_config()
    seed = pipeline._resolve_seed()
    setup_training_environment(training_config, seed)

    print("\n" + "=" * 70)
    print("STAGE 1: BUILD DATA (Graph-Aligned Toy)")
    print("=" * 70)
    data_result = pipeline.build_data(config)
    print(f"\n  Elementary PnL shape: {data_result.elementary_pnl.shape}")
    print(f"  Target PnL shape:     {data_result.target_pnl.shape}")

    print("\n" + "=" * 70)
    print("STAGE 2: BUILD MODEL")
    print("=" * 70)
    model = pipeline.build_model(config, data_result)

    print("\n" + "=" * 70)
    print("STAGE 3: COMPILE + TRAIN")
    print("=" * 70)
    trainer = Trainer(model=model, config=training_config, seed=seed)
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
    print("STAGE 5: EVALUATION")
    print("=" * 70)
    eval_ds = data_result.test_ds if data_result.test_ds is not None else data_result.val_ds

    evaluator = Evaluator(model, loss_fn=torch.nn.L1Loss())
    eval_result = evaluator.run(
        eval_ds,
        additional_metrics={"rmse": rmse, "mae": mae, "r_squared": r_squared, "mape": mape},
    )
    print(f"\n  {eval_result.summary()}")

    train_eval = evaluator.run(
        data_result.train_ds,
        additional_metrics={"r_squared": r_squared},
        return_predictions=False,
    )
    r2_train = train_eval.metrics.get("r_squared")
    r2_test = eval_result.metrics.get("r_squared")
    print(f"\n  Train R^2 = {r2_train:.4f} | Test R^2 = {r2_test:.4f}")

    if r2_test is not None:
        if r2_test > 0.5:
            print("  -> PASS: Model learned graph-aligned aggregation")
        elif r2_train > 0.5:
            print("  -> Generalization gap: model fits train but not test")
        else:
            print("  -> Model did not fit train (investigate)")

    print("\n" + "=" * 70)
    print("STAGE 6: MODEL REGISTRATION")
    print("=" * 70)
    registry_dir = Path(config.artifacts_dir) / "registry"
    registry = ModelRegistry(str(registry_dir))
    registry.register(
        model=model,
        training_result=result,
        tags=["hybrid_gnn_rnn", "graph_aligned_toy", "latest"],
        description="Hybrid GNN-RNN PT graph-aligned toy (target=graph-weighted mean)",
    )
    print("  Registered.")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)

    from src.rade_ml_pt.training.plots import show_training_plots
    show_training_plots(result)

    return result


# ======================================================================
# Main
# ======================================================================

def main():
    workdir = Path(tempfile.mkdtemp(prefix="rade_ml_pt_graph_aligned_toy_"))
    logger.info(f"Working directory: {workdir}")

    job = make_graph_aligned_toy_data(workdir, n_scenarios=1000, n_elementary=20, n_target=4)
    config = build_configs(workdir, job)
    run_step_by_step(config)

    logger.info(f"Artifacts in: {workdir}")


if __name__ == "__main__":
    main()
