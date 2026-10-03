#!/usr/bin/env python3
"""
Hyperparameter Tuning Example: Hybrid GNN-RNN Model (PyTorch)
=============================================================

This script demonstrates the ``HybridGnnRnnTunePipeline`` using synthetic
data.  It is designed to:

  1. Validate that the tuning pipeline is wired correctly end-to-end.
  2. Show how to configure the search space, tuner, and optional retrain.
  3. Be runnable out-of-the-box:
     ``python examples/rade_ml_pt/hybrid_gnn_rnn/05_tune_hybrid_gnn_rnn.py``

What it does
------------
  0. Generates synthetic PnL and trade attributes (same as 01_train).
  1. Builds a ``PipelineConfig`` with data, training, and metadata settings.
  2. Instantiates ``HybridGnnRnnTunePipeline`` with Optuna tuner kwargs.
  3. Runs the tuning study — each trial samples architecture + training
     params, builds a fresh model, trains it, and reports val loss.
  4. Logs the best trial's params and score.
  5. (Optional) Retrains the best configuration via the full train pipeline
     when ``retrain_best=True`` is set in metadata.

Tuning tips
-----------
  - Start with a small ``n_trials`` (5-10) to verify the pipeline runs.
  - Increase to 50-100+ for a real search.
  - Use ``pruner="hyperband"`` to cut unpromising trials early.
  - Keep ``epochs`` low (10-30) during tuning; use full epochs for retrain.
  - The ``batch_size`` param is in the search space but only affects the
    retrain step (the DataLoader is built once with the config's batch_size).

Run::

    python examples/rade_ml_pt/hybrid_gnn_rnn/05_tune_hybrid_gnn_rnn.py
"""
from __future__ import annotations

import sys
import logging
import tempfile
from pathlib import Path

# ensure project root is importable.
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
logger = logging.getLogger("example.hybrid_gnn_rnn_pt.tune")


# ======================================================================
# 0.  Synthetic Data Generation
# ======================================================================
# Reuses the same data generation as 01_train — small Gaussian PnL with
# target = noisy linear combination of elementary PnL.
# ======================================================================

def make_synthetic_data(
    workdir: Path,
    n_scenarios: int = 200,
    n_elementary: int = 20,
    n_target: int = 4,
    seed: int = 42,
) -> dict:
    """Generate synthetic PnL and attributes, return a job dict."""
    rng = np.random.RandomState(seed)

    # build trade IDs across 2 underlyings × 2 product types.
    underlyings = ["EURUSD", "GBPUSD"]
    product_types = ["vanilla_option", "forward"]

    elem_ids = []
    idx = 1
    for und in underlyings:
        for prod in product_types:
            n_per_group = n_elementary // (len(underlyings) * len(product_types))
            for _ in range(n_per_group):
                elem_ids.append(f"{und}|{prod}|{idx}")
                idx += 1
    while len(elem_ids) < n_elementary:
        elem_ids.append(f"EURUSD|vanilla_option|{idx}")
        idx += 1
    elem_ids = elem_ids[:n_elementary]

    tgt_ids = [f"EURUSD|vanilla_option|tgt_{i + 1}" for i in range(n_target)]

    logger.info(f"Trade universe: {n_elementary} elementary + {n_target} target")

    # elementary PnL: small Gaussian daily changes.
    elem_pnl_arr = rng.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01

    # target PnL: noisy linear combination of elementary (learnable signal).
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

    # persist to disk so the data pipeline can load them via CacheLoader.
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
# The tune pipeline needs the same data and training configs as training,
# but with lower epochs (trials should be fast) and no model_config
# (architecture is varied per trial by the search space).
# ======================================================================

def build_tune_config(workdir: Path, job: dict, n_trials: int = 5) -> tuple:
    """
    Build the PipelineConfig and tuner_kwargs for a tuning run.

    Returns (config, tuner_kwargs) tuple.
    """
    from src.rade_ml_pt.pipelines.config import PipelineConfig
    from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        BasisSelectionConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml_pt.core.config import TrainingConfig, OptimizerConfig, EarlyStoppingConfig

    artifacts_dir = workdir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # -- Data config (same as training) --
    data_config = HybridGnnRnnDataConfig(
        folders=FolderEnvironmentConfig(root_folder=str(workdir)),
        # splits: 85% train, 10% val, 5% test.
        validation_split=0.10,
        test_split=0.05,
        # single-day PnL snapshots.
        seq_length=1,
        # batch size for the shared DataLoader (built once before all trials).
        batch_size=16,
        shuffle=True,
        cache=False,
        drop_remainder=False,
        # PnL standardisation — fix this before tuning; evaluate separately.
        transform_type="standard",
        # dimensionality reduction: PCA basis selection.
        dimensionality=DimensionalityConfig(
            reduction_mode="basis_selection",
            basis_selection=BasisSelectionConfig(
                var_threshold=0.999999,
                method="pca",
                max_components=10,
            ),
        ),
        # k-NN trade graph.
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
        # attribute encoder.
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

    # -- Training config --
    # keep epochs LOW during tuning — each trial should finish fast.
    # the search space tunes lr and loss; early stopping handles convergence.
    training_config = TrainingConfig(
        epochs=15,
        loss="mae",
        metrics=["mae"],
        optimizer=OptimizerConfig(
            name="adam",
            learning_rate=1e-3,
        ),
        early_stopping=EarlyStoppingConfig(
            patience=5,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),
        strategy="auto",
        mixed_precision=False,
        verbose=False,
    )

    # -- Pipeline config --
    # set retrain_best=True to automatically retrain the winning config
    # with the full train pipeline after tuning completes.
    config = PipelineConfig(
        training_config=training_config.to_dict(),
        data_config=data_config,
        model_config=None,      # architecture is varied per trial.
        registry_dir=None,
        tracking_dir=None,
        artifacts_dir=str(artifacts_dir),
        metadata={
            "job": job,
            "run_name": "hybrid_gnn_rnn_tune_example",
            "retrain_best": False,  # set True to retrain winner after tuning.
        },
    )

    # -- Tuner kwargs --
    # these are forwarded to the Optuna-backed Tuner.
    # note: seed is NOT needed here — the base TunePipeline.run() automatically
    # injects the pipeline seed (from data_config.seed) into the Tuner.
    tuner_kwargs = {
        "n_trials": n_trials,       # number of Optuna trials to run.
        "direction": "minimize",    # minimise validation loss.
        "pruner": "median",         # prune trials worse than running median.
    }

    logger.info(f"Tune config: n_trials={n_trials}, epochs={training_config.epochs}, "
                f"batch_size={data_config.batch_size}, transform={data_config.transform_type}")

    return config, tuner_kwargs


# ======================================================================
# 2.  Run Tuning Pipeline
# ======================================================================

def run_tuning(config, tuner_kwargs) -> None:
    """Instantiate and run the HybridGnnRnnTunePipeline."""
    from src.rade_ml_pt.pipelines.hybrid_gnn_rnn.tune import HybridGnnRnnTunePipeline

    print("\n" + "=" * 70)
    print("TUNING: HybridGnnRnnTunePipeline")
    print("=" * 70)
    print(f"\n  Tuner settings: {tuner_kwargs}")
    print(f"  Retrain best:   {config.metadata.get('retrain_best', False)}")

    # instantiate the tuning pipeline.
    pipeline = HybridGnnRnnTunePipeline(config, tuner_kwargs=tuner_kwargs)

    # run() builds data once, then runs n_trials Optuna trials,
    # each sampling from the search space, building a model, and training.
    result = pipeline.run()

    # -- display results --
    print("\n" + "=" * 70)
    print("TUNING RESULTS")
    print("=" * 70)
    print(f"\n  Best trial:      #{result.best_trial_number}")
    print(f"  Best val loss:   {result.best_value:.6f}")
    print(f"  Trials:          {result.n_completed} completed, {result.n_pruned} pruned")
    print(f"  Total time:      {result.elapsed_seconds:.1f}s")

    print(f"\n  Best hyperparameters:")
    for param_name, param_value in sorted(result.best_params.items()):
        print(f"    {param_name:25s} = {param_value}")

    # -- per-trial summary --
    print(f"\n  All trials:")
    print(f"    {'#':>4s}  {'val_loss':>10s}  {'duration':>8s}  {'state':>10s}")
    print(f"    {'─' * 4}  {'─' * 10}  {'─' * 8}  {'─' * 10}")
    for trial in result.all_trials:
        val = f"{trial['value']:.6f}" if trial["value"] is not None else "     N/A"
        dur = f"{trial['duration_seconds']:.1f}s" if trial["duration_seconds"] else "    N/A"
        marker = " <-- best" if trial["number"] == result.best_trial_number else ""
        print(f"    {trial['number']:4d}  {val:>10s}  {dur:>8s}  {trial['state']:>10s}{marker}")

    print("\n" + "=" * 70)
    print("TUNING COMPLETE")
    print("=" * 70)

    return result


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="rade_ml_pt_tune_"))
    logger.info(f"Working directory: {workdir}")

    # 0. generate synthetic data.
    job = make_synthetic_data(workdir, n_scenarios=200, n_elementary=20, n_target=4)

    # 1. build tune config — start with n_trials=5 for a quick smoke test.
    #    increase to 50-100+ for real hyperparameter search.
    config, tuner_kwargs = build_tune_config(workdir, job, n_trials=5)

    # 2. run tuning.
    result = run_tuning(config, tuner_kwargs)

    logger.info(f"All artifacts in: {workdir}")


if __name__ == "__main__":
    main()
