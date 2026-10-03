#!/usr/bin/env python3
"""
Test: Ensemble Training Pipeline (Hybrid GNN-RNN)
==================================================

Generates synthetic toy data for 2 clusters, builds an EnsembleConfig,
runs EnsembleTrainPipeline, and validates the outputs (member versions,
ensemble registration, artifacts on disk).

Uses small data (500 scenarios, 10 elementary, 3 target per cluster)
and short training (30 epochs) so the full run completes in under a minute.

Usage::

    python examples/rade_ml_pt/hybrid_gnn_rnn/06_test_ensemble_train.py
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.ensemble_train_test")


# ======================================================================
# 1.  Synthetic Data Generation (per cluster)
# ======================================================================

def make_cluster_data(
    cluster_dir: Path,
    cluster_id: str,
    n_scenarios: int = 500,
    n_elementary: int = 10,
    n_target: int = 3,
    seed: int = 42,
) -> dict:
    """
    Generate toy data for one cluster.

    target_pnl = mean(elementary_pnl) + small noise

    Returns a job dict with cluster_info paths, plus the target trade IDs.
    """
    from src.rade_ml_pt.data.io import CacheLoader

    rng = np.random.RandomState(seed)
    cluster_dir.mkdir(parents=True, exist_ok=True)

    elem_ids = [f"{cluster_id}|elem_{i}" for i in range(n_elementary)]
    tgt_ids = [f"{cluster_id}|tgt_{i}" for i in range(n_target)]

    elem_pnl_arr = rng.randn(n_scenarios, n_elementary).astype(np.float32) * 0.01
    mean_pnl = np.mean(elem_pnl_arr, axis=1, keepdims=True)
    tgt_pnl_arr = np.broadcast_to(mean_pnl, (n_scenarios, n_target)).copy()
    tgt_pnl_arr += rng.randn(n_scenarios, n_target).astype(np.float32) * 1e-5

    elem_pnl = pd.DataFrame(elem_pnl_arr, columns=elem_ids)
    tgt_pnl = pd.DataFrame(tgt_pnl_arr, columns=tgt_ids)

    def _make_attrs(trade_ids, trade_type="option"):
        n = len(trade_ids)
        return {
            "trade_id": trade_ids,
            "moneyness": rng.uniform(0.8, 1.2, n).tolist(),
            "yrs_to_maturity": rng.uniform(0.1, 2.0, n).tolist(),
            "delta": rng.uniform(-1.0, 1.0, n).tolist(),
            "vega": rng.uniform(0.0, 0.5, n).tolist(),
            "product_type": ["vanilla_option"] * n,
            "product_subtype": ["european"] * n,
            "trade_type": [trade_type] * n,
            "underlying_risk_factors": [["FX"]] * n,
        }

    paths = {
        "elementary_pnl_path": str(cluster_dir / "elem_pnl.pkl"),
        "target_pnl_path": str(cluster_dir / "tgt_pnl.pkl"),
        "elementary_attribs_path": str(cluster_dir / "elem_attrs.pkl"),
        "target_attribs_path": str(cluster_dir / "tgt_attrs.pkl"),
    }

    CacheLoader.save_data(elem_pnl, paths["elementary_pnl_path"])
    CacheLoader.save_data(tgt_pnl, paths["target_pnl_path"])
    CacheLoader.save_data(_make_attrs(elem_ids, "elementary"), paths["elementary_attribs_path"])
    CacheLoader.save_data(_make_attrs(tgt_ids, "target"), paths["target_attribs_path"])

    logger.info(
        "Cluster '%s': %d scenarios, %d elementary, %d target trades",
        cluster_id, n_scenarios, n_elementary, n_target,
    )

    return {
        "job": {"cluster_info": paths},
        "trade_ids": tgt_ids,
    }


# ======================================================================
# 2.  Build EnsembleConfig
# ======================================================================

def build_ensemble_config(
    workdir: Path,
    cluster_data: dict,
) -> "EnsembleConfig":
    from src.rade_ml_pt.ensemble.config import EnsembleConfig
    from src.rade_ml_pt.data.hybrid_gnn_rnn.config import (
        HybridGnnRnnDataConfig,
        FolderEnvironmentConfig,
        DimensionalityConfig,
        GraphBuilderConfig,
        AttributeEncoderConfig,
    )
    from src.rade_ml_pt.core.config import (
        TrainingConfig,
        OptimizerConfig,
        EarlyStoppingConfig,
    )

    registry_dir = str(workdir / "registry")
    artifacts_dir = str(workdir / "artifacts")

    training_config = TrainingConfig(
        epochs=30,
        loss="mae",
        metrics=["mae"],
        optimizer=OptimizerConfig(name="adam", learning_rate=1e-3),
        early_stopping=EarlyStoppingConfig(
            patience=10,
            monitor="val_loss",
            mode="min",
            restore_best_weights=True,
        ),
        strategy="cpu",
        verbose=False,
    )

    cluster_mapping = {}
    member_configs = {}
    cluster_key_values = {}

    for cluster_id, cdata in cluster_data.items():
        cluster_mapping[cluster_id] = cdata["trade_ids"]

        cluster_dir = workdir / "data" / cluster_id

        data_config = HybridGnnRnnDataConfig(
            folders=FolderEnvironmentConfig(root_folder=str(cluster_dir)),
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
                k=5,
                distance_metric="euclidean",
                include_quota=False,
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

        member_configs[cluster_id] = {
            "data_config": data_config,
            "training_config": training_config.to_dict(),
            "model_config": None,
            "metadata": {
                "job": cdata["job"],
                "run_name": f"ensemble_test_{cluster_id}",
                "tags": [cluster_id, "ensemble_test"],
            },
        }

        cluster_key_values[cluster_id] = ["GBP" if "0" in cluster_id else "USD", "FLOW_RATES"]

    config = EnsembleConfig(
        member_configs=member_configs,
        cluster_mapping=cluster_mapping,
        cluster_key=["ccy", "desk"],
        cluster_key_values=cluster_key_values,
        aggregation="concat",
        execution_strategy="sequential",
        registry_dir=registry_dir,
        artifacts_dir=artifacts_dir,
        metadata={"run_name": "ensemble_train_test"},
    )

    return config


# ======================================================================
# 3.  Run and Validate
# ======================================================================

def run_test(config: "EnsembleConfig") -> None:
    from src.rade_ml_pt.pipelines.ensemble.train import EnsembleTrainPipeline

    print("\n" + "=" * 70)
    print("ENSEMBLE TRAINING PIPELINE TEST")
    print("=" * 70)
    print(f"  Clusters:       {config.cluster_ids}")
    print(f"  Total trades:   {len(config.all_trade_ids)}")
    print(f"  Aggregation:    {config.aggregation}")
    print(f"  Strategy:       {config.execution_strategy}")
    print(f"  Registry dir:   {config.registry_dir}")
    print(f"  Artifacts dir:  {config.artifacts_dir}")

    router_keys = config.get_cluster_keys_for_router()
    if router_keys:
        print(f"  Routing keys:")
        for cid, keys in router_keys.items():
            print(f"    {cid}: {keys}")

    # --- Run the pipeline ---
    print("\n" + "-" * 70)
    print("Running EnsembleTrainPipeline...")
    print("-" * 70)

    pipeline = EnsembleTrainPipeline(config, tags=["test", "latest"])
    result = pipeline.run()

    # --- Validate outputs ---
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    ensemble_version = result["ensemble_version"]
    member_versions = result["member_versions"]
    member_results = result["member_results"]

    print(f"\n  Ensemble version: {ensemble_version}")
    print(f"\n  Member results:")
    for cid in sorted(member_versions.keys()):
        ver = member_versions[cid]
        res = member_results[cid]
        print(
            f"    {cid}: version={ver}, "
            f"best_val_loss={res.best_val_loss:.6f}, "
            f"best_epoch={res.best_epoch}, "
            f"stopped_early={res.stopped_early}"
        )

    # --- Verify registry files ---
    print("\n" + "=" * 70)
    print("REGISTRY VERIFICATION")
    print("=" * 70)

    registry_dir = Path(config.registry_dir)
    ens_dir = registry_dir / "ensemble" / ensemble_version

    expected_files = [
        "ensemble_config.json",
        "member_versions.json",
        "trade_cluster_map.json",
        "member_summary.json",
    ]
    print(f"\n  Ensemble registry dir: {ens_dir}")
    for fname in expected_files:
        fpath = ens_dir / fname
        status = "OK" if fpath.exists() else "MISSING"
        print(f"    [{status}] {fname}")

    # Verify member model files
    print(f"\n  Member model directories:")
    for cid, ver in sorted(member_versions.items()):
        ver_dir = registry_dir / ver
        model_exists = (ver_dir / "model.pt").exists()
        test_exists = (ver_dir / "datasets" / "test.pt").exists()
        graph_exists = (ver_dir / "graph_builder.pkl").exists()
        encoder_exists = (ver_dir / "encoder.pkl").exists()
        print(
            f"    {cid} ({ver}):"
            f" model.pt={'OK' if model_exists else 'MISSING'},"
            f" test.pt={'OK' if test_exists else 'MISSING'},"
            f" graph_builder={'OK' if graph_exists else 'MISSING'},"
            f" encoder={'OK' if encoder_exists else 'MISSING'}"
        )

    # --- Verify the ensemble can be loaded back ---
    print("\n" + "=" * 70)
    print("ROUND-TRIP VERIFICATION")
    print("=" * 70)

    from src.rade_ml_pt.ensemble.registry import EnsembleRegistry

    ens_registry = EnsembleRegistry(config.registry_dir)
    loaded_config, loaded_versions, loaded_version = ens_registry.load("latest")

    print(f"\n  Loaded ensemble version: {loaded_version}")
    print(f"  Loaded cluster_ids:     {loaded_config.cluster_ids}")
    print(f"  Loaded n_members:       {loaded_config.n_members}")
    print(f"  Loaded total trades:    {len(loaded_config.all_trade_ids)}")
    print(f"  Loaded member versions: {loaded_versions}")

    assert loaded_version == ensemble_version, "Version mismatch!"
    assert set(loaded_versions.keys()) == set(member_versions.keys()), "Member version keys mismatch!"
    assert loaded_config.n_members == config.n_members, "Member count mismatch!"
    assert len(loaded_config.all_trade_ids) == len(config.all_trade_ids), "Trade count mismatch!"

    # --- Verify EnsembleBuilder can reconstruct the ensemble ---
    from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
    from src.rade_ml_pt.registry.store import ModelRegistry

    model_registry = ModelRegistry(config.registry_dir)
    builder = EnsembleBuilder(model_registry)
    ensemble = builder.build(loaded_config, loaded_versions)

    print(f"\n  EnsembleModel rebuilt:")
    print(f"    Members:       {sorted(ensemble.members.keys())}")
    print(f"    Router trades: {ensemble.router.n_trades}")
    print(f"    Aggregation:   {ensemble.aggregation}")

    member_meta = ensemble.get_member_metadata()
    for cid, meta in sorted(member_meta.items()):
        print(f"    {cid}: {meta['model_class']}, {meta['n_parameters']:,} params, {meta['n_trades']} trades")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("ALL CHECKS PASSED")
    print("=" * 70)


# ======================================================================
# Main
# ======================================================================

def main():
    workdir = Path(tempfile.mkdtemp(prefix="rade_ensemble_train_test_"))
    logger.info(f"Working directory: {workdir}")

    # Generate synthetic data for 2 clusters
    data_dir = workdir / "data"
    cluster_data = {}
    for i, seed in enumerate([42, 99]):
        cid = f"cluster_{i}"
        cdata = make_cluster_data(
            cluster_dir=data_dir / cid,
            cluster_id=cid,
            n_scenarios=500,
            n_elementary=10,
            n_target=3,
            seed=seed,
        )
        cluster_data[cid] = cdata

    config = build_ensemble_config(workdir, cluster_data)
    run_test(config)

    logger.info(f"All artifacts in: {workdir}")


if __name__ == "__main__":
    main()
