#!/usr/bin/env python3
"""
Ensemble Evaluation Pipeline — Standalone Example
===================================================

Loads an ensemble config and member versions from the registry (as saved
by ``EnsembleTrainPipeline``) and runs ``EnsembleEvalPipeline`` across
all available splits (train / val / test).

The ``registry_dir`` is read from the saved ``EnsembleConfig`` — the only
inputs are where the registry lives and where to write evaluation
artifacts.

Usage::

    python examples/rade_ml_pt/hybrid_gnn_rnn/07_test_ensemble_eval.py \\
        --registry-dir /data/model_store \\
        --artifacts-dir /data/eval_output \\
        --ensemble-version production
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("example.ensemble_eval")

DIVIDER = "=" * 70
SUB_DIVIDER = "-" * 70


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Ensemble Evaluation Pipeline from saved registry artifacts.",
    )
    parser.add_argument(
        "--registry-dir", type=str, required=True,
        help="Root registry directory (where EnsembleTrainPipeline saved models and ensemble metadata).",
    )
    parser.add_argument(
        "--artifacts-dir", type=str, required=True,
        help="Directory to write evaluation artifacts (can differ from training artifacts).",
    )
    parser.add_argument(
        "--ensemble-version", type=str, default="latest",
        help="Ensemble version or tag to evaluate (default: 'latest').",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    from src.rade_ml_pt.ensemble.config import EnsembleConfig
    from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
    from src.rade_ml_pt.pipelines.ensemble.eval import EnsembleEvalPipeline

    # ------------------------------------------------------------------
    # 1. Load ensemble config + member versions from registry
    # ------------------------------------------------------------------
    ens_registry = EnsembleRegistry(args.registry_dir)
    saved_config, member_versions, resolved_version = ens_registry.load(args.ensemble_version)

    registry_dir = saved_config.registry_dir or args.registry_dir

    print(f"\n{DIVIDER}")
    print("ENSEMBLE LOADED FROM REGISTRY")
    print(DIVIDER)
    print(f"  Requested:    '{args.ensemble_version}'")
    print(f"  Resolved:     {resolved_version}")
    print(f"  Registry:     {registry_dir}")
    print(f"  Artifacts:    {args.artifacts_dir}")
    print(f"  Clusters:     {saved_config.cluster_ids}")
    print(f"  Total trades: {len(saved_config.all_trade_ids)}")
    print(f"  Aggregation:  {saved_config.aggregation}")
    print(f"\n  Members:")
    for cid, ver in sorted(member_versions.items()):
        n_trades = len(saved_config.cluster_mapping.get(cid, []))
        ds_dir = Path(registry_dir) / ver / "datasets"
        splits = [s for s in ("train", "val", "test") if (ds_dir / f"{s}.pt").exists()]
        print(f"    {cid}: version={ver}, trades={n_trades}, cached_splits={splits}")

    # ------------------------------------------------------------------
    # 2. Build eval config — registry_dir from saved config, only
    #    artifacts_dir is overridden from CLI
    # ------------------------------------------------------------------
    eval_config = EnsembleConfig(
        cluster_mapping=saved_config.cluster_mapping,
        aggregation=saved_config.aggregation,
        execution_strategy="sequential",
        registry_dir=registry_dir,
        artifacts_dir=args.artifacts_dir,
    )

    # ------------------------------------------------------------------
    # 3. Run evaluation
    # ------------------------------------------------------------------
    print(f"\n{SUB_DIVIDER}")
    print("Running EnsembleEvalPipeline...")
    print(SUB_DIVIDER)

    eval_output = EnsembleEvalPipeline(
        ensemble_config=eval_config,
        ensemble_version=resolved_version,
    ).run()

    # ------------------------------------------------------------------
    # 4. Primary split (test)
    # ------------------------------------------------------------------
    ens_metrics = eval_output["ensemble_metrics"]
    pm_metrics = eval_output["per_member_metrics"]
    summary = eval_output["member_summary"]

    print(f"\n{DIVIDER}")
    print("PRIMARY SPLIT (test)")
    print(DIVIDER)

    if ens_metrics:
        print(f"\n  Ensemble metrics:")
        for k, v in sorted(ens_metrics.items()):
            print(f"    {k:10s}: {v:.6f}")

    if pm_metrics:
        print(f"\n  Per-member metrics:")
        for cid in sorted(pm_metrics.keys()):
            m = pm_metrics[cid]
            print(
                f"    {cid}: "
                f"MAE={m.get('mae', 0):.6f}  "
                f"MSE={m.get('mse', 0):.6f}  "
                f"RMSE={m.get('rmse', 0):.6f}  "
                f"scenarios={m.get('n_scenarios', '?')}  "
                f"targets={m.get('n_targets', '?')}"
            )

    if summary:
        print(f"\n  Member rollup:")
        for k, v in sorted(summary.items()):
            if k == "per_member":
                continue
            print(f"    {k}: {v:.6f}" if isinstance(v, float) else f"    {k}: {v}")

    if not ens_metrics and not pm_metrics:
        print("\n  No test data available for any member.")

    # ------------------------------------------------------------------
    # 5. Additional splits (train / val)
    # ------------------------------------------------------------------
    additional = eval_output.get("additional_splits", {})
    if additional:
        print(f"\n{DIVIDER}")
        print("ADDITIONAL SPLITS")
        print(DIVIDER)

        for split_name in ("train", "val"):
            results = additional.get(split_name)
            if not results:
                continue
            ens_m = results.get("ensemble_metrics", {})
            pm_m = results.get("per_member_metrics", {})

            print(f"\n  [{split_name.upper()}]")
            if ens_m:
                print(f"    Ensemble — MAE: {ens_m.get('mae', 0):.6f}, RMSE: {ens_m.get('rmse', 0):.6f}")
            if pm_m:
                for cid in sorted(pm_m.keys()):
                    m = pm_m[cid]
                    print(f"    {cid}: MAE={m.get('mae', 0):.6f}, MSE={m.get('mse', 0):.6f}")

    # ------------------------------------------------------------------
    # 6. Verify artifacts on disk
    # ------------------------------------------------------------------
    eval_dir = Path(args.artifacts_dir) / "ensemble" / resolved_version / "evaluation"

    print(f"\n{DIVIDER}")
    print("ARTIFACTS")
    print(DIVIDER)
    print(f"  Dir: {eval_dir}")

    for fname in ("ensemble_metrics.json", "per_member_metrics.json", "member_rollup.json"):
        fpath = eval_dir / fname
        status = "OK" if fpath.exists() else "MISSING"
        print(f"    [{status}] {fname}")

    for split in ("train", "val"):
        for base in ("ensemble_metrics", "per_member_metrics", "member_rollup"):
            fpath = eval_dir / f"{base}_{split}.json"
            if fpath.exists():
                print(f"    [OK] {base}_{split}.json")

    plots_dir = eval_dir / "plots"
    if plots_dir.exists():
        for split_dir in sorted(plots_dir.iterdir()):
            if split_dir.is_dir():
                pngs = list(split_dir.glob("*.png"))
                print(f"    plots/{split_dir.name}/: {len(pngs)} plot(s)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    n_splits = 1 + len(additional)
    print(f"\n{DIVIDER}")
    print(f"DONE — {resolved_version} — {n_splits} split(s) evaluated")
    print(DIVIDER)


if __name__ == "__main__":
    main()
