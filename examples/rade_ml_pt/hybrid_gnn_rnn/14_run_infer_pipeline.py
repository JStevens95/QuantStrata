#!/usr/bin/env python3
"""
Ensemble Inference Pipeline — Standalone Example
==================================================

Loads an ensemble config + member versions from the registry (as saved
by ``EnsembleTrainPipeline``) and runs ``EnsembleInferencePipeline``
against either:

* **new-scenario shocks** (default) — the production path.  Reads risk-
  factor shock CSVs from a folder, re-prices every cluster's elementary
  trades, runs the ensemble forward pass.  Mirrors what the *Inference
  Console* will trigger for an analyst.

* a **synthetic smoke check** (``--synthetic-smoke``) — short-circuits
  the scenario-build step by handing each member a random tensor of
  the requested shape via the model-agnostic ``member_inputs``
  metadata shortcut.  This lets you exercise the **full pipeline
  wiring** (registry load → ensemble assembly → activity-log emit →
  forward pass → result build → CSV / JSON save) without needing
  any scenario CSVs or model-specific artifacts on disk.  Works
  against any registered ensemble — including the synthetic 2-cluster
  fixture used by the pytest suite.

Phase 1 highlight
-----------------
Demonstrates the new ``on_event`` hook on
:class:`EnsembleInferencePipeline`.  Every lifecycle transition
(``Pipeline started → Loading ensemble → Cluster context loaded →
Forward pass complete → Pipeline complete``) is streamed live to
stdout via a custom emit callback that wraps a thread-safe
:class:`EventCollector`, so you can see exactly what the UI activity
log will receive once Phase 2 wires it to a Dash callback.

Output sections
---------------
1. **HEADER** — what got loaded from the registry.
2. **ACTIVITY LOG** — live narration as events fire (Phase 1 contract).
3. **PREDICTION SUMMARY** — shape + per-target stats from
   :class:`InferenceResult`.
4. **DISK ARTIFACTS** — paths to ``predictions.csv`` and
   ``inference_result.json`` that ``post_infer()`` wrote.
5. **EVENT TIMELINE** — recap of the full event sequence the
   ``EventCollector`` captured (same data the UI Store will hold).

Usage
-----
::

    # production path — needs a folder of risk-factor shock CSVs
    python examples/rade_ml_pt/hybrid_gnn_rnn/14_run_infer_pipeline.py \
        --registry-dir   /data/model_store \
        --artifacts-dir  /data/infer_output \
        --ensemble-version production \
        --new-scenario-dir /data/scenarios/2026Q2_stress

    # synthetic smoke check — no scenarios required, works against
    # any registered ensemble (including the pytest fixture)
    python examples/rade_ml_pt/hybrid_gnn_rnn/14_run_infer_pipeline.py \
        --registry-dir   /tmp/infer_smoke/registry \
        --artifacts-dir  /tmp/infer_smoke/artifacts \
        --ensemble-version latest \
        --synthetic-smoke

    # silent path — proves on_event=None is a behavioural no-op
    python examples/rade_ml_pt/hybrid_gnn_rnn/14_run_infer_pipeline.py \
        --registry-dir   /tmp/infer_smoke/registry \
        --artifacts-dir  /tmp/infer_smoke/artifacts \
        --ensemble-version latest \
        --synthetic-smoke --no-events
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-45s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
# Quiet the pipeline's own info-level noise so the activity-log
# narration stays readable; the events themselves are the more
# structured channel for what's going on.
logging.getLogger("src.rade_ml_pt").setLevel(logging.WARNING)

logger = logging.getLogger("example.ensemble_infer")

DIVIDER     = "=" * 70
SUB_DIVIDER = "-" * 70

# Status icon palette — mirrors the UI's `_STATUS_ICON` table in
# `layouts/inference.py` so eyeballing this script feels the same as
# eyeballing the Inference Console activity log.
_STATUS_GLYPH = {
    "ok":      "[OK ]",
    "fail":    "[FAIL]",
    "running": "[... ]",
    "pending": "[   ]",
}


# ─────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the EnsembleInferencePipeline against either a folder "
            "of new-scenario shocks (the production path) or against "
            "the saved baseline elementary PnL (smoke check)."
        ),
    )
    parser.add_argument(
        "--registry-dir", type=str, required=True,
        help="Root registry directory (where EnsembleTrainPipeline "
             "saved member models + ensemble metadata).",
    )
    parser.add_argument(
        "--artifacts-dir", type=str, required=True,
        help="Where post_infer() should write predictions.csv + "
             "inference_result.json.  Created if missing.",
    )
    parser.add_argument(
        "--ensemble-version", type=str, default="latest",
        help="Ensemble version or tag to load (default: 'latest').",
    )
    parser.add_argument(
        "--new-scenario-dir", type=str, default=None,
        help="Folder containing one CSV per shocked risk factor "
             "(filename minus '.csv' is the risk-factor key).  "
             "Required unless --synthetic-smoke is set.",
    )
    parser.add_argument(
        "--synthetic-smoke", action="store_true",
        help="Skip new-scenario ingest; feed each cluster a random "
             "tensor of shape `[--smoke-batch, --smoke-features]` via "
             "the model-agnostic `member_inputs` shortcut.  Lets you "
             "exercise the full pipeline wiring (load → assemble → "
             "predict → save) against any registered ensemble without "
             "needing scenario CSVs on disk.",
    )
    parser.add_argument(
        "--smoke-batch", type=int, default=4,
        help="Batch size for --synthetic-smoke inputs (default: 4).",
    )
    parser.add_argument(
        "--smoke-features", type=int, default=4,
        help="Feature dim for --synthetic-smoke inputs (default: 4 — "
             "matches the pytest 2-cluster fixture).",
    )
    parser.add_argument(
        "--no-events", action="store_true",
        help="Pass on_event=None (the pre-Phase-1 default) so the "
             "pipeline runs silently.  Proves the hook is opt-in and "
             "doesn't change behaviour when not used.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=10,
        help="How many prediction rows to print in the per-scenario "
             "preview (default: 10).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────
# Live event printer — wraps EventCollector so we get *both* a
# real-time narration on stdout *and* the buffered list at the end.
# ─────────────────────────────────────────────────────────────────────


def _make_emit_callback(collector):
    """Return a callable that prints + buffers each :data:`ActivityEntry`.

    The pipeline's contract is just ``EmitFn = Callable[[entry], None]``
    — anything callable qualifies, so we can compose ``print`` with
    ``collector.append`` cleanly.
    """
    def _emit(entry: Dict[str, Any]) -> None:
        glyph  = _STATUS_GLYPH.get(entry["status"], "[?]")
        target = entry.get("target") or ""
        detail = entry.get("detail")
        line = f"  {glyph}  {entry['phase']:<32}"
        if target:
            line += f"  →  {target}"
        if detail:
            line += f"   ({detail})"
        # flush=True so the line lands before the pipeline's next
        # synchronous step; keeps the narration in real time even
        # with output buffering.
        print(line, flush=True)
        collector(entry)
    return _emit


# ─────────────────────────────────────────────────────────────────────
# Mode helpers
# ─────────────────────────────────────────────────────────────────────


def _build_synthetic_member_inputs(
    cluster_ids: List[str],
    *,
    batch:    int,
    features: int,
) -> Dict[str, Any]:
    """Build a model-agnostic ``member_inputs`` dict for the smoke path.

    Mirrors the pattern used by the pytest fixture
    (``test_infer_new_scenarios``): one ``{"features": tensor}`` per
    cluster of shape ``[batch, features]``.  When the pipeline sees
    ``metadata['inference']['member_inputs']`` populated, it
    short-circuits the input-build phase entirely (see
    ``_build_member_inputs`` in ``infer.py``) and routes the dict
    straight to ``EnsembleModel.predict``.

    Works against any registered ensemble whose member ``forward()``
    accepts a ``{"features": tensor}`` dict (the fixture's
    ``PipelineTestModel``); for production hybrid_gnn_rnn members
    you'll want ``--new-scenario-dir`` instead so the real input
    pipeline runs.
    """
    import torch

    return {
        cid: {"features": torch.randn(batch, features)}
        for cid in cluster_ids
    }


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main():  # noqa: C901 — sequential script; complexity is linear narration
    args = parse_args()

    if not args.synthetic_smoke and not args.new_scenario_dir:
        raise SystemExit(
            "Either --new-scenario-dir or --synthetic-smoke is required."
        )

    from src.rade_ml_pt.ensemble.config import EnsembleConfig
    from src.rade_ml_pt.ensemble.registry import EnsembleRegistry
    from src.rade_ml_pt.pipelines.ensemble.infer import (
        EnsembleInferencePipeline,
    )
    from src.rade_ml_pt.pipelines.ensemble.infer_events import EventCollector

    # ──────────────────────────────────────────────────────────────
    # 1. Load + report what the registry gave us
    # ──────────────────────────────────────────────────────────────
    ens_registry = EnsembleRegistry(args.registry_dir)
    saved_config, member_versions, resolved_version = ens_registry.load(
        args.ensemble_version,
    )
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
        print(f"    {cid}: version={ver}, trades={n_trades}")

    print(f"\n  Mode:         "
          f"{'synthetic-smoke' if args.synthetic_smoke else 'new_scenarios'}")
    if args.synthetic_smoke:
        print(f"  Smoke shape:  "
              f"[batch={args.smoke_batch}, features={args.smoke_features}]")
    if args.new_scenario_dir:
        print(f"  Scenarios:    {args.new_scenario_dir}")
    print(f"  Events:       "
          f"{'silent (on_event=None)' if args.no_events else 'streaming'}")

    # ──────────────────────────────────────────────────────────────
    # 2. Build the inference config — same EnsembleConfig shape the
    #    registry returned, plus a metadata['inference'] dict that
    #    tells the pipeline which input mode to take.
    # ──────────────────────────────────────────────────────────────
    infer_config = EnsembleConfig(
        member_configs=saved_config.member_configs,
        pipeline_class=saved_config.pipeline_class,
        cluster_mapping=saved_config.cluster_mapping,
        cluster_keys=saved_config.cluster_keys,
        cluster_key=saved_config.cluster_key,
        cluster_key_values=saved_config.cluster_key_values,
        aggregation=saved_config.aggregation,
        weights=saved_config.weights,
        execution_strategy="sequential",
        registry_dir=registry_dir,
        artifacts_dir=args.artifacts_dir,
    )

    if args.synthetic_smoke:
        member_inputs = _build_synthetic_member_inputs(
            cluster_ids=saved_config.cluster_ids,
            batch=args.smoke_batch,
            features=args.smoke_features,
        )
        infer_config.metadata["inference"] = {
            "input_mode":    "new_scenarios",
            "member_inputs": member_inputs,
        }
    else:
        infer_config.metadata["inference"] = {
            "input_mode":       "new_scenarios",
            "new_scenario_dir": args.new_scenario_dir,
        }

    # ──────────────────────────────────────────────────────────────
    # 3. Wire the on_event hook — this is the Phase 1 surface the
    #    UI activity log binds to.  --no-events flips this off so
    #    you can confirm pipeline behaviour is identical either way.
    # ──────────────────────────────────────────────────────────────
    collector = EventCollector()
    on_event = None if args.no_events else _make_emit_callback(collector)

    print(f"\n{SUB_DIVIDER}")
    print("ACTIVITY LOG  (live — pipeline emits in real time)")
    print(SUB_DIVIDER)
    if args.no_events:
        print("  (silenced — re-run without --no-events to see the stream)")

    pipeline = EnsembleInferencePipeline(
        ensemble_config=infer_config,
        ensemble_version=resolved_version,
        on_event=on_event,
    )

    t0 = time.perf_counter()
    try:
        result = pipeline.run()
    except Exception as exc:
        # The pipeline always emits a `Pipeline failed` event before
        # re-raising, so the activity log above already shows what
        # broke; just exit non-zero.
        print(f"\n{DIVIDER}")
        print("PIPELINE FAILED")
        print(DIVIDER)
        print(f"  {type(exc).__name__}: {exc}")
        sys.exit(2)
    wall = time.perf_counter() - t0

    # ──────────────────────────────────────────────────────────────
    # 4. Prediction summary
    # ──────────────────────────────────────────────────────────────
    import numpy as np

    preds = result.predictions
    print(f"\n{DIVIDER}")
    print("PREDICTION SUMMARY  (InferenceResult)")
    print(DIVIDER)
    print(f"  shape:           {preds.shape}")
    print(f"  n_samples:       {result.n_samples}")
    print(f"  model_version:   {result.model_version}")
    print(f"  pipeline.latency:{result.latency_seconds * 1000:8.1f} ms")
    print(f"  wall (incl I/O): {wall * 1000:8.1f} ms")
    if result.sample_ids:
        print(f"  sample_ids:      {len(result.sample_ids)} ids "
              f"(first: {result.sample_ids[0]!r}, "
              f"last: {result.sample_ids[-1]!r})")
    else:
        print(f"  sample_ids:      <none>")

    print(f"\n  global stats:")
    print(f"    mean = {float(np.mean(preds)):+.6f}")
    print(f"    std  = {float(np.std(preds)):+.6f}")
    print(f"    min  = {float(np.min(preds)):+.6f}")
    print(f"    max  = {float(np.max(preds)):+.6f}")

    if preds.ndim == 2 and preds.shape[1] > 0:
        print(f"\n  per-target stats (first 6 columns):")
        for j in range(min(6, preds.shape[1])):
            col = preds[:, j]
            print(f"    target_{j:02d}: mean={float(col.mean()):+.4f}  "
                  f"std={float(col.std()):+.4f}  "
                  f"min={float(col.min()):+.4f}  "
                  f"max={float(col.max()):+.4f}")

        n_show = min(args.max_rows, preds.shape[0])
        print(f"\n  preview (first {n_show} of {preds.shape[0]} rows · "
              f"first 6 of {preds.shape[1]} targets):")
        for i in range(n_show):
            row = "  ".join(
                f"{preds[i, j]:+.4f}" for j in range(min(6, preds.shape[1]))
            )
            label = (
                result.sample_ids[i]
                if result.sample_ids and i < len(result.sample_ids)
                else f"row_{i:03d}"
            )
            print(f"    {label:>14s}:  {row}")

    if result.metadata:
        print(f"\n  metadata:")
        for k, v in sorted(result.metadata.items()):
            if k == "per_member_sample_ids" and isinstance(v, dict):
                print(f"    {k}: " + ", ".join(
                    f"{cid}={len(ids) if ids else 0}" for cid, ids in v.items()
                ))
            else:
                print(f"    {k}: {v}")

    # ──────────────────────────────────────────────────────────────
    # 5. Disk artifacts post_infer() wrote
    # ──────────────────────────────────────────────────────────────
    out_dir = Path(args.artifacts_dir) / "inference"
    print(f"\n{DIVIDER}")
    print("DISK ARTIFACTS  (written by post_infer)")
    print(DIVIDER)
    print(f"  Dir: {out_dir}")
    for fname in ("predictions.csv", "inference_result.json"):
        fpath = out_dir / fname
        status = "OK" if fpath.exists() else "MISSING"
        size = f"{fpath.stat().st_size:,} bytes" if fpath.exists() else "—"
        print(f"    [{status}] {fname:<24s}  {size}")

    # ──────────────────────────────────────────────────────────────
    # 6. Event timeline recap — the same data the UI Store will hold
    # ──────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"EVENT TIMELINE  ({len(collector)} events captured)")
    print(DIVIDER)
    if args.no_events:
        print("  (silenced — re-run without --no-events to populate)")
    else:
        for i, e in enumerate(collector.snapshot()):
            glyph = _STATUS_GLYPH.get(e["status"], "[?]")
            target = e.get("target") or ""
            detail = e.get("detail")
            line = f"  {i:>3d}.  {glyph}  {e['phase']:<32}"
            if target:
                line += f"  →  {target}"
            if detail:
                line += f"   ({detail})"
            print(line)

    # ──────────────────────────────────────────────────────────────
    # 7. Done
    # ──────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print(f"DONE — {resolved_version} — "
          f"{result.n_samples} samples · "
          f"{result.latency_seconds * 1000:.0f} ms")
    print(DIVIDER)


if __name__ == "__main__":
    main()
