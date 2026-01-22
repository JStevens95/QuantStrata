"""
Orchestrator Smoke Test (no domain dependencies, no dataset paths).

Purpose
-------
This file is a *true* end-to-end validation of your orchestrator runtime building blocks
without touching MarketDataset, providers, pricers, ML, etc.

It demonstrates:
1) Defining Steps inline (in the same file) and composing a Pipeline dynamically.
2) Running the Pipeline via PipelineRunner and passing Context forward step-by-step.
3) Producing real on-disk outputs using ArtifactStore + logger:
   - logs/<run.log>
   - artifacts/state.json
   - (optionally) manifest.json if you decide to write it here too.

Design notes (Vn-proof)
-----------------------
- No reliance on "built-in pipeline discovery" or registry wiring; this isolates the core runner.
- Uses atomic-ish file writing for artifacts (write temp -> replace).
- Uses explicit, stable state keys that scale as you add more steps.
- Keeps steps small, deterministic, and side-effect boundaries obvious.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from src.orchestrator.artifacts.store import ArtifactStore
from src.orchestrator.config.schemas import IOConfig, RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import Pipeline, PipelineRunner
from src.orchestrator.core.step import Step
from src.orchestrator.logging.setup import build_run_logger


# =============================================================================
# Small utilities (kept local to this example so we don't pollute the library)
# =============================================================================

def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _write_json_atomic(*, path: Path, payload: Mapping[str, Any]) -> None:
    """
    Write JSON to disk using a simple atomic-ish pattern:
    write to `*.tmp` then replace the destination.

    This reduces the chance of partially-written artifacts when runs are interrupted.
    """
    # Ensure parent directory exists so the write never fails due to missing folders.
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize JSON with stable formatting (diff-friendly).
    content = json.dumps(dict(payload), indent=2, sort_keys=True)

    # Write to a temporary file first.
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")

    # Replace destination (atomic on most OS/filesystems).
    tmp_path.replace(path)


# =============================================================================
# Step implementations (inline, no extra modules)
# =============================================================================

@dataclass(frozen=True, slots=True)
class InitNumbersStep(Step):
    """
    Initialize two integers in ctx.state.

    Outputs
    -------
    ctx.state["a"] : int
    ctx.state["b"] : int
    """

    def run(self, ctx: Context) -> Context:
        # Put deterministic values into state (simple, testable).
        ctx.put("a", 10)
        ctx.put("b", 5)

        # Log an explicit message so you can see Step ordering in run.log.
        ctx.logger.info("InitNumbersStep | a=%s | b=%s", ctx.get("a"), ctx.get("b"))
        return ctx


@dataclass(frozen=True, slots=True)
class ComputeSumStep(Step):
    """
    Compute a + b and store the result.

    Inputs
    ------
    ctx.state["a"] : int (or int-like)
    ctx.state["b"] : int (or int-like)

    Outputs
    -------
    ctx.state["sum"] : int
    """

    def run(self, ctx: Context) -> Context:
        # Pull values from state with explicit defaults.
        a_raw = ctx.get("a", 0)
        b_raw = ctx.get("b", 0)

        # Coerce to ints (in real steps you might validate types more strictly).
        a = int(a_raw)
        b = int(b_raw)

        # Compute and persist result.
        total = a + b
        ctx.put("sum", total)

        ctx.logger.info("ComputeSumStep | %d + %d = %d", a, b, total)
        return ctx


@dataclass(frozen=True, slots=True)
class PrintStateArtifactStep(Step):
    """
    "Save" an artifact by printing it to console (no filesystem IO).

    This is ideal for smoketests because it:
    - runs the pipeline fully
    - validates state wiring + step order
    - shows outputs clearly
    - writes zero files
    """

    def run(self, ctx: Context) -> Context:
        payload: Dict[str, Any] = {
            "workdir": str(ctx.artifact_store.workdir),
            "run_id": ctx.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "state": {k: ctx.state[k] for k in sorted(ctx.state.keys())},
        }

        # Print a pretty JSON blob to console (your “artifact”).
        ctx.logger.info("ARTIFACT_PRINT_START | state.json")
        print(json.dumps(payload, indent=2, sort_keys=True))
        ctx.logger.info("ARTIFACT_PRINT_END   | state.json")

        # Still store something in state so downstream steps can reference it.
        ctx.put("state_json_path", None)
        return ctx


@dataclass(frozen=True, slots=True)
class SaveStateJsonStep(Step):
    """
    Persist (a small subset of) ctx.state to an artifact JSON file.

    This simulates "saving an artifact" without involving MarketDataset, models, etc.

    Outputs
    -------
    ctx.state["artifact.state_json_path"] : str
    """

    filename: str = "state.json"

    def run(self, ctx: Context) -> Context:
        # Access the store (Context owns it; steps should not create stores).
        store: ArtifactStore = ctx.artifact_store

        # Keep the payload small and deterministic (avoid serializing huge objects).
        payload: Dict[str, Any] = {
            "workdir": str(ctx.artifact_store.workdir),
            "run_id": ctx.run_id,
            "timestamp_utc": _utc_now_iso(),
            "state": {k: ctx.state[k] for k in sorted(ctx.state.keys())},
        }

        # Choose a stable artifact location.
        out_path = store.artifacts_root / self.filename

        # Write using atomic-ish IO to avoid partial files.
        _write_json_atomic(path=out_path, payload=payload)

        # Store the path back into state under a namespaced key (scales well in Vn).
        ctx.put("artifact.state_json_path", str(out_path))

        ctx.logger.info("SaveStateJsonStep | wrote=%s", out_path)
        return ctx


# =============================================================================
# Main runner
# =============================================================================

def main(save_files: bool = False) -> None:
    """
    Execute a minimal inline pipeline and print where outputs landed.

    You can run this file directly:
        python examples/orchestrator/run_orchestrator_smoketest.py
    """
    # -------------------------------------------------------------------------
    # Build a minimal RunConfig (this keeps the Context shape consistent with Vn)
    # -------------------------------------------------------------------------
    cfg = RunConfig(
        pipeline="__inline_smoketest__",
        io=IOConfig(workdir="./.runs"),
        params={},
    )

    # -------------------------------------------------------------------------
    # Create a stable run id and materialize the run folder layout
    # -------------------------------------------------------------------------
    run_id = f"smoketest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # Resolve the workdir once; avoids surprises with relative paths.
    workdir = Path(str(cfg.io.workdir)).expanduser().resolve()

    # Construct store using the same names your tests expect: run_path/artifacts_path/logs_path.
    store = ArtifactStore(
        workdir=workdir,
        run_id=run_id,
        artifacts_dirname=str(cfg.io.artifacts_dir),
        logs_dirname=str(cfg.io.logs_dir),
    )

    # Ensure disk layout exists before logging/writing artifacts.
    if save_files:
        store.ensure_layout()

    # -------------------------------------------------------------------------
    # Build a run logger that logs to console + file
    # -------------------------------------------------------------------------
    logger = build_run_logger(
        "QuantStrata.Orchestrator.SmokeTest",
        log_file=store.logs_root / "run.log" if save_files else None,
    )

    # -------------------------------------------------------------------------
    # Create the base Context passed between steps
    # -------------------------------------------------------------------------
    ctx = Context(
        run_id=run_id,
        cfg=cfg,
        logger=logger,
        artifact_store=store,
        provider=None,  # intentionally unused in this smoke test
        state={},
    )

    # -------------------------------------------------------------------------
    # Define Steps inline and build the Pipeline
    # -------------------------------------------------------------------------
    steps = [
        InitNumbersStep(name="init_numbers"),
        ComputeSumStep(name="compute_sum"),
        PrintStateArtifactStep(name="print_state_artifact"),
        # SaveStateJsonStep(name="save_state_json", filename="state.json"),
    ]

    pipeline = Pipeline(name="inline_smoketest_v1", steps=steps)

    # -------------------------------------------------------------------------
    # Run the pipeline with common runner controls
    # -------------------------------------------------------------------------
    runner = PipelineRunner(
        only=None,         # Example: {"compute_sum"} to run only a subset
        skip=None,         # Example: {"save_state_json"} to avoid IO step
        resume_from=None,  # Example: "compute_sum" to start execution mid-pipeline
        dry_run=False,     # Set True to validate selection without executing steps
    )

    final_ctx = runner.run(pipeline, ctx)

    # -------------------------------------------------------------------------
    # Print where outputs are and what state keys exist
    # -------------------------------------------------------------------------
    print("\nDONE")
    print(f"Run folder: {store.run_root if save_files else "save_files: False"}")
    print(f"Artifacts : {store.artifacts_root if save_files else "save_files: False"}")
    print(f"Logs      : {store.logs_root if save_files else "save_files: False"}")
    print(f"State keys: {sorted(final_ctx.state.keys())}")
    print(f"state.json: {final_ctx.get('artifact.state_json_path')}")


if __name__ == "__main__":
    main(save_files=False)