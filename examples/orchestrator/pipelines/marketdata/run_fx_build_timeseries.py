"""
FX Marketdata Example — using a built-in pipeline (registry + discovery).

This script demonstrates the *intended* user workflow for built-in pipelines:

  1) Build a RunConfig that targets a pipeline name (registered via discovery).
  2) Build a Pipeline from the registry using that RunConfig.
  3) Execute the pipeline via PipelineRunner.
  4) Read outputs from Context.state (and optionally Context.provider).

Important: print at the END
---------------------------
- The orchestrator logger emits messages during execution (STEP_START/STEP_END).
- To keep console output clean, we do *no printing* until the pipeline completes.
"""

from __future__ import annotations

# Standard library imports (pure Python).
from datetime import datetime, timedelta, timezone  # time utilities for run_id and request dates
from pathlib import Path  # filesystem path handling (workdir, optional logs/artifacts)
from typing import Any, Dict, Optional  # type hints for config/state dictionaries

# Orchestrator imports (your orchestration framework).
from src.orchestrator.artifacts.store import ArtifactStore  # run folder layout (optional)
from src.orchestrator.config.schemas import IOConfig, RunConfig  # run configuration schema objects
from src.orchestrator.core.context import Context  # Context: shared carrier passed step-to-step
from src.orchestrator.core.pipeline import PipelineRunner  # executor that runs pipelines
from src.orchestrator.core.registry import PipelineRegistry  # maps pipeline name -> builder(cfg)->Pipeline
from src.orchestrator.logging.setup import build_run_logger  # convenience logger builder
from src.orchestrator.runtime import discovery  # registers built-in pipelines into a registry


# =============================================================================
# Small helpers (kept local to the example)
# =============================================================================

def utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.

    Why:
    - Avoid naive datetimes.
    - Keep all timestamps consistent across logs and filenames.
    """
    return datetime.now(timezone.utc)


def _as_dict(state: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    """
    Safely read a dictionary from ctx.state.

    Returns None if:
    - key doesn't exist, OR
    - the value is not a dict
    """
    value = state.get(key)
    return value if isinstance(value, dict) else None


def _as_list(state: Dict[str, Any], key: str) -> Optional[list]:
    """
    Safely read a list from ctx.state.

    Returns None if:
    - key doesn't exist, OR
    - the value is not a list
    """
    value = state.get(key)
    return value if isinstance(value, list) else None


def _format_header(title: str) -> str:
    """
    Create a readable header block for printing.

    We return a string (rather than printing) so callers can decide when to print.
    """
    bar = "=" * len(title)
    return f"\n{title}\n{bar}"


# =============================================================================
# Main
# =============================================================================

def main(*, save_files: bool = False) -> None:
    """
    Execute the built-in marketdata timeseries pipeline for a small FX request.

    Parameters
    ----------
    save_files:
        If True:
          - create run folders (ArtifactStore.ensure_layout)
          - write logs to <run>/logs/run.log
          - allow steps to write artifacts under <run>/artifacts
        If False:
          - no folders are created by this script
          - logging goes to console only (still fully runs)
    """

    # -------------------------------------------------------------------------
    # 1) Run metadata (run_id + workdir)
    # -------------------------------------------------------------------------
    # A run_id is an identifier for a single execution of a pipeline.
    run_id = f"fx_marketdata_{utc_now().strftime('%Y%m%d_%H%M%S')}"

    # Workdir is the root folder under which run folders live (when save_files=True).
    workdir = Path("./.runs").expanduser().resolve()

    # -------------------------------------------------------------------------
    # 2) Build RunConfig.params in the EXACT schema expected by build_timeseries pipeline
    # -------------------------------------------------------------------------
    # We keep a small date window so the synthetic dataset is quick to generate.
    end_dt = utc_now().date()  # end date (inclusive/exclusive depends on provider; pipeline just passes the string)
    start_dt = (utc_now() - timedelta(days=2)).date()  # start date a couple days earlier

    # Example-only: a readable mapping of "labels" -> "MarketId strings".
    # This is *not* required by the pipeline; we use it to keep the example readable.
    ids_by_name: Dict[str, str] = {
        "spot": "FX.SPOT.EURUSD",
        "vol": "FX.VOL.EURUSD.ATM",
        "rd": "IR.RATE.USD.OIS",
        "rf": "IR.RATE.EUR.OIS",
    }

    # The pipeline expects `marketdata.ids: list[str]`.
    # So we flatten the mapping to a list of strings here.
    ids_list = list(ids_by_name.values())

    # params is pipeline-owned: the pipeline decides its schema.
    # This block MUST match what your pipeline reads in src/orchestrator/pipelines/marketdata/build_timeseries.py.
    params: Dict[str, Any] = {
        "marketdata": {
            # BuildMarketIdsStep expects this key.
            "ids": ids_list,

            # BuildTimeseriesRequestStep expects these keys.
            "start": str(start_dt),  # provider expects string dates
            "end": str(end_dt),      # provider expects string dates
            "freq": "D",             # optional; pipeline defaults to "D" if missing/blank
            "scenarios": 8,          # optional; pipeline defaults to 1

            # Provider factory config (BuildProviderStep reads provider.type/seed/name).
            "provider": {
                "type": "synthetic",         # your pipeline uses "type" (not "kind")
                "seed": 7,                   # deterministic output
                "name": "SyntheticProvider", # human-friendly label
            },

            # Optional snapshot step config (BuildSnapshotStep is a no-op if absent).
            "snapshot": {
                "time": "last",       # "last" resolves to final time index explicitly
                "scenario_idx": 1,    # which scenario to snapshot
            },
        }
    }

    # RunConfig is the top-level config object passed into the pipeline builder and steps.
    cfg = RunConfig(
        pipeline="marketdata.build_timeseries",  # registry lookup key
        io=IOConfig(workdir=str(workdir)),       # where outputs would go (if save_files=True)
        params=params,                           # pipeline-specific configuration
    )

    # -------------------------------------------------------------------------
    # 3) ArtifactStore carrier (only creates folders when save_files=True)
    # -------------------------------------------------------------------------
    # ArtifactStore is kept on Context so steps can save artifacts in a standard layout.
    store = ArtifactStore(
        workdir=workdir,
        run_id=run_id,
        artifacts_dirname=str(cfg.io.artifacts_dir),
        logs_dirname=str(cfg.io.logs_dir),
        enable_save=save_files
    )

    # -------------------------------------------------------------------------
    # 4) Logger setup (console always; file only if save_files=True)
    # -------------------------------------------------------------------------
    # build_run_logger returns a configured logger that emits structured step logs.
    # If we pass log_file=None, it only prints to console.
    logger = build_run_logger(
        logger_name="QuantStrata.Examples.FxMarketdata",
        log_file=(store.logs_root / "run.log") if save_files else None,
    )

    # -------------------------------------------------------------------------
    # 5) Registry + discovery -> resolve the pipeline builder -> build pipeline
    # -------------------------------------------------------------------------
    # PipelineRegistry holds a mapping: name -> builder function.
    registry = PipelineRegistry()

    # discovery registers all built-in pipelines into the registry.
    # This centralizes imports to avoid side effects elsewhere.
    discovery.register_builtin_pipelines(registry)

    # Get the builder function for this pipeline name.
    # The builder signature is: builder(cfg: RunConfig) -> Pipeline
    builder = registry.get(cfg.pipeline)

    # Build the pipeline instance (ordered list of Step objects).
    pipeline = builder(cfg)

    # -------------------------------------------------------------------------
    # 6) Context + runner -> execute pipeline
    # -------------------------------------------------------------------------
    # Context is the shared carrier passed to each Step.
    # Steps store outputs in ctx.state and may also attach ctx.provider.
    ctx = Context(
        run_id=run_id,
        cfg=cfg,
        logger=logger,
        artifact_store=store,
        provider=None,  # BuildProviderStep will set this
        state={},       # Step outputs accumulate here
    )

    # PipelineRunner runs the pipeline step-by-step and returns the final Context.
    runner = PipelineRunner(
        only=None,        # run all steps (or set a subset of names)
        skip=None,        # skip none (or set a subset of names)
        resume_from=None, # start at beginning (or resume at a step name)
        dry_run=False,    # False => actually execute steps
    )

    # This ensures the logger stream appears first and uninterrupted.
    final_ctx = runner.run(pipeline, ctx)
    state = final_ctx.state

    # -------------------------------------------------------------------------
    # 7) Plain console summary
    # -------------------------------------------------------------------------

if __name__ == "__main__":
    # Default behaviour:
    main(save_files=False)