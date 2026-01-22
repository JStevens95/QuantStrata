# src/orchestrator/runtime/entrypoints.py

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Optional

from src.orchestrator.artifacts.manifest import RunManifest
from src.orchestrator.artifacts.store import ArtifactStore
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.registry import PipelineRegistry
from src.orchestrator.logging.setup import build_run_logger

# IMPORTANT: import the module, not the function, so tests can monkeypatch it reliably.
from src.orchestrator.runtime import discovery


def run_pipeline_from_config(cfg: RunConfig, *, run_id: Optional[str] = None) -> Context:
    """
    Build and run a pipeline from a RunConfig.

    Parameters
    ----------
    cfg:
        Fully-validated run configuration.
    run_id:
        Optional explicit run identifier. If not provided, a UTC timestamp-based id is used.

    Returns
    -------
    Context
        Final context after pipeline execution (contains state, store, logger, etc.).
    """
    # --- Resolve / generate a stable run id ---
    resolved_run_id = str(run_id).strip() if run_id is not None else ""
    if not resolved_run_id:
        resolved_run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")

    # --- Capture run start timestamp (required by RunManifest) ---
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Build artifact store and ensure directory layout exists on disk ---
    store = ArtifactStore.from_config(cfg=cfg, run_id=resolved_run_id)
    store.ensure_layout()

    # --- Create a logger that writes both console + run log file ---
    logger = build_run_logger(
        logger_name="QuantStrata.Orchestrator",
        log_file=store.logs_root / "run.log",
    )

    # --- Build pipeline registry and register built-ins (dynamic module call) ---
    registry = PipelineRegistry()
    discovery.register_builtin_pipelines(registry)

    # --- Resolve pipeline builder from registry ---
    builder = registry.get(cfg.pipeline)

    # --- Build pipeline from config (builder decides how to interpret cfg.params) ---
    pipeline = builder(cfg)

    # --- Create the base execution context ---
    ctx = Context(
        run_id=resolved_run_id,
        cfg=cfg,
        logger=logger,
        artifact_store=store,
        provider=None,
        state={},
    )

    # --- Execute pipeline ---
    runner = PipelineRunner(
        only=set(cfg.only) if cfg.only else None,
        skip=set(cfg.skip) if cfg.skip else None,
        resume_from=cfg.resume_from,
        dry_run=bool(cfg.dry_run),
    )
    ctx = runner.run(pipeline, ctx)

    # --- Capture run end timestamp (optional but useful) ---
    finished_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- Write manifest at end (captures config + outputs) ---
    manifest = RunManifest(
        run_id=resolved_run_id,
        pipeline=str(cfg.pipeline),
        started_at_utc=started_at_utc,
        config=asdict(cfg),
        outputs={"state_keys": sorted(ctx.state.keys())},
    )
    store.write_manifest(manifest)

    return ctx