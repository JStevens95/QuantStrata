"""
Programmatic orchestrator entrypoints.

This module provides:
- run_pipeline_from_config(cfg)
- run_pipeline_by_name(path)

We do not force a CLI in V1; that can be layered later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.orchestrator.artifacts.manifest import RunManifest
from src.orchestrator.artifacts.store import ArtifactStore
from src.orchestrator.config.loader import load_run_config
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.context import Context
from src.orchestrator.core.pipeline import PipelineRunner
from src.orchestrator.core.registry import PipelineRegistry
from src.orchestrator.logging.setup import build_run_logger
from src.orchestrator.runtime.discovery import register_builtin_pipelines


def run_pipeline_from_config(cfg: RunConfig, *, run_id: Optional[str] = None) -> Context:
    """
    Run a pipeline given a validated RunConfig.

    Parameters
    ----------
    cfg:
        Typed run configuration.
    run_id:
        Optional explicit run id. If omitted, a timestamp-based id is created.

    Returns
    -------
    Context
        Final run context.
    """
    # Create a stable run id (UTC timestamp) if not provided.
    rid = str(run_id or f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")

    # Create an artifact store for this run.
    store = ArtifactStore(workdir=Path(cfg.io.workdir), run_id=rid,
                          artifacts_dirname=cfg.io.artifacts_dir, logs_dirname=cfg.io.logs_dir)
    store.ensure_layout()

    # Create a run logger that logs to both console and file.
    logger = build_run_logger(
        "QuantStrata.Orchestrator",
        log_file=store.logs_path / "run.log",
    )

    # Build a registry and register built-in pipelines.
    registry = PipelineRegistry()
    register_builtin_pipelines(registry)

    # Resolve pipeline builder by name.
    builder = registry.get(cfg.pipeline)

    # Build pipeline from config.
    pipeline = builder(cfg)

    # Create context passed to steps.
    ctx = Context(
        run_id=rid,
        cfg=cfg,
        logger=logger,
        artifact_store=store,
        provider=None,
        state={},
    )

    # Prepare runner options from config.
    runner = PipelineRunner(
        only=set(cfg.only) if cfg.only else None,
        skip=set(cfg.skip) if cfg.skip else None,
        resume_from=cfg.resume_from,
        dry_run=bool(cfg.dry_run),
    )

    # Record start time for manifest.
    started_at_utc = datetime.now(timezone.utc).isoformat()

    # Execute pipeline.
    ctx = runner.run(pipeline, ctx)

    # Write a small manifest for reproducibility / audit.
    manifest = RunManifest(
        run_id=rid,
        pipeline=str(cfg.pipeline),
        started_at_utc=started_at_utc,
        config={
            "pipeline": cfg.pipeline,
            "only": list(cfg.only) if cfg.only else None,
            "skip": list(cfg.skip) if cfg.skip else None,
            "resume_from": cfg.resume_from,
            "dry_run": cfg.dry_run,
            "io": {
                "workdir": cfg.io.workdir,
                "artifacts_dir": cfg.io.artifacts_dir,
                "logs_dir": cfg.io.logs_dir,
            },
            "params": dict(cfg.params or {}),
        },
        outputs={
            # Store just keys in V1 (avoid serializing large objects).
            "state_keys": sorted(ctx.state.keys()),
        },
    )
    store.save_manifest(manifest)

    return ctx


def run_pipeline_by_name(config_path: str | Path, *, run_id: Optional[str] = None) -> Context:
    """
    Load a config file and run the pipeline.

    Parameters
    ----------
    config_path:
        Path to JSON/YAML run config.
    run_id:
        Optional explicit run id override.

    Returns
    -------
    Context
        Final run context.
    """
    cfg = load_run_config(config_path)
    return run_pipeline_from_config(cfg, run_id=run_id)