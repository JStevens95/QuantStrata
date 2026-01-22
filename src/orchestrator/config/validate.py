"""
Validation helpers for RunConfig.

We validate only cross-cutting invariants here.
Pipeline-specific invariants should be validated inside pipeline builders/steps.
"""

from __future__ import annotations

from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.core.errors import ConfigError


def validate_run_config(cfg: RunConfig) -> RunConfig:
    """
    Validate basic invariants for RunConfig.

    Returns
    -------
    RunConfig
        The same cfg object (useful for functional-style composition).
    """
    # Validate pipeline name.
    pipeline_name = str(cfg.pipeline).strip()
    if not pipeline_name:
        raise ConfigError("RunConfig.pipeline must be a non-empty string.")

    # Validate optional step filters.
    for field_name, seq in (("only", cfg.only), ("skip", cfg.skip)):
        if seq is None:
            continue
        for step_name in seq:
            if not str(step_name).strip():
                raise ConfigError(f"RunConfig.{field_name} contains an empty step name.")

    # Validate IO fields.
    if not str(cfg.io.workdir).strip():
        raise ConfigError("RunConfig.io.workdir must be non-empty.")
    if not str(cfg.io.artifacts_dir).strip():
        raise ConfigError("RunConfig.io.artifacts_dir must be non-empty.")
    if not str(cfg.io.logs_dir).strip():
        raise ConfigError("RunConfig.io.logs_dir must be non-empty.")

    return cfg