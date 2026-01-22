"""
Run configuration schemas.

We keep RunConfig small and Vn-friendly:
- pipeline chooses which builder to run (string)
- only/skip/resume_from/dry_run control orchestration
- params is a dict for pipeline-specific settings (domain-owned)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass(frozen=True, slots=True)
class IOConfig:
    """
    Output locations for a run.

    workdir:
        Root folder where all runs are placed (each run gets its own subfolder).
    artifacts_dir:
        Subfolder under run for artifacts.
    logs_dir:
        Subfolder under run for logs.
    """
    workdir: str = "./.runs"
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"


@dataclass(frozen=True, slots=True)
class RunConfig:
    """
    Top-level orchestrator configuration.

    pipeline:
        Registry name for the pipeline (e.g., "marketdata.replay_static_v1").
    only/skip:
        Optional step-name filters.
    resume_from:
        Optional step name to resume from.
    dry_run:
        If True, run selection + logging without executing steps.
    io:
        IOConfig controlling output paths.
    params:
        Pipeline-specific settings. Treated as opaque by core orchestrator.
    """
    pipeline: str
    only: Optional[Sequence[str]] = None
    skip: Optional[Sequence[str]] = None
    resume_from: Optional[str] = None
    dry_run: bool = False

    io: IOConfig = field(default_factory=IOConfig)
    params: Dict[str, object] = field(default_factory=dict)