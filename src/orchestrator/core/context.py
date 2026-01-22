"""
Context passed between steps.

A Context is a mutable container holding:
- cfg: validated RunConfig
- logger: run logger
- artifact_store: filesystem store for run outputs
- provider: optional MarketDataProvider (pipelines may attach)
- state: dict for any intermediate objects

We keep Context mutable because pipelines are inherently incremental:
each step adds outputs for subsequent steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import logging

from src.orchestrator.core.types import RunId, State


@dataclass(slots=True)
class Context:
    """
    Mutable run context.

    Parameters
    ----------
    run_id:
        Unique identifier for this run (used for folder naming and manifests).
    cfg:
        RunConfig (typed) which controls the pipeline and parameters.
    logger:
        Logger configured for this run.
    artifact_store:
        ArtifactStore for writing datasets/models/reports and run manifest.
    provider:
        Optional domain provider (e.g., StaticProvider), attached by pipelines.
    state:
        Mutable dictionary for step outputs and intermediate values.
    """

    run_id: RunId
    cfg: object
    logger: logging.Logger
    artifact_store: object

    provider: Optional[object] = None
    state: State = field(default_factory=dict)

    def put(self, key: str, value: object) -> None:
        """Store a value in the shared run state."""
        self.state[str(key)] = value

    def get(self, key: str, default: object | None = None) -> object | None:
        """Retrieve a value from the shared run state."""
        return self.state.get(str(key), default)