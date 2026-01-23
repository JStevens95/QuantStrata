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

import logging
from dataclasses import dataclass, field
from typing import Optional, Any, Type, TypeVar, cast

from src.orchestrator.core.types import RunId, State
from src.orchestrator.config.schemas import RunConfig
from src.orchestrator.artifacts.store import ArtifactStore

T = TypeVar("T")


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
    cfg: RunConfig
    logger: logging.Logger | None
    artifact_store: ArtifactStore | None

    provider: Optional[object] = None
    state: State = field(default_factory=dict)

    def has(self, key: str) -> bool:
        """Return True if the given state key exists."""
        return key in self.state

    def put(self, key: str, value: object) -> None:
        """Store a value in the shared run state."""
        self.state[str(key)] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from state.

        - If default is provided, returns default when key is missing.
        - If default is NOT provided (left as None intentionally), this behaves
          like a normal dict access: KeyError if missing.
        """
        if key in self.state:
            return self.state[key]
        if default is not None:
            return default
        raise KeyError(f"Missing ctx.state[{key!r}]")

    def require(self, key: str, *, expected_type: Optional[Type[T]] = None) -> T:
        """
        Get a required key from state with a clean, consistent error.

        Parameters
        ----------
        key:
            State key that must exist.
        expected_type:
            Optional runtime type check for clearer failures.

        Returns
        -------
        T
            The stored value (optionally type-checked).
        """
        if key not in self.state:
            raise KeyError(f"Missing required ctx.state[{key!r}] (run_id={self.run_id})")

        value = self.state[key]

        if expected_type is not None and not isinstance(value, expected_type):
            raise TypeError(
                f"ctx.state[{key!r}] has wrong type (run_id={self.run_id}). "
                f"expected={expected_type.__name__}, got={type(value).__name__}"
            )

        return cast(T, value)