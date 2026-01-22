"""
Orchestrator error hierarchy.

Why this exists:
- We want callers to catch orchestrator-specific failures cleanly.
- We don't want domain exceptions (pricing, marketdata, ML) to leak as-is.
"""

from __future__ import annotations


class OrchestratorError(RuntimeError):
    """Base class for orchestrator failures."""


class ConfigError(OrchestratorError):
    """Raised when a run configuration is invalid or internally inconsistent."""


class StepError(OrchestratorError):
    """Raised when a step fails (wraps underlying exception)."""


class ArtifactError(OrchestratorError):
    """Raised when saving/loading artifacts fails."""