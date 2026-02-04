"""
Step abstraction.

A Step is the smallest orchestration unit:
- reads inputs from Context (cfg/state/artifacts/provider)
- performs deterministic work
- writes outputs back into Context
"""

from __future__ import annotations

from dataclasses import dataclass

from src.orchestrator.core.context import Context
from src.orchestrator.core.errors import StepError
from src.orchestrator.core.types import StepName, Tags


@dataclass(slots=True)
class Step:
    """
    Base Step.

    Subclasses should override `run(ctx)`.

    Attributes
    ----------
    name:
        Stable step identifier used for selection (only/skip/resume).
    tags:
        Optional metadata (useful for later grouping / filtering).
    """

    name: StepName
    tags: Tags = ()

    def run(self, ctx: Context) -> Context:
        """
        Execute the step.

        Must be overridden in subclasses.
        """
        raise NotImplementedError("Step.run must be implemented by subclasses.")

    def safe_run(self, ctx: Context) -> Context:
        """
        Execute the step and wrap unexpected errors as StepError.

        This gives a clean orchestrator error surface while preserving the original
        exception as `__cause__` for debugging.
        """
        try:
            return self.run(ctx)
        except StepError:
            # If the step already raised a StepError, preserve it.
            raise
        except Exception as exc:  # noqa: BLE001 (we wrap to unify orchestrator errors)
            raise StepError(f"Step failed: {self.name}") from exc