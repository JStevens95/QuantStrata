"""
Pipeline and execution runner.

Pipeline:
- A named ordered list of Step objects.

PipelineRunner:
- Executes steps in order
- Supports only/skip filters
- Supports resume_from
- Supports dry_run
- Emits consistent logging events
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Set

from src.orchestrator.core.context import Context
from src.orchestrator.core.errors import ConfigError
from src.orchestrator.core.step import Step
from src.orchestrator.logging.events import log_step_end, log_step_start


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A named ordered sequence of steps."""
    name: str
    steps: Sequence[Step]


@dataclass(slots=True)
class PipelineRunner:
    """
    Execute a Pipeline.

    Parameters
    ----------
    only:
        If provided, run only steps whose names are in this set.
    skip:
        If provided, skip steps whose names are in this set.
    resume_from:
        If provided, skip steps until this step name is reached.
    dry_run:
        If True, log which steps would run but do not execute them.
    """

    only: Optional[Set[str]] = None
    skip: Optional[Set[str]] = None
    resume_from: Optional[str] = None
    dry_run: bool = False

    def run(self, pipeline: Pipeline, ctx: Context) -> Context:
        """
        Execute a pipeline end-to-end.

        Returns
        -------
        Context
            The final context after all selected steps have completed.
        """
        # Convert steps to a list so we can scan names multiple times cheaply.
        all_steps = list(pipeline.steps)

        # Validate resume_from early to fail fast on typos.
        if self.resume_from is not None:
            step_names = [s.name for s in all_steps]
            if self.resume_from not in step_names:
                raise ConfigError(
                    "resume_from does not match any step name.\n"
                    f"resume_from={self.resume_from!r}\n"
                    f"available={step_names}"
                )

        # Select steps according to only/skip/resume rules.
        selected_steps = self._select_steps(all_steps)

        ctx.logger.info("PIPELINE_START | %s | selected_steps=%d", pipeline.name, len(selected_steps))

        # Execute steps in order.
        for step in selected_steps:
            log_step_start(ctx.logger, step_name=step.name, tags=step.tags)

            if self.dry_run:
                # Dry-run means we do not execute, only log the intent.
                ctx.logger.info("DRY_RUN | step=%s | skipped_execution=True", step.name)
                log_step_end(ctx.logger, step_name=step.name, ok=True)
                continue

            # Run step with safety wrapper.
            ctx = step.safe_run(ctx)

            log_step_end(ctx.logger, step_name=step.name, ok=True)

        ctx.logger.info("PIPELINE_END   | %s", pipeline.name)
        return ctx

    def _select_steps(self, steps: List[Step]) -> List[Step]:
        """
        Apply resume_from / only / skip selection logic to a list of steps.
        """
        selected: List[Step] = []

        # If resume_from is None, we start immediately.
        resumed = self.resume_from is None

        for step in steps:
            # Resume logic: ignore steps until we hit resume_from.
            if not resumed:
                if step.name == self.resume_from:
                    resumed = True
                else:
                    continue

            # only-filter: if provided, require membership.
            if self.only is not None and step.name not in self.only:
                continue

            # skip-filter: if provided, exclude membership.
            if self.skip is not None and step.name in self.skip:
                continue

            selected.append(step)

        return selected