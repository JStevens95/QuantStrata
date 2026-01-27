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
- Optionally writes a run manifest to ArtifactStore (when enabled)
"""

from __future__ import annotations

from dataclasses import dataclass  # dataclass helpers for simple containers
from datetime import datetime, timezone  # stable UTC timestamps for manifests
from typing import List, Optional, Sequence, Set  # typing helpers

from src.orchestrator.core.context import Context  # orchestrator execution context
from src.orchestrator.core.errors import ConfigError  # config validation errors
from src.orchestrator.core.step import Step  # pipeline step interface
from src.orchestrator.logging.events import log_step_end, log_step_start  # consistent logging events


def _utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string (stable for logging/manifests)."""
    return datetime.now(timezone.utc).isoformat()  # timezone-aware UTC timestamp


@dataclass(frozen=True, slots=True)
class Pipeline:
    """A named ordered sequence of steps."""
    name: str  # pipeline identifier (used in logs/manifest)
    steps: Sequence[Step]  # ordered sequence of Step objects


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

    only: Optional[Set[str]] = None  # run only these step names (if set)
    skip: Optional[Set[str]] = None  # skip these step names (if set)
    resume_from: Optional[str] = None  # start execution from this step name (if set)
    dry_run: bool = False  # if True, do not actually execute steps

    def run(self, pipeline: Pipeline, ctx: Context) -> Context:
        """
        Execute a pipeline end-to-end.

        Returns
        -------
        Context
            The final context after all selected steps have completed.
        """
        started_at = _utc_now_iso()  # capture start timestamp for manifest/debugging

        all_steps = list(pipeline.steps)  # make steps re-iterable and cheap to scan

        # Validate resume_from early to fail fast on typos (better UX).
        if self.resume_from is not None:
            step_names = [s.name for s in all_steps]  # collect available step names
            if self.resume_from not in step_names:  # verify requested step exists
                raise ConfigError(
                    "resume_from does not match any step name.\n"
                    f"resume_from={self.resume_from!r}\n"
                    f"available={step_names}"
                )

        selected_steps = self._select_steps(all_steps)  # apply only/skip/resume selection rules
        selected_step_names = [s.name for s in selected_steps]  # record selection for manifest

        ok = True  # optimistic success flag (flips to False on exception)
        failed_step: Optional[str] = None  # name of the step that failed (if any)
        error_type: Optional[str] = None  # exception class name (if any)
        error_message: Optional[str] = None  # exception string (if any)

        # Pipeline-level start log (if logger is present).
        if ctx.logger is not None:
            ctx.logger.info("PIPELINE_START | %s | selected_steps=%d", pipeline.name, len(selected_steps))

        try:
            for step in selected_steps:  # execute selected steps in order
                log_step_start(ctx.logger, step_name=step.name, tags=step.tags)  # step start log

                if self.dry_run:
                    # In dry-run mode, we do not execute anything; we only log intent.
                    if ctx.logger is not None:
                        ctx.logger.info("DRY_RUN | step=%s | skipped_execution=True", step.name)
                    log_step_end(ctx.logger, step_name=step.name, ok=True)  # step end log (successful dry run)
                    continue  # proceed to next step

                # Execute with step-level safety wrapper (Step.safe_run unifies errors).
                ctx = step.safe_run(ctx)

                # If we got here, the step ran successfully.
                log_step_end(ctx.logger, step_name=step.name, ok=True)

        except Exception as exc:
            # Record failure metadata for manifest/debugging (do not swallow the error).
            ok = False  # pipeline failed
            failed_step = getattr(step, "name", None)  # step still in scope from loop
            error_type = type(exc).__name__  # short error type
            error_message = str(exc)  # error message string
            raise  # re-raise to preserve behavior and traceback

        finally:
            ended_at = _utc_now_iso()  # capture end timestamp for manifest/debugging

            # Attempt to write manifest if an ArtifactStore exists AND saving is enabled.
            store = getattr(ctx, "artifact_store", None)  # pull store from context safely
            if store is not None and bool(getattr(store, "enable_save", False)):
                manifest_payload = {
                    "run_id": str(ctx.run_id),  # run identity
                    "pipeline": str(pipeline.name),  # pipeline identity
                    "started_at": started_at,  # start timestamp
                    "ended_at": ended_at,  # end timestamp
                    "ok": bool(ok),  # overall status
                    "selected_steps": list(selected_step_names),  # selected steps executed (or intended)
                    "only": sorted(self.only) if self.only else None,  # selection config (normalized)
                    "skip": sorted(self.skip) if self.skip else None,  # selection config (normalized)
                    "resume_from": self.resume_from,  # resume config
                    "dry_run": bool(self.dry_run),  # dry run flag
                    "failed_step": failed_step,  # step name that failed
                    "error_type": error_type,  # exception type
                    "error_message": error_message,  # exception message
                }

                try:
                    store.write_run_manifest(manifest_payload)  # best-effort write (atomic-ish)
                except Exception:
                    # Never allow artifact writing to break the run.
                    if ctx.logger is not None:
                        ctx.logger.exception("MANIFEST_WRITE_FAILED | pipeline=%s", pipeline.name)

            # Always log pipeline end (status included) if logger is present.
            if ctx.logger is not None:
                ctx.logger.info("PIPELINE_END   | %s | ok=%s", pipeline.name, ok)

        return ctx  # return the final context (state updated by steps)

    def _select_steps(self, steps: List[Step]) -> List[Step]:
        """
        Apply resume_from / only / skip selection logic to a list of steps.
        """
        selected: List[Step] = []  # selected steps accumulate here
        resumed = self.resume_from is None  # if no resume_from, we start immediately

        for step in steps:  # iterate in declared pipeline order
            # Resume logic: ignore steps until we hit resume_from (inclusive).
            if not resumed:
                if step.name == self.resume_from:
                    resumed = True  # we reached resume point, start selecting from here
                else:
                    continue  # keep skipping until resume point is reached

            # only-filter: if provided, require membership.
            if self.only is not None and step.name not in self.only:
                continue  # not selected by only-filter

            # skip-filter: if provided, exclude membership.
            if self.skip is not None and step.name in self.skip:
                continue  # explicitly skipped

            selected.append(step)  # include this step

        return selected  # return ordered selected steps