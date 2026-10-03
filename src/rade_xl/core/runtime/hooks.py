"""
Observation points a caller can attach to a pipeline without subclassing it.

Hooks exist to keep the customisation tiers separate. A user who wants to log
to an experiment tracker, stream progress to a dashboard or assert an
invariant between stages should not have to subclass a pipeline to do it --
subclassing is for changing *what a stage computes*, and conflating the two
produces overrides whose only purpose is to add a print statement and which
then silently drift from the base implementation.

The central design rule
-----------------------
**A hook may observe, but it may not alter.** Hooks receive payloads and return
``None``. A hook that could rewrite a payload would make the pipeline's
behaviour depend on observation, and two runs with identical specs would stop
being comparable because one had a dashboard attached.

**A failing hook does not fail the run.** A broken tracker credential must not
destroy four hours of training. Hook exceptions are caught, logged at warning
level with the hook and stage named, and execution continues. The one
exception is :meth:`PipelineHook.on_run_start`, where a hook that cannot
initialise should say so before any compute is spent -- see
:class:`~rade_xl.core.runtime.pipeline.Pipeline` for where that line is drawn.
"""

from __future__ import annotations

from collections.abc import Mapping

__all__ = ["PipelineHook"]


class PipelineHook:
    """
    Base class for pipeline observers.

    Every method is a no-op, so a subclass overrides only what it needs and
    gains new hook points without modification when the framework adds them.
    This is the opposite trade-off from a protocol -- here the default
    behaviour is genuinely useful, so inheritance costs nothing and spares
    every hook from stubbing out seven methods.

    Notes
    -----
    A hook instance is not shared across processes. Under a parallel job set
    each worker constructs its own, so a hook holding an open file handle or a
    network session is safe, but a hook accumulating state will see only its
    own worker's events.
    """

    def on_run_start(self, run_id: str, spec_digest: str) -> None:
        """
        Observe the start of the run, before any stage executes.

        The place to open a tracker run or create a directory. Of all the
        hook points, this is the one where failing loudly may be preferable,
        because nothing has been computed yet.

        Parameters
        ----------
        run_id
            Identifier for this run.
        spec_digest
            Digest of the run specification, for correlating with a bundle.
        """

    def on_run_end(self, run_id: str, *, succeeded: bool) -> None:
        """
        Observe the end of the run, whether or not it succeeded.

        Called after the last stage, including when a stage failed, so a hook can close a file or finalise a tracker run
        whatever happened. Check ``succeeded`` rather than assuming: marking a
        crashed run as complete is worse than not marking it at all.

        Parameters
        ----------
        run_id
            Identifier for this run.
        succeeded
            Whether every stage completed.
        """

    def on_stage_start(self, stage: str) -> None:
        """
        Observe the start of a pipeline stage.

        Parameters
        ----------
        stage
            Stage name, such as ``build_data`` or ``fit``.
        """

    def on_stage_end(self, stage: str, *, seconds: float) -> None:
        """
        Observe the successful completion of a stage.

        Not called for a stage that raised; see :meth:`on_stage_error`.

        Parameters
        ----------
        stage
            Stage name.
        seconds
            Wall time for the stage.
        """

    def on_stage_error(self, stage: str, error: BaseException) -> None:
        """
        Observe a stage failure.

        The exception is re-raised after every hook has been notified, so this
        is for recording, not for recovery. A hook cannot suppress the error.

        Parameters
        ----------
        stage
            Stage name.
        error
            The exception, before it is wrapped in a
            :class:`~rade_xl.core.runtime.errors.StageError`.
        """

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, float]) -> None:
        """
        Observe the end of an epoch, or of a boosting round.

        Deliberately engine-agnostic: a gradient loop and a boosted-tree fit
        both report here, so a progress dashboard works for either without
        knowing which it is watching.

        Parameters
        ----------
        epoch
            Zero-based epoch index.
        metrics
            Metrics for the epoch, including losses.
        """

    def on_metrics(self, stage: str, metrics: Mapping[str, float]) -> None:
        """
        Observe metrics produced by a stage.

        Parameters
        ----------
        stage
            Stage that produced them.
        metrics
            Metric name to value, in original target units.
        """

    def on_artifact(self, name: str, path: str) -> None:
        """
        Observe a file a stage wrote.

        A figure, a report, a bundle. Reported as a path rather than as
        contents, because a hook that wanted to upload a two-gigabyte
        checkpoint should decide that for itself.

        Parameters
        ----------
        name
            Logical name of the artifact.
        path
            Where it was written.
        """
