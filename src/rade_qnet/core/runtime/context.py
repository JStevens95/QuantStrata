"""
The ambient state every pipeline stage can reach.

:class:`RunContext` is what a stage is given besides its typed input: where to
write, which run it belongs to, which seed it was given, who is observing it,
and where to record what it produced. Threading those through every signature
individually would make each stage's parameters mostly plumbing; hiding them
in module globals would make two concurrent runs in one process impossible.

Why the storage protocols are declared here
-------------------------------------------
:class:`Catalog` and :class:`Tracker` are *implemented* in ``rade_qnet.storage``
but *declared* in ``core``. That inversion is forced and is worth
understanding, because it recurs across the framework.

``core`` may not import ``storage`` -- the dependency runs the other way, and
the layering test enforces it. But a ``RunContext`` living in ``core`` has to
be able to hold a catalog. Declaring the protocol here resolves it: ``core``
owns the *interface*, ``storage`` owns the *implementation*, and the arrow
still points from ``storage`` to ``core``. This is dependency inversion, and
it is also what lets a test substitute an in-memory catalog with no filesystem
at all.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .hooks import PipelineHook
from .logging import bound_context, get_logger
from .seeding import derive_seed

if TYPE_CHECKING:
    from logging import Logger

    from ..contract.bundle import Manifest

__all__ = ["Catalog", "RunContext", "Tracker"]

_LOGGER = get_logger(__name__)


@runtime_checkable
class Catalog(Protocol):
    """
    The registry of what has been trained, and where it was put.

    Implemented by ``rade_qnet.storage.catalog``; declared here for the reason
    given in the module docstring.

    The implementation is single-writer by construction. Version assignment
    must be atomic, because the obvious implementation -- read the catalog,
    compute ``max + 1``, write it back -- loses a bundle whenever two runs
    finish together, and a parallel job set finishes runs together by design.

    Only the three methods a *pipeline* calls are declared here. The query
    side -- ``entries`` and ``records``, which
    :class:`~rade_qnet.storage.registry.RunRegistry` reads -- is deliberately
    absent, for the reason given at
    :class:`~rade_qnet.core.capability.supervised.RebuildableDataModule`: an
    ``isinstance`` check against a runtime-checkable protocol only tests that
    the methods exist, so widening this one would make every three-method
    stub in the suite stop satisfying it, failing training runs over methods
    the training path never calls. The registry constructs a concrete
    :class:`~rade_qnet.storage.catalog.JsonlCatalog` instead of accepting any
    ``Catalog``, which is what makes that safe.
    """

    def next_version(self, model_name: str, *, job_id: str | None = None) -> int:
        """
        Reserve the next version number for a model, atomically.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier within a job set, or ``None`` for a single run.
            Versions are numbered per job, so each member of a job set has its
            own sequence.

        Returns
        -------
        int
            A version number no other caller will be given.
        """
        ...

    def record(self, manifest: Manifest, *, location: Path | None = None) -> None:
        """
        Record a written bundle.

        Called after the bundle is on disk, never before: a catalog entry
        pointing at a directory that does not exist is worse than a bundle
        with no entry, because the first breaks every reader and the second
        is recoverable by rescanning.

        Parameters
        ----------
        manifest
            The manifest of the bundle that was written.
        location
            The bundle's directory. Recorded so a run selected from the
            catalog -- by tag, by alias, by best metric -- can be opened
            without the reader knowing the directory layout, which differs
            between a single run and a job set. Optional so an entry can
            still be recorded where no directory exists.
        """
        ...

    def latest(self, model_name: str, *, job_id: str | None = None) -> Manifest | None:
        """
        Return the most recent recorded manifest.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.

        Returns
        -------
        Manifest or None
            The latest manifest, or ``None`` if nothing is recorded.
        """
        ...


@runtime_checkable
class Tracker(Protocol):
    """
    An experiment tracker.

    Declared in ``core`` for the same reason as :class:`Catalog`, and with the
    same consequence: the framework's default implementation does nothing, and
    a tracking failure degrades to a warning rather than ending the run. A
    model that trained successfully but could not reach a tracking server has
    still trained successfully.
    """

    def log_params(self, params: Mapping[str, object]) -> None:
        """
        Record the run's configuration.

        Parameters
        ----------
        params
            Flattened specification values.
        """
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Record metrics, optionally against a step.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None`` for a final metric.
        """
        ...

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Record a written file.

        Parameters
        ----------
        path
            The file.
        name
            Logical name, defaulting to the file name.
        """
        ...

    def finish(self, *, succeeded: bool) -> None:
        """
        Close the tracking run.

        Parameters
        ----------
        succeeded
            Whether the run completed.
        """
        ...


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Ambient state for one run, passed to every stage.

    Frozen, so a stage cannot reconfigure the run it is part of. Derived
    contexts -- one per job in a job set -- are produced by :meth:`for_job`,
    which returns a new context rather than mutating this one.

    Parameters
    ----------
    run_id
        Identifier for this run, used in log lines, directory names and
        catalog entries.
    spec_digest
        Digest of the run specification.
    output_directory
        Root for everything this run writes.
    seed
        Base seed. A job's seed is derived from it rather than shared; see
        :meth:`for_job`.
    job_id
        Job identifier within a job set, or ``None`` for a single run.
    hooks
        Observers, notified in order.
    catalog
        Where bundles are recorded, or ``None`` to skip recording.
    tracker
        Where metrics are tracked, or ``None`` to skip tracking.
    """

    run_id: str
    spec_digest: str
    output_directory: Path
    seed: int = 0
    job_id: str | None = None
    hooks: tuple[PipelineHook, ...] = ()
    catalog: Catalog | None = None
    tracker: Tracker | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def logger(self) -> Logger:
        """
        A logger whose records carry this run's identifiers.

        The identifiers are attached by the contextvar-based filter rather
        than by this property, so a log line emitted deep inside a model's
        own code is tagged too, without that code knowing about the context.
        """
        return get_logger("rade_qnet.run")

    @property
    def reports_directory(self) -> Path:
        """Where reports and figures are written."""
        return self.output_directory / "reports"

    @property
    def bundles_directory(self) -> Path:
        """Where bundles are written."""
        return self.output_directory / "bundles"

    def activate(self, *, stage: str | None = None) -> AbstractContextManager[None]:
        """
        Return a context manager binding this run's identifiers for logging.

        Parameters
        ----------
        stage
            Stage name to bind alongside the run and job identifiers.

        Returns
        -------
        AbstractContextManager
            A manager that binds on entry and restores on exit, so a nested
            stage does not leak its name to its parent.
        """
        return bound_context(run_id=self.run_id, job_id=self.job_id, stage=stage)

    def for_job(self, job_id: str, *, output_directory: Path | None = None) -> RunContext:
        """
        Derive a context for one member of a job set.

        The derived seed is a hash of the base seed and the job identifier,
        not ``base + index``. Two properties follow, and both matter for a
        parallel job set:

        - **Order independence.** A job's seed depends on its identifier, so
          re-running a job set with the jobs reordered, or re-running one
          failed job alone, reproduces the same result.
        - **No accidental correlation.** Consecutive integer seeds produce
          correlated streams in some generators, which would make members of
          a job set less independent than they appear.

        Parameters
        ----------
        job_id
            Identifier for the job.
        output_directory
            Override the derived directory. Defaults to a subdirectory named
            after the job.

        Returns
        -------
        RunContext
            A new context; this one is unchanged.
        """
        return RunContext(
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            output_directory=output_directory or (self.output_directory / "jobs" / job_id),
            seed=derive_seed(self.seed, job_id),
            job_id=job_id,
            hooks=self.hooks,
            catalog=self.catalog,
            tracker=self.tracker,
            metadata=self.metadata,
        )

    def notify(self, call: Callable[[PipelineHook], None], *, description: str) -> None:
        """
        Invoke a callable against every hook, tolerating failures.

        Takes a callable rather than a method name so the call is type
        checked: ``context.notify(lambda hook: hook.on_stage_start(stage))``
        is verified against :class:`PipelineHook`, where a string name would
        not be.

        Centralising the error handling here is what makes the "a hook may
        not break a run" rule true everywhere rather than in each of the
        dozen places a hook is called. A raising hook is logged with its class
        and the description below, and the remaining hooks still run -- one
        broken observer must not silence the others.

        Parameters
        ----------
        call
            What to invoke on each hook.
        description
            What was being reported, for the warning message.
        """
        for hook in self.hooks:
            try:
                call(hook)
            # Deliberately broad: see the rule above -- observers never fail runs.
            except Exception:
                _LOGGER.warning(
                    "hook %s raised while reporting %s; continuing",
                    type(hook).__name__,
                    description,
                    exc_info=True,
                )

    def track_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Send metrics to the tracker, if there is one, tolerating failures.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None``.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_metrics(metrics, step=step)
        except Exception:  # Broad by design: tracking is never load-bearing.
            _LOGGER.warning("tracker failed to log metrics; continuing", exc_info=True)

    def track_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Send an artifact to the tracker, if there is one, tolerating failures.

        Parameters
        ----------
        path
            The file that was written.
        name
            Logical name, defaulting to the file name.
        """
        if self.tracker is None:
            return
        try:
            self.tracker.log_artifact(path, name=name)
        except Exception:  # Broad by design: tracking is never load-bearing.
            _LOGGER.warning("tracker failed to log artifact %s; continuing", path, exc_info=True)

    def describe(self) -> Sequence[str]:
        """
        Return a short human-readable description, for the run's opening log.

        Returns
        -------
        Sequence of str
            One line per salient field.
        """
        lines = [
            f"run_id         {self.run_id}",
            f"job_id         {self.job_id or '-'}",
            f"spec_digest    {self.spec_digest}",
            f"seed           {self.seed}",
            f"output         {self.output_directory}",
            f"hooks          {', '.join(type(h).__name__ for h in self.hooks) or '-'}",
            f"catalog        {type(self.catalog).__name__ if self.catalog else '-'}",
            f"tracker        {type(self.tracker).__name__ if self.tracker else '-'}",
        ]
        return tuple(lines)
