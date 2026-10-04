"""
The job-set runner: expand, merge, dispatch, aggregate.

Four steps, in that order, and the order carries most of the design.

**Expand and merge first, and validate everything.** Every job's
specification is merged and validated before any of them starts, so a typo
in the fortieth job's overrides is reported in the first second rather than
three hours in -- and every typo is reported at once, so a user who mistyped
two keys learns both now instead of discovering the second after fixing the
first and waiting again.

**Dispatch through an executor.** The runner does not know where jobs run.
It builds work items and hands them over, which is what makes sequential and
parallel runs provably equivalent: there is no branch here to get wrong.

**Aggregate last.** Results come back in input order, are turned into
manifest rows, and the manifest is written atomically at the end.

Why the runner cannot ask a domain for its jobs
-----------------------------------------------
``orchestration`` may not import ``domains`` or ``models``. So this runner
cannot ask ``domains.pnl`` to partition a portfolio, and cannot import the
flagship.

It does not need to. Expanding a portfolio into jobs happens *before* the
runner sees anything -- by the user, by ``domains.pnl.clusters``, or by
``rade_qnet.api`` -- and arrives as a ``JobSetSpec``. The model is resolved
through the registry by name, exactly as ``TrainPipeline`` does it.

This is the layering doing its job rather than obstructing it: the
constraint forced portfolio expansion to be a separate, independently
testable function instead of a branch inside the runner.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ...core.runtime.components import ENGINES, MODELS, registration_modules
from ...core.runtime.context import RunContext
from ...core.runtime.hashing import abbreviate_digest, digest_spec
from ...core.runtime.logging import get_logger
from ..compute.base import WorkItem
from ..compute.policy import Placement, choose_placement
from .manifest import JobRecord, JobSetManifest
from .unit import JobPayload, run_job

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from ...core.spec.jobs import JobSetSpec
    from ...core.spec.run import SupervisedRunSpec
    from ..compute.base import Executor, WorkResult
    from .unit import JobOutcome

__all__ = ["JobSetRunner"]

_LOGGER = get_logger(__name__)

#: Subdirectory each job's output goes under, beneath the set's directory.
#: A level of nesting so the set's own files -- the manifest, its figures --
#: are not mixed in with forty job directories.
JOBS_SUBDIRECTORY = "jobs"


class JobSetRunner:
    """
    Runs every job in a set and records what happened.

    Parameters
    ----------
    spec
        The validated job-set specification.
    run_id
        Identifier for the set. Defaults to the set's name and its
        specification digest, which makes two runs of the same file land in
        the same place -- re-running a set should extend it, not scatter it.
    executor
        Where jobs run. ``None`` asks the placement policy, which is the
        normal path and the one ``placement.executor: auto`` means.
    catalog_root
        Where bundles are recorded, or ``None`` to skip recording. Defaults
        to the set's output directory, so a set's bundles are indexed
        together.
    metadata
        Free-form annotations carried into every job's run context.
    """

    def __init__(
        self,
        spec: JobSetSpec,
        *,
        run_id: str | None = None,
        executor: Executor | None = None,
        catalog_root: Path | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.spec = spec
        self.spec_digest = digest_spec(spec)
        self.run_id = run_id or self._default_run_id()
        self.executor = executor
        self.metadata = dict(metadata or {})
        self.output_directory = spec.output_root / self.run_id
        self.catalog_root = catalog_root if catalog_root is not None else self.output_directory

    def run(self) -> JobSetManifest:
        """
        Run every job and write the set's manifest.

        Returns
        -------
        JobSetManifest
            One record per job, in declaration order.

        Raises
        ------
        SpecError
            If any job's merged specification is invalid. Raised before any
            job starts, with every failure listed -- a set that trained
            thirty-nine models and then discovered the fortieth was
            misconfigured has wasted the thirty-nine runs' worth of time it
            took to find out.
        """
        specs = self.spec.validate_jobs()
        placement = self._placement(n_jobs=len(specs))
        items = [self._item(job_id, spec) for job_id, spec in specs.items()]

        _LOGGER.info(
            "job set %s: %d job(s) via %s",
            self.run_id,
            len(items),
            placement.executor.description,
        )

        started = time.perf_counter()
        results = placement.executor.map(items)
        elapsed = time.perf_counter() - started

        manifest = self._manifest(
            results,
            placement=placement.executor.description,
            reason=placement.reason,
            elapsed=elapsed,
        )
        manifest.write(self.output_directory)
        return manifest

    def payloads(self) -> dict[str, JobPayload]:
        """
        Return each job's payload without running anything.

        Exposed because it is the natural unit to inspect, to test against,
        and to re-run one failed job from. A job that failed inside a set
        should be reproducible on its own, and this is what makes that a
        one-line operation rather than a reconstruction of the merge, the
        seed derivation and the directory layout.

        Returns
        -------
        dict
            Job identifier to payload, in declaration order.
        """
        return {
            job_id: self._payload(job_id, spec)
            for job_id, spec in self.spec.validate_jobs().items()
        }

    def _item(self, job_id: str, spec: SupervisedRunSpec) -> WorkItem[JobPayload, JobOutcome]:
        """
        Build the work item for one job.

        Parameters
        ----------
        job_id
            The job's identifier.
        spec
            Its merged, validated run specification.

        Returns
        -------
        WorkItem
            Keyed by the job identifier, so results, log lines and manifest
            rows all agree on what to call it.
        """
        return WorkItem(key=job_id, function=run_job, payload=self._payload(job_id, spec))

    def _payload(self, job_id: str, spec: SupervisedRunSpec) -> JobPayload:
        """
        Build one job's payload.

        Parameters
        ----------
        job_id
            The job's identifier.
        spec
            Its merged, validated run specification.

        Returns
        -------
        JobPayload
            Plain data, picklable, carrying no live objects.
        """
        return JobPayload(
            job_id=job_id,
            spec=spec,  # type: ignore[arg-type]
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            output_directory=self.output_directory / JOBS_SUBDIRECTORY / job_id,
            seed=spec.seed,
            catalog_root=self.catalog_root,
            metadata=self.metadata,
            # Resolved in the parent, so an unregistered name fails here --
            # where the error can list what is available -- rather than
            # inside a worker, where it would surface as a dead process.
            registration_modules=registration_modules(
                (MODELS, spec.model.name), (ENGINES, spec.training.engine)
            ),
        )

    def _placement(self, *, n_jobs: int) -> Placement:
        """
        Resolve where the jobs run.

        Parameters
        ----------
        n_jobs
            How many jobs there are, which the policy uses.

        Returns
        -------
        Placement
            The executor and the reason for it. An executor supplied to the
            constructor is wrapped in the same shape rather than special-
            cased downstream, so the manifest records a placement and a
            reason either way.
        """
        if self.executor is None:
            return choose_placement(self.spec.placement, n_jobs=n_jobs)

        return Placement(
            executor=self.executor,
            name=self.spec.placement.executor,
            reason="supplied directly by the caller",
        )

    def _manifest(
        self,
        results: Sequence[WorkResult[JobOutcome]],
        *,
        placement: str,
        reason: str,
        elapsed: float,
    ) -> JobSetManifest:
        """
        Turn the executor's results into the set's manifest.

        Parameters
        ----------
        results
            One per job, in declaration order.
        placement
            How the executor described itself.
        reason
            Why that placement was used.
        elapsed
            Wall time for the whole set.

        Returns
        -------
        JobSetManifest
            The manifest, not yet written.
        """
        records = tuple(
            JobRecord.of(result, relative_to=self.output_directory) for result in results
        )
        for record in records:
            if not record.succeeded:
                _LOGGER.error(
                    "job %s failed: %s: %s",
                    record.job_id,
                    record.failure_kind,
                    record.failure_message,
                )
        return JobSetManifest(
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            name=self.spec.name,
            placement=placement,
            placement_reason=reason,
            wall_seconds=elapsed,
            jobs=records,
        )

    def context(self) -> RunContext:
        """
        Return the set-level run context.

        Not the context any job runs under -- each job derives its own, in
        its own process. This one exists so set-level work, such as the
        figures a summary renders, has somewhere to write and something to
        log against.

        Returns
        -------
        RunContext
            A context for the set as a whole.
        """
        return RunContext(
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            output_directory=self.output_directory,
            metadata=self.metadata,
        )

    def _default_run_id(self) -> str:
        """
        Derive an identifier for the set from its specification.

        Derived rather than timestamped, deliberately. A timestamped
        identifier makes every run of the same file a new directory, which
        looks tidy and means a re-run after a crash produces a second
        partial set rather than completing the first. Deriving it from the
        digest means the same specification lands in the same place, and a
        changed one does not.

        Returns
        -------
        str
            For example ``portfolio-7f3a9c21`` or ``jobset-7f3a9c21``.
        """
        prefix = self.spec.name or "jobset"
        return f"{prefix}-{abbreviate_digest(self.spec_digest)}"
