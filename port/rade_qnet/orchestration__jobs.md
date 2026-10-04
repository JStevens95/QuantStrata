# `src/rade_qnet/orchestration/jobs`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 56 | 2160 | `6d1b581f5bd21ac3` |
| 2 | `manifest.py` | 407 | 12734 | `0a5173658b335f6b` |
| 3 | `set.py` | 335 | 11474 | `309bbb0a049e589c` |
| 4 | `unit.py` | 301 | 11485 | `dbcca9ec2df2b2c8` |

---

## 1. `src/rade_qnet/orchestration/jobs/__init__.py`

2160 bytes · SHA-256 `6d1b581f5bd21ac3`

```python
"""
One model, many jobs.

A job set is deliberately simple: a list of jobs, each one a full independent
training run of the same model, differing in its data slice and -- optionally
-- in its architecture complexity.  A liquid cluster with abundant history can
be given a wider, deeper configuration than a sparse one, from the same
specification file.

What a job set is *not* is a special kind of model.  There is no ensemble
model class, no shared parameters and no joint optimisation.  Fan-out is an
execution concern, which is why it lives beside ``compute`` rather than in
``models``.

Modules
-------
``unit.py``
    ``run_job(payload)`` -- a module-level, picklable function that trains a
    single job, with the ``JobPayload`` it takes and the ``JobOutcome`` it
    returns.  Module-level so it can cross a process boundary under the spawn
    start method, which a bound method or a closure cannot.
``manifest.py``
    The job-set manifest: per-job status, metrics, bundle version, wall time
    and failure reason, written last and written atomically.
``set.py``
    ``JobSetRunner``: expands the specification into jobs, merges defaults
    with per-job overrides, dispatches through an executor, and aggregates the
    results.  Partial failure is first-class -- one failed job does not
    discard the others.

A layering note
---------------
``orchestration`` may not import ``domains`` or ``models``, so nothing here
can ask a domain to partition a portfolio into jobs.  It does not need to:
that expansion happens before the runner sees anything and arrives as a
``JobSetSpec``, and the model is resolved by name through the registry.  The
constraint is what turned portfolio expansion into a separate, independently
testable function rather than a branch inside the runner.
"""

from __future__ import annotations

from .manifest import MANIFEST_FILENAME, JobRecord, JobSetManifest, JobStatus
from .set import JobSetRunner
from .unit import JobOutcome, JobPayload, run_job

__all__ = [
    "MANIFEST_FILENAME",
    "JobOutcome",
    "JobPayload",
    "JobRecord",
    "JobSetManifest",
    "JobSetRunner",
    "JobStatus",
    "run_job",
]
```

---

## 2. `src/rade_qnet/orchestration/jobs/manifest.py`

12734 bytes · SHA-256 `0a5173658b335f6b`

```python
"""
The set-level record: what ran, what it produced, and what failed.

A job set's bundles are each recorded in the catalog on their own. This is
the record of the *set* -- which jobs there were, which of them succeeded,
how long each took, and for the ones that did not, why.

Written last and written atomically
-----------------------------------
Last, because a manifest that named a bundle before the bundle was on disk
would be a record of something that had not happened. Atomically, because a
set is read while it is being worked on: a dashboard, a watching script, or
the next stage of a pipeline will open this file at some point during the
hours a forty-job set takes, and a half-written one should read as the
previous version rather than as a parse error.

Atomic here means written to a temporary file in the same directory and
renamed over the target. The rename is atomic within a filesystem, and the
same directory is what guarantees the two are on one.

Why the metrics are flattened
-----------------------------
Each record carries plain ``{split: {metric: value}}`` floats rather than the
contract types a training run produces. A set's summary has to rank, compare
and plot across jobs that need not share a metric set -- different clusters,
different reports, a job that failed before evaluating -- and the one thing
all of them do share is "a number by name, if it has one".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field

from ... import __version__
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from ..compute.base import WorkResult
    from .unit import JobOutcome

__all__ = ["MANIFEST_FILENAME", "JobRecord", "JobSetManifest", "JobStatus"]

_LOGGER = get_logger(__name__)

#: Name of the manifest within a job set's output directory.
MANIFEST_FILENAME = "jobset.json"

#: Suffix for the temporary file an atomic write goes through. Visible in a
#: directory listing on purpose: a leftover one is evidence of a crash
#: mid-write, which is worth being able to see.
_WRITING_SUFFIX = ".writing"

#: What happened to one job.
JobStatus = Literal["succeeded", "failed"]


class _Record(BaseModel):
    """
    Base for the records written here.

    Not frozen, unlike a spec: a manifest is assembled rather than declared,
    and the build-then-freeze dance would buy nothing. Unknown fields are
    still forbidden, so a manifest written by a newer version and read by an
    older one fails loudly rather than silently dropping a field somebody is
    relying on.
    """

    model_config = ConfigDict(extra="forbid")


class JobRecord(_Record):
    """
    One row: what happened to one job.

    Parameters
    ----------
    job_id
        The job's identifier.
    status
        Whether it completed.
    wall_seconds
        How long it took, including a failure -- "it failed after four
        hours" and "it failed immediately" call for different responses.
    seed
        The seed actually applied, so one job can be re-run on its own and
        reproduce what it produced here.
    epochs
        How many epochs ran.
    stopped_early
        Whether early stopping ended the run. A set where every job stopped
        at epoch three is a finding about the configuration; one where none
        stopped means the epoch budget was the binding constraint. The epoch
        count alone cannot tell those apart.
    metrics
        Metrics per split.
    bundle_directory
        Where the bundle was written, relative to the set's output
        directory. Relative so the record survives the directory being
        copied, archived or mounted somewhere else -- an absolute path is
        correct exactly once, on the machine that produced it.
    bundle_version
        The version the catalog assigned.
    model_name
        The registered model name, so a bundle can be reloaded from this
        record alone.
    failure_kind
        The exception's class name, for a set that failed the same way
        forty times -- which is one fix, not forty.
    failure_message
        The exception's message.
    failure_traceback
        The formatted traceback, captured in whichever process failed.
    """

    job_id: str
    status: JobStatus
    wall_seconds: float = 0.0
    seed: int = 0
    epochs: int = 0
    stopped_early: bool = False
    metrics: Mapping[str, Mapping[str, float]] = Field(default_factory=dict)
    bundle_directory: str | None = None
    bundle_version: int | None = None
    model_name: str = ""
    failure_kind: str | None = None
    failure_message: str | None = None
    failure_traceback: str | None = None

    @classmethod
    def of(cls, result: WorkResult[JobOutcome], *, relative_to: Path) -> JobRecord:
        """
        Build a record from one executor result.

        Parameters
        ----------
        result
            What the executor returned for this job.
        relative_to
            The set's output directory, which bundle paths are recorded
            relative to.

        Returns
        -------
        JobRecord
            The record.
        """
        if not result.succeeded:
            failure = result.failure
            return cls(
                job_id=result.key,
                status="failed",
                wall_seconds=result.wall_seconds,
                failure_kind=failure.kind if failure else None,
                failure_message=failure.message if failure else None,
                failure_traceback=failure.traceback_text if failure else None,
            )

        outcome = result.value
        return cls(
            job_id=result.key,
            status="succeeded",
            wall_seconds=result.wall_seconds,
            seed=outcome.seed,
            epochs=outcome.epochs,
            stopped_early=outcome.stopped_early,
            metrics={split: dict(values) for split, values in outcome.metrics.items()},
            bundle_directory=_relative(outcome.bundle_directory, relative_to),
            bundle_version=outcome.bundle_version,
            model_name=outcome.model_name,
        )

    @property
    def succeeded(self) -> bool:
        """
        Whether the job completed.

        Returns
        -------
        bool
            True when it did.
        """
        return self.status == "succeeded"

    def metric(self, split: str, name: str) -> float | None:
        """
        Return one metric, or ``None`` if this job does not have it.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float or None
            The value, or ``None``.
        """
        return self.metrics.get(split, {}).get(name)


class JobSetManifest(_Record):
    """
    The record of one job set.

    Parameters
    ----------
    run_id
        Identifier for the set.
    spec_digest
        Digest of the job-set specification.
    name
        The set's label, if it was given one.
    created_at
        When the set finished.
    framework_version
        Which version of ``rade_qnet`` ran it.
    placement
        Where the jobs actually ran, as the executor described itself. Not
        what was configured: those differ whenever the placement policy
        chose, which is the default.
    placement_reason
        Why that placement was used. A set that chose its own placement and
        a set that was told one are not the same experiment, and six months
        later this is the only thing that remembers which it was.
    wall_seconds
        Elapsed time for the whole set, as the caller experienced it.
    jobs
        One record per job, in the order the specification declared them --
        never in completion order, which would differ between two identical
        runs.
    """

    run_id: str
    spec_digest: str
    name: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    framework_version: str = __version__
    placement: str = ""
    placement_reason: str = ""
    wall_seconds: float = 0.0
    jobs: tuple[JobRecord, ...] = ()

    @property
    def succeeded(self) -> tuple[JobRecord, ...]:
        """
        The jobs that completed.

        Returns
        -------
        tuple of JobRecord
            In declaration order.
        """
        return tuple(record for record in self.jobs if record.succeeded)

    @property
    def failed(self) -> tuple[JobRecord, ...]:
        """
        The jobs that did not.

        Returns
        -------
        tuple of JobRecord
            In declaration order.
        """
        return tuple(record for record in self.jobs if not record.succeeded)

    def record(self, job_id: str) -> JobRecord | None:
        """
        Return one job's record.

        Parameters
        ----------
        job_id
            The identifier.

        Returns
        -------
        JobRecord or None
            The record, or ``None`` if the set has no such job.
        """
        return next((record for record in self.jobs if record.job_id == job_id), None)

    def metric_by_job(self, split: str, name: str) -> dict[str, float]:
        """
        Return one metric across every job that has it.

        The shape every job-set figure and ranking needs. Jobs without the
        metric are omitted rather than given a placeholder: a zero would
        plot, and would be indistinguishable from a genuinely zero score.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        dict
            Job identifier to value, in declaration order.
        """
        values: dict[str, float] = {}
        for job in self.jobs:
            value = job.metric(split, name)
            if value is not None:
                values[job.job_id] = value
        return values

    def summary(self) -> str:
        """
        Return a one-line summary, for a log and a terminal.

        Returns
        -------
        str
            For example ``39 of 40 job(s) succeeded in 1842.3s``.
        """
        return (
            f"{len(self.succeeded)} of {len(self.jobs)} job(s) succeeded "
            f"in {self.wall_seconds:.1f}s"
        )

    def write(self, directory: Path) -> Path:
        """
        Write the manifest into a directory, atomically.

        Parameters
        ----------
        directory
            The set's output directory. Created if absent.

        Returns
        -------
        Path
            The manifest's path.
        """
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / MANIFEST_FILENAME
        # In the same directory as the target, because `os.replace` is atomic
        # only within a filesystem -- a temporary directory elsewhere could
        # be on a different one, and the rename would silently degrade into
        # a copy that a reader can observe half-finished.
        temporary = path.with_name(path.name + _WRITING_SUFFIX)
        temporary.write_text(self.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(path)
        _LOGGER.info("wrote job set manifest to %s: %s", path, self.summary())
        return path

    @classmethod
    def read(cls, directory: Path) -> JobSetManifest:
        """
        Read a manifest from a directory.

        Parameters
        ----------
        directory
            The set's output directory.

        Returns
        -------
        JobSetManifest
            The manifest.

        Raises
        ------
        FileNotFoundError
            If there is no manifest there.
        """
        path = directory / MANIFEST_FILENAME
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _relative(path: Path | None, root: Path) -> str | None:
    """
    Render a path relative to the set's directory, where it is beneath it.

    Parameters
    ----------
    path
        The path, or ``None``.
    root
        The set's output directory.

    Returns
    -------
    str or None
        A relative path, the absolute one where it is not beneath the root,
        or ``None``.
    """
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        # A job writing outside the set's directory is unusual but legal --
        # an explicit `output_root` on one job, say. Recording the absolute
        # path is better than failing to record it.
        return str(path)
```

---

## 3. `src/rade_qnet/orchestration/jobs/set.py`

11474 bytes · SHA-256 `309bbb0a049e589c`

```python
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
```

---

## 4. `src/rade_qnet/orchestration/jobs/unit.py`

11485 bytes · SHA-256 `dbcca9ec2df2b2c8`

```python
"""
One job: train one model, in whichever process is running this.

:func:`run_job` is the function a job set hands to an executor. It is
module-level, and its payload and its return value are plain data, because
both have to survive a round trip through pickle to reach a worker and come
back.

What crosses the boundary, and what does not
---------------------------------------------
:class:`JobPayload` carries the **ingredients** of a
:class:`~rade_qnet.core.runtime.context.RunContext`, not a context.

A context holds hooks, a catalog and a tracker. Hooks are arbitrary user
objects -- a progress bar bound to a terminal, a client holding a socket, a
closure over a notebook's state -- and none of that need be picklable, nor
would it mean anything in another process if it were. So the payload carries
primitives and the worker builds its own context from them.

That has a second benefit worth more than the first. A worker's context is
*provably derived from the specification*, rather than inherited from
whatever the parent process happened to be holding. Two runs of the same
specification therefore configure their workers identically, which is half
of why placement cannot change results. The other half is the seed, which is
derived from the run seed and the job identifier by hash -- never from a
position, a process identifier or a clock.

What comes back
---------------
:class:`JobOutcome` is metrics and locations, not a model.

Returning the trained model would mean pickling a network and its weights
back to the parent for every job, which is slow, memory-hungry and pointless:
the bundle is already on disk, written by the worker, and the parent wants to
know *where* far more often than it wants the object. A caller who needs the
model loads the bundle, which is the same path they would take tomorrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.runtime.components import get_model, import_registrations
from ...core.runtime.context import RunContext
from ...core.runtime.errors import SpecError
from ...core.runtime.logging import get_logger
from ...storage.catalog import JsonlCatalog
from ..pipelines.resolve import pipeline_for
from ..pipelines.train import TrainPipeline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.capability.definition import PredictorDefinition
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["JobOutcome", "JobPayload", "run_job"]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JobPayload:
    """
    Everything one job needs, in a form that survives pickling.

    Parameters
    ----------
    job_id
        Identifier for this job within its set. Names the output directory
        and the catalog entry, and -- importantly -- derives the seed, so
        changing it changes the model.
    spec
        The merged, validated run specification for this job. A pydantic
        model, so it pickles by value and arrives in the worker identical to
        the one validated in the parent.
    run_id
        Identifier for the whole set, shared by every job in it.
    spec_digest
        Digest of the job-set specification, recorded against every bundle
        so a set's members can be found together afterwards.
    output_directory
        Where this job writes. Derived in the parent rather than the worker,
        so the layout of a set is decided in one place.
    seed
        The set's base seed. The job's own seed is derived from it and the
        job identifier inside :meth:`RunContext.for_job`.
    catalog_root
        Where bundles are recorded, or ``None`` to skip recording. A path
        rather than a :class:`~rade_qnet.storage.catalog.Catalog`, because the
        catalog holds a lock file handle and a handle is meaningless in
        another process. Each worker opens its own against the same path,
        which is exactly what the single-writer design is built for.
    metadata
        Free-form annotations carried into the run context.
    registration_modules
        Modules the worker must import before it can resolve the names this
        job's specification uses.

        A spawned worker starts with a bare interpreter, so a component
        registered purely as an import side effect is absent there. It
        *appears* to work without this, because spawn re-imports the main
        module and a script that trains a model has usually imported it --
        which means the failure arrives the first time the entry point
        changes, with the symptom "no model named ..." from a specification
        that is correct and that worked yesterday.
    """

    job_id: str
    spec: SupervisedRunSpec
    run_id: str
    spec_digest: str
    output_directory: Path
    seed: int = 0
    catalog_root: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    registration_modules: tuple[str, ...] = ()

    def context(self) -> RunContext:
        """
        Rebuild the run context for this job, in this process.

        Parameters
        ----------
        None

        Returns
        -------
        RunContext
            A context whose seed is derived from the set's seed and this
            job's identifier.
        """
        parent = RunContext(
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            output_directory=self.output_directory.parent.parent,
            seed=self.seed,
            catalog=JsonlCatalog(self.catalog_root) if self.catalog_root is not None else None,
            metadata=self.metadata,
        )
        # Derived rather than constructed directly, so the seed-derivation
        # rule has exactly one implementation and a job run alone reproduces
        # what it would have produced inside its set.
        return parent.for_job(self.job_id, output_directory=self.output_directory)


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """
    What one job produced: metrics and locations, never a model.

    Parameters
    ----------
    job_id
        The job this came from.
    metrics
        Metrics per split, as plain floats. Flattened out of
        :class:`~rade_qnet.core.contract.result.TrainingResult` so the manifest
        can be written without the contract types, and so a job set's summary
        does not depend on a model's own result shape.
    seed
        The seed actually applied, recorded so one job can be reproduced
        without re-deriving it.
    epochs
        How many epochs ran.
    stopped_early
        Whether early stopping ended the run. Recorded alongside the epoch
        count rather than inferred from it: a set where every job stopped at
        epoch three is a finding about the configuration, and one where none
        of them stopped at all means the epoch budget was the binding
        constraint -- two quite different conclusions that the epoch count
        alone cannot distinguish.
    bundle_directory
        Where the bundle was written, or ``None`` if nothing was persisted.
    bundle_version
        The version the catalog assigned, or ``None``.
    model_name
        The registered model name, so a bundle can be reloaded from the
        manifest alone.
    """

    job_id: str
    metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    seed: int = 0
    epochs: int = 0
    stopped_early: bool = False
    bundle_directory: Path | None = None
    bundle_version: int | None = None
    model_name: str = ""

    def metric(self, split: str, name: str) -> float | None:
        """
        Return one metric, or ``None`` if it was not recorded.

        ``None`` rather than an exception, because a job-set summary ranks
        and plots across jobs that need not all have the same metrics -- a
        job that failed before evaluating has none at all, and asking for a
        missing one is a normal occurrence rather than a mistake.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float or None
            The value, or ``None``.
        """
        return self.metrics.get(split, {}).get(name)


def run_job(payload: JobPayload) -> JobOutcome:
    """
    Train one job and return its metrics and locations.

    Module-level, and deliberately so. Under the spawn start method a bound
    method or a closure cannot cross the process boundary, and the error it
    produces names the pickle protocol rather than the design mistake.

    Parameters
    ----------
    payload
        Everything this job needs.

    Returns
    -------
    JobOutcome
        Metrics and locations.

    Raises
    ------
    SpecError
        If the job's specification is interactive rather than supervised.
        Reinforcement-learning runs arrive in Phase 7 and fan out through
        the same machinery; until then this says so, rather than failing
        several stages later with a message about a missing data source.
    """
    # First, before anything looks a name up. Idempotent, so in a sequential
    # run where the parent already imported everything this is a handful of
    # dictionary lookups.
    import_registrations(payload.registration_modules)

    if payload.spec.task != "supervised":
        raise SpecError(
            f"job {payload.job_id!r} is a {payload.spec.task!r} run; job sets "
            f"currently fan out supervised training only"
        )

    context = payload.context()
    with context.activate():
        _LOGGER.info("starting job %s with seed %d", payload.job_id, context.seed)
        definition = _definition(payload.spec)
        # Resolved rather than fixed, so a model's training override runs
        # in a portfolio exactly as it does in a single run. A job set that
        # quietly dropped a model's overrides would produce bundles that
        # differ from the single-run ones in a way no metric reveals.
        pipeline_cls = pipeline_for(definition, "train", TrainPipeline)
        pipeline = pipeline_cls(context=context, spec=payload.spec, definition=definition)
        result = pipeline.run()

    saved = pipeline.saved
    return JobOutcome(
        job_id=payload.job_id,
        metrics={
            split: dict(evaluation.metrics) for split, evaluation in result.evaluations.items()
        },
        seed=result.seed,
        epochs=len(result.fit.history),
        stopped_early=result.fit.stopped_early,
        bundle_directory=saved.directory if saved is not None else None,
        bundle_version=saved.manifest.version if saved is not None else None,
        model_name=payload.spec.model.name,
    )


def _definition(spec: SupervisedRunSpec) -> PredictorDefinition:
    """
    Resolve the model definition this job trains.

    Resolved in the worker rather than passed in the payload. A definition is
    a class the registry already knows how to find from a name, so sending
    the object would pickle a class reference that the worker then has to
    import anyway -- with the difference that a failure to import it would
    surface as a pickle error rather than as "no model named ...".

    Parameters
    ----------
    spec
        The job's run specification.

    Returns
    -------
    PredictorDefinition
        A fresh definition.
    """
    return get_model(spec.model.name)()
```

