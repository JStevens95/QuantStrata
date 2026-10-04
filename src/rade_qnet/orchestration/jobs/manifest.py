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
from ...core.provenance.logging import get_logger

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
