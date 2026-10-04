# `src/rade_qnet/orchestration/jobs`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 64 | 2610 | `38c0c4fe2f3800a8` |
| 2 | `fanout.py` | 165 | 6078 | `53090f7ef26e54f6` |
| 3 | `groups.py` | 441 | 13842 | `e51d04f043e63356` |
| 4 | `manifest.py` | 407 | 12734 | `0a5173658b335f6b` |
| 5 | `set.py` | 336 | 11660 | `8aa5850984437273` |
| 6 | `unit.py` | 301 | 11485 | `dbcca9ec2df2b2c8` |

---

## 1. `src/rade_qnet/orchestration/jobs/__init__.py`

2610 bytes · SHA-256 `38c0c4fe2f3800a8`

```python
"""
One model, many jobs.

A job set is deliberately simple: a list of jobs, each one a full independent
training run of the same model, differing in its data slice and -- optionally
-- in its architecture complexity.  A data group with abundant history can be
given a wider, deeper configuration than a sparse one, from the same
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
``groups.py``
    ``read_group_set(root)``: which data groups exist on disk, read from a
    declared ``groups.json`` manifest, with a cheap snapshot fingerprint for
    provenance.  Knows nothing about what a group *means*.
``fanout.py``
    ``job_set_for_groups``: turns a group set into a ``JobSetSpec``, one job
    per group, with a per-group override hook and the snapshot fingerprint
    carried as a tag onto every bundle.

Why expansion is separate from the runner
-----------------------------------------
The runner runs jobs and nothing else.  Expanding a group set into jobs
happens before the runner sees anything and arrives as an ordinary
``JobSetSpec``, so a hand-written set and an expanded one are
indistinguishable to it -- which is what keeps the runner a job runner
rather than a group runner.  The model is resolved by name through the
registry, because ``orchestration`` may not import ``models``.
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

## 2. `src/rade_qnet/orchestration/jobs/fanout.py`

6078 bytes · SHA-256 `53090f7ef26e54f6`

```python
"""
Turning a set of data groups into the jobs a job set fans out over.

This is the bridge between "a directory holding four groups" and "a job set
with four jobs". It produces a :class:`~rade_qnet.core.spec.jobs.JobSetSpec`:
shared defaults, one job per group, each job overriding the directory it
reads and anything the caller wants to vary.

Why expansion is separate from the runner
-----------------------------------------
:mod:`rade_qnet.orchestration.jobs.set` knows how to run *jobs* and nothing
else. By the time it sees anything, there are only jobs -- a caller wrote the
set by hand, or this module expanded a group set into one, and the runner
cannot tell the difference. That is exactly the property that makes the
runner reusable: the moment it knew what a group was, it would stop being a
job runner and start being a group runner.

Settings per group
------------------
The reason a group set is a job set rather than a loop is that groups are not
alike. One with abundant history supports a wider, deeper model than a thin
one, and forcing both to the same configuration means either underfitting the
first or overfitting the second. :func:`job_set_for_groups` therefore takes a
per-group override hook, which is the mechanism behind that whole
proposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.runtime.logging import get_logger
from ...core.spec.jobs import JobSetSpec, parse_job_set_spec
from ...core.spec.merge import deep_merge

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .groups import DataGroup, GroupSet

__all__ = ["FINGERPRINT_KEY", "fingerprint_tag", "group_overrides", "job_set_for_groups"]

_LOGGER = get_logger(__name__)

#: Prefix the snapshot digest is tagged with, so every bundle in the set can
#: be traced back to the data it was trained on.
FINGERPRINT_KEY = "data_fingerprint"


def fingerprint_tag(fingerprint: str) -> str:
    """
    Encode a snapshot digest as a set tag.

    A set's tags are free-form labels rather than a mapping, so a key-value
    fact has to be encoded into one. Done here, in one place, because the
    writer and every future reader have to agree on the encoding, and a
    convention applied at two call sites is a convention that will eventually
    be applied at one.

    Parameters
    ----------
    fingerprint
        The group set's snapshot digest.

    Returns
    -------
    str
        For example ``data_fingerprint=7f3a9c21...``.
    """
    return f"{FINGERPRINT_KEY}={fingerprint}"


def group_overrides(group: DataGroup) -> dict[str, Any]:
    """
    Return the overrides every group job needs, whatever the model.

    Only the source directory. Deliberately minimal: anything else would be
    this module deciding how a model should be configured, which it is in no
    position to know -- the whole point of the framework is that the model
    declares that.

    Parameters
    ----------
    group
        The group.

    Returns
    -------
    dict
        An override fragment, merged over the set's defaults.
    """
    return {"source": {"params": {"directory": str(group.directory)}}}


def job_set_for_groups(
    groups: GroupSet,
    *,
    defaults: Mapping[str, Any],
    output_root: Path,
    name: str | None = None,
    placement: Mapping[str, Any] | None = None,
    overrides_for: Callable[[DataGroup], Mapping[str, Any]] | None = None,
) -> JobSetSpec:
    """
    Expand a group set into a job set, one job per group.

    Parameters
    ----------
    groups
        The groups to train over.
    defaults
        The run-specification fragment every job shares: the model, the
        source kind, the training settings. Merged *under* each job's
        overrides, so a per-group setting wins and everything it does not
        mention survives.
    output_root
        Where the set writes.
    name
        The set's label, used in its run identifier and so in its directory
        name. Defaults to ``groups``.
    placement
        Where jobs run. ``None`` leaves it to the placement policy, which is
        the normal path.
    overrides_for
        Per-group overrides, merged over :func:`group_overrides`. This is the
        hook that lets a data-rich group get a wider model than a thin one,
        which is the reason a group set is a job set at all.

    Returns
    -------
    JobSetSpec
        A validated job set. Validated here rather than at dispatch, so a
        set that expands into something misconfigured fails at expansion --
        where the error can say which group -- rather than after some of its
        jobs have already trained.
    """
    jobs = []
    for group in groups:
        overrides: dict[str, Any] = dict(group_overrides(group))
        if overrides_for is not None:
            # Merged by the job-set loader's own rules rather than ad hoc
            # here, so a per-group fragment follows exactly the same merge
            # semantics as one written by hand in a file. Two merge
            # implementations would eventually disagree, and the
            # disagreement would look like a model behaving differently
            # depending on how it was launched.
            overrides = deep_merge(overrides, overrides_for(group))
        jobs.append({"id": group.name, "overrides": overrides, "description": group.describe()})

    spec = parse_job_set_spec(
        {
            "name": name or "groups",
            "output_root": str(output_root),
            "defaults": dict(defaults),
            "jobs": jobs,
            **({"placement": dict(placement)} if placement is not None else {}),
            # Carried as a tag so it reaches every bundle. A result that
            # looks surprising six months from now is asked one question
            # first: was this trained on the data I think it was.
            "tags": [fingerprint_tag(groups.fingerprint)],
        }
    )
    _LOGGER.info("expanded %s into %d job(s)", groups.describe(), len(spec.jobs))
    return spec
```

---

## 3. `src/rade_qnet/orchestration/jobs/groups.py`

13842 bytes · SHA-256 `e51d04f043e63356`

```python
"""
Which data groups exist on disk, and which snapshot of them a run used.

A **data group** is one self-contained training problem: its own directory,
its own input and target columns, its own history. A set of them is what
turns a single run into a job set -- one model trained per group, in
parallel, each writing its own bundle.

This is the general mechanism behind "train this model over every slice of my
data". A slice might be a region, a desk, a product line, a book of
instruments; the framework does not care, and that is the point. Nothing here
knows what a group *means*, only that it has a name, a directory and a set of
column identifiers.

Why a declared manifest rather than a directory listing
-------------------------------------------------------
A group set is described by a JSON file, not by whichever subdirectories
happen to be present. A listing silently changes meaning the moment somebody
leaves a scratch folder behind, and a job set that quietly grew a
forty-first member is a result nobody asked for. The manifest makes the
membership a decision rather than an accident.

The fingerprint
---------------
Every group set read here is fingerprinted, and the fingerprint is tagged
onto every bundle the resulting job set produces. The reason is the question
asked about every result that looks surprising: *was this trained on the data
I think it was?* Six months later the directory has been refreshed, columns
have been added, and a digest captured at the time is the only way to answer.

The digest covers what identifies the snapshot -- the group names, their
column identifiers, and the size and modification time of every file read --
rather than the file contents. Hashing gigabytes on every run to detect a
change that a size and a timestamp already reveal would make the check
expensive enough that somebody eventually turns it off, and a provenance
check nobody runs is worth nothing.

What this module does not do
----------------------------
It does not read the data. A group's history is read by whatever data module
the model declares, through the ordinary source machinery, because the shape
that history needs to be in is the model's business. This module answers the
prior question: which groups exist, what is in them, and which snapshot is
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.runtime.errors import SpecError
from ...core.runtime.hashing import digest_payload
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = ["MANIFEST_FILENAME", "DataGroup", "GroupSet", "read_group_set"]

_LOGGER = get_logger(__name__)

#: The file a group set directory is described by.
MANIFEST_FILENAME = "groups.json"

#: Manifest keys consumed directly by this module. Anything else in an entry
#: is carried through as a free-form attribute, which is what lets a caller
#: annotate a group without this module growing a field for it.
_RESERVED_KEYS = frozenset({"name", "directory", "input_ids", "target_ids"})


@dataclass(frozen=True, slots=True)
class DataGroup:
    """
    One self-contained training problem.

    Parameters
    ----------
    name
        The group's identifier, which becomes a job identifier and therefore
        a directory name and a seed component. Keep it to characters that are
        safe in a path, for that reason.
    directory
        Where this group's data lives.
    input_ids, target_ids
        Identifiers for the group's input and target columns, in column
        order.

        Carried for **provenance and reporting**, not for computation: they
        go into the snapshot fingerprint and into a log line, and that is
        all this module does with them. A model that needs to interpret its
        columns reads them through its own ``data.py``, in whatever
        vocabulary it uses -- which is why these are plain tuples here
        rather than a framework type that would have to name the two sides
        of somebody else's problem.
    attributes
        Free-form annotations from the manifest -- a region, a desk, a
        currency. Carried rather than interpreted, because every caller
        partitions their data differently and a fixed set of fields here
        would be wrong for the second one.
    """

    name: str
    directory: Path
    input_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    attributes: dict[str, str]

    @property
    def n_columns(self) -> int:
        """
        How many columns the group covers, inputs and targets together.

        Returns
        -------
        int
            The count.
        """
        return len(self.input_ids) + len(self.target_ids)

    def describe(self) -> str:
        """
        Return a one-line description, for a log and a progress line.

        Returns
        -------
        str
            For example ``FX__G10: 24 input, 6 target column(s)``.
        """
        return f"{self.name}: {len(self.input_ids)} input, {len(self.target_ids)} target column(s)"


@dataclass(frozen=True, slots=True)
class GroupSet:
    """
    A set of data groups and the fingerprint of the snapshot they came from.

    Parameters
    ----------
    groups
        The groups, in manifest order. Order is preserved rather than sorted
        so that a job set's directory listing matches the file somebody
        wrote.
    root
        The directory the set was read from.
    fingerprint
        Digest of the snapshot, recorded against every bundle trained from
        it.
    """

    groups: tuple[DataGroup, ...]
    root: Path
    fingerprint: str

    def __len__(self) -> int:
        """
        Return how many groups there are.

        Returns
        -------
        int
            The count.
        """
        return len(self.groups)

    def __iter__(self) -> Iterator[DataGroup]:
        """
        Iterate the groups in manifest order.

        Yields
        ------
        DataGroup
            Each group.
        """
        return iter(self.groups)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Every group name, in manifest order.

        Returns
        -------
        tuple of str
            The names.
        """
        return tuple(group.name for group in self.groups)

    def group(self, name: str) -> DataGroup:
        """
        Return one group by name.

        Parameters
        ----------
        name
            The group's identifier.

        Returns
        -------
        DataGroup
            The group.

        Raises
        ------
        KeyError
            If there is no such group, listing the ones there are. The usual
            cause is a typo in a caller's group filter, and the list is what
            turns that into a one-second fix.
        """
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"no group named {name!r}; this set has {list(self.names)}")

    def select(self, names: Iterable[str] | None) -> GroupSet:
        """
        Return a set restricted to some groups.

        The fingerprint is **kept**, not recomputed. A subset of a snapshot
        was still trained on that snapshot, and recomputing would make two
        runs over different subsets of identical data look like runs over
        different data.

        Parameters
        ----------
        names
            Which groups to keep, in the order given. ``None`` keeps
            everything, which is the common case and avoids the caller
            branching.

        Returns
        -------
        GroupSet
            The restricted set.

        Raises
        ------
        KeyError
            If any name is not in the set.
        """
        if names is None:
            return self
        return GroupSet(
            groups=tuple(self.group(name) for name in names),
            root=self.root,
            fingerprint=self.fingerprint,
        )

    def describe(self) -> str:
        """
        Return a one-line summary.

        Returns
        -------
        str
            For example ``4 group(s), 138 column(s), snapshot a1b2c3d4``.
        """
        columns = sum(group.n_columns for group in self.groups)
        return f"{len(self.groups)} group(s), {columns} column(s), snapshot {self.fingerprint[:8]}"


def read_group_set(root: Path) -> GroupSet:
    """
    Read a group set from a directory and fingerprint the snapshot.

    Parameters
    ----------
    root
        The directory, containing a manifest named ``groups.json``.

    Returns
    -------
    GroupSet
        The groups and the snapshot digest.

    Raises
    ------
    SpecError
        If the manifest is absent, unreadable, or describes a group whose
        directory does not exist. Checked here, once, rather than left to
        surface inside whichever job happened to be scheduled first: a set
        with a missing directory is knowable before any training starts, and
        discovering it thirty-nine jobs in is the framework's fault.
    """
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SpecError(
            f"no group manifest at {manifest_path}. A group set directory is "
            f"described by a {MANIFEST_FILENAME} listing its groups, rather than "
            f"by whatever subdirectories happen to be present"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SpecError(f"{manifest_path} is not readable JSON: {error}") from error

    groups = tuple(
        _group(entry, root=root, origin=manifest_path) for entry in _entries(payload, manifest_path)
    )
    if not groups:
        raise SpecError(f"{manifest_path} declares no groups; a set needs at least one")

    group_set = GroupSet(
        groups=groups,
        root=root,
        fingerprint=_fingerprint(groups, manifest_path),
    )
    _LOGGER.info("read group set from %s: %s", root, group_set.describe())
    return group_set


def _entries(payload: object, origin: Path) -> Sequence[dict[str, object]]:
    """
    Pull the group list out of a manifest payload.

    Parameters
    ----------
    payload
        The parsed JSON.
    origin
        The file, for error messages.

    Returns
    -------
    sequence of dict
        The group entries.

    Raises
    ------
    SpecError
        If the payload is not a mapping with a list of groups.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise SpecError(
            f"{origin} must be a mapping with a 'groups' list, received {type(payload).__name__}"
        )
    return payload["groups"]


def _group(entry: object, *, root: Path, origin: Path) -> DataGroup:
    """
    Build one group from a manifest entry.

    Parameters
    ----------
    entry
        The entry.
    root
        The set's directory, which relative group paths are resolved
        against.
    origin
        The manifest file, for error messages.

    Returns
    -------
    DataGroup
        The group.

    Raises
    ------
    SpecError
        If the entry is malformed or names a directory that is not there.
    """
    if not isinstance(entry, dict) or "name" not in entry:
        raise SpecError(f"{origin}: each group needs at least a 'name', received {entry!r}")

    name = str(entry["name"])
    directory = root / str(entry.get("directory", name))
    if not directory.is_dir():
        raise SpecError(f"{origin}: group {name!r} points at {directory}, which is not a directory")

    return DataGroup(
        name=name,
        directory=directory,
        input_ids=tuple(str(value) for value in entry.get("input_ids", ())),
        target_ids=tuple(str(value) for value in entry.get("target_ids", ())),
        attributes={
            str(key): str(value) for key, value in entry.items() if key not in _RESERVED_KEYS
        },
    )


def _fingerprint(groups: Sequence[DataGroup], manifest_path: Path) -> str:
    """
    Digest what identifies this snapshot.

    Covers the manifest's own size and modification time, plus each group's
    name, column identifiers and directory contents by size and timestamp.
    Not the data itself: hashing gigabytes on every run to detect a change a
    timestamp already reveals makes the check expensive enough that somebody
    eventually disables it, and a provenance check nobody runs is worth
    nothing.

    Parameters
    ----------
    groups
        The groups.
    manifest_path
        The manifest, which is included in the digest.

    Returns
    -------
    str
        A hex digest.
    """
    payload: dict[str, object] = {"manifest": _stat(manifest_path)}
    for group in groups:
        payload[group.name] = {
            "input_ids": list(group.input_ids),
            "target_ids": list(group.target_ids),
            # Sorted, because a directory listing's order is filesystem
            # specific and would make the same snapshot fingerprint
            # differently on two machines.
            "files": {
                path.name: _stat(path)
                for path in sorted(group.directory.rglob("*"))
                if path.is_file()
            },
        }
    return digest_payload(payload)


def _stat(path: Path) -> dict[str, int]:
    """
    Return the size and modification time a fingerprint uses.

    Parameters
    ----------
    path
        The file.

    Returns
    -------
    dict
        Size in bytes and modification time in nanoseconds.
    """
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
```

---

## 4. `src/rade_qnet/orchestration/jobs/manifest.py`

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

## 5. `src/rade_qnet/orchestration/jobs/set.py`

11660 bytes · SHA-256 `8aa5850984437273`

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

Why the runner does not expand anything itself
----------------------------------------------
Expanding a group set into jobs happens *before* the runner sees anything --
by the user writing a file, by
:func:`~rade_qnet.orchestration.jobs.fanout.job_set_for_groups`, or by
``rade_qnet.api`` -- and arrives as a ``JobSetSpec``. The model is resolved
through the registry by name, exactly as ``TrainPipeline`` does it, because
``orchestration`` may not import ``models``.

Keeping the two apart is what makes expansion a separate, independently
testable function instead of a branch inside the runner, and keeps the
runner ignorant of where its jobs came from.
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
        Where bundles are recorded. Defaults to the set's ``output_root`` --
        the same place a single run records -- so every run under one root
        shares one catalog, and a registry over it can compare a sweep's
        variants or this week's retrains with last week's. Defaulting to the
        set's own directory, as this once did, gave every variant of a set
        its own catalog, and no query could see across them.
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
        self.catalog_root = catalog_root if catalog_root is not None else spec.output_root

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

## 6. `src/rade_qnet/orchestration/jobs/unit.py`

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

