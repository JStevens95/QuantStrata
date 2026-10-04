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
