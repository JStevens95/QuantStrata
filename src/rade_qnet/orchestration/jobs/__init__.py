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
