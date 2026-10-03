"""
Tests for ``rade_xl.orchestration.jobs`` -- fan-out across many jobs.

Partial failure is the case these tests exist for. A job set of forty clusters
where one cluster has insufficient history must return thirty-nine trained
models and one recorded failure. Discarding the other thirty-nine because of
one bad input is the behaviour that makes a framework untrustworthy at scale.

Planned modules
---------------
``test_jobs_set.py``
    Job expansion from a spec, per-job override merging including differing
    architecture complexity, result aggregation, and partial failure.
    [Phase 4]
``test_jobs_unit.py``
    ``run_job`` is importable and picklable at module level -- the property
    that lets it cross a process boundary under the spawn start method, which
    a bound method or closure cannot.  [Phase 4]
``test_jobs_manifest.py``
    Manifest contents, and that it is written atomically by a single owner so a
    concurrent set cannot lose entries.  [Phase 4]
"""
