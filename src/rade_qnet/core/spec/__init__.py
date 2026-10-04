"""
Configuration schemas -- the declarative description of a run.

Every specification is a ``pydantic`` model with ``extra="forbid"``, so a typo
in a YAML key is a load-time error rather than a silently ignored setting.
Specs round-trip exactly: ``Spec.model_validate(spec.model_dump())`` returns an
equal object, which is what makes a persisted spec a faithful record of how a
model was produced.

Specs describe *intent* only.  They never hold fitted state, file handles,
tensors or live objects, which keeps them cheap to hash, log and send to a
worker process.

Modules
-------
``run.py``
    ``RunSpec``, discriminated on ``task`` into a supervised run and a
    reinforcement-learning run.  The single object a pipeline is built from.
    [Phase 1, delivered]
``jobs.py``
    ``JobSpec`` (one training job) and ``JobSetSpec`` (shared defaults, a list
    of jobs with per-job overrides, and placement).  [Phase 4, delivered]
``data.py``
    ``SourceSpec`` union, ``SplitSpec`` and ``LoaderSpec``.  [Phase 1, delivered]
``environment.py``
    ``EnvSpec``, ``VectorSpec`` and ``WrapperSpec`` for interactive sources.
    [Phase 7]
``batching.py``
    ``DatasetBatchingSpec``, ``RolloutSpec``, ``ReplaySpec`` and
    ``SimulationSpec``, discriminated on ``kind``.  [Phase 1 / Phase 7, delivered]
``training.py``
    Engine-discriminated training specs: ``TorchTrainSpec``, ``XGBTrainSpec``,
    ``SklearnTrainSpec`` and ``RlTrainSpec``.  [Phase 1, delivered]
``hardware.py``
    ``HardwareSpec`` -- how a *single* job uses its machine (device,
    precision, compilation, distribution, determinism).  Deliberately separate
    from placement, which decides *where* jobs run -- with the exception of
    ``threads_per_worker``, which lives here precisely because it must *not*
    be a placement decision: the thread count fixes the order a reduction
    accumulates in, so a budget travelling with the executor would let
    placement change results.  [Phase 1, delivered; see defect 13]
``reports.py``
    ``ReportsSpec`` -- which report writers are enabled for a run.  [Phase 1, delivered]
``merge.py``
    Deep-merge of job-set defaults with per-job overrides, with the merge rules
    stated explicitly so an override never silently drops a nested default.
    Merges *raw mappings*, before validation: once a mapping has been through
    the schema, a field the user never set is indistinguishable from one they
    did, so merging validated specs would have every job overwrite the shared
    defaults with its own copy of them.  [Phase 4, delivered]
"""

__all__: tuple[str, ...] = ()
