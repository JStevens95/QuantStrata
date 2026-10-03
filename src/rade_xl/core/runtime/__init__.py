"""
The machinery that executes a pipeline.

Where ``spec`` and ``contract`` describe *what* a run is, ``runtime`` provides
the apparatus that makes a run happen and makes it observable: a context object
threaded through every stage, an instrumented step runner, lifecycle hooks,
component lookup by name, deterministic seeding and the framework's error
hierarchy.

This package is deliberately small and dependency-free.  It is imported by
every pipeline, every engine and every report writer, so anything heavy placed
here would be paid for by every process the framework starts.

Modules
-------
``errors.py``
    ``RadeXLError`` and its specialisations: ``SpecError``, ``StageError``,
    ``ContractError``, ``BundleError``, ``CapabilityError``, ``ComponentError``.
    [Phase 1, delivered]
``hashing.py``
    Stable hashes for a spec, a payload, an array set and a file, used for
    cache keys and bundle provenance.  SHA-256 over canonical JSON, never
    Python's ``hash()``, which is salted per process.  [Phase 1, delivered]
``logging.py``
    Structured logging with contextual identifiers (run, job, stage) carried on
    ``contextvars`` so a worker's output is attributable without threading a
    logger through every call.  Configuration is explicit and never happens at
    import.  [Phase 1, delivered]
``seeding.py``
    ``seed_everything(seed, determinism)`` where determinism is ``off``,
    ``warn`` or ``strict``.  Unlike a blanket deterministic flag, this is
    explicit about the performance and kernel-support trade-off being made.
    Engines register their own seeding callbacks, because ``core`` may not
    import a training library.  [Phase 1, delivered]
``components.py``
    The name registry: ``@model``, ``@engine``, ``@learner``, ``@report`` and
    the lookups that resolve a string in a spec to a class.  Resolution by name
    (rather than by importable dotted path) keeps specs stable across
    refactors.  [Phase 1, delivered]
``hooks.py``
    ``PipelineHook`` -- the run, stage, epoch, metric and artifact observation
    points.  A hook may observe but never alter, and a failing hook never fails
    a run.  [Phase 1, delivered]
``context.py``
    ``RunContext`` -- run identifier, working directory, resolved seed, bound
    logging identifiers, hook fan-out and handles to the catalog and tracker.
    One object passed down instead of a dozen keyword arguments.  Also declares
    the ``Catalog`` and ``Tracker`` protocols that ``storage`` implements, since
    the dependency may not run the other way.  [Phase 1, delivered]
``pipeline.py``
    The ``Pipeline`` base and its ``step()`` runner.  Every stage goes through
    ``step()``, which times it, emits start and end events, and wraps failures
    in a ``StageError`` naming the stage.  [Phase 1, delivered]

Planned modules
---------------
``cache.py``
    The step cache, so tuning and evaluation can reuse an expensive data build
    across trials keyed by spec digest.  [Phase 5]
"""

__all__: tuple[str, ...] = ()
