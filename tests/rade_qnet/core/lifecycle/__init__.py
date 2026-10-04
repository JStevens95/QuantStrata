"""
Tests for ``rade_qnet.core.lifecycle`` -- pipeline execution machinery.

This sub-tree is where the framework's debuggability is defended. A failing
stage must produce an error naming that stage, a hook must fire in a
predictable order, a seed must reproduce a run exactly, and a spec hash must be
stable across processes -- otherwise the cache silently misses and tuning
re-runs work it already did.

Planned modules
---------------
``test_lifecycle_context.py``
    ``RunContext`` construction, directory layout and logger binding.
    [Phase 1]
``test_lifecycle_pipeline.py``
    The ``step()`` runner: timing, event emission, cache hits and misses, and
    failure wrapped in a ``StageError`` that names the stage.  [Phase 1]
``test_lifecycle_hooks.py``
    Hook ordering, and the guarantee that a misbehaving hook cannot abort a
    run.  [Phase 1]
``test_lifecycle_components.py``
    Registration and lookup by name, duplicate detection and the error raised
    for an unknown name.  [Phase 1]
``test_lifecycle_seeding.py``
    Reproducibility under each determinism level.  [Phase 1]
``test_lifecycle_hashing.py``
    Hash stability: equal specs hash equally, key order does not matter, and
    the hash survives a fresh interpreter.  [Phase 1]
``test_lifecycle_logging.py``
    Contextual identifiers surviving a process boundary.  [Phase 1]
``test_lifecycle_errors.py``
    The error hierarchy and its messages.  [Phase 1]
"""
