"""
The system of record.

Everything a run produces lands here, and nothing else writes to the model
store.  A single writer is what makes concurrent job sets safe: a catalog
updated by eight worker processes through read-modify-write will lose entries,
which is a defect this package exists to prevent.

A bundle is self-describing.  Given only a bundle directory, the framework can
state which model produced it, from which spec, against which data
fingerprint, at which code version, and can rebuild the model and invert its
target transforms without consulting the original run.

Modules
-------
``manifest.py``
    The manifest schema plus content hashes for every file, making corruption
    and silent drift detectable at load time.  Verification is an explicit step
    rather than automatic, because hashing a large checkpoint to populate a
    listing would make the listing unusable.  [Phase 1, delivered]
``bundle.py``
    Writing and reading a versioned bundle: weights, fitted state, spec,
    signature, metrics and manifest.  Written to a staging directory and
    renamed into place, so a crash can never leave a half-written bundle that
    looks valid.  Parameters are stored, never pickled model objects.
    [Phase 1, delivered]
``catalog.py``
    The index of bundles, queryable by model, job and tag, with a single-writer
    discipline and an append-oriented log rather than whole-file rewrites.
    ``JsonlCatalog`` for real runs, ``InMemoryCatalog`` for tests and
    notebooks.  [Phase 1, delivered]
``registry.py``
    ``RunRegistry``: choose a trained run by tag, by best metric or by an
    alias such as ``production``, and record promotions and tags added after
    training as append-only events beside the catalog -- never by rewriting a
    bundle.  [Delivered after Phase 6]
``locking.py``
    The exclusive file lock that makes the catalog's single-writer discipline
    real, with one implementation per platform behind one function.  Separate
    from ``catalog.py`` because ``fcntl`` is POSIX-only: imported there, it
    made the entire library fail to load on Windows rather than lose a
    feature.  [Phase 1, delivered]
``tracker.py``
    Experiment tracking behind one interface, with a no-op default.  Tracking
    is optional infrastructure; a run must never fail because a tracking server
    is unreachable.  [Phase 1, delivered]

Planned modules
---------------
``predictions.py``
    Writing a prediction set alongside its bundle reference, for the inference
    pipeline.  [Phase 5]

Dependency rule
---------------
May import: ``core``.
May not import: ``engines``.  Serialising weights is the engine's job; this
package stores the bytes the engine hands it.
"""

__all__: tuple[str, ...] = ()
