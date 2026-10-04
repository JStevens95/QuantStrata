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

One run, or all of them
-----------------------
That is the seam this package is organised on, and it was invisible while
six modules sat flat.  ``bundle.py`` and ``manifest.py`` are about a single
directory: what is in it, whether it is intact, how to write it without ever
leaving a half-finished one that looks valid.  ``runs/`` is about the
collection: which runs happened, which are blessed, which is running now.

The two halves are read by different people at different times.  A pipeline
writes a bundle once and never looks at the index; a quant choosing a model
for Monday queries the index and never opens a bundle by hand.

Modules
-------
``bundle.py``
    Writing and reading a versioned bundle: weights, fitted state, spec,
    signature, metrics and manifest.  Written to a staging directory and
    renamed into place, so a crash can never leave a half-written bundle that
    looks valid.  Parameters are stored, never pickled model objects.
    [Phase 1, delivered]
``manifest.py``
    The manifest schema plus content hashes for every file, making corruption
    and silent drift detectable at load time.  Verification is an explicit step
    rather than automatic, because hashing a large checkpoint to populate a
    listing would make the listing unusable.  [Phase 1, delivered]
``locking.py``
    The exclusive file lock that makes the catalog's single-writer discipline
    real, with one implementation per platform behind one function.  It sits
    at this level rather than inside ``runs/`` because it is infrastructure
    both halves may take, and because ``fcntl`` is POSIX-only: imported from
    the catalog directly, it made the entire library fail to load on Windows
    rather than lose a feature.  [Phase 1, delivered]

Sub-packages
------------
``runs/``
    The index across runs: ``catalog.py`` records every run that happened,
    ``registry.py`` records which ones are blessed and under what tag, and
    ``tracker.py`` follows one that is still going.

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
