"""
Fixed, finite training data.

A data module is responsible for turning raw inputs into a ``DataBundle``: the
split arrays, the state that was fitted while preparing them, the input
signature, and the lineage needed to reproduce the result.

Two rules are enforced here because getting them wrong is the most common
source of silently optimistic backtests:

1. **Fitting along the scenario (time) axis may only observe training rows.**
   Scalers, basis selection and any other time-axis estimator are fitted on
   train indices alone.
2. **Fitting along the entity axis may observe the full universe.**  An
   attribute encoder or a nearest-neighbour graph over instruments is not
   leakage -- instrument identity is known at decision time.  Conflating these
   two axes would wrongly reject legitimate models.

Modules
-------
``module.py``
    ``DataModule`` -- the base a model's data build subclasses.  Declares the
    stages (load, fit state, transform, split, signature, package) that the
    train pipeline drives.  The *contract*, with no implementation of it.
    [Phase 2]
``tabular.py``
    ``TabularDataModule`` -- one implementation of that contract, and the one
    most models use unchanged: read a file, scale, reduce, split, package.
    Separate from ``module.py`` so the abstraction can be read without the
    file handling, and the file handling changed without touching the
    abstraction.  [Phase 2]
``tables.py``
    Reading raw tabular data from disk, and ``fingerprint_source``, which
    decides what counts as "the input".  [Phase 2]
``cache.py``
    The content-addressed cache for expensive builds, and
    ``PreparedDataset`` -- the cacheable intermediate that sits between the
    expensive work and the engine-specific packaging.  Keyed on the source
    fingerprint from ``tables.py``, among other things; the two modules were
    one until that coupling was made an import rather than an adjacency.
    [Phase 2]
``splits.py``
    Split strategies: ``chronological``, ``purged_kfold``, ``grouped`` and
    ``explicit``.  Sequence-aware, so a window may not straddle a split
    boundary.  [Phase 2]
``rebuild.py``
    Reconstructing a dataset from a bundle's recorded lineage, which is what
    lets evaluation and inference run without the original build.  [Phase 5]
``transforms/``
    Fitted transforms with explicit inverses.  [Phase 2]

Planned modules
---------------
``transitions.py``
    Reads a stored transition table into a dataset source, which is how
    *offline* reinforcement learning reuses this package unchanged.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
