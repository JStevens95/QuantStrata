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
    train pipeline drives.  [Phase 2]
``splits.py``
    Split strategies: ``chronological``, ``purged_kfold``, ``grouped`` and
    ``explicit``.  Sequence-aware, so a window may not straddle a split
    boundary.  [Phase 2]
``transforms/``
    Fitted transforms with explicit inverses.  [Phase 2]
``io.py``
    Readers and the content-addressed cache for expensive builds.  Also
    defines ``PreparedDataset``, the cacheable intermediate that sits between
    the expensive work and the engine-specific packaging.  [Phase 2]

Planned modules
---------------
``transitions.py``
    Reads a stored transition table into a dataset source, which is how
    *offline* reinforcement learning reuses this package unchanged.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
