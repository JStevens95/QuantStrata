"""
Tests for ``rade_xl.sources.dataset`` -- fixed training data.

This is the highest-value sub-tree in the suite, because the defects it guards
against do not announce themselves. A leaky split does not raise an error; it
produces an encouraging validation score and a model that fails in production.
The tests therefore assert leakage properties directly rather than inferring
them from metrics.

Planned modules
---------------
``test_dataset_module.py``
    ``DataModule`` stage ordering, and the ``DataBundle`` it produces.
    [Phase 2]
``test_dataset_splits.py``
    Each split strategy. Explicitly covered: splits are disjoint and cover the
    index exactly, chronological order is respected, no sequence window
    straddles a boundary, and the scenario split is unaffected by the batch
    shuffle flag -- which is the conflation that made splits random in the
    implementation this framework replaces.  [Phase 2]
``test_dataset_io.py``
    Readers and cache behaviour: a cache hit returns an identical bundle, and a
    changed source fingerprint invalidates the entry.  [Phase 2]
``test_dataset_transitions.py``
    Reading a stored transition table as a dataset source.  [Phase 7]
"""
