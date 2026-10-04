"""
Tests for ``rade_qnet.storage`` -- bundles, catalog and tracking.

Storage is tested for its failure behaviour as much as its success behaviour,
because the failures are silent. An interrupted write must leave no bundle that
looks valid; a concurrent catalog update must lose no entry; a corrupted file
must be detected at load rather than producing quiet nonsense.

Planned modules
---------------
``test_storage_bundle.py``
    Write and read round-trips, and that an interrupted write leaves nothing
    loadable -- verified by writing to a temporary location and failing before
    the rename.  [Phase 1]
``test_storage_manifest.py``
    Manifest schema, content hashes, and a modified file failing verification.
    [Phase 1]
``test_storage_catalog.py``
    Registration, query by model and job, and concurrent writes from several
    processes losing no entry.  [Phase 1]
``test_storage_tracker.py``
    The tracking interface, and that an unreachable backend degrades to a
    warning instead of failing a completed run.  [Phase 1]
"""
