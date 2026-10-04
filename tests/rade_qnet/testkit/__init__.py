"""
Tests for ``rade_qnet.testkit`` -- the conformance tooling.

The tooling that decides whether a model is correct has to be correct itself.
A conformance suite that passes everything is worse than none, because it
provides false assurance, so these tests check the suite in both directions: it
passes a model built to the contract, and it fails a model deliberately built
to violate each clause.

Planned modules
---------------
``test_testkit_conformance.py``
    Each conformance check verified against a compliant model and against a
    model that breaks exactly that check.  [Phase 1]
``test_testkit_parity.py``
    Comparison against a golden fixture: identical inputs pass, and a
    perturbation larger than the tolerance fails.  [Phase 0]
``test_testkit_fixtures.py``
    The synthetic sources and specs are deterministic, so a test built on them
    cannot flake.  [Phase 1]
"""
