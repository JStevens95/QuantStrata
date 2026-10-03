"""
Tests for ``rade_xl.sources.environment.adapters``.

Adapter tests run against a stub implementing the external interface rather
than against the real library, so the suite neither installs nor depends on a
third-party environment package. The assertion is about translation only: the
external interface in, the framework protocol out.

Planned modules
---------------
``test_adapters_gymnasium.py``
    Space mapping and step-return normalisation, including the five-tuple
    convention and the distinction between termination and truncation.
    [Phase 7]
"""
