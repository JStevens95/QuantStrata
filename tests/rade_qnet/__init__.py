"""
Unit tests for the rade_qnet framework.

The tree below mirrors ``src/rade_qnet`` one directory at a time.  Mirroring is
not tidiness for its own sake: it means the question "is this component
tested?" is answered by looking at one predictable path, and an untested
sub-package shows up as an empty directory rather than as an absence nobody
notices.

Tests are built in the same order as the framework, so that each phase is
verified against components already proven in the phase before it.  See
``src/rade_qnet/docs/IMPLEMENTATION.md`` for the order and for each phase's
definition of done.

Scope
-----
These tests cover the **model-independent** framework.  Model-specific suites
arrive with the phase that builds the model, which is why ``models`` and
``domains`` hold placeholder packages rather than full mirrors.

Naming
------
``test_<package>_<module>.py`` -- matching the convention used by the sibling
test suites in this repository, so a file name states which module it covers
without reference to its directory.

Running
-------
From the repository root::

    .venv/bin/python -m pytest tests/rade_qnet -q
"""
