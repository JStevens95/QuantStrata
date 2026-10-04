"""
Tests for ``rade_qnet.core.capability`` -- the model-facing interface.

Two properties matter most here, and both are about keeping the framework
welcoming. A model that implements only the required minimum must work, which
is what keeps a simple model to one file. And a model that implements no
capabilities must be unaffected by every capability that exists, which is what
lets new capabilities be added without a migration.

Planned modules
---------------
``test_capability_definition.py``
    The ``ModelDefinition`` bases and the ``@model`` decorator, including the
    error raised when two models claim the same name.  [Phase 1]
``test_capability_protocols.py``
    Runtime detection of each opt-in capability, verified both ways: present
    and detected, absent and ignored.  [Phase 1]
``test_capability_simple.py``
    The convenience bases, asserting that a minimal model needs no data code.
    [Phase 2]
"""
