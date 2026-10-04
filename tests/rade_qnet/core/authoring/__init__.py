"""
Tests for ``rade_qnet.core.authoring`` -- the model-facing interface.

Two properties matter most here, and both are about keeping the framework
welcoming. A model that implements only the required minimum must work, which
is what keeps a simple model to one file. And a model that implements no
capabilities must be unaffected by every authoring that exists, which is what
lets new capabilities be added without a migration.

Planned modules
---------------
``test_authoring_definition.py``
    The ``ModelDefinition`` bases and the ``@model`` decorator, including the
    error raised when two models claim the same name.  [Phase 1]
``test_authoring_protocols.py``
    Runtime detection of each opt-in authoring, verified both ways: present
    and detected, absent and ignored.  [Phase 1]
``test_authoring_supervised.py``
    ``SupervisedModel``, asserting that a minimal model needs one line of data
    code, and that every fault it detects names the model package.  [Phase 2]
"""
