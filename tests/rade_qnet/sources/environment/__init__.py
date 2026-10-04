"""
Tests for ``rade_qnet.sources.environment`` -- interactive data.

Environments are tested against deterministic toy dynamics with known closed
forms, so an assertion can be exact rather than statistical. A test that only
checks "reward went up" cannot distinguish a correct implementation from a
lucky one.

Planned modules
---------------
``test_environment_protocol.py``
    Reset and step contracts, episode termination, and -- for the
    differentiable variant -- that a gradient actually flows through the
    dynamics.  [Phase 7]
``test_environment_spaces.py``
    Space descriptions and the derived ``PolicySignature``.  [Phase 7]
``test_environment_vector.py``
    Batched stepping, asserting that the vectorised form matches the serial
    form element for element.  [Phase 7]
``test_environment_wrappers.py``
    Wrapper composition and order, and that each wrapper records itself in the
    run lineage.  [Phase 7]
``test_environment_recording.py``
    Episode capture, and that a recorded episode replays identically.
    [Phase 7]
"""
