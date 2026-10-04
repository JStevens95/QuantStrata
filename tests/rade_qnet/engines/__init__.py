"""
Tests for ``rade_qnet.engines`` -- the training-library adapters.

As with sources, there is one shared contract suite that every engine is run
through. It is the suite that keeps the engine interface honest: a one-shot
tree fit and a multi-epoch gradient loop must both satisfy it, and if only the
gradient loop can, then the interface is a PyTorch interface with a generic
name.

Planned modules
---------------
``test_engines_base.py``
    The ``Engine`` protocol and capability flags, plus the shared conformance
    suite each adapter is parametrised into.  [Phase 1]
"""
