"""
Bridges from third-party environment interfaces to ``Environment``.

Adapters are thin and one-directional: they translate an external interface
into the framework's protocol and nothing more.  Keeping them isolated here
means an upstream API change is a single-file fix, and the framework never
takes a hard dependency on an external environment library.

Planned modules
---------------
``gymnasium.py``
    Wraps a Gymnasium environment, mapping its spaces onto framework spaces and
    normalising the step-return convention.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
