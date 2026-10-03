"""
Tests for ``rade_xl.sources`` -- where a training signal comes from.

The suite's organising idea is that every source, however different its
origin, satisfies one protocol. So alongside the per-source tests there is a
shared contract suite that each source implementation is run through, which is
what makes "add a new source" a safe operation.
"""
