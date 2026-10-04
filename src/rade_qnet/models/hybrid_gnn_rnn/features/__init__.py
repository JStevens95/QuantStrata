"""
Feature construction specific to the hybrid graph-temporal network.

These transforms are fitted along the **entity axis**: they look across the
whole instrument universe rather than across time.  That is not leakage --
which instruments exist, and what their attributes are, is known before any
P&L is observed.  The framework's conformance suite distinguishes the two axes
precisely so that these transforms are not wrongly flagged.

Planned modules
---------------
``encoder.py``
    Encodes instrument attributes (currency pair, tenor, product type) into the
    dense feature matrix the graph block consumes.  [Phase 3]
``graph.py``
    Builds the sparse instrument graph from encoded attributes, emitting edge
    indices, edge weights and the dense shape.  [Phase 3]
"""

__all__: tuple[str, ...] = ()
