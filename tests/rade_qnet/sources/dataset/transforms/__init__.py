"""
Tests for ``rade_qnet.sources.dataset.transforms`` -- fitted transforms.

Every transform is held to three properties: it is fitted only on the rows it
is permitted to see, its fitted state round-trips through save and load, and --
if it touches the target -- its inverse recovers the original values to
floating-point tolerance. The third matters because a metric is only meaningful
in original units, so a broken inverse corrupts every number a user reads.

Planned modules
---------------
``test_transforms_scaling.py``
    Fit statistics computed from training rows only; inverse recovers the
    input.  [Phase 2]
``test_transforms_sequence.py``
    Window construction, including the boundary cases at the start of a series
    and at a split edge.  [Phase 2]
``test_transforms_reduction.py``
    Basis selection under both ``fit_on`` settings, with an explicit test that
    the default never observes validation or test rows.  [Phase 2]
``test_transforms_encoding.py``
    Entity-axis encoding, including an unseen category at inference time.
    [Phase 2]
"""
