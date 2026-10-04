"""
Tests for ``rade_qnet.sources.batching`` -- the ``BatchSource`` adapters.

Every module here is run through one shared contract suite, because the whole
value of the abstraction is that a training loop cannot tell the sources apart.
If a source satisfies the protocol only loosely, the loop acquires a special
case and the abstraction has failed.

Planned modules
---------------
``test_batching_contract.py``
    The shared suite applied to every source: batch shapes are as declared,
    iteration is reproducible under a fixed seed, and exhaustion is signalled
    rather than silently looping.  [Phase 2, extended in Phase 7]
``test_batching_dataset.py``
    Iterating a ``DataBundle`` split, and that static inputs are presented once
    per batch rather than once per sample.  [Phase 2]
``test_batching_rollout.py``
    On-policy collection against a toy environment.  [Phase 7]
``test_batching_replay.py``
    Buffer capacity, eviction order and sampling distribution.  [Phase 7]
``test_batching_offline.py``
    Serving batches from a stored transition table.  [Phase 7]
``test_batching_simulation.py``
    Path batches from a differentiable environment, with the computation graph
    preserved.  [Phase 7]
"""
