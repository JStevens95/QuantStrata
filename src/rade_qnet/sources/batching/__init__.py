"""
Adapters that present any source as a uniform stream of batches.

Each module here implements ``BatchSource``, the one protocol a training loop
consumes.  The loop therefore never branches on whether it is doing supervised
learning, on-policy reinforcement learning, off-policy reinforcement learning
or pathwise optimisation -- it asks for batches and applies the learner's
update rule.

Batches are NumPy, not tensors
------------------------------
``sources`` may import ``core`` and nothing else, so no module here can build
an engine's native tensor.  That is deliberate: batch ordering, window
arithmetic and split membership are identical for every engine, so they are
written once against NumPy and every engine inherits the same implementation.
Conversion to the native type belongs to the engine's loader.

Modules
-------
``dataset.py``
    ``DatasetSource`` -- iterates one split of a prepared dataset.  Supervised
    learning.  [Phase 2]
``rollout.py``
    ``RolloutSource`` -- collects a fresh batch of on-policy experience from an
    environment before each update.  Unbounded, which is what selects the
    ``fit_steps`` driver.  [Phase 7]

Planned modules
---------------
``replay.py``
    ``ReplaySource`` -- maintains a buffer and samples from it, uniformly or by
    priority.  Off-policy value-based methods.  [Phase 7]
``offline.py``
    ``OfflineSource`` -- serves batches from a stored transition table, with no
    live environment.  Offline reinforcement learning.  [Phase 7]
``simulation.py``
    ``SimulationSource`` -- draws batches of simulated paths from a
    differentiable environment, keeping the computation graph intact so the loss
    can be back-propagated through the dynamics.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
