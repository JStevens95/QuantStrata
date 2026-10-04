"""
Update rules -- what one optimisation step means.

A learner receives a batch and the current parameters and returns the losses
and metrics for that step.  It knows nothing about epochs, checkpoints,
schedules, logging or devices; the loop owns all of those.  This is what allows
a new algorithm to be added as one focused module.

Two protocols, not one
----------------------
A supervised learner implements ``Learner`` -- ``(model, inputs, target)`` --
and an interactive one implements ``PolicyLearner`` -- ``act`` and ``update``
over whole transitions.  An earlier version of this charter claimed the split
between loop and learner "allows Phase 7 to add four reinforcement-learning
learners against an unchanged loop".  Half of that held and half did not: the
split itself did its job, and the two drivers in ``loops.py`` are the same
shape, but a policy has no target and so could not be served by the
supervised signature.  The reasoning is recorded in full in that module.

Modules
-------
``supervised.py``
    Forward pass, loss against a target, backward pass.  ``Learner``.
    [Phase 2]
``random.py``
    Samples an action from an untrained policy and updates nothing.
    ``PolicyLearner``.  The control an algorithm is compared against, and the
    reader that keeps the interactive contracts honest before any algorithm
    exists.  [Phase 7]

Planned modules
---------------
``dqn.py``
    Temporal-difference update with a target network.  [Phase 7]
``ppo.py``
    Clipped surrogate objective with generalised advantage estimation.
    [Phase 7]
``sac.py``
    Soft actor-critic with entropy regularisation.  [Phase 7]
``pathwise.py``
    Back-propagates a risk measure directly through a differentiable
    simulation.  No value function and no policy-gradient estimator -- the
    gradient is exact.  [Phase 7]
"""

__all__: tuple[str, ...] = ()
