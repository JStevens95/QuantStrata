"""
Update rules -- what one optimisation step means.

A learner receives a batch and the current parameters and returns the losses
and metrics for that step.  It knows nothing about epochs, checkpoints,
schedules, logging or devices; the loop owns all of those.  This is what allows
a new algorithm to be added as one focused module.

Modules
-------
``supervised.py``
    Forward pass, loss against a target, backward pass.  [Phase 2]

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
