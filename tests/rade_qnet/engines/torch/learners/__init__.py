"""
Tests for ``rade_qnet.engines.torch.learners`` -- update rules.

A learner is tested in isolation, one update at a time, against a problem whose
correct gradient is known. Learning-rate schedules, checkpoints and logging are
the loop's business and are absent here, which is precisely why a learner bug
is findable.

Planned modules
---------------
``test_learners_supervised.py``
    One update reduces the loss on a convex problem; the gradient matches a
    finite-difference estimate.  [Phase 2]
``test_learners_dqn.py``
    Temporal-difference target construction and target-network
    synchronisation.  [Phase 7]
``test_learners_ppo.py``
    Advantage estimation and the clipping behaviour at the ratio bounds.
    [Phase 7]
``test_learners_sac.py``
    Entropy term and twin-critic updates.  [Phase 7]
``test_learners_pathwise.py``
    The gradient through a differentiable simulation matches the analytic
    gradient on a problem with a closed form.  [Phase 7]
"""
