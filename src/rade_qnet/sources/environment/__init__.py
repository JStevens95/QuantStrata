"""
Interactive data: environments an agent learns by acting in.

An environment is the interactive counterpart of a dataset. A dataset is
asked for rows; an environment is acted in, and what it yields next depends
on what was done. That difference is the whole of what reinforcement
learning adds to this framework, and
:mod:`rade_qnet.sources.batching.rollout` is where it is flattened back into
the one stream shape a training loop consumes.

Modules
-------
``protocol.py``
    ``Environment`` -- the step-and-reward interface, with ``StepOutcome``.
    [Phase 7]

Two kinds of environment, one of them not yet here
--------------------------------------------------
``Environment``
    The familiar step-and-reward interface. Gradients, if any, come from the
    learner's own estimator. Delivered.

``DifferentiableEnvironment``
    The dynamics are themselves differentiable, so a loss can be
    back-propagated straight through a simulated path. Hedging and
    replication problems are naturally of this form, and forcing them into a
    transition-based interface discards the exact gradients that make them
    tractable. Most reinforcement-learning libraries model only the first
    kind; supporting the second directly is a deliberate choice for a
    quantitative-finance framework, and it remains the plan.

    It is **not** declared yet, on purpose. A ``runtime_checkable`` protocol
    needs a distinguishing member or it is satisfied by every environment,
    and the member it needs is whatever the pathwise learner reads -- which
    does not exist to be asked. Declaring it early is how
    ``EngineCapabilities`` spent four phases asserting a contract nothing
    honoured. It arrives with the learner that gives it meaning, in one
    change, together with ``simulation.py``.

Planned modules
---------------
``spaces.py``
    Helpers for building observation and action spaces. The descriptions
    themselves are already in ``core.contract.signature`` -- ``SpaceSpec``
    and ``PolicySignature`` -- because the run spec references them and
    ``core`` may not depend on ``sources``.
``vector.py``
    Batched environments stepped in lockstep, synchronously or in worker
    processes.
``wrappers.py``
    Composable observation, action and reward wrappers. Each records itself
    in the run lineage, so a reward-shaping change is visible in the bundle.
``recording.py``
    Episode capture to disk, producing the transition tables that an offline
    source reads back.
``gymnasium.py``
    A bridge from the Gymnasium interface, mapping its spaces onto framework
    spaces and normalising its step-return convention.

    This was an ``adapters/`` sub-package holding nothing but a charter, and
    it was deleted for the same reason ``DifferentiableEnvironment`` is not
    yet declared: a package with no modules is a promise with no reader, and
    it costs a directory level on every import for a file that does not
    exist.  One bridge is one module.  A second one can make it a package.
"""

from .protocol import Environment, StepOutcome

__all__ = ["Environment", "StepOutcome"]
