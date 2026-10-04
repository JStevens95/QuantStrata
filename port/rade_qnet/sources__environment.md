# `src/rade_qnet/sources/environment`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 62 | 2660 | `8f8efee07fa1f4fa` |
| 2 | `protocol.py` | 187 | 6817 | `c6a527da39da2b1a` |

---

## 1. `src/rade_qnet/sources/environment/__init__.py`

2660 bytes · SHA-256 `8f8efee07fa1f4fa`

```python
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
``adapters/``
    Bridges to third-party environment interfaces.
"""

from .protocol import Environment, StepOutcome

__all__ = ["Environment", "StepOutcome"]
```

---

## 2. `src/rade_qnet/sources/environment/protocol.py`

6817 bytes · SHA-256 `c6a527da39da2b1a`

```python
"""
What an environment is: reset it, step it, and ask what it accepts.

:class:`~rade_qnet.core.capability.definition.PolicyDefinition.build_environment`
has promised since Phase 1 that it returns "an environment satisfying one of
the environment protocols in ``rade_qnet.sources.environment``". Until this
module there were no such protocols -- the promise named something that did
not exist. This is that definition.

Why a protocol rather than a base class
---------------------------------------
The same reason as :class:`~rade_qnet.core.contract.source.BatchSource`: an
environment satisfies this by having the right members, so a hedging simulator
written in another repository needs no import of ours, and the framework's own
environments are checked against the definition a user's environment is.

Why the spaces are declared, not inferred
-----------------------------------------
:attr:`Environment.observation_space` and :attr:`Environment.action_space`
exist so a policy can be constructed *before* anything is stepped, and
rebuilt later from a saved bundle with no environment at all. Inferring the
spaces by resetting the environment and measuring the result would make
policy construction depend on a live environment, which is exactly what a
six-month-old bundle does not have.

Why the step result is one frozen object
----------------------------------------
Returning a tuple of five positional values is the convention elsewhere, and
it is the source of a specific, silent defect: the ``terminated`` and
``truncated`` flags are both booleans sitting next to each other, so
transposing them type-checks, runs, and produces an agent that treats a
time-limited episode as a genuine failure. Those are different events --
one means the task ended, the other means the clock ran out -- and conflating
them biases every value estimate that bootstraps past the boundary. Naming
them on a frozen object makes the mistake unwriteable.

What is deliberately not here yet
---------------------------------
``DifferentiableEnvironment``. The package charter and
``PHASE_7_REINFORCEMENT_LEARNING.md`` §3.2 both describe it, and it is a
genuinely important case for this framework -- when the dynamics are
differentiable, a risk measure can be back-propagated straight through a
simulated path, with no value function and no policy-gradient estimator.

It is absent because nothing would read it. A ``runtime_checkable`` protocol
with no distinguishing member is satisfied by *every* environment, so an
``isinstance`` check against it would answer ``True`` for a plainly
non-differentiable one; and a protocol with members invented ahead of the
learner that needs them is how ``EngineCapabilities`` came to spend four
phases declaring a contract nothing honoured. It arrives with the pathwise
learner, which is the thing that gives it meaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.contract.data import TensorLike
    from ...core.contract.signature import SpaceSpec

__all__ = ["Environment", "StepOutcome"]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """
    The result of one step: what was seen, what it was worth, and whether it ended.

    Parameters
    ----------
    observation
        What the policy sees next. For a step that ended the episode this is
        the terminal observation, which a bootstrapping learner needs and a
        reset would discard.
    reward
        The scalar reward for the step that was taken.
    terminated
        The episode reached a terminal state of the task itself -- the
        position was closed, the option expired, the agent failed.
    truncated
        The episode was cut short from outside the task: a step limit, a
        wall-clock budget. Distinct from :attr:`terminated` because a value
        estimate should bootstrap past a truncation and must not bootstrap
        past a termination. Collapsing the two is a silent bias, not an
        error.
    info
        Free-form diagnostics. Never read by the framework, so an environment
        may put anything here without affecting a run.
    """

    observation: TensorLike
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, object] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        """
        Whether the episode is over for any reason.

        Convenience for a collector that only needs to know when to reset.
        A *learner* should read the two flags separately, for the reason in
        :attr:`truncated`.

        Returns
        -------
        bool
            True if the episode terminated or was truncated.
        """
        return self.terminated or self.truncated


@runtime_checkable
class Environment(Protocol):
    """
    Step-and-reward dynamics an agent learns by acting in.

    The interactive counterpart of a dataset. Four members: the two spaces,
    which are declared up front so a policy can be built without stepping
    anything, and the two methods that drive an episode.
    """

    @property
    def observation_space(self) -> SpaceSpec:
        """
        What the policy sees.

        Returns
        -------
        SpaceSpec
            The observation space.
        """
        ...

    @property
    def action_space(self) -> SpaceSpec:
        """
        What the policy must produce.

        Returns
        -------
        SpaceSpec
            The action space.
        """
        ...

    def reset(self, *, seed: int | None = None) -> TensorLike:
        """
        Start a new episode and return its first observation.

        Parameters
        ----------
        seed
            Seeds the environment's own randomness. Passed explicitly rather
            than read from a global, because a job set runs forty of these in
            separate processes and a shared global would make two members
            either identical or unreproducible. ``None`` leaves the existing
            stream alone, which is what a mid-run reset wants: re-seeding on
            every episode would make every episode the same one.

        Returns
        -------
        TensorLike
            The first observation, matching :attr:`observation_space`.
        """
        ...

    def step(self, action: TensorLike) -> StepOutcome:
        """
        Apply one action and return what happened.

        Parameters
        ----------
        action
            An action drawn from :attr:`action_space`.

        Returns
        -------
        StepOutcome
            The next observation, the reward, and whether the episode ended.
        """
        ...
```

