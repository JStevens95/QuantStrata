"""
The learner that learns nothing, on purpose.

:class:`RandomLearner` satisfies
:class:`~rade_qnet.engines.torch.training.loops.PolicyLearner` completely and updates
no parameter. It exists so that every seam in the interactive path has a real
consumer before any algorithm is written: a policy is built from a signature,
an environment is constructed, actions are drawn from the policy, an episode
runs to its end, transitions are batched, the driver counts its budget, a
bundle is written, and the bundle re-loads. All of that is exercised here,
with nothing learned.

Why a no-op learner is worth shipping
-------------------------------------
It is the control. When PPO is added and the agent does not improve, the
question is whether the algorithm is wrong or the plumbing is -- and the only
way to answer it cheaply is to have a configuration that is *known* to do
nothing, whose episode return is therefore whatever an untrained policy
earns. A PPO run that matches it has not learned; a PPO run that cannot even
match it is broken somewhere this learner already proved works.

It is also what keeps the contracts honest in the meantime. A protocol with
no implementation is a guess about what an implementation will need.
``EngineCapabilities`` spent four phases as such a guess, and the fields it
guessed at turned out to be read by nothing. :class:`PolicyLearner` is not
in that position: it has a reader from the day it is declared.

Why it is random rather than fixed
----------------------------------
A fixed action would make a one-line environment look like it worked while
leaving the interesting paths untested: an episode that never varies rarely
terminates for an interesting reason, and the stacking, resetting and
bootstrapping flags are all exercised by *variety*. So the action is sampled
through the policy -- see :meth:`act` -- which also means the policy is
genuinely forward-passed on every step rather than merely constructed.

(The module name matches the registered learner name, as ``supervised.py``
does. It does not shadow the standard library's ``random``, because Python 3
resolves imports absolutely; this module does not use it in any case.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from ....core.contract.data import TARGET_KEY
from ....core.runtime.components import learner as register_learner
from ....core.runtime.errors import EngineError
from ....core.runtime.logging import get_logger
from ..training.loops import LOSS_KEY

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ....core.contract.signature import PolicySignature, SpaceSpec

__all__ = ["RandomLearner"]

_LOGGER = get_logger(__name__)

#: Reported alongside the flat loss so a block's record carries something that
#: actually moves. Without it the history is a column of zeros, and a reader
#: cannot tell a working no-op run from a run that never collected anything.
_MEAN_REWARD_KEY = "mean_reward"


@register_learner("random", engine="torch")
class RandomLearner:
    """
    Draws actions from an untrained policy and never updates it.

    Resolved by name through the learner registry, like every other learner,
    so that the registry has a second entry and the resolution path is
    exercised by something other than its own tests.

    Parameters
    ----------
    signature
        The policy signature, which supplies the action space. Needed because
        how a network's output becomes an action depends on the space: a
        discrete space wants one index, a box space wants a vector within its
        bounds.
    seed
        Seeds the action sampling, so a run with this learner is reproducible
        end to end. Kept in a dedicated generator rather than taken from the
        global Torch stream for every draw, because a job set runs several of
        these in one process and a shared stream would make their sampling
        interleave.

        ``None`` -- the default, and what the engine passes -- draws the seed
        *once* from the global stream, which the pipeline has already seeded.
        So the run's declared seed governs this learner without the engine
        having to forward it, and the per-draw independence is still kept.
    training, optimiser, hardware, gradient_norms
        Accepted and unused.

        The engine resolves a learner by name and therefore cannot know which
        arguments the algorithm it was handed wants, so the contract is that
        every policy learner is offered all of them and takes what it needs.
        A learner that performs no update needs none of these: there is no
        objective to configure, no parameter to step, no precision to
        autocast under and no gradient to measure. Accepting them is what
        lets this learner be resolved by exactly the same code path as a real
        one, which is the point -- a seam only tested through a special case
        is not tested.
    """

    def __init__(
        self,
        *,
        signature: PolicySignature,
        seed: int | None = None,
        training: object = None,
        optimiser: object = None,
        hardware: object = None,
        gradient_norms: object = None,
    ) -> None:
        del training, optimiser, hardware, gradient_norms
        self.signature = signature
        if seed is None:
            # Drawn from the global stream, which `seed_everything` has
            # already set from the run's spec. Taken as a single draw so the
            # generator below stays independent of anything else sampling
            # later in the process.
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        self.seed = seed
        self.generator = torch.Generator().manual_seed(seed)

    def act(self, policy: torch.nn.Module, observation: object) -> object:
        """
        Forward the observation through the policy and sample an action from it.

        The policy is really called, under ``no_grad``, for two reasons. It
        proves the policy was built and materialised -- a lazily initialised
        network that is never forward-passed has no parameters, which is the
        defect the materialise stage exists to catch. And it means the actions
        reflect an untrained network rather than a uniform draw, so the
        episode statistics this learner produces are the baseline an
        algorithm should be compared against.

        Sampling rather than taking the argmax, because an untrained
        network's argmax is very nearly constant, and a constant action
        rarely reaches a terminal state -- leaving exactly the episode-boundary
        handling that most needs exercising unexercised.

        Parameters
        ----------
        policy
            The prepared policy. Called with the observation as a keyword
            argument, so a network declares what it consumes in its own
            signature rather than depending on positional order -- the same
            convention as :class:`~..learners.supervised.SupervisedLearner`.
        observation
            One observation, unbatched, as the environment produced it.

        Returns
        -------
        object
            For a discrete space, an ``int64`` tensor holding one index. For
            a box space, a tensor shaped like the space and within its
            bounds.
        """
        batched = torch.as_tensor(observation).unsqueeze(0)
        with torch.no_grad():
            output = policy(observation=batched)
        return self._action_from(output.squeeze(0))

    def _action_from(self, output: torch.Tensor) -> torch.Tensor:
        """
        Turn one unbatched policy output into one action.

        Sampled and returned on the host, whatever device the policy is on,
        for two reasons that happen to agree. The environment runs on the
        host, so the action has to cross back anyway. And a dedicated
        generator is bound to one device -- an accelerator generator cannot
        draw from a host tensor and vice versa -- so keeping the draw on the
        CPU is what makes a seeded run produce the same actions on a laptop
        and on a GPU box. A run whose trajectory depended on the accelerator
        would be reproducible only on the machine that produced it.

        Parameters
        ----------
        output
            The policy's output for a single observation.

        Returns
        -------
        torch.Tensor
            An action the environment will accept, on the host.

        Raises
        ------
        EngineError
            If the action space names a kind this learner has not been taught,
            or if a discrete policy's output does not have one value per
            action -- refused rather than reshaped, because an output of the
            wrong width means the policy was built against a different space
            than the environment declares, and sampling from it would produce
            actions the environment rejects at an unrelated moment.
        """
        space: SpaceSpec = self.signature.action
        output = output.cpu()
        if space.kind == "discrete":
            if space.n is None or output.numel() != space.n:
                raise EngineError(
                    f"the policy produced {output.numel()} value(s) for a "
                    f"discrete action space of {space.n}; build_policy must "
                    f"size its output head from the signature's action space"
                )
            probabilities = torch.softmax(output.to(dtype=torch.float32), dim=-1)
            return torch.multinomial(probabilities, num_samples=1, generator=self.generator)[0]

        if space.kind == "box":
            action = output.to(dtype=torch.float32).reshape(space.shape)
            # Clamped rather than trusted: an untrained head emits whatever
            # its initialisation gives, which for a bounded space is usually
            # outside the bounds, and an out-of-range action is a failure in
            # the environment rather than here.
            return action.clamp(
                min=space.low if space.low is not None else float("-inf"),
                max=space.high if space.high is not None else float("inf"),
            )

        raise EngineError(
            f"action space kind {space.kind!r} is not supported by RandomLearner; "
            f"expected 'box' or 'discrete'"
        )

    def update(
        self,
        policy: torch.nn.Module,
        experience: Mapping[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        Report the batch's scalars and change nothing.

        No optimiser, no backward pass, no parameter touched. The reported
        loss is ``0.0`` by construction and is flat for the whole run, which
        is the honest answer: this learner has no objective. The mean reward
        is reported beside it so the history is not a column of zeros.

        Parameters
        ----------
        policy
            The prepared policy. Accepted and deliberately unused, because
            the protocol passes it and a signature that omitted it would not
            satisfy the protocol.
        experience
            One batch of transitions, on the device.

        Returns
        -------
        dict
            A flat ``loss`` and the batch's mean reward.

        Raises
        ------
        EngineError
            If the batch carries no reward, which means it is not experience.
        """
        del policy  # Named for the protocol; there is nothing to update.

        reward = experience.get(TARGET_KEY)
        if reward is None:
            raise EngineError(
                f"the experience batch has {sorted(experience)} but no "
                f"{TARGET_KEY!r}; a batch with no reward is not experience"
            )

        mean_reward = float(reward.to(dtype=torch.float32).mean().item())
        _LOGGER.debug("no-op update over %d transition(s)", reward.shape[0])
        return {LOSS_KEY: 0.0, _MEAN_REWARD_KEY: mean_reward}
