"""
The base class for reinforcement-learning models: learn by acting.

The third of the paradigm bases promised by :mod:`rade_qnet.core.authoring`,
and the one ``supervised.py`` says "arrives with Phase 7".

Where :class:`~.supervised.SupervisedModel` learns a mapping from inputs to
known targets, a :class:`PolicyModel` has no targets at all. It is given an
environment, it acts, and the only feedback is a scalar reward for what it
did. That is a different *paradigm*, not a different data shape, which is why
it is a separate base rather than a flag on the supervised one.

What this base saves a model author
-----------------------------------
:class:`~.definition.PolicyDefinition` demands three methods. This class
supplies :meth:`signature` -- read the two spaces off the environment -- so a
model author writes two:

.. code-block:: python

    @model("my_agent")
    class MyAgent(PolicyModel):
        def build_environment(self, spec):
            return MyEnvironment(**spec.environment.params)

        def build_policy(self, spec, signature):
            return MyNetwork(signature)

That is the same bargain :class:`~.supervised.SupervisedModel` strikes, and
it matters for the same reason: a baseline agent that needed fifty lines of
plumbing would not be a useful control.

Why the signature is read, not measured
---------------------------------------
:meth:`signature` asks the environment what its spaces *are*, rather than
resetting it and measuring the observation that comes back. The difference
shows up six months later: a saved bundle has a ``PolicySignature`` and no
environment, and :meth:`~.definition.PolicyDefinition.build_policy` is
required to be a pure function of the spec and that signature so the policy
can be rebuilt regardless. Measuring would also quietly get the action space
wrong, since nothing an environment *returns* describes what it accepts.

An environment whose spaces genuinely are not known until it is constructed
is still fine -- it is constructed before :meth:`signature` is called. What
is not fine is an environment that has to be *stepped* first, and such an
environment should compute its spaces in ``__init__``.

Why ``EnvironmentLike`` is declared here as well as in ``sources``
------------------------------------------------------------------
For the same reason :class:`~.supervised.DataModuleLike` is: ``core`` may
import nothing from its sibling layers, so it cannot name
:class:`~rade_qnet.sources.environment.protocol.Environment`. This protocol
names the two members this base actually reads, and nothing else -- a real
environment satisfies it, and so does a two-property stub in a test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..contract.signature import PolicySignature
from ..lifecycle.errors import ComponentError
from ..provenance.logging import get_logger
from .definition import PolicyDefinition

if TYPE_CHECKING:
    from ..contract.signature import SpaceSpec

__all__ = ["EnvironmentLike", "PolicyModel"]

_LOGGER = get_logger(__name__)


@runtime_checkable
class EnvironmentLike(Protocol):
    """
    The two properties :class:`PolicyModel` needs from an environment.

    Narrow on purpose; see the module docstring. Note that it names neither
    ``reset`` nor ``step``: this base never drives the environment, it only
    describes it. The engine is what needs the full
    :class:`~rade_qnet.sources.environment.protocol.Environment`, and the
    engine is in a layer allowed to say so.
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


class PolicyModel(PolicyDefinition):
    """
    A policy that learns from interaction rather than from labelled data.

    Supplies :meth:`signature`, so a subclass implements
    :meth:`~.definition.PolicyDefinition.build_environment` and
    :meth:`~.definition.PolicyDefinition.build_policy`.

    A model whose spaces cannot be read off its environment -- a multi-agent
    setup, say, where the spaces are per-agent -- subclasses
    :class:`~.definition.PolicyDefinition` directly and writes its own
    ``signature``.
    """

    def signature(self, environment: object) -> PolicySignature:
        """
        Read the observation and action spaces off the environment.

        Parameters
        ----------
        environment
            The environment from
            :meth:`~.definition.PolicyDefinition.build_environment`.

        Returns
        -------
        PolicySignature
            The two declared spaces, saved into the bundle so the policy can
            be rebuilt without reconstructing the environment.

        Raises
        ------
        ComponentError
            If the environment does not declare both spaces. Reported as a
            component error, not a contract error, because the fault is in
            the model package's ``build_environment`` rather than in the
            framework or the data.
        """
        if not isinstance(environment, EnvironmentLike):
            raise ComponentError(
                f"{type(self).__name__}.build_environment() returned "
                f"{type(environment).__name__}, which does not declare both "
                f"observation_space and action_space; PolicyModel reads the "
                f"signature off the environment rather than measuring it"
            )

        signature = PolicySignature(
            observation=environment.observation_space,
            action=environment.action_space,
        )
        _LOGGER.info(
            "declared spaces for %s: observation=%s action=%s",
            type(self).__name__,
            signature.observation.kind,
            signature.action.kind,
        )
        return signature
