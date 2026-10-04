"""
Tests for the reinforcement-learning capability base.

:class:`PolicyModel` supplies one of the three methods
:class:`PolicyDefinition` demands, so there are only two things to test and
one of them is a refusal.

What the base does: it reads the spaces off the environment rather than
measuring them. The test that matters is therefore not that the signature is
correct -- it is that the environment was never *stepped* to find out. A
base that reset the environment to measure its observation would work fine
in every test and fail the one thing the signature exists for: rebuilding a
policy six months later from a bundle, with no environment to reset.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.authoring.definition import PolicyDefinition
from src.rade_qnet.core.authoring.policy import EnvironmentLike, PolicyModel
from src.rade_qnet.core.contract.signature import PolicySignature, SpaceSpec
from src.rade_qnet.core.lifecycle.errors import ComponentError

OBSERVATION = SpaceSpec(kind="box", shape=(4,), dtype="float32", low=-1.0, high=1.0)
ACTION = SpaceSpec(kind="discrete", n=2)


class Declared:
    """
    An environment that declares its spaces and refuses to be driven.

    ``reset`` and ``step`` raise, which is how these tests prove the base
    never touches them.
    """

    observation_space = OBSERVATION
    action_space = ACTION

    def reset(self, *, seed=None):
        """
        Fail loudly.

        Parameters
        ----------
        seed
            Unused.

        Raises
        ------
        AssertionError
            Always. Declaring a signature must not drive the environment.
        """
        del seed
        raise AssertionError("the signature must be read, not measured")

    def step(self, action):
        """
        Fail loudly.

        Parameters
        ----------
        action
            Unused.

        Raises
        ------
        AssertionError
            Always.
        """
        del action
        raise AssertionError("the signature must be read, not measured")


class Agent(PolicyModel):
    """The two-method model a user writes."""

    def build_environment(self, spec):
        """
        Construct the environment.

        Parameters
        ----------
        spec
            Unused by this stub.

        Returns
        -------
        Declared
            An environment that declares its spaces.
        """
        del spec
        return Declared()

    def build_policy(self, spec, signature):
        """
        Construct an untrained policy.

        Parameters
        ----------
        spec
            Unused by this stub.
        signature
            Recorded, so a test can confirm what the policy was built from.

        Returns
        -------
        PolicySignature
            The signature itself, standing in for a network.
        """
        del spec
        return signature


class TestWhatTheBaseSupplies:
    """One of three methods, which is the point of the base."""

    def test_a_subclass_needs_only_two_methods(self):
        """
        ``signature`` is inherited, so the model is concrete without it.

        The same bargain ``SupervisedModel`` strikes. A base that saved
        nothing would not be worth having.
        """
        assert isinstance(Agent(), PolicyDefinition)

    def test_the_signature_is_the_environment_s_two_spaces(self):
        """Read straight off the environment, unmodified."""
        agent = Agent()
        signature = agent.signature(agent.build_environment(spec=None))

        assert signature == PolicySignature(observation=OBSERVATION, action=ACTION)

    def test_the_environment_is_never_driven_to_find_out(self):
        """
        The behaviour the bundle depends on.

        ``Declared.reset`` and ``Declared.step`` raise, so this passes only
        if the base reads the declared spaces. A base that measured them
        would fail here -- and, more importantly, would fail to rebuild a
        policy from a bundle, where there is no environment at all.
        """
        agent = Agent()
        # Would raise AssertionError from the stub if either were called.
        agent.signature(agent.build_environment(spec=None))

    def test_the_policy_is_built_from_the_signature_alone(self):
        """
        Which is what makes a saved policy rebuildable.

        ``build_policy`` is required to be a pure function of the spec and
        the signature; this records that the signature it receives is the one
        the environment declared.
        """
        agent = Agent()
        signature = agent.signature(agent.build_environment(spec=None))
        assert agent.build_policy(None, signature) is signature


class TestWhatIsRefused:
    """An environment that does not declare its spaces."""

    def test_an_environment_without_spaces_is_refused_by_name(self):
        """
        Reported as a component error, naming the method at fault.

        The fault is in the model package's ``build_environment``, not in the
        framework, so the message points there rather than at the pipeline
        stage that happened to be running.
        """

        class Undeclared(PolicyModel):
            def build_environment(self, spec):
                """Return something that is not an environment."""
                del spec
                return object()

            def build_policy(self, spec, signature):
                """Unreachable in this test."""
                del spec, signature

        model = Undeclared()
        with pytest.raises(ComponentError, match="does not declare both"):
            model.signature(model.build_environment(spec=None))

    def test_the_narrow_protocol_asks_for_only_the_two_spaces(self):
        """
        It names neither ``reset`` nor ``step``.

        Deliberate: this base describes an environment, it never drives one.
        A protocol that asked for all four would make a two-property stub --
        including the one above -- stop satisfying it, for members the base
        does not use.
        """

        class SpacesOnly:
            observation_space = OBSERVATION
            action_space = ACTION

        assert isinstance(SpacesOnly(), EnvironmentLike)
