"""
Tests for the opt-in capability protocols.

Three properties are being protected here, and each one is a design claim that
would be easy to lose to a careless edit.

A model never pays for a capability it does not use -- so a plain object must
satisfy none of these. Adding a capability cannot break an existing model --
so detection must be structural, with no base class to inherit. And the
capability is visible in the type rather than in a flag -- so the framework
asks ``isinstance`` rather than reading ``model.has_static_inputs``, which
cannot then disagree with what the model implements.

The known limit is also tested: ``isinstance`` verifies method *presence*, not
signatures. That is why the conformance suite exists, and a test asserting the
limit keeps anyone from mistaking the check for more than it is.
"""

from __future__ import annotations

import pytest

from src.rade_xl.core.capability.protocols import (
    CustomStep,
    Inductive,
    Precomputable,
    Routable,
    StaticInputs,
)

ALL_PROTOCOLS = (StaticInputs, Precomputable, CustomStep, Routable, Inductive)


class PlainModel:
    """A model that declares no capabilities at all."""

    def forward(self, batch):
        """
        Return the batch unchanged.

        Parameters
        ----------
        batch
            One batch.

        Returns
        -------
        object
            The batch.
        """
        return batch


class FullyCapable:
    """A model implementing every capability, written against no base class."""

    def static_inputs(self):
        """Return one static input."""
        return {"adjacency": object()}

    def precompute(self, static):
        """Return a reusable encoding."""
        return {"node_embeddings": static}

    def forward_with_precomputed(self, batch, precomputed):
        """Return predictions using the encoding."""
        return (batch, precomputed)

    def training_step(self, batch, static):
        """Return a scalar loss."""
        return (batch, static)

    def covered_targets(self):
        """Return the targets trained on."""
        return ("EURUSD",)

    def supports_unseen_entities(self):
        """Return whether unseen entities are supported."""
        return True



class TestOptingOut:
    """A model that declares nothing is charged for nothing."""

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_a_plain_model_satisfies_no_capability(self, protocol):
        """
        No base-class method to stub out and no ``None`` to return.

        A simple regressor should be a simple class, not a class obliged to
        say "not applicable" five times.
        """
        assert not isinstance(PlainModel(), protocol)

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_a_bare_object_satisfies_no_capability(self, protocol):
        """
        A protocol that matched everything would route everything.

        The framework branches on these, so a false positive sends a model
        down a path it cannot handle.
        """
        assert not isinstance(object(), protocol)


class TestOptingIn:
    """Detection is structural: no import, no inheritance."""

    @pytest.mark.parametrize("protocol", ALL_PROTOCOLS)
    def test_a_capable_model_is_detected(self, protocol):
        """
        ``FullyCapable`` inherits from nothing and imports nothing.

        Which is the property that makes adding a sixth capability a non-event
        for every model that already exists.
        """
        assert isinstance(FullyCapable(), protocol)

    def test_capabilities_are_detected_independently(self):
        """
        Implementing one must not imply another.

        A model with a custom loss and no static inputs is an ordinary case,
        and coupling the two would force it to invent a static input.
        """

        class OnlyCustomStep:
            """A model with a custom loss and nothing else."""

            def training_step(self, batch, static):
                """Return a scalar loss."""
                return 0.0

        model = OnlyCustomStep()
        assert isinstance(model, CustomStep)
        assert not isinstance(model, StaticInputs)

    def test_precomputable_requires_both_of_its_methods(self):
        """
        One without the other is unusable.

        An encoding nothing consumes is wasted work; a consumer with nothing
        to consume cannot run.
        """

        class HalfPrecomputable:
            """Computes an encoding but offers no way to use it."""

            def precompute(self, static):
                """Return an encoding."""
                return {}

        assert not isinstance(HalfPrecomputable(), Precomputable)

    def test_an_inherited_method_counts(self):
        """
        A capability may come from a mixin or a shared base.

        Refusing inherited methods would push authors into copy-paste.
        """

        class Base:
            """Supplies the capability."""

            def supports_unseen_entities(self):
                """Return True."""
                return True

        class Derived(Base):
            """Inherits it."""

        assert isinstance(Derived(), Inductive)


class TestTheKnownLimit:
    """``isinstance`` routes; the conformance suite verifies."""

    def test_a_wrong_signature_still_passes_the_isinstance_check(self):
        """
        The documented limit, asserted so nobody mistakes the check for more.

        This model would be routed down the static-input path and then fail
        when called. Catching that is the conformance suite's job, and the
        division of labour only holds if this limit is understood.
        """

        class WrongShape:
            """Has the name but not the signature."""

            def static_inputs(self, unexpected_argument):
                """Take an argument the framework will not supply."""
                return {}

        assert isinstance(WrongShape(), StaticInputs)

    def test_a_non_callable_attribute_also_passes(self):
        """
        Protocol checking looks for the attribute, not for a method.

        Another reason the conformance suite calls what it finds rather than
        trusting that it is callable.
        """

        class NotAMethod:
            """Declares the name as data."""

            covered_targets = ("EURUSD",)

        assert isinstance(NotAMethod(), Routable)
