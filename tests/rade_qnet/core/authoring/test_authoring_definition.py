"""
Tests for the model definition bases.

The claim under test is that the required surface is small: three methods for
a predictor, three for a policy, and nothing else. A linear model should be
forty lines, and the flagship multi-cluster graph model should be a package,
with the framework treating them identically.

The other claim is structural. ``build_model`` must be a pure function of the
spec and the signature, receiving no data -- that is exactly what guarantees a
model can be rebuilt from a saved bundle six months later without re-running
the data build. A signature that accepted a bundle would quietly destroy that
guarantee, so the signature itself is asserted here.
"""

from __future__ import annotations

import inspect

import pytest

from src.rade_qnet.core.authoring.definition import (
    ModelDefinition,
    PolicyDefinition,
    PredictorDefinition,
    model,
)
from src.rade_qnet.core.contract.signature import InputSignature, PolicySignature, SpaceSpec
from src.rade_qnet.core.contract.state import FittedState
from src.rade_qnet.core.lifecycle.components import MODELS
from src.rade_qnet.testkit.fixtures import (
    StandardisingState,
    isolated_registries,
    make_signature,
    make_tensor_bundle,
)


class MinimalPredictor(PredictorDefinition):
    """The smallest complete predictor definition."""

    def build_data(self, spec):
        """Return a synthetic bundle."""
        return make_tensor_bundle()

    def signature(self, bundle):
        """Pass the bundle's signature through unchanged."""
        return bundle.signature

    def build_model(self, spec, signature):
        """Return a stand-in model object."""
        return object()


class MinimalPolicy(PolicyDefinition):
    """The smallest complete policy definition."""

    def build_environment(self, spec):
        """Return a stand-in environment."""
        return object()

    def signature(self, environment):
        """Declare trivial spaces."""
        return PolicySignature(
            observation=SpaceSpec(kind="box", shape=(4,), low=-1.0, high=1.0),
            action=SpaceSpec(kind="discrete", n=2, dtype="int64"),
        )

    def build_policy(self, spec, signature):
        """Return a stand-in policy object."""
        return object()


class TestRequiredSurface:
    """Three methods, and the framework asks for nothing more."""

    @pytest.mark.parametrize("method", ["build_data", "signature", "build_model"])
    def test_each_predictor_method_is_required(self, method):
        """
        All three, because the pipeline calls all three in order.

        A definition missing any of them cannot complete a run, so the
        failure belongs at class-definition time rather than mid-run.
        """
        assert getattr(PredictorDefinition, method).__isabstractmethod__

    @pytest.mark.parametrize("method", ["build_environment", "signature", "build_policy"])
    def test_each_policy_method_is_required(self, method):
        """The structural counterpart, with the same reasoning."""
        assert getattr(PolicyDefinition, method).__isabstractmethod__

    def test_an_incomplete_predictor_cannot_be_instantiated(self):
        """The requirement, demonstrated rather than asserted from a flag."""

        class Incomplete(PredictorDefinition):
            """Builds data and nothing else."""

            def build_data(self, spec):
                """Return nothing useful."""
                return

        with pytest.raises(TypeError, match="build_model"):
            Incomplete()

    def test_a_complete_predictor_needs_nothing_else(self):
        """
        Forty lines is genuinely enough.

        If anything else were required, this class would not construct, and
        the claim that simple models stay simple would be false.
        """
        assert MinimalPredictor() is not None

    def test_a_complete_policy_needs_nothing_else(self):
        """The same, for the interactive case."""
        assert MinimalPolicy() is not None

    def test_the_shared_base_is_neither_kind_of_definition(self):
        """
        It holds only what both styles share, and trains nothing itself.

        Python permits instantiating it, since it declares no abstract
        methods of its own -- but it supplies neither a data build nor an
        environment build, so nothing can run it. A pipeline resolving a
        model checks for the concrete base rather than for this one.
        """
        assert not isinstance(ModelDefinition(), PredictorDefinition | PolicyDefinition)


class TestRebuildability:
    """The signatures that make a saved bundle loadable.

    These are assertions about method shape rather than behaviour, which is
    unusual -- but the guarantee lives in the shape. A ``build_model`` that
    could see data would be free to depend on it, and nothing downstream
    could detect that until a reload failed.
    """

    def test_building_a_model_sees_only_the_spec_and_the_signature(self):
        """
        No data parameter, so it cannot depend on data.

        Which is precisely what lets a bundle be reopened and its model
        reconstructed without the original dataset.
        """
        parameters = set(inspect.signature(PredictorDefinition.build_model).parameters)
        assert parameters == {"self", "spec", "signature"}

    def test_building_a_policy_sees_only_the_spec_and_the_signature(self):
        """
        So a saved policy is rebuildable without an environment.

        Reconstructing a trading environment to load a policy would make
        inference depend on market data access it does not need.
        """
        parameters = set(inspect.signature(PolicyDefinition.build_policy).parameters)
        assert parameters == {"self", "spec", "signature"}

    def test_a_predictor_signature_is_derived_from_the_bundle(self):
        """
        Separate from the bundle's own signature so a model may narrow it.

        A data build may offer more inputs than a given model consumes, and
        the bundle should record what was used rather than what was offered.
        """
        bundle = make_tensor_bundle()
        assert isinstance(MinimalPredictor().signature(bundle), InputSignature)

    def test_a_model_may_narrow_the_declared_interface(self):
        """
        Passing the bundle's signature through is normal, not mandatory.

        Narrowing has to work, or the separation above would be decorative.
        """

        class Narrowing(MinimalPredictor):
            """Consumes only the features, ignoring anything else offered."""

            def signature(self, bundle):
                """Declare a narrower interface than the bundle offers."""
                return make_signature(n_features=2)

        narrowed = Narrowing().signature(make_tensor_bundle())
        assert narrowed.dynamic["features"].shape == (None, 2)


class TestFittedState:
    """The optional hook, and why it is optional."""

    def test_a_model_fits_nothing_by_default(self):
        """
        Scalers and encoders belong to the data build.

        Requiring every model to say so would be five words of boilerplate
        in every definition in existence.
        """
        assert MinimalPredictor().fitted_state() is None

    def test_a_model_may_declare_state_it_fitted_itself(self):
        """
        For the minority that fit something the data build could not know.

        A learned graph or a selected feature subset is discovered during
        training, so it has nowhere else to live.
        """

        class WithState(MinimalPredictor):
            """Fits a transform during training."""

            def fitted_state(self):
                """Return the state fitted during training."""
                return StandardisingState(mean=1.0, scale=2.0)

        assert isinstance(WithState().fitted_state(), FittedState)

    def test_the_hook_is_not_abstract(self):
        """Optional means optional, including for a direct subclass."""
        assert not getattr(ModelDefinition.fitted_state, "__isabstractmethod__", False)


class TestRegistration:
    """A definition is reached by the name that appears in a YAML spec."""

    def test_the_decorator_records_the_name_and_engine(self):
        """
        Both, because the engine is declared by the model, not chosen by the run.

        A graph network cannot be trained by the XGBoost engine, and letting
        a spec ask for it would only defer the error to a worse place.
        """
        with isolated_registries():

            @model("demo", engine="sklearn")
            class Decorated(MinimalPredictor):
                """A registered definition."""

            assert Decorated.component_name == "demo"
            assert Decorated.component_engine == "sklearn"

    def test_the_decorator_returns_the_class(self):
        """
        So registration composes with other decorators and with inheritance.

        Returning a wrapper would break ``super()`` in any subclass.
        """
        with isolated_registries():

            @model("demo", engine="sklearn")
            class Decorated(MinimalPredictor):
                """A registered definition."""

            assert issubclass(Decorated, PredictorDefinition)

    def test_registration_does_not_leak_between_tests(self):
        """
        The isolation helper exists because registries are process-global.

        Without it, one test's registration changes what another test sees,
        and the failure appears in whichever test happens to run second.
        """
        with isolated_registries():

            @model("demo", engine="sklearn")
            class Decorated(MinimalPredictor):
                """A registered definition."""

        assert "demo" not in MODELS.names()
