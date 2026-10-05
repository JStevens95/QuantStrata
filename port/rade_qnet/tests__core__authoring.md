# `tests/rade_qnet/core/authoring`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 21 | 990 | `5fa0437358e278ef` |
| 2 | `test_authoring_capabilities.py` | 209 | 6428 | `0a7e4938fcbfb1b8` |
| 3 | `test_authoring_definition.py` | 267 | 9802 | `deb2cc813fc5215f` |
| 4 | `test_authoring_policy.py` | 200 | 6257 | `fe80eb7d05705468` |
| 5 | `test_authoring_supervised.py` | 292 | 10715 | `996747fa07aa4b1c` |

---

## 1. `tests/rade_qnet/core/authoring/__init__.py`

990 bytes · SHA-256 `5fa0437358e278ef`

```python
"""
Tests for ``rade_qnet.core.authoring`` -- the model-facing interface.

Two properties matter most here, and both are about keeping the framework
welcoming. A model that implements only the required minimum must work, which
is what keeps a simple model to one file. And a model that implements no
capabilities must be unaffected by every authoring that exists, which is what
lets new capabilities be added without a migration.

Planned modules
---------------
``test_authoring_definition.py``
    The ``ModelDefinition`` bases and the ``@model`` decorator, including the
    error raised when two models claim the same name.  [Phase 1]
``test_authoring_protocols.py``
    Runtime detection of each opt-in authoring, verified both ways: present
    and detected, absent and ignored.  [Phase 1]
``test_authoring_supervised.py``
    ``SupervisedModel``, asserting that a minimal model needs one line of data
    code, and that every fault it detects names the model package.  [Phase 2]
"""
```

---

## 2. `tests/rade_qnet/core/authoring/test_authoring_capabilities.py`

6428 bytes · SHA-256 `0a7e4938fcbfb1b8`

```python
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

from src.rade_qnet.core.authoring.capabilities import (
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
```

---

## 3. `tests/rade_qnet/core/authoring/test_authoring_definition.py`

9802 bytes · SHA-256 `deb2cc813fc5215f`

```python
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
```

---

## 4. `tests/rade_qnet/core/authoring/test_authoring_policy.py`

6257 bytes · SHA-256 `fe80eb7d05705468`

```python
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
```

---

## 5. `tests/rade_qnet/core/authoring/test_authoring_supervised.py`

10715 bytes · SHA-256 `996747fa07aa4b1c`

```python
"""
Tests for ``SupervisedModel``, the base class for supervised models.

``SupervisedModel`` exists so that a straightforward model is a short file. It
supplies ``build_data`` and ``signature``, leaving a subclass to write
``data_module`` -- one line -- and ``build_model``. Without it, every
supervised model would reimplement the same twenty lines of source-to-payload wrapping,
and each copy would be a separate opportunity to get it wrong.

The design point worth recording is that ``data_module`` is *abstract* rather
than defaulted to the built-in tabular module. It cannot be defaulted:
``core`` has an empty dependency set, so it cannot import ``sources``, and a
default would invert the dependency direction that the whole architecture
rests on. The cost is one line in every subclass; the benefit is that ``core``
stays importable without any of the layers above it.

The rest of the tests are about the errors. Every fault this base can detect
is in the *model package* -- a data module that is not one, a prepared dataset
missing a field, a source that is not a source -- so each is reported as a
component error naming the offending type. The alternative is an
``AttributeError`` several stages later, where nothing says which of the
user's methods returned the wrong thing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np
import pytest

from src.rade_qnet.core.authoring.supervised import DataModuleLike, SupervisedModel
from src.rade_qnet.core.lifecycle.errors import ComponentError
from src.rade_qnet.core.spec.data import TabularSourceSpec
from src.rade_qnet.core.spec.run import SupervisedRunSpec
from src.rade_qnet.sources.dataset.tabular import TabularDataModule


@pytest.fixture
def spec(tmp_path):
    """
    Build a supervised run spec over a small CSV.

    Returns
    -------
    SupervisedRunSpec
        The spec.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    target = features.sum(axis=1)

    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "c", "target"])
        writer.writerows(np.column_stack([features, target]).tolist())

    return SupervisedRunSpec.model_validate(
        {
            "model": "synthetic_tabular",
            "source": TabularSourceSpec(path=path),
            "reports": {"enabled": ()},
        }
    )


class CsvRegressor(SupervisedModel):
    """The whole of a supervised model's data wiring, as a subclass should write it."""

    def data_module(self, spec):
        """Return the built-in tabular module."""
        del spec
        return TabularDataModule()

    def build_model(self, spec, signature):
        """Return a stand-in model, since these tests never train."""
        del spec, signature
        return object()


class TestTheDependencyDirection:
    """Why ``data_module`` is abstract rather than defaulted."""

    def test_data_module_is_abstract(self):
        """
        Because ``core`` cannot import ``sources``.

        Defaulting it to the built-in tabular module would invert the
        dependency direction the architecture rests on -- ``core`` would then
        import a layer above it, and the one-way stack would no longer be one
        way. One line per subclass is the price.
        """
        assert "data_module" in SupervisedModel.__abstractmethods__

    def test_the_expected_interface_is_declared_structurally(self):
        """
        As a protocol, so the consumer declares what it needs.

        That is what lets a user's own data module satisfy it without
        importing anything from ``core``, and what lets ``core`` describe its
        requirement without naming the class that meets it.
        """
        assert isinstance(TabularDataModule(), DataModuleLike)

    def test_an_object_missing_a_method_does_not_satisfy_the_protocol(self):
        """
        So the check is worth making before the build runs.

        An object with ``build`` but no ``batch_sources`` would otherwise fail
        halfway through the build, after the expensive read.
        """

        class HalfModule:
            """A module that can load but not batch."""

            def build(self, spec, *, seed=0):
                """Return nothing useful."""
                del spec, seed
                return object()

        assert not isinstance(HalfModule(), DataModuleLike)


class TestBuildData:
    """The twenty lines a supervised model no longer has to write."""

    def test_one_payload_per_split(self, spec):
        """
        So a model gets the splits its configuration asked for.

        A fixed three-payload assumption would break on a run with no
        validation fraction, which is a legitimate configuration.
        """
        bundle = CsvRegressor().build_data(spec)
        assert set(bundle.splits) == {"train", "validation", "test"}

    def test_each_payload_reports_its_sample_count(self, spec):
        """
        Because it is the denominator of every averaged metric.

        A zero would make a reported mean loss infinite, and an inflated one
        would make it quietly too small.
        """
        bundle = CsvRegressor().build_data(spec)
        assert all(payload.n_samples > 0 for payload in bundle.splits.values())

    def test_the_source_is_stored_as_the_loader(self, spec):
        """
        One object rather than a payload plus a parallel source map.

        A ``DatasetSource`` satisfies both ``BatchSource`` and
        ``Iterable[Batch]``, so keeping one reference means the two cannot
        drift apart -- which they would, the first time a stage rebuilt one
        and not the other.
        """
        bundle = CsvRegressor().build_data(spec)
        loader = bundle.splits["train"].loader
        assert loader.n_samples == bundle.splits["train"].n_samples

    def test_the_signature_and_state_come_from_the_data_build(self, spec):
        """
        Not reconstructed, so the bundle describes what was actually built.

        A reconstructed signature agrees with the real one only while two
        pieces of code agree about the transforms.
        """
        bundle = CsvRegressor().build_data(spec)
        assert bundle.signature.dynamic
        assert bundle.state is not None

    def test_the_lineage_is_carried_through(self, spec):
        """
        Because it is the only record of which data this run saw.

        Lost here, a run could not be compared against another one, and
        comparison is how anyone finds out the data changed.
        """
        assert CsvRegressor().build_data(spec).lineage.source_fingerprint

    def test_the_signature_defaults_to_the_data_build_s(self, spec):
        """
        Which is right for a model that consumes everything on offer.

        A model using only some of its inputs should narrow it, so the saved
        signature describes what the model used rather than what was
        available -- but that is an override, not the default.
        """
        model = CsvRegressor()
        bundle = model.build_data(spec)
        assert model.signature(bundle) is bundle.signature


class TestComponentErrors:
    """Every detectable fault is in the model package, and is named as such."""

    def test_a_data_module_that_is_not_one_is_reported(self, spec):
        """
        Naming the type that was returned.

        The alternative is an ``AttributeError`` on ``build`` several lines
        later, which says nothing about which of the user's methods was
        wrong.
        """

        class NotAModule(CsvRegressor):
            """A model whose data module is not a data module."""

            def data_module(self, spec):
                """Return the wrong kind of thing."""
                del spec
                return object()

        with pytest.raises(ComponentError, match="data_module"):
            NotAModule().build_data(spec)

    def test_a_prepared_dataset_missing_a_field_is_reported(self, spec):
        """
        Naming the fields, because a custom data module is the likely cause.

        A user whose own module forgot to attach the lineage gets told which
        field is absent, rather than an attribute error from inside the
        bundle construction.
        """

        @dataclass
        class Incomplete:
            """A prepared dataset with no lineage."""

            signature: object = None
            state: object = None

        class BadModule:
            """A data module whose build returns the wrong shape."""

            def build(self, spec, *, seed=0):
                """Return an incomplete prepared dataset."""
                del spec, seed
                return Incomplete()

            def batch_sources(self, prepared, spec, *, seed=0):
                """Return nothing, since the build already failed."""
                del prepared, spec, seed
                return {}

        class WithBadModule(CsvRegressor):
            """A model wired to the broken module."""

            def data_module(self, spec):
                """Return the broken module."""
                del spec
                return BadModule()

        with pytest.raises(ComponentError, match="lineage"):
            WithBadModule().build_data(spec)

    def test_a_source_that_is_not_a_batch_source_is_reported(self, spec):
        """
        Naming the split and the type.

        A training loop could not consume it, and the failure without this
        check is inside the loop's first iteration -- where the message is
        about an object not being iterable.
        """

        class BadSourceModule:
            """A data module whose sources are not sources."""

            def __init__(self) -> None:
                """Hold a real module to delegate the build to."""
                self.inner = TabularDataModule()

            def build(self, spec, *, seed=0):
                """Delegate, so the prepared dataset is valid."""
                return self.inner.build(spec, seed=seed)

            def batch_sources(self, prepared, spec, *, seed=0):
                """Return something that is not a batch source."""
                del prepared, spec, seed
                return {"train": object()}

        class WithBadSources(CsvRegressor):
            """A model wired to the broken module."""

            def data_module(self, spec):
                """Return the broken module."""
                del spec
                return BadSourceModule()

        with pytest.raises(ComponentError, match="BatchSource"):
            WithBadSources().build_data(spec)
```

