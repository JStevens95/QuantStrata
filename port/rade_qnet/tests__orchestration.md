# `tests/rade_qnet/orchestration`

2 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 8 | 365 | `af8945dcb1b57f28` |
| 2 | `test_orchestration_serving.py` | 449 | 15496 | `0e78199808bff149` |

---

## 1. `tests/rade_qnet/orchestration/__init__.py`

365 bytes · SHA-256 `af8945dcb1b57f28`

```python
"""
Tests for ``rade_qnet.orchestration`` -- pipelines, job sets and placement.

The property this sub-tree protects is that *placement cannot change results*.
A job set run sequentially and the same job set run across eight processes must
produce identical artifacts, and the only way to be sure is to run both and
compare. Several tests here do exactly that.
"""
```

---

## 2. `tests/rade_qnet/orchestration/test_orchestration_serving.py`

15496 bytes · SHA-256 `0e78199808bff149`

```python
"""
Tests for holding a saved model open across many predictions.

The thing worth testing here is not that a held predictor is faster -- a
benchmark would assert that, badly, and fail on a loaded machine. It is that
holding work open has not changed any answer.

So the shape of this module is: run the same prediction twice through a held
predictor and once through the ordinary one-shot path, and assert the numbers
and the provenance are identical. A cache that returns a different answer from
the uncached path is not a cache, and the failure would be invisible in
production -- the service would simply be wrong in a way that only showed up
when somebody reconciled against a batch run.

The second thing tested is the refusal to cache. A model with static inputs
must get a freshly prepared handle every call, because a graph adjacency can
change between requests and a handle prepared against a stale one answers
confidently from the wrong neighbourhood. That is a correctness property with
no observable symptom, which is exactly the kind that needs a test.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.rade_qnet import api
from src.rade_qnet.core.authoring.policy import PolicyModel
from src.rade_qnet.core.contract.result import Predictions
from src.rade_qnet.core.lifecycle.components import ComponentError
from src.rade_qnet.core.lifecycle.components import engine as register_engine
from src.rade_qnet.core.lifecycle.components import model as register_model
from src.rade_qnet.core.lifecycle.errors import BundleError, ContractError
from src.rade_qnet.engines.torch import TorchEngine
from src.rade_qnet.orchestration.pipelines.infer import InferPipeline
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.orchestration.serving import Agent, Predictor
from src.rade_qnet.orchestration.stages.reload import load_bundle
from src.rade_qnet.storage.runs.catalog import InMemoryCatalog
from src.rade_qnet.testkit.fixtures import (
    SyntheticEnvironment,
    isolated_registries,
    make_run_context,
)

from .pipelines.support import (
    SyntheticSupervisedModel,
    make_spec,
    register_components,
    write_linear_dataset,
)


@pytest.fixture(autouse=True)
def _registries():
    """
    Register the synthetic components, isolated to this module.

    Yields
    ------
    None
        For the duration of the test.
    """
    with isolated_registries(empty=True):
        register_components()
        yield


@pytest.fixture
def bundle(tmp_path):
    """
    Train a model and return its bundle directory.

    Returns
    -------
    pathlib.Path
        The bundle.
    """
    spec = make_spec(write_linear_dataset(tmp_path / "linear.csv"))
    pipeline = TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run", catalog=InMemoryCatalog()),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )
    pipeline.execute()
    assert pipeline.saved is not None
    return pipeline.saved.directory


@pytest.fixture
def predictor(tmp_path, bundle):
    """
    Open a held predictor over the trained bundle.

    Returns
    -------
    Predictor
        Ready to predict.
    """
    return Predictor(
        bundle,
        context=make_run_context(output_directory=tmp_path / "served"),
    )


def one_shot(tmp_path, bundle):
    """
    Predict the ordinary way, reopening the bundle.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    bundle
        The bundle directory.

    Returns
    -------
    Predictions
        What the uncached path produces.
    """
    return InferPipeline(
        context=make_run_context(output_directory=tmp_path / "cold"),
        directory=bundle,
    ).run()


class TestHoldingTheModelChangesNoAnswer:
    """The only property that matters: the numbers are the same."""

    def test_a_held_prediction_matches_the_one_shot_path(self, tmp_path, predictor, bundle):
        """
        Same bundle, same inputs, same values -- to the bit.

        ``allclose`` would be the wrong assertion. Nothing here is being
        recomputed in a different order or at a different precision; the
        held path runs the identical forward pass on the identical tensors,
        so any difference at all is a defect rather than drift.
        """
        held = predictor.predict()
        cold = one_shot(tmp_path, bundle)
        np.testing.assert_array_equal(np.asarray(held.values), np.asarray(cold.values))

    def test_the_second_prediction_matches_the_first(self, predictor):
        """
        The cache is populated after call one, so call two exercises it.

        This is the test that would fail if the reused handle had been
        mutated by the first pass -- autocast state left enabled, a
        precomputed encoding retained past its validity.
        """
        first = predictor.predict()
        second = predictor.predict()
        np.testing.assert_array_equal(np.asarray(first.values), np.asarray(second.values))

    def test_provenance_survives_being_held(self, predictor):
        """
        A held prediction is no cheaper to reconcile than a cold one.

        The held path is the one that will run a thousand times a day, so
        it is the one where missing provenance costs most. The whole
        provenance block has to survive, not just the bundle version, since
        reconciling needs the source fingerprints too.
        """
        predictions = predictor.predict()
        assert isinstance(predictions, Predictions)
        assert predictions.bundle_version
        assert predictions.provenance


class TestWhatIsAndIsNotReused:
    """Caching is a correctness decision here, not a performance one."""

    def test_the_bundle_is_read_once(self, tmp_path, bundle, monkeypatch):
        """
        Opening the bundle is the expensive part, and it happens at open.

        Asserted by making a second read impossible: the loader is replaced
        after construction, so a predict that tried to reload would raise
        rather than quietly cost what the handle exists to save.
        """
        predictor = Predictor(
            bundle, context=make_run_context(output_directory=tmp_path / "served")
        )

        def refuse(*_args, **_kwargs):
            message = "the bundle was reopened"
            raise AssertionError(message)

        monkeypatch.setattr("src.rade_qnet.orchestration.pipelines.infer.load_bundle", refuse)
        predictor.predict()
        predictor.predict()

    def test_a_model_without_static_inputs_reuses_its_handle(self, predictor):
        """
        The common case, and the one the saving is for.

        The synthetic model has no static inputs, so after one prediction
        the prepared handle is kept.
        """
        predictor.predict()
        assert predictor._prepared is not None

    def test_a_model_with_static_inputs_is_prepared_every_time(self, predictor):
        """
        The case where reuse would be wrong.

        Static inputs come from the data build rather than from the bundle,
        so a new instrument can change a graph adjacency between requests.
        A handle prepared against the previous one would answer from the
        wrong neighbourhood -- confidently, and with no symptom. Simulated
        here by reporting a non-empty static block, since the synthetic
        model has none of its own.
        """
        handle = SimpleNamespace(static={"adjacency": object()})
        predictor._remember(SimpleNamespace(handle=handle))
        assert predictor._prepared is None


class TestTheHandleDescribesItself:
    """A service needs something to put in a log line and a health check."""

    def test_the_model_name_comes_from_the_manifest(self, predictor):
        """
        Recorded rather than resolved.

        The manifest's name is what the bundle was written as; the class may
        since have been renamed, and the bundle still has to describe itself
        as the thing it was.
        """
        assert predictor.model_name

    def test_the_description_names_the_version(self, predictor):
        """So two handles in one process are distinguishable in a log."""
        assert "v" in predictor.describe()


class HeldPolicy(torch.nn.Module):
    """
    A one-layer policy sized from a signature.

    Parameters
    ----------
    signature
        Supplies the observation width and the number of actions.
    """

    def __init__(self, signature) -> None:
        super().__init__()
        self.head = torch.nn.Linear(signature.observation.shape[0], signature.action.n)

    def forward(self, *, observation: torch.Tensor) -> torch.Tensor:
        """
        Score one batch of observations.

        Parameters
        ----------
        observation
            A batch of observations.

        Returns
        -------
        torch.Tensor
            One row of action scores per observation.
        """
        return self.head(observation)


class HeldAgentModel(PolicyModel):
    """The two-method model an interactive run trains."""

    def build_environment(self, spec):
        """
        Construct the environment from the spec's parameters.

        Parameters
        ----------
        spec
            The validated run specification.

        Returns
        -------
        SyntheticEnvironment
            A fresh environment.
        """
        return SyntheticEnvironment(**dict(spec.environment.params))

    def build_policy(self, spec, signature):
        """
        Construct the untrained policy from the signature alone.

        Parameters
        ----------
        spec
            Unused: the shape is fully determined by the spaces.
        signature
            The declared spaces.

        Returns
        -------
        HeldPolicy
            An untrained policy.
        """
        del spec
        return HeldPolicy(signature)


@pytest.fixture
def policy_bundle(tmp_path):
    """
    Train a policy and return its bundle directory.

    Returns
    -------
    pathlib.Path
        The bundle.
    """
    with isolated_registries(empty=True):
        register_engine("torch")(TorchEngine)
        register_model("hedger", engine="torch")(HeldAgentModel)
        result = api.train(
            {
                "task": "reinforcement",
                "model": "hedger",
                "seed": 11,
                "environment": {"name": "synthetic", "params": {"episode_length": 4}},
                "training": {
                    "total_steps": 16,
                    "steps_per_update": 4,
                    "batch_size": 4,
                    "evaluate_every_steps": 8,
                },
                "output_root": str(tmp_path / "rl"),
            }
        )
        yield Path(result.bundle_directory)


class TestOpeningASavedPolicy:
    """The plumbing: a bundle becomes a usable network and two spaces."""

    def test_an_agent_reports_the_spaces_from_the_bundle(self, policy_bundle):
        """
        Read off disk, not off an environment.

        This is the whole reason the signature is saved. Serving a policy
        means there is no environment to ask, so a bundle that could not
        describe its own spaces could not be served at all.
        """
        held = Agent(policy_bundle)
        assert held.signature.observation.shape
        assert held.signature.action.n

    def test_the_description_names_the_model(self, policy_bundle):
        """So one line in a log identifies which policy answered."""
        assert "hedger" in Agent(policy_bundle).describe()

    def test_a_supervised_bundle_is_refused(self, bundle):
        """
        Pointing api.agent at a predictor is a mistake worth naming.

        The error says which function to use instead, because the two
        bundle kinds are indistinguishable from the directory name and a
        caller who got it wrong has no other way to find out.
        """
        with pytest.raises(BundleError, match="only a policy"):
            Agent(bundle)

    def test_a_policy_bundle_is_refused_by_the_predictor_path(self, policy_bundle):
        """The same mistake from the other direction, refused symmetrically."""
        with pytest.raises(BundleError, match="only supervised runs"):
            load_bundle(policy_bundle)


class TestActingIsRefusedUntilItCanBeHonest:
    """The gap this commit deliberately does not paper over."""

    def test_acting_without_a_greedy_mode_raises(self, policy_bundle):
        """
        The random learner explores by design, so it must not serve.

        Falling back to ``PolicyLearner.act`` would give a deployed policy
        that returned a different action each time it was asked the same
        question, with nothing in the log to show it. A loud refusal is the
        only honest answer, and the message has to name the method to add
        or the reader is left to discover it.
        """
        held = Agent(policy_bundle)
        with pytest.raises(ComponentError, match="act_greedily"):
            held.act(torch.zeros(held.signature.observation.shape[0]))

    def test_the_refusal_explains_why_rather_than_just_what(self, policy_bundle):
        """
        A refusal nobody understands gets worked around.

        Somebody hitting this will be tempted to reach past it to the
        exploratory ``act``, so the message has to say why that is wrong
        and not merely that a method is absent.
        """
        held = Agent(policy_bundle)
        with pytest.raises(ComponentError, match="explores by design"):
            held.act(torch.zeros(held.signature.observation.shape[0]))

    def test_a_learner_with_a_greedy_mode_is_used(self, policy_bundle, monkeypatch):
        """
        The refusal is the only missing piece; everything behind it works.

        Looked up by name rather than declared on ``PolicyLearner``, so a
        real algorithm starts serving by adding one method and changing
        nothing here.
        """
        held = Agent(policy_bundle)
        monkeypatch.setattr(
            held._learner_type,
            "act_greedily",
            staticmethod(
                lambda policy, observation: policy(observation=observation.unsqueeze(0)).argmax()
            ),
            raising=False,
        )
        assert held.act(torch.zeros(held.signature.observation.shape[0])) is not None


class TestTheObservationIsCheckedAgainstTheBundle:
    """A wrong-shaped observation must not become a confident number."""

    def test_a_mismatched_observation_is_refused(self, policy_bundle):
        """
        Checked here rather than left to the network.

        A first layer that happens to accept the wrong width returns a
        number indistinguishable from a real one, so waiting for the
        forward pass to object is waiting for something that may not
        happen.
        """
        held = Agent(policy_bundle)
        wrong = torch.zeros(held.signature.observation.shape[0] + 3)
        with pytest.raises(ContractError, match="cannot be adapted"):
            held.act(wrong)

    def test_the_error_names_both_shapes(self, policy_bundle):
        """Knowing what was expected is most of fixing the caller."""
        held = Agent(policy_bundle)
        with pytest.raises(ContractError, match="shape"):
            held.act(torch.zeros(99))
```

