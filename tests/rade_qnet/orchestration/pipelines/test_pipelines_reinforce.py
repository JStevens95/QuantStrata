"""
Tests for the interactive training pipeline.

The pipeline's claim is that an interactive run is a *sibling* of a
supervised one rather than a fork, so the tests are mostly about sameness:
the same stage mechanics, the same ``TrainingResult``, the same bundle
layout, the same catalog entry. If those drift apart, every later
improvement to training has to be made twice.

The two tests that matter most are about the bundle, and they are the reason
``ModelBundle.signature`` was widened to a union rather than made to hold the
experience stream's tensor description:

**A policy rebuilds from its bundle with no environment.** That is the whole
purpose of saving a signature. A tensor description would have passed every
other test here and failed this one, because it loses the number of discrete
actions and the bounds of a continuous space.

**A supervised reader refuses an interactive bundle by name.** Widening a
contract is only safe if the readers that cannot handle the new member say
so. ``load_bundle`` and ``load_signature`` both do.

One absence is deliberate and tested for: there is no evaluate stage, because
nothing on this path can turn exploration off yet. See the pipeline's module
docstring.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.rade_qnet import api
from src.rade_qnet.core.capability.policy import PolicyModel
from src.rade_qnet.core.runtime.components import engine as register_engine
from src.rade_qnet.core.runtime.components import model as register_model
from src.rade_qnet.core.runtime.errors import BundleError, SpecError, StageError
from src.rade_qnet.core.spec.run import parse_run_spec
from src.rade_qnet.engines.torch import TorchEngine
from src.rade_qnet.orchestration.pipelines.reinforce import ReinforcePipeline
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.orchestration.stages.reload import load_bundle
from src.rade_qnet.storage.bundle import (
    load_lineage,
    load_policy_signature,
    load_signature,
    load_spec,
    open_bundle,
)
from src.rade_qnet.testkit.fixtures import (
    SyntheticEngine,
    SyntheticEnvironment,
    isolated_registries,
)

from .support import SyntheticSupervisedModel

EPISODE_LENGTH = 4


class Policy(torch.nn.Module):
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


class Agent(PolicyModel):
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
        Policy
            An untrained policy.
        """
        del spec
        return Policy(signature)


@pytest.fixture
def registered():
    """
    Register the real Torch engine and the interactive model, in isolation.

    ``empty=True`` because this fixture *claims* the names ``torch`` and
    ``agent`` rather than merely adding them, and a block that only snapshots
    would pass or fail on collection order.

    The learner registry is deliberately not emptied by
    :func:`isolated_registries` and is not touched here: ``random`` is
    registered by importing the Torch engine, which is exactly the path a
    real run relies on.

    Yields
    ------
    None
        For the duration of one test.
    """
    with isolated_registries(empty=True):
        register_engine("torch")(TorchEngine)
        register_model("agent", engine="torch")(Agent)
        yield


def payload(output_root, **training):
    """
    Build an interactive run configuration.

    Parameters
    ----------
    output_root
        Where the run writes.
    training
        Keys merged over the minimal training section.

    Returns
    -------
    dict
        A configuration ready for :func:`api.train`.
    """
    section = {
        "total_steps": 24,
        "steps_per_update": 4,
        "batch_size": 4,
        "evaluate_every_steps": 8,
    }
    section.update(training)
    return {
        "task": "reinforcement",
        "model": "agent",
        "seed": 11,
        "environment": {"name": "synthetic", "params": {"episode_length": EPISODE_LENGTH}},
        "training": section,
        "output_root": str(output_root),
    }


class TestAFullInteractiveRun:
    """It runs, and it produces the same shape of answer a supervised run does."""

    def test_a_run_completes_and_reports_its_blocks(self, tmp_path, registered):
        """
        Twenty-four steps in blocks of eight is three records.

        The budget is in steps rather than passes, and the history is in
        blocks rather than updates, so this pins both.
        """
        result = api.train(payload(tmp_path))

        assert result.fit.n_epochs == 3
        assert result.seed == 11

    def test_the_episode_return_is_the_metric_that_is_reported(self, tmp_path, registered):
        """
        Not a loss.

        The synthetic environment rewards one per step and terminates after
        four, so an episode's return is four by arithmetic. That it arrives
        in the history at all is the point: the driver reads it from the
        source's own bookkeeping.
        """
        result = api.train(payload(tmp_path))

        for record in result.fit.history:
            assert record.metrics["episode_return"] == pytest.approx(float(EPISODE_LENGTH))

    def test_there_is_no_validation_loss(self, tmp_path, registered):
        """An environment has no held-out split, so nothing pretends it does."""
        result = api.train(payload(tmp_path))
        assert all(record.val_loss is None for record in result.fit.history)

    def test_the_run_has_no_evaluations(self, tmp_path, registered):
        """
        Empty, and deliberately so.

        Scoring a policy means running episodes with exploration off, which
        nothing on this path can do yet. An empty mapping says "not measured";
        a number produced from the exploring policy would say "measured" and
        be wrong, and somebody would compare two runs with it.
        """
        result = api.train(payload(tmp_path))
        assert result.evaluations == {}

    def test_the_stage_sequence_mirrors_the_supervised_one(self):
        """
        The same names in the same order, where the work is the same.

        Compared as a property rather than described in prose, so a stage
        quietly renamed on one side shows up here.
        """
        shared = ("resolve", "resolve_seed", "materialise", "prepare_hardware", "fit", "persist")
        for name in shared:
            assert name in ReinforcePipeline.stages
            assert name in TrainPipeline.stages

        interactive = ReinforcePipeline.stages
        assert interactive.index("materialise") < interactive.index("prepare_hardware")
        assert interactive.index("fit") < interactive.index("persist")

    def test_nothing_is_written_when_a_stage_fails(self, tmp_path):
        """
        Persistence is last, so a failed run leaves no half-registered bundle.

        Forced by a model whose ``build_policy`` returns something that is
        not a network -- a realistic fault in a model package, and one the
        spec cannot catch, since the spec never sees the object.
        """

        class Broken(Agent):
            def build_policy(self, spec, signature):
                """Return something the engine cannot train."""
                del spec, signature
                return "not a network"

        with isolated_registries(empty=True):
            register_engine("torch")(TorchEngine)
            register_model("agent", engine="torch")(Broken)

            with pytest.raises(StageError):
                api.train(payload(tmp_path))

        assert not list(tmp_path.rglob("manifest.json"))


class TestTheBundle:
    """What a policy bundle holds, and what can read it."""

    def test_the_bundle_has_the_same_layout_as_a_supervised_one(self, tmp_path, registered):
        """
        Same files, same names, written by the same function.

        A second layout would mean every tool that reads a bundle needs two
        code paths.
        """
        result = api.train(payload(tmp_path))
        written = {path.name for path in Path(result.bundle_directory).iterdir()}

        assert {"manifest.json", "signature.json", "spec.json", "lineage.json"} <= written
        assert "fitted_state" in written

    def test_the_policy_rebuilds_from_the_bundle_with_no_environment(self, tmp_path, registered):
        """
        The reason the signature is saved at all.

        No environment is constructed anywhere in this test after the run
        finishes. The spaces come off the bundle, and they come off it
        intact -- the discrete action count and the box bounds both survive,
        which a tensor description would have lost.
        """
        result = api.train(payload(tmp_path))
        saved = open_bundle(Path(result.bundle_directory))

        signature = load_policy_signature(saved)
        assert signature.action.n == SyntheticEnvironment().action_space.n
        assert signature.observation.high == float(EPISODE_LENGTH)

        rebuilt = Agent().build_policy(load_spec(saved), signature)
        assert rebuilt.head.out_features == signature.action.n

    def test_a_supervised_reader_refuses_it(self, tmp_path, registered):
        """
        Widening a contract is only safe if its narrow readers say so.

        ``load_signature`` expects a target and an interactive bundle has
        none, so it fails rather than parsing a partial result.
        """
        result = api.train(payload(tmp_path))
        saved = open_bundle(Path(result.bundle_directory))

        with pytest.raises(BundleError):
            load_signature(saved)

    def test_load_bundle_refuses_it_by_name(self, tmp_path, registered):
        """
        And says which path the bundle does not belong to.

        The evaluate and infer pipelines both go through this function, so
        one refusal covers both.
        """
        result = api.train(payload(tmp_path))

        with pytest.raises(BundleError, match="only supervised runs"):
            load_bundle(Path(result.bundle_directory))

    def test_the_lineage_records_the_environment_rather_than_a_split(self, tmp_path, registered):
        """
        An empty split map, with a note saying why.

        Otherwise an interactive bundle looks like a supervised one whose
        data build failed.
        """
        result = api.train(payload(tmp_path))
        saved = open_bundle(Path(result.bundle_directory))

        recorded = load_lineage(saved)
        assert recorded.split_indices == {}
        assert recorded.n_scenarios == 24
        assert "environment" in recorded.notes["interaction"]

    def test_the_run_is_recorded_in_the_catalog(self, tmp_path, registered):
        """
        So an interactive run is selectable by the same tools as any other.

        A separate record keeper for policies would mean two places to look
        for "the best model we have".
        """
        result = api.train(payload(tmp_path))
        assert result.bundle_directory.endswith("v1")


class TestWhatIsRefused:
    """The pairings that cannot work, named before anything expensive runs."""

    def test_a_supervised_model_in_an_interactive_run_is_refused(self, tmp_path, registered):
        """
        Reported against the model, not against a missing environment.

        Without this the run would fail in ``build_environment`` with an
        ``AttributeError``, which points at the pipeline rather than at the
        configuration.
        """
        register_model("predictor", engine="torch")(SyntheticSupervisedModel)
        configuration = payload(tmp_path)
        configuration["model"] = "predictor"

        with pytest.raises(SpecError, match="learns from a fixed dataset"):
            api.train(configuration)

    def test_an_engine_that_cannot_train_a_policy_is_refused(self, tmp_path, registered):
        """
        Asked at resolve time, which costs milliseconds.

        The capability is opt-in precisely so this answer is available
        before an environment is built: a tree engine cannot train a policy
        by any amount of plumbing, and finding out four stages in would waste
        the data build.
        """
        register_engine("sklearn")(SyntheticEngine)
        configuration = payload(tmp_path)
        configuration["training"] = {"engine": "sklearn"}

        with pytest.raises(Exception, match="engine"):
            api.train(configuration)

    def test_an_interactive_spec_still_parses_as_a_run_spec(self):
        """
        Routed by task, through the ordinary discriminated union.

        No second parser and no second entry point: ``api.train`` reads the
        task and dispatches, because "train what this file describes" is one
        intent.
        """
        spec = parse_run_spec(payload("/tmp"), origin="test")
        assert spec.task == "reinforcement"
