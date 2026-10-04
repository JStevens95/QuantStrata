"""
Tests for producing predictions from a saved model.

Inference is evaluation without targets, so the reload half is already
covered by the evaluation tests and is not repeated here. What these tests
cover is what is *different* once nobody is computing a metric:

- **Nothing reduces the numbers.** An evaluation that misattributes every
  prediction to the wrong row still produces a mean absolute error, and a
  wrong one is hard to distinguish from a right one. An inference pass that
  does the same hands somebody a number for the wrong instrument. So the
  tests here assert on individual values and their identifiers rather than
  on aggregates.

- **Provenance is load-bearing.** A prediction detached from the bundle and
  source behind it cannot be reconciled later, and reconciling predictions
  after the fact is the ordinary case: somebody will ask why Tuesday's
  number differed from Monday's.

- **Refusal is a feature.** Asking a transductive model about an instrument
  it has never seen tends to return a default embedding -- a confident
  number indistinguishable from a real one. The test that this is refused
  matters more than most of the tests that something succeeds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.rade_qnet.core.contract.result import Predictions
from src.rade_qnet.core.lifecycle.errors import ContractError, StageError
from src.rade_qnet.core.spec.data import ScalingSpec, TabularSourceSpec, TransformsSpec
from src.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline
from src.rade_qnet.orchestration.pipelines.infer import InferPipeline
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.orchestration.stages.scoring import scoring_source
from src.rade_qnet.storage.bundle import load_lineage, open_bundle
from src.rade_qnet.storage.runs.catalog import InMemoryCatalog
from src.rade_qnet.testkit.fixtures import isolated_registries, make_run_context

from .support import (
    SyntheticSupervisedModel,
    make_spec,
    register_components,
    write_linear_dataset,
)

#: Entities a model with a per-entity axis would have been trained on. Used
#: to give the synthetic model's data bundle an entity list it would not
#: otherwise have, so the unseen-entity branch can be exercised at all.
KNOWN_ENTITIES = ("EURUSD", "GBPUSD", "USDJPY")


@pytest.fixture(autouse=True)
def _registries():
    """
    Register the synthetic components, isolated to this module.

    Yields
    ------
    None
        For the duration of the test.
    """
    # ``empty=True`` because the stand-in engine below claims a shipped
    # engine's name. Without it, whether this fixture succeeds depends on
    # whether an earlier test imported the real one -- which made the suite
    # pass or fail on collection order.
    with isolated_registries(empty=True):
        register_components()
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write the standardised training dataset.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


@pytest.fixture
def spec(dataset):
    """
    Build a spec that standardises its features.

    Scaling is on so that the inversion stage has something to invert.
    Without it a prediction in the model's output space and one in the
    target's original units are the same number, and every test about units
    would pass vacuously.

    Returns
    -------
    SupervisedRunSpec
        The spec.
    """
    return make_spec(
        dataset,
        source=TabularSourceSpec(
            path=dataset,
            transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
        ),
    )


@pytest.fixture
def bundle(tmp_path, spec):
    """
    Train a model and return its bundle directory.

    Returns
    -------
    pathlib.Path
        The bundle.
    """
    pipeline = TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run", catalog=InMemoryCatalog()),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )
    pipeline.execute()
    assert pipeline.saved is not None
    return pipeline.saved.directory


def entity_aware(tmp_path, directory, entities):
    """
    Build the pieces needed to exercise the unseen-entity check.

    The synthetic model has no entity axis, so its data bundle carries no
    identifiers and the check returns early -- correctly, since a tabular
    model's "entities" are the caller's row labels rather than anything the
    model resolves. To test the branch that matters, the rebuilt bundle is
    given an entity list, which is what a model with a per-entity embedding
    table would produce.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    directory
        The bundle to load.
    entities
        Identifiers to request predictions for.

    Returns
    -------
    tuple
        The pipeline, the loaded bundle, and a data bundle that knows which
        entities the model was trained on.
    """
    pipeline = InferPipeline(
        context=make_run_context(output_directory=tmp_path / "infer"),
        directory=directory,
        entities=entities,
    )
    loaded = pipeline.load()
    data = loaded.definition.rebuild_data(loaded.spec, state=loaded.state, lineage=loaded.lineage)
    return pipeline, loaded, replace(data, entity_ids=KNOWN_ENTITIES)


def infer(tmp_path, directory, **kwargs) -> Predictions:
    """
    Run the inference pipeline over a bundle.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    directory
        The bundle to predict with.
    **kwargs
        Passed through to the pipeline.

    Returns
    -------
    Predictions
        The predictions.
    """
    return InferPipeline(
        context=make_run_context(output_directory=tmp_path / "infer"),
        directory=directory,
        **kwargs,
    ).execute()


class TestTheForwardPass:
    """Predictions come back, in the right space, for the right rows."""

    def test_the_stage_sequence_is_declared(self, tmp_path, bundle):
        """Declared as data, so a hook can enumerate it before the run."""
        pipeline = InferPipeline(
            context=make_run_context(output_directory=tmp_path / "infer"),
            directory=bundle,
        )
        pipeline.execute()

        assert pipeline.stages == (
            "load",
            "prepare_inputs",
            "restore",
            "predict",
            "invert",
            "attribute",
        )
        assert pipeline.completed_stages == pipeline.stages

    def test_one_prediction_per_row_of_the_split(self, tmp_path, bundle):
        """
        The count matches the split, not the whole source.

        A pass that silently predicted over everything would still return
        plausible numbers, just the wrong ones for the question asked.
        """
        predictions = infer(tmp_path, bundle)
        saved = open_bundle(bundle)
        expected = len(load_lineage(saved).split_indices["test"])
        assert predictions.n_predictions == expected

    def test_predictions_are_in_original_target_units(self, tmp_path, bundle, dataset):
        """
        Compared against the actual target column, not against a scale.

        The relationship is exactly linear and the engine solves it in
        closed form, so a correctly inverted prediction equals the observed
        value. A prediction left in standardised space would be off by the
        training mean and standard deviation -- a large, obvious gap, which
        is what makes this assertion worth making as an equality.
        """
        predictions = infer(tmp_path, bundle)
        assert predictions.in_original_units is True

        rows = dataset.read_text(encoding="utf-8").splitlines()[1:]
        targets = np.array([float(row.split(",")[-1]) for row in rows])
        indices = np.asarray(load_lineage(open_bundle(bundle)).split_indices["test"])

        assert np.allclose(predictions.values, targets[indices], atol=1e-6)

    def test_predictions_match_the_evaluation_pass(self, tmp_path, bundle):
        """
        The two pipelines must not produce different numbers.

        They share their reload path as functions rather than as a base
        class, and this is the test that the sharing is real: if one of them
        ever grows its own version of a stage, the vectors stop agreeing.
        """
        predictions = infer(tmp_path, bundle)

        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=bundle,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)
        handle = pipeline.restore(loaded, data)
        raw = pipeline._engine(loaded).predict(handle, scoring_source(data, "test"))
        expected = loaded.state.inverse_transform_targets(np.asarray(raw, dtype=np.float64))

        assert np.allclose(predictions.values, np.ravel(expected))

    def test_an_absent_split_names_the_available_ones(self, tmp_path, bundle):
        """So a typo is diagnosable from the message alone."""
        with pytest.raises(StageError, match=r"no 'holdout' split"):
            infer(tmp_path, bundle, split="holdout")


class TestProvenance:
    """What a prediction must carry to be reconciled with anything later."""

    def test_predictions_carry_provenance(self, tmp_path, bundle):
        """Bundle version, spec digest and the time of inference."""
        manifest = open_bundle(bundle).manifest
        predictions = infer(tmp_path, bundle)

        assert predictions.bundle_version == str(manifest.version)
        assert predictions.provenance["spec_digest"] == manifest.spec_digest
        assert predictions.provenance["model_name"] == manifest.model_name
        assert predictions.provenance["predicted_at"]

    def test_provenance_records_both_fingerprints(self, tmp_path, bundle):
        """
        The data predicted over, and the data the model was trained on.

        Both, because the question people ask later is whether they differ.
        Recording only one makes that unanswerable from the record alone.
        """
        predictions = infer(tmp_path, bundle)

        assert (
            predictions.provenance["source_fingerprint"]
            == predictions.provenance["trained_on_fingerprint"]
        )

    def test_predicting_over_new_data_records_the_difference(self, tmp_path, bundle):
        """So a prediction made against other inputs is identifiable as such."""
        other = write_linear_dataset(tmp_path / "other.csv", seed=77)
        predictions = infer(
            tmp_path,
            bundle,
            source=TabularSourceSpec(
                path=other,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
            ),
        )

        assert (
            predictions.provenance["source_fingerprint"]
            != predictions.provenance["trained_on_fingerprint"]
        )

    def test_scenario_indices_pair_values_with_source_rows(self, tmp_path, bundle):
        """Without which a prediction cannot be attributed to an observation."""
        predictions = infer(tmp_path, bundle)
        expected = np.asarray(load_lineage(open_bundle(bundle)).split_indices["test"])

        assert predictions.scenario_indices is not None
        assert np.array_equal(predictions.scenario_indices, expected)


class TestUnseenEntities:
    """A refusal that matters more than most successes."""

    def test_a_model_with_no_entity_axis_accepts_any_labels(self, tmp_path, bundle):
        """
        Because they are the caller's labels for rows, not model entities.

        A tabular model predicts per scenario. Refusing a label it was never
        going to resolve would be a false alarm, and false alarms are how a
        useful check gets switched off.
        """
        predictions = infer(tmp_path, bundle)
        labels = tuple(f"row-{index}" for index in range(predictions.n_predictions))

        relabelled = infer(tmp_path, bundle, entities=labels)
        assert relabelled.entity_ids == labels

    def test_non_inductive_model_raises_for_unseen_entity(self, tmp_path, bundle):
        """
        Rather than returning a plausible number for an unknown instrument.

        The quiet failure this check exists for: a model with a per-entity
        embedding table has no row for a new instrument, so it falls back to
        a default and returns a confident prediction that looks exactly like
        the real ones.
        """
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("EURUSD", "BRAND_NEW"))

        with pytest.raises(ContractError, match=r"does not declare Inductive"):
            pipeline._check_entities(loaded, data)

    def test_the_refusal_names_the_offending_entities(self, tmp_path, bundle):
        """So the caller can see whether it is one stale ticker or a whole feed."""
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("BRAND_NEW",))

        with pytest.raises(ContractError, match=r"BRAND_NEW"):
            pipeline._check_entities(loaded, data)

    def test_a_known_entity_is_not_refused(self, tmp_path, bundle):
        """Otherwise the check would reject every request, including valid ones."""
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("EURUSD",))

        pipeline._check_entities(loaded, data)

    def test_an_inductive_model_is_allowed_through(self, tmp_path, bundle):
        """Declaring the capability is what turns the refusal into a route."""
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("BRAND_NEW",))

        class InductiveDefinition:
            """Declares that unseen entities are predictable."""

            def supports_unseen_entities(self):
                """Return True."""
                return True

        pipeline._check_entities(replace(loaded, definition=InductiveDefinition()), data)

    def test_answering_false_is_refused_like_not_declaring_at_all(self, tmp_path, bundle):
        """
        Because the answer is a method, not the mere presence of one.

        A model may be inductive or transductive according to how it was
        configured -- an entity embedding table versus a feature encoder --
        so the capability is read by calling it, and a class that answers
        False is in exactly the position of a class that never declared it.
        """
        pipeline, loaded, data = entity_aware(tmp_path, bundle, ("BRAND_NEW",))

        class Transductive:
            """Declares the capability and then disclaims it."""

            def supports_unseen_entities(self):
                """Return False."""
                return False

        with pytest.raises(ContractError, match=r"does not declare Inductive"):
            pipeline._check_entities(replace(loaded, definition=Transductive()), data)
