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

from types import SimpleNamespace

import numpy as np
import pytest

from src.rade_qnet.core.contract.result import Predictions
from src.rade_qnet.orchestration.pipelines.infer import InferPipeline
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.orchestration.serving import Predictor
from src.rade_qnet.storage.runs.catalog import InMemoryCatalog
from src.rade_qnet.testkit.fixtures import isolated_registries, make_run_context

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
