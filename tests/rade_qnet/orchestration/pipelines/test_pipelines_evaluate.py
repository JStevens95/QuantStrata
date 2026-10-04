"""
Tests for scoring a saved model.

Almost every test here is about a failure that produces a *number* rather
than an error, because that is the characteristic failure of this pipeline.
A re-scored model with a re-derived split reports a plausible metric. A
re-fitted scaler produces plausible predictions. Nothing raises, and the
mistake surfaces weeks later as two numbers that should match and do not.

So the tests are built to make the wrong behaviour visibly wrong rather than
slightly wrong:

- The data is exactly linear and the engine solves least squares in closed
  form, so a correct reproduction is *equal*, not close. Equality is the only
  assertion strong enough to distinguish a correct reload from a nearly
  correct one.
- The "new data" used for re-scoring is shifted, not merely reseeded. A
  shifted feature distribution makes a re-fitted scaler produce different
  numbers from a correctly reapplied one, which a same-distribution redraw
  would not.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.contract.result import EvaluationResult
from src.rade_qnet.core.lifecycle.components import MODELS
from src.rade_qnet.core.lifecycle.errors import StageError
from src.rade_qnet.core.spec.data import ScalingSpec, TabularSourceSpec, TransformsSpec
from src.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.storage.bundle import load_lineage, load_spec, open_bundle
from src.rade_qnet.storage.runs.catalog import InMemoryCatalog
from src.rade_qnet.testkit.fixtures import isolated_registries, make_run_context

from .support import (
    MODEL_NAME,
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

    Scaling is switched on deliberately. Without a fitted transform there is
    nothing for a reload to get wrong, and most of this module would pass
    against a pipeline that re-fitted everything on the data in front of it.

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
def trained(tmp_path, spec):
    """
    Train a model and return its run result and bundle directory.

    Returns
    -------
    tuple
        The :class:`TrainingResult` and the bundle directory.
    """
    pipeline = TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run", catalog=InMemoryCatalog()),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )
    result = pipeline.execute()
    assert pipeline.saved is not None
    return result, pipeline.saved.directory


def evaluate(tmp_path, directory, **kwargs) -> EvaluationResult:
    """
    Run the evaluation pipeline over a bundle.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    directory
        The bundle to score.
    **kwargs
        Passed through to the pipeline.

    Returns
    -------
    EvaluationResult
        The scored result.
    """
    return EvaluatePipeline(
        context=make_run_context(output_directory=tmp_path / "eval"),
        directory=directory,
        **kwargs,
    ).execute()


class TestReproduction:
    """The gate: a saved model re-scored must give back its own numbers."""

    def test_evaluate_reproduces_training_metrics(self, tmp_path, trained):
        """
        Every metric on every split, exactly.

        Exactly rather than approximately, which is affordable because the
        engine is closed-form and the weights are loaded rather than
        re-fitted. A tolerance here would let through a reload that
        re-derived the split into something *almost* the same -- and an
        almost-correct split is the failure mode that costs most, because
        it means the model is being scored partly on rows it trained on.
        """
        result, directory = trained
        rescored = evaluate(tmp_path, directory)

        assert set(rescored.evaluations) == set(result.evaluations)
        for split, original in result.evaluations.items():
            assert rescored.evaluations[split].metrics == original.metrics
            assert rescored.evaluations[split].n_samples == original.n_samples

    def test_the_baseline_is_reproduced_too(self, tmp_path, trained):
        """
        Not only the model's metrics.

        The baseline is computed from the *training* targets, so reproducing
        it is a second, independent check that the split came back intact:
        a re-derived split would change which rows the baseline mean is
        taken over, even where the model's own score happened to survive.
        """
        result, directory = trained
        rescored = evaluate(tmp_path, directory)

        for split, original in result.evaluations.items():
            assert rescored.evaluations[split].baseline_metrics == original.baseline_metrics

    def test_metrics_are_in_original_target_units(self, tmp_path, trained):
        """
        A scaled target still reports an error on the original scale.

        An error in standardised space is not a quantity anyone can act on,
        and two runs' standardised errors -- each scaled by its own training
        mean -- are not comparable to one another.
        """
        _, directory = trained
        rescored = evaluate(tmp_path, directory)

        assert all(result.in_original_units for result in rescored.evaluations.values())
        # The relationship is exact, so a correctly inverted prediction has
        # essentially no error. A metric left in standardised space would be
        # a different number entirely.
        assert rescored.metric("test", "mae") < 1e-6

    def test_a_reproduction_is_marked_as_one(self, tmp_path, trained):
        """
        The result says the source did not change.

        Which is the half of provenance a reader actually acts on: a metric
        that matches a recorded one means nothing unless it was computed
        over the same rows.
        """
        _, directory = trained
        rescored = evaluate(tmp_path, directory)

        assert rescored.source_changed is False
        assert rescored.source_fingerprint == rescored.trained_on_fingerprint
        assert "CHANGED SOURCE" not in rescored.describe()


class TestTheSavedSplitIsUsed:
    """The split is read from lineage, never re-derived."""

    def test_rebuild_uses_the_saved_split(self, tmp_path, trained, dataset):
        """
        Changed split settings do not move an old bundle's boundaries.

        The naive implementation re-derives the split from the spec it is
        handed. It would pass every other test in this module and fail this
        one, which is why the fraction is changed to something that could
        not possibly coincide.
        """
        result, directory = trained
        rescored = evaluate(
            tmp_path,
            directory,
            source=TabularSourceSpec(
                path=dataset,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
                split={"kind": "chronological", "test_fraction": 0.5, "validation_fraction": 0.3},
            ),
        )

        for split, original in result.evaluations.items():
            assert rescored.evaluations[split].n_samples == original.n_samples

    def test_the_rebuilt_split_indices_are_the_saved_ones(self, tmp_path, trained):
        """
        Identical index lists, not merely identical sizes.

        Two different splits of the same data can have the same sizes, so
        comparing counts alone would pass against a re-derived split that
        happened to use the same fractions. The indices are the thing that
        decides which rows a metric is computed over.
        """
        _, directory = trained
        saved_lineage = load_lineage(open_bundle(directory))

        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)

        assert data.lineage.split_indices == saved_lineage.split_indices


class TestTheSavedStateIsUsed:
    """The fitted state is applied, never re-fitted."""

    def test_rebuild_does_not_refit_the_state(self, tmp_path, trained, dataset):
        """
        The classic production failure, asserted directly.

        Scoring against a *shifted* source: the relationship is unchanged, so
        a model whose saved scaler is reapplied still predicts well. A model
        whose scaler was re-fitted on the shifted data sees inputs centred
        where the training inputs were, which is no longer where these
        inputs are, and its error explodes.
        """
        _, directory = trained
        shifted = write_linear_dataset(tmp_path / "shifted.csv", seed=29, shift=3.0)

        rescored = evaluate(
            tmp_path,
            directory,
            source=TabularSourceSpec(
                path=shifted,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
            ),
            splits=("test",),
        )

        # Reapplied correctly, the exact linear relationship still holds.
        assert rescored.metric("test", "mae") < 1e-6

    def test_the_bundle_state_object_is_the_one_applied(self, tmp_path, trained):
        """
        Not merely an equivalent one.

        Checked through the pipeline's own stages rather than its result,
        because the result cannot distinguish "the saved state" from "a
        state that happens to agree".
        """
        _, directory = trained
        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)

        assert data.state is loaded.state


class TestChangedSources:
    """A re-score against new data is allowed, and is never silent."""

    def test_rebuild_reports_a_changed_fingerprint(self, tmp_path, trained):
        """
        Surfaced on the result, not left for a reader to infer.

        Re-scoring on new data is a legitimate and common thing to do. Doing
        it while believing you reproduced an old number is not, and the only
        difference between the two is whether anybody was told.
        """
        _, directory = trained
        shifted = write_linear_dataset(tmp_path / "other.csv", seed=77)

        rescored = evaluate(
            tmp_path,
            directory,
            source=TabularSourceSpec(
                path=shifted,
                transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
            ),
            splits=("test",),
        )

        assert rescored.source_changed is True
        assert rescored.source_fingerprint != rescored.trained_on_fingerprint
        assert "CHANGED SOURCE" in rescored.describe()

    def test_a_source_with_different_columns_is_refused(self, tmp_path, trained):
        """
        Rather than scored against the wrong inputs.

        A changed column count means the weights were shaped for one
        interface and are being fed another. Here the saved scaler refuses
        first, which is the better place for it: the fault is caught while
        reapplying the state rather than after a forward pass.
        """
        _, directory = trained
        narrow = tmp_path / "narrow.csv"
        rows = (tmp_path / "linear.csv").read_text(encoding="utf-8").splitlines()
        narrow.write_text(
            "\n".join(",".join(line.split(",")[1:]) for line in rows) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(StageError, match=r"3 column\(s\) but the scaler was fitted on 4"):
            evaluate(
                tmp_path,
                directory,
                source=TabularSourceSpec(
                    path=narrow,
                    transforms=TransformsSpec(scaling=ScalingSpec(method="standard")),
                ),
                splits=("test",),
            )


class TestLoading:
    """Opening a bundle, and failing usefully when it cannot be opened."""

    def test_an_unregistered_model_names_the_problem(self, tmp_path, trained):
        """
        The error must say the package was not imported.

        A bundle records a model *name*, so loading one needs the model's
        package importable -- the same constraint job sets hit as defect 12,
        reached from the other direction. A bare registry miss here would
        send the reader looking at the bundle rather than at their imports.
        """
        _, directory = trained

        with isolated_registries():
            # Emptied rather than simply not re-registered: the fixture
            # snapshots the registries rather than clearing them, so without
            # this the model would still be resolvable inside the block.
            MODELS.restore({})
            with pytest.raises(StageError, match=r"not registered in this process"):
                evaluate(tmp_path, directory)

    def test_the_error_lists_what_is_registered(self, tmp_path, trained):
        """So a typo in a model name is diagnosable from the message alone."""
        _, directory = trained

        with isolated_registries():
            MODELS.restore({"other_model": MODELS.entry(MODEL_NAME)})
            with pytest.raises(StageError) as caught:
                evaluate(tmp_path, directory)

        assert "other_model" in str(caught.value)

    def test_the_loaded_spec_round_trips(self, tmp_path, trained, spec):
        """
        ``load_spec`` returns the specification the run was configured with.

        Phase 1 wrote ``spec.json`` and provided no way to read it back,
        which went unnoticed for four phases because nothing re-loaded a
        bundle.
        """
        del tmp_path
        _, directory = trained
        assert load_spec(open_bundle(directory)) == spec

    def test_the_stage_sequence_is_declared(self, tmp_path, trained):
        """Declared as data, so a hook can enumerate it before the run."""
        _, directory = trained
        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        pipeline.execute()

        assert pipeline.stages == ("load", "rebuild_data", "restore", "score", "report")
        assert pipeline.completed_stages == pipeline.stages


class TestProvenance:
    """What a metric must carry to be comparable to another metric."""

    def test_the_result_identifies_the_bundle(self, tmp_path, trained):
        """Model name, version and spec digest, from the manifest."""
        _, directory = trained
        manifest = open_bundle(directory).manifest
        rescored = evaluate(tmp_path, directory)

        assert rescored.model_name == manifest.model_name
        assert rescored.bundle_version == manifest.version
        assert rescored.spec_digest == manifest.spec_digest

    def test_selecting_splits_scores_only_those(self, tmp_path, trained):
        """
        A re-score against new observations usually wants only ``test``.

        On fresh data the training split is no longer a meaningful category,
        and scoring it would invite a comparison that means nothing.
        """
        _, directory = trained
        rescored = evaluate(tmp_path, directory, splits=("test",))

        assert set(rescored.evaluations) == {"test"}

    def test_a_missing_metric_names_the_alternatives(self, tmp_path, trained):
        """So a typo is diagnosable without opening the contract."""
        _, directory = trained
        rescored = evaluate(tmp_path, directory)

        with pytest.raises(Exception, match=r"no metric 'rmsee'"):
            rescored.metric("test", "rmsee")


class TestPredictionsAreUnchanged:
    """The restored model is the trained model, not a lookalike."""

    def test_restored_weights_predict_identically(self, tmp_path, trained, spec):
        """
        Compared row by row against the training run's own forward pass.

        Metrics are a reduction, and two different prediction vectors can
        reduce to the same mean absolute error. Comparing the vectors
        themselves is the stronger statement.
        """
        from src.rade_qnet.orchestration.stages.scoring import scoring_source  # noqa: PLC0415

        _, directory = trained

        trainer = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "again"),
            spec=spec,
            definition=SyntheticSupervisedModel(),
        )
        trainer.execute()
        trained_data = trainer.build_data(spec)
        assert trainer.handle is not None
        expected = trainer._engine().predict(trainer.handle, scoring_source(trained_data, "test"))

        pipeline = EvaluatePipeline(
            context=make_run_context(output_directory=tmp_path / "eval"),
            directory=directory,
        )
        loaded = pipeline.load()
        data = pipeline.rebuild_data(loaded)
        handle = pipeline.restore(loaded, data)
        actual = pipeline._engine(loaded).predict(handle, scoring_source(data, "test"))

        assert np.allclose(np.ravel(actual), np.ravel(expected))
