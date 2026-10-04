"""
Tests for the training pipeline, now that it runs end to end.

In Phase 1 these tests covered a skeleton whose last three stages raised. Phase
2 supplies an engine, so what is worth asserting changes completely: not "the
unfinished stages are instrumented" but "the finished sequence produces the
right artifacts, in the right order, with the right numbers".

Everything here runs against
:class:`~rade_qnet.testkit.fixtures.SyntheticEngine`, which solves least squares
in closed form. That is a deliberate choice over the real torch engine: the
solution is *exact*, so these tests assert on the metrics themselves rather
than on a tolerance. A pipeline test that can only check "the loss went down"
cannot distinguish a correct pipeline from one that scores the wrong split --
which is exactly the bug Phase 2 found in this file's subject.

Four orderings are load-bearing and are asserted rather than assumed:

- resolution precedes the data build, so a misspelled name costs milliseconds
  rather than a full build;
- seeding precedes the data build, because a data build that samples is part
  of what a seed must reproduce;
- materialisation precedes hardware preparation, which is defect 6;
- persistence precedes reporting, and nothing registers before it succeeds.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

# The three report modules are imported for their registration side effect:
# `isolated_registries` snapshots the registries rather than emptying them, so
# a report is available inside the isolated context only if its module was
# imported before the context opened.
from src.rade_qnet.analysis.reports import curves, quality, summary  # noqa: F401
from src.rade_qnet.analysis.reports.base import Report, report
from src.rade_qnet.core.capability.simple import TabularModel
from src.rade_qnet.core.contract.bundle import ModelBundle
from src.rade_qnet.core.contract.data import DataBundle, TensorBatchData
from src.rade_qnet.core.runtime.components import ENGINES
from src.rade_qnet.core.runtime.components import engine as register_engine
from src.rade_qnet.core.runtime.errors import StageError
from src.rade_qnet.core.spec.data import (
    ScalingSpec,
    TabularSourceSpec,
    TransformsSpec,
)
from src.rade_qnet.orchestration.pipelines.scoring import scoring_source, source_for
from src.rade_qnet.orchestration.pipelines.train import TrainPipeline
from src.rade_qnet.sources.dataset.module import TabularDataModule
from src.rade_qnet.storage.bundle import load_signature, open_bundle
from src.rade_qnet.storage.catalog import InMemoryCatalog
from src.rade_qnet.testkit.fixtures import (
    LinearModel,
    RecordingHook,
    SyntheticEngine,
    isolated_registries,
    make_run_context,
    make_signature,
    make_tensor_bundle,
)

#: Engine tag the synthetic engine is registered under. See the registry
#: fixture for why it borrows an existing name.
ENGINE_TAG = "sklearn"

#: Width of the synthetic problem. Small, and solvable exactly.
N_FEATURES = 4

#: True coefficients of the synthetic target, so a correct run must recover
#: them and an incorrect one cannot.
TRUE_COEFFICIENTS = np.array([1.5, -0.7, 0.3, 2.0])
TRUE_INTERCEPT = 0.4


@pytest.fixture(autouse=True)
def _registries():
    """
    Isolate the component registries for every test in this module.

    Autouse because every test here registers an engine, and a registration
    that leaked would fail whichever *other* test happened to run next rather
    than the one at fault.

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
        # Registered under 'sklearn' rather than a name of its own, because
        # TrainingSpec is a discriminated union over the engine names the
        # framework ships and a test cannot invent a fourth tag. The fit is
        # closed-form least squares, so 'sklearn' is also the honest label for
        # what this engine does.
        register_engine(ENGINE_TAG)(SyntheticEngine)
        yield


@pytest.fixture
def dataset(tmp_path):
    """
    Write a small exactly-linear dataset to a CSV file.

    Exactly linear, with no noise, because the engine solves least squares
    exactly -- so a correct pipeline must report an r-squared of one, and any
    misalignment anywhere in the chain makes that impossible to fake.

    Returns
    -------
    pathlib.Path
        The file.
    """
    rng = np.random.default_rng(11)
    features = rng.normal(size=(400, N_FEATURES))
    targets = features @ TRUE_COEFFICIENTS + TRUE_INTERCEPT

    path = tmp_path / "linear.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"f{index}" for index in range(N_FEATURES)] + ["target"])
        writer.writerows(np.column_stack([features, targets]).tolist())
    return path


class SyntheticTabularModel(TabularModel):
    """A model definition needing one line of data code, as advertised."""

    component_name = "synthetic_tabular"
    component_engine = ENGINE_TAG

    def data_module(self, spec):
        """Return the standard tabular data module."""
        return TabularDataModule()

    def build_model(self, spec, signature):
        """Return an unmaterialised linear model."""
        del spec
        return LinearModel(n_features=int(signature.dynamic["features"].shape[-1]))


def make_spec(dataset, **overrides):
    """
    Build a run spec pointing at a dataset and the synthetic engine.

    Parameters
    ----------
    dataset
        Path to the CSV file.
    **overrides
        Fields to replace.

    Returns
    -------
    SupervisedRunSpec
        The spec.
    """
    from src.rade_qnet.core.spec.run import SupervisedRunSpec  # noqa: PLC0415

    fields = {
        "model": "synthetic_tabular",
        "source": TabularSourceSpec(path=dataset),
        "training": {"engine": ENGINE_TAG},
        "reports": {"enabled": ()},
    }
    fields.update(overrides)
    return SupervisedRunSpec.model_validate(fields)


@pytest.fixture
def pipeline(tmp_path, dataset):
    """
    Provide a pipeline wired to the synthetic engine and dataset.

    Returns
    -------
    TrainPipeline
        The pipeline.
    """
    return TrainPipeline(
        context=make_run_context(output_directory=tmp_path / "run"),
        spec=make_spec(dataset),
        definition=SyntheticTabularModel(),
    )


class TestStageSequence:
    """The canonical order, and the four orderings that carry meaning."""

    def test_the_declared_stages_are_the_full_sequence(self, pipeline):
        """
        Declared as data, so a hook or a report can enumerate them.

        A sequence that existed only in the body of ``run`` could not be
        introspected before the run started.
        """
        assert pipeline.stages == (
            "resolve",
            "resolve_seed",
            "build_data",
            "declare_signature",
            "build_model",
            "materialise",
            "prepare_hardware",
            "fit",
            "evaluate",
            "persist",
            "report",
        )

    def test_every_stage_runs(self, pipeline):
        """The sequence completes, which is the Phase 2 deliverable."""
        pipeline.execute()
        assert pipeline.completed_stages == pipeline.stages

    def test_resolution_precedes_the_data_build(self, tmp_path, dataset):
        """
        So a misspelled component name costs milliseconds, not a full build.

        Finding an unregistered engine after a twenty-minute data build wastes
        the twenty minutes, and the information was available before it
        started.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("resolve") < started.index("build_data")

    def test_seeding_precedes_the_data_build(self, tmp_path, dataset):
        """
        The seed is applied before anything that could consume randomness.

        A data build that samples is part of what a seed must reproduce -- a
        subgraph, a negative set, a shuffled split. Seeding afterwards makes a
        run reproducible in its training and not in its data, which is the
        harder half to notice.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("resolve_seed") < started.index("build_data")

    def test_materialisation_precedes_hardware_preparation(self, tmp_path, dataset):
        """
        Defect 6, asserted on the sequence rather than on its consequences.

        A lazily shaped model has no parameters until it has seen a batch, and
        an optimiser built over the empty set reports success and updates
        nothing. The stage order is the fix, so the stage order is the test.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("materialise") < started.index("prepare_hardware")

    def test_reporting_follows_persistence(self, tmp_path, dataset):
        """
        A report reads a finished bundle, so it cannot run before one exists.

        And because a report never fails a run, running it last means a report
        fault cannot cost a trained model.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset, reports={"enabled": ("summary",)}),
            definition=SyntheticTabularModel(),
        )
        pipeline.execute()
        started = [event[1] for event in hook.events if event[0] == "stage_start"]
        assert started.index("persist") < started.index("report")


class TestResolution:
    """Unresolvable names fail first, and say which name."""

    def test_an_unregistered_engine_fails_at_resolve(self, tmp_path, dataset):
        """
        Named, so the reader knows what to register.

        A generic failure would send them to the source to find out what the
        spec asked for.
        """
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )
        # The engine is removed rather than misnamed in the spec, because the
        # spec's discriminated union rejects an unknown engine tag before the
        # pipeline ever sees it. So the only way to reach the resolve stage
        # with an unresolvable engine is an empty registry -- which is also
        # the real-world case: a host that never imported the engine package.
        del ENGINES._entries[ENGINE_TAG]

        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "resolve"
        assert ENGINE_TAG in str(caught.value)

    def test_an_unregistered_report_fails_at_resolve(self, tmp_path, dataset):
        """
        Before the data build, not after the training run.

        A report name is checked by instantiating it, which is what turns a
        typo into an immediate failure rather than a warning logged once
        training has already finished.
        """
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("no_such_report",)}),
            definition=SyntheticTabularModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "resolve"
        assert "no_such_report" in str(caught.value)


class TestTheNumbers:
    """The point of an exactly solvable problem: assert on the metrics."""

    def test_an_exactly_linear_target_is_recovered(self, pipeline):
        """
        Every split scores an r-squared of one, to numerical precision.

        This single assertion covers an improbable amount of the pipeline. It
        can only hold if the state was fitted on the training rows, inverted
        correctly, the splits are disjoint, the engine trained the caller's
        model, and predictions are paired with their own targets. Any one of
        those being wrong makes it unreachable.
        """
        result = pipeline.execute()
        for split in ("train", "validation", "test"):
            assert result.evaluations[split].metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_the_training_split_is_scored_too(self, pipeline):
        """
        Not redundant: it is how overfitting becomes visible.

        A model scoring well on train and badly on validation has overfitted;
        one scoring badly on both has underfitted. Without the training score
        the two are indistinguishable in the saved bundle.
        """
        result = pipeline.execute()
        assert "train" in result.evaluations

    def test_metrics_are_reported_in_original_units(self, pipeline):
        """
        A mean absolute error in standardised space is not actionable.

        And it is incomparable with another run's, which was scaled by a
        different training mean.
        """
        result = pipeline.execute()
        assert all(evaluation.in_original_units for evaluation in result.evaluations.values())

    def test_a_scaled_target_gives_the_same_errors_as_an_unscaled_one(self, tmp_path, dataset):
        """
        The verification the ``in_original_units`` flag cannot provide.

        That flag is self-reported: it would stay true while the values
        underneath were in standard deviations. The r-squared test above
        cannot catch it either, because r-squared is scale-invariant -- it is
        1.0 whether or not the inversion happened, which is exactly why a
        units bug here would survive a green suite.

        ``mae`` and ``rmse`` are not scale-invariant. Scaling the target
        divides them by its training standard deviation, so running the same
        data once with ``scale_target`` on and once off must produce the same
        errors if and only if both predictions and targets are inverted
        before scoring. Inverting only one of the two would be far worse than
        inverting neither, and this catches that case as well.
        """
        errors = {}
        for scale_target in (False, True):
            spec = make_spec(
                dataset,
                source=TabularSourceSpec(
                    path=dataset,
                    transforms=TransformsSpec(
                        scaling=ScalingSpec(method="standard", scale_target=scale_target)
                    ),
                ),
            )
            result = TrainPipeline(
                context=make_run_context(output_directory=tmp_path / f"scaled_{scale_target}"),
                spec=spec,
                definition=SyntheticTabularModel(),
            ).execute()
            errors[scale_target] = result.evaluations["test"].metrics

        # The target's own scale, which is what a missing inversion would
        # divide the errors by. Asserted to be far from one, so the comparison
        # below could not pass by coincidence on a target that happened to be
        # standardised already.
        raw = np.loadtxt(dataset, delimiter=",", skiprows=1)
        assert raw[:, -1].std() > 2.0

        for metric in ("mae", "rmse"):
            assert errors[True][metric] == pytest.approx(errors[False][metric], abs=1e-8)

    def test_every_split_is_scored_against_a_baseline(self, pipeline):
        """
        So no headline number is reported without a reference point.

        A mean absolute error of 0.03 is either excellent or embarrassing, and
        only the baseline says which.
        """
        result = pipeline.execute()
        assert all(evaluation.baseline_metrics for evaluation in result.evaluations.values())

    def test_the_sample_counts_match_the_splits(self, pipeline):
        """
        A metric computed over the wrong number of rows is a wrong metric.

        Checked against the lineage rather than against a literal, so the test
        does not encode the split fractions. At a sequence length of one there
        is no boundary gap, so every scenario belongs to exactly one split and
        the counts must account for all of them.
        """
        result = pipeline.execute()
        scored = {name: value.n_samples for name, value in result.evaluations.items()}
        bundle = pipeline.build_data(pipeline.spec)
        assert scored == bundle.lineage.split_sizes
        assert sum(scored.values()) == bundle.lineage.n_scenarios


class TestScoringAlignment:
    """The defect this stage had, and the guard that now prevents it."""

    def test_a_shuffling_source_is_scored_in_a_stable_order(self, pipeline):
        """
        The bug Phase 2 found: two passes over a training source disagree.

        Scoring takes two passes -- one for the forward pass, one for the
        targets -- and pairs them row for row. A training source reshuffles
        between passes, so pairing them compares each prediction against an
        unrelated target. The counts still match and every metric still
        computes, so the only symptom is a model that appears to score at
        random on the very split it was trained on.

        Asserted through the training r-squared, because that is where the
        symptom appeared.
        """
        result = pipeline.execute()
        assert result.evaluations["train"].metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_the_ordered_view_has_the_same_samples(self, pipeline):
        """
        A stable order must not be achieved by dropping or adding rows.

        Otherwise the fix for the alignment bug would introduce a sampling
        bug, which is harder to see.
        """
        pipeline.execute()
        data = pipeline.build_data(pipeline.spec)
        shuffled = source_for(data, "train")
        ordered = scoring_source(data, "train")
        assert ordered.n_samples == shuffled.n_samples
        assert sorted(_targets_of(ordered)) == pytest.approx(sorted(_targets_of(shuffled)))

    def test_a_source_that_cannot_be_ordered_is_refused(self, tmp_path, dataset):
        """
        A plausible wrong number is worse than no number.

        A source whose order varies and which offers no ordered view is
        refused rather than scored, because the metrics would be computed
        against mismatched targets and nothing in them would look wrong.
        """

        class Reshuffling:
            """Reshuffles every pass and offers no ordered view."""

            def __init__(self) -> None:
                self._pass = 0

            @property
            def signature(self):
                """The declared interface."""
                return make_signature(n_features=N_FEATURES)

            @property
            def static(self):
                """No static inputs."""
                return {}

            @property
            def steps_per_epoch(self):
                """Two batches per pass."""
                return 2

            @property
            def n_samples(self):
                """Twenty samples."""
                return 20

            def batches(self):
                """Yield a differently ordered pass each time."""
                self._pass += 1
                order = np.random.default_rng(self._pass).permutation(20)
                for start in (0, 10):
                    rows = order[start : start + 10]
                    yield {
                        "features": np.zeros((10, N_FEATURES)),
                        "target": rows.astype(np.float64),
                    }

        class ShufflingModel(SyntheticTabularModel):
            """Replaces the training split with a source that cannot be ordered."""

            def build_data(self, spec):
                """Return a bundle whose training split reshuffles."""
                bundle = super().build_data(spec)
                source = Reshuffling()
                splits = dict(bundle.splits)
                splits["train"] = TensorBatchData(loader=source, n_samples=20, n_batches=2)
                return DataBundle(
                    splits=splits,
                    signature=bundle.signature,
                    state=bundle.state,
                    lineage=bundle.lineage,
                )

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=ShufflingModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert "OrderedSource" in str(caught.value)


class TestPersistence:
    """What lands on disk, and what does not when a run fails."""

    def test_the_bundle_is_written(self, pipeline):
        """Every file a bundle needs to be reopened without the data."""
        pipeline.execute()
        written = {
            path.name for path in pipeline.context.bundles_directory.rglob("*") if path.is_file()
        }
        assert {
            "manifest.json",
            "signature.json",
            "lineage.json",
            "result.json",
            "spec.json",
        } <= written

    def test_the_weights_are_written_through_the_engine(self, pipeline):
        """
        The seam that keeps ``storage`` free of any engine import.

        Storage decides where and when; the engine decides how.
        """
        pipeline.execute()
        weights = list(pipeline.context.bundles_directory.rglob("weights.*"))
        assert weights

    def test_a_reloaded_bundle_predicts_identically(self, pipeline):
        """
        The whole purpose of persisting a model, asserted end to end.

        Every other test in this class checks that a *file* appeared. None of
        them checks that the file means anything. A bundle whose weights were
        written from a stale reference, or reloaded into a differently shaped
        model, produces a complete and well-formed bundle that predicts
        something else -- and the discrepancy only surfaces in production,
        against a model nobody can reproduce.

        Identity is asserted exactly rather than approximately. Weights
        round-trip through the file without arithmetic, so anything other
        than bit-for-bit equality means a dtype was narrowed or a tensor was
        reordered, and both are bugs rather than tolerances.
        """
        bundle = _execute_and_persist(pipeline)
        data = pipeline.build_data(pipeline.spec)
        # The same ordered view scoring uses, so the two passes below are
        # comparable row for row rather than merely the same multiset.
        source = scoring_source(data, "test")
        engine = pipeline._engine()

        def predict_with(model):
            """Prepare a model the way the pipeline does, then predict."""
            return engine.predict(
                engine.prepare(
                    model,
                    hardware=pipeline.spec.hardware,
                    training=pipeline.spec.training,
                ),
                source,
            )

        before = predict_with(bundle.model)

        # Reopened from disk, and the model rebuilt from the saved signature:
        # this is the path a consumer who never saw the training run has to
        # take, so it is the one worth testing.
        manifest_path = next(pipeline.context.bundles_directory.rglob("manifest.json"))
        saved = open_bundle(manifest_path.parent)
        rebuilt = engine.materialise(
            pipeline.definition.build_model(pipeline.spec, load_signature(saved)),
            bundle.signature,
        )

        # Before loading, so the assertion below cannot pass on a weights file
        # that was never read. An untrained model predicting the same thing as
        # a trained one would make the round-trip test vacuous, and that is
        # precisely the state a silently skipped `load_weights` leaves.
        assert not np.array_equal(before, predict_with(rebuilt))

        after = predict_with(engine.load_weights(rebuilt, saved.weights_path))
        assert np.array_equal(before, after)

    def test_quality_metrics_are_recorded_in_the_lineage(self, pipeline):
        """
        Captured at training time, because afterwards they cannot be.

        Once the run is over the dataset may not be reconstructable, so
        numbers describing it are recorded in the bundle or lost.
        """
        bundle = _execute_and_persist(pipeline)
        assert "feature_completeness" in bundle.lineage.quality

    def test_the_catalog_is_updated_only_on_success(self, tmp_path, dataset):
        """
        A run that failed leaves nothing registered.

        Otherwise the catalog advertises a bundle that is not there, and the
        failure surfaces as a missing file in whatever reads it next.
        """
        catalog = InMemoryCatalog()
        succeeding = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "ok", catalog=catalog),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )
        succeeding.execute()
        # `entries` rather than `next_version`, because the latter *reserves* a
        # version rather than reporting one -- reading it would advance the
        # counter and the assertion would be measuring the test.
        registered = len(catalog.entries())

        failing = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "bad", catalog=catalog),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )

        def explode(handle, data):
            """Fail before anything is written."""
            del handle, data
            message = "no"
            raise RuntimeError(message)

        failing.fit = explode
        with pytest.raises(StageError):
            failing.execute()
        assert len(catalog.entries()) == registered == 1


class TestReports:
    """Enabled declaratively, and never load-bearing."""

    def test_enabled_reports_are_rendered(self, tmp_path, dataset):
        """Enabling a report is a specification change, not a code change."""
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("summary", "curves", "quality")}),
            definition=SyntheticTabularModel(),
        )
        pipeline.execute()
        written = {
            path.name for path in pipeline.context.reports_directory.rglob("*") if path.is_file()
        }
        assert {"summary.md", "training_history.csv", "data_quality.md"} <= written

    def test_a_failing_report_does_not_fail_the_run(self, tmp_path, dataset):
        """
        The property that makes reports safe to enable.

        A completed, scored, persisted training run is not discarded because a
        figure could not be drawn.
        """

        @report("explodes")
        class Explodes(Report):
            """A report that always raises."""

            def render(self, context):
                """Raise."""
                message = "this report is broken"
                raise RuntimeError(message)

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("explodes",)}),
            definition=SyntheticTabularModel(),
        )
        result = pipeline.execute()
        assert result.evaluations["test"].metrics["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_fail_fast_turns_a_report_failure_into_a_run_failure(self, tmp_path, dataset):
        """
        For the case where the report *is* the deliverable.

        Opt-in, because the default must not let a figure discard a model.
        """

        @report("explodes_too")
        class ExplodesToo(Report):
            """A report that always raises."""

            def render(self, context):
                """Raise."""
                message = "this report is broken"
                raise RuntimeError(message)

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("explodes_too",), "fail_fast": True}),
            definition=SyntheticTabularModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "report"


class TestTheReportHook:
    """
    How a model adds a report of its own without overriding a stage.

    Some reports are not optional: a graph model's connectivity diagnostics
    are the only place its largest assumption is visible, and switching
    them off to save a few seconds should not be something a user can do by
    accident. Without this hook the only way to guarantee one is to
    override the ``report`` stage, which means reimplementing the
    fail-fast handling, the ordering and the registration -- for a model
    whose only quarrel with the base is the *list*.
    """

    def test_the_default_is_exactly_what_the_spec_asked_for(self, tmp_path, dataset):
        """
        So adding the hook changed nothing for any existing model.

        The first thing to check about a new seam in a base class.
        """
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ("summary", "curves")}),
            definition=SyntheticTabularModel(),
        )
        assert pipeline.report_names() == ("summary", "curves")

    def test_a_name_the_hook_adds_is_rendered(self, tmp_path, dataset):
        """
        Even though the specification never mentions it.

        This is the behaviour the hook exists for, asserted through a real
        run rather than against the hook's return value, because the
        failure mode worth catching is a stage that reads the spec
        directly and bypasses the hook.
        """

        @report("added_by_the_model")
        class AddedByTheModel(Report):
            """A report a model insists on."""

            def render(self, context):
                """Write a marker file."""
                path = context.directory / "insisted.md"
                path.write_text("rendered", encoding="utf-8")
                return (path,)

        class Insisting(TrainPipeline):
            """A pipeline that appends one report to whatever was asked for."""

            def report_names(self):
                """Return the spec's reports plus this model's own."""
                return (*super().report_names(), "added_by_the_model")

        pipeline = Insisting(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ()}),
            definition=SyntheticTabularModel(),
        )
        pipeline.execute()
        assert (pipeline.context.reports_directory / "insisted.md").is_file()

    def test_a_name_the_hook_adds_is_checked_at_resolve(self, tmp_path, dataset):
        """
        At the first stage, not the last.

        Resolution instantiates every report precisely so that a
        misspelled name costs nothing instead of costing a full training
        run. A hook that added names after that check would reintroduce
        the cost for exactly the reports a model considers mandatory.
        """

        class Insisting(TrainPipeline):
            """A pipeline that insists on a report nobody registered."""

            def report_names(self):
                """Return a name that does not resolve."""
                return (*super().report_names(), "never_registered")

        pipeline = Insisting(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset, reports={"enabled": ()}),
            definition=SyntheticTabularModel(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "resolve"
        assert "never_registered" in str(caught.value)


class TestInstrumentation:
    """Every stage is timed, logged and attributed, finished or not."""

    def test_a_failure_is_attributed_to_its_stage(self, tmp_path, dataset):
        """Not "something raised during training" but "fit raised"."""

        class BrokenFit(SyntheticTabularModel):
            """A definition whose fit stage fails."""

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=BrokenFit(),
        )

        def explode(handle, data):
            """Raise instead of fitting."""
            del handle, data
            message = "no"
            raise RuntimeError(message)

        pipeline.fit = explode
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "fit"
        assert isinstance(caught.value.__cause__, RuntimeError)

    def test_every_stage_that_started_is_timed(self, pipeline):
        """
        Including the one that failed, which is the useful part.

        "Fit failed after four hours" and "fit failed immediately" call for
        completely different investigations, and only the timing distinguishes
        them.
        """
        pipeline.execute()
        assert set(pipeline.timings) == set(pipeline.stages)

    def test_the_run_end_hook_fires_after_a_failure(self, tmp_path, dataset):
        """
        The guarantee that makes cleanup possible.

        A hook that only fired on success could not close a progress bar,
        release a GPU or mark a job failed.
        """
        hook = RecordingHook()
        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run", hooks=(hook,)),
            spec=make_spec(dataset, reports={"enabled": ("missing",)}),
            definition=SyntheticTabularModel(),
        )
        with pytest.raises(StageError):
            pipeline.execute()
        assert hook.events[-1][:1] == ("run_end",)
        assert hook.events[-1][2] is False


class TestCustomisationTiers:
    """Overriding one stage must inherit the rest, instrumentation included."""

    def test_one_stage_can_be_replaced(self, tmp_path, dataset):
        """
        Tier three: subclass, replace one stage, inherit the other ten.

        With their timing, logging and error attribution, which is what makes
        the tier worth having over writing a pipeline from scratch.
        """

        class ScoresTestOnly(TrainPipeline):
            """Replaces only the evaluate stage."""

            def evaluate(self, handle, data):
                """Score the test split and nothing else."""
                full = super().evaluate(handle, data)
                return {"test": full["test"]}

        pipeline = ScoresTestOnly(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=SyntheticTabularModel(),
        )
        result = pipeline.execute()
        assert set(result.evaluations) == {"test"}
        assert pipeline.completed_stages == pipeline.stages

    def test_the_applied_seed_is_recorded(self, pipeline):
        """
        What was applied, not what was requested.

        A job-set member derives a per-job seed, and recording the derived
        value is what lets a single failed job be re-run alone and reproduce.
        """
        result = pipeline.execute()
        assert result.seed == pipeline.context.seed


class TestDelegation:
    """The three stages the framework cannot implement for a model."""

    def test_the_data_build_delegates_to_the_definition(self, pipeline):
        """
        A thin delegation by design.

        The framework does not know how to build any model's data, and a base
        implementation that tried would be overridden by every non-trivial
        model.
        """
        bundle = pipeline.build_data(pipeline.spec)
        assert isinstance(bundle, DataBundle)

    def test_the_signature_delegates_to_the_definition(self, pipeline):
        """So a model may narrow the interface the data offers."""
        bundle = make_tensor_bundle()
        assert pipeline.declare_signature(bundle) == bundle.signature

    def test_a_data_build_with_no_training_split_is_refused(self, tmp_path, dataset):
        """
        Refused by the contract, before any engine sees it.

        "No batches" from inside a training loop does not say that the split
        was missing entirely, which is a different problem with a different
        fix. ``DataBundle`` enforces this on construction, which is why the
        pipeline stage adds no check of its own.
        """

        class NoTrainSplit(SyntheticTabularModel):
            """Drops the training split."""

            def build_data(self, spec):
                """Return a bundle with no training split."""
                bundle = super().build_data(spec)
                splits = {
                    name: payload for name, payload in bundle.splits.items() if name != "train"
                }
                return DataBundle(
                    splits=splits,
                    signature=bundle.signature,
                    state=bundle.state,
                    lineage=bundle.lineage,
                )

        pipeline = TrainPipeline(
            context=make_run_context(output_directory=tmp_path / "run"),
            spec=make_spec(dataset),
            definition=NoTrainSplit(),
        )
        with pytest.raises(StageError) as caught:
            pipeline.execute()
        assert caught.value.stage == "build_data"
        assert "requires a 'train' split" in str(caught.value)


def _targets_of(source):
    """
    Drain one pass of a source's targets.

    Parameters
    ----------
    source
        A bounded source.

    Returns
    -------
    list of float
        The targets.
    """
    return [
        float(value)
        for batch in source.batches()
        for value in np.ravel(np.asarray(batch["target"]))
    ]


def _execute_and_persist(pipeline) -> ModelBundle:
    """
    Run a pipeline and return the bundle its persist stage assembled.

    Parameters
    ----------
    pipeline
        The pipeline to run.

    Returns
    -------
    ModelBundle
        The bundle.
    """
    captured: list[ModelBundle] = []
    original = pipeline.persist

    def capture(handle, data, signature, result):
        """Record the bundle on its way past."""
        bundle = original(handle, data, signature, result)
        captured.append(bundle)
        return bundle

    pipeline.persist = capture
    pipeline.execute()
    return captured[0]
