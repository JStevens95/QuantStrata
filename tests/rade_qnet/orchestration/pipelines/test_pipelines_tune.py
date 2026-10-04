"""
Tests for searching a space of configurations.

A search that is subtly wrong still completes and still names a winner, so
most of these tests are about the bookkeeping rather than the optimisation:

- **Defect 7.** An unknown path in the space must fail at construction,
  before anything trains. Left unchecked it is explored, measured and
  reported as a dimension that makes no difference -- which reads as a
  finding rather than a bug.
- **Build once.** Asserted by a call count rather than by timing, because a
  timing assertion measures the machine.
- **Failures are recorded.** A search that drops a failing trial reports the
  best of a partial sweep while looking like a complete one.
- **Selection is deterministic.** Including the tie rule, since a search
  whose winner moves between identical runs cannot be acted on.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.runtime.errors import ContractError, SpecError, StageError
from src.rade_qnet.core.spec.tune import parse_tune_spec
from src.rade_qnet.orchestration.pipelines.search import expand, propose
from src.rade_qnet.orchestration.pipelines.tune import TunePipeline
from src.rade_qnet.testkit.fixtures import isolated_registries, make_run_context

from .support import (
    ENGINE_TAG,
    MODEL_NAME,
    SyntheticSupervisedModel,
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
    Write the training dataset.

    Returns
    -------
    pathlib.Path
        The file.
    """
    return write_linear_dataset(tmp_path / "linear.csv")


def tune_spec(dataset, **overrides):
    """
    Build a search over the synthetic model.

    The default space varies ``seed``, which is a real field of a run
    specification and has no effect on a closed-form least-squares fit. That
    is deliberate: it makes every trial succeed and score identically, so
    the tests about bookkeeping are not entangled with tests about whether
    the search found anything.

    Parameters
    ----------
    dataset
        Path to the CSV file.
    **overrides
        Fields to replace.

    Returns
    -------
    TuneSpec
        The validated search.
    """
    fields = {
        "model": MODEL_NAME,
        "base": {
            "source": {"kind": "tabular", "path": str(dataset)},
            "training": {"engine": ENGINE_TAG},
            "reports": {"enabled": ()},
        },
        "space": {"seed": [1, 2, 3, 4]},
        "trials": 4,
        "sampler": "grid",
        "objective": "mae",
        "direction": "minimise",
    }
    fields.update(overrides)
    return parse_tune_spec(fields)


def pipeline_for(tmp_path, spec, *, name="tune"):
    """
    Build a tuning pipeline over a search.

    Parameters
    ----------
    tmp_path
        Temporary directory for the run's output.
    spec
        The search.
    name
        Subdirectory to write into. Two searches in one test need two,
        because every trial persists a model and a bundle refuses to
        overwrite an existing one -- correctly, and for the same reason a
        training run does.

    Returns
    -------
    TunePipeline
        The pipeline.
    """
    return TunePipeline(
        context=make_run_context(output_directory=tmp_path / name),
        spec=spec,
        definition=SyntheticSupervisedModel(),
    )


class TestTheSearchRuns:
    """The sequence completes and produces what it promises."""

    def test_the_stage_sequence_is_declared(self, tmp_path, dataset):
        """Declared as data, so a hook can enumerate it before the run."""
        pipeline = pipeline_for(tmp_path, tune_spec(dataset))
        pipeline.execute()

        assert pipeline.stages == (
            "resolve",
            "propose",
            "build_data",
            "run_trials",
            "select",
            "refit",
        )
        assert pipeline.completed_stages == pipeline.stages

    def test_every_proposal_becomes_a_trial(self, tmp_path, dataset):
        """Four points, four records, in order."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert len(result.trials) == 4
        assert [record.trial for record in result.trials] == [0, 1, 2, 3]

    def test_the_trials_record_what_was_proposed(self, tmp_path, dataset):
        """
        Flat, so two trials can be read side by side.

        A nested fragment has to be mentally flattened before it can be
        compared against another, which is the only thing anyone does with
        a column of proposals.
        """
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert [record.overrides for record in result.trials] == [
            {"seed": 1},
            {"seed": 2},
            {"seed": 3},
            {"seed": 4},
        ]

    def test_the_winner_carries_its_metrics(self, tmp_path, dataset):
        """So it can be examined on more than the one number it was chosen by."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert result.best.objective is not None
        assert "mae" in result.best.metrics


class TestDataIsBuiltOnce:
    """Forty trials over one dataset should read it once."""

    def test_data_build_is_reused_across_trials(self, tmp_path, dataset):
        """
        Asserted by a call count, not by timing.

        A timing assertion measures the machine; a count measures the
        pipeline.
        """
        pipeline = pipeline_for(tmp_path, tune_spec(dataset, trials=4))
        pipeline.execute()

        assert pipeline.data_builds == 1

    def test_the_same_dataset_object_reaches_every_trial(self, tmp_path, dataset):
        """
        Reuse means the same object, not an equal one.

        An equal rebuild would also pass a count-free test while costing
        exactly what the reuse was meant to save.
        """
        pipeline = pipeline_for(tmp_path, tune_spec(dataset))
        pipeline.execute()

        assert pipeline.shared_data is not None

    def test_a_search_over_the_source_builds_per_trial(self, tmp_path, dataset):
        """
        Because the trials then have genuinely different datasets.

        Sharing one would mean the search measured something other than
        what it proposed, which is the quietest way for a search to be
        wrong: every number is plausible and every one is about the wrong
        configuration.
        """
        other = write_linear_dataset(tmp_path / "other.csv", seed=5)
        spec = tune_spec(
            dataset,
            space={"source.path": [str(dataset), str(other)]},
            trials=2,
        )
        pipeline = pipeline_for(tmp_path, spec)
        pipeline.execute()

        assert pipeline.data_builds == 0
        assert pipeline.shared_data is None


class TestDefectSeven:
    """An unknown override fails before anything trains."""

    def test_unknown_trial_override_fails_at_construction(self, tmp_path, dataset):
        """
        The structural cure.

        ``rade_ml_pt`` passed ``learning_rate`` into
        ``dataclasses.replace(TrainingConfig)`` and raised ``TypeError``; it
        was latent only because the hybrid model overrode the method. Here
        the proposal goes through the same validated merge a job's
        overrides do, so the path fails at trial construction.
        """
        spec = tune_spec(dataset, space={"training.not_a_field": [1, 2]}, trials=2)

        with pytest.raises(StageError, match=r"do not\s+produce a valid run specification"):
            pipeline_for(tmp_path, spec).execute()

    def test_the_failure_names_the_offending_path(self, tmp_path, dataset):
        """
        Because nobody typed the value -- it was generated.

        A message that said only "validation failed" would send the reader
        looking at a proposal they did not write.
        """
        spec = tune_spec(dataset, space={"training.not_a_field": [1, 2]}, trials=2)

        with pytest.raises(StageError, match=r"not_a_field"):
            pipeline_for(tmp_path, spec).execute()

    def test_nothing_trains_before_the_check(self, tmp_path, dataset):
        """
        The point of checking at construction rather than at trial time.

        A space with one bad path breaks every trial, so discovering it on
        the fortieth wastes the thirty-nine before it.
        """
        spec = tune_spec(dataset, space={"training.not_a_field": [1, 2]}, trials=2)
        pipeline = pipeline_for(tmp_path, spec)

        with pytest.raises(StageError):
            pipeline.execute()

        assert pipeline.data_builds == 0


class TestFailingTrials:
    """Recorded and surfaced, never swallowed."""

    def test_a_failed_trial_does_not_end_the_search(self, tmp_path, dataset, monkeypatch):
        """Thirty-nine good trials must survive one bad one."""
        spec = tune_spec(dataset, trials=4)
        pipeline = pipeline_for(tmp_path, spec)

        original = SyntheticSupervisedModel.build_model
        calls = {"n": 0}

        def flaky(self, run_spec, signature):
            """Fail the second trial only."""
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("deliberate trial failure")
            return original(self, run_spec, signature)

        monkeypatch.setattr(SyntheticSupervisedModel, "build_model", flaky)
        result = pipeline.execute()

        assert len(result.trials) == 4
        assert len(result.failed) == 1
        assert result.failed[0].trial == 1

    def test_a_failure_records_its_reason(self, tmp_path, dataset, monkeypatch):
        """So the search report says why, not merely that something went wrong."""
        spec = tune_spec(dataset, trials=2)
        pipeline = pipeline_for(tmp_path, spec)

        def always_fails(self, run_spec, signature):
            """Fail every trial."""
            raise RuntimeError("deliberate trial failure")

        monkeypatch.setattr(SyntheticSupervisedModel, "build_model", always_fails)

        with pytest.raises(StageError, match=r"nothing to\s+select between"):
            pipeline.execute()

    def test_the_summary_states_the_failure_count(self, tmp_path, dataset, monkeypatch):
        """Even when it is zero, because a missing clause is not information."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert "0 failed" in result.describe()


class TestSelection:
    """One rule, stated, applied once."""

    def test_best_trial_selection_is_deterministic(self, tmp_path, dataset):
        """Same seed, same winner, across two independent searches."""
        first = pipeline_for(tmp_path, tune_spec(dataset), name="a").execute()
        second = pipeline_for(tmp_path, tune_spec(dataset), name="b").execute()

        assert first.best_trial == second.best_trial
        assert [record.objective for record in first.trials] == [
            record.objective for record in second.trials
        ]

    def test_ties_go_to_the_earlier_trial(self, dataset):
        """
        Which is what makes the winner stable under a re-run.

        Asserted against the comparison rule itself rather than through a
        search. Contriving a genuine tie between two trained models is
        harder than it sounds -- varying the seed moves the split, so even
        a closed-form fit scores slightly differently -- and a test that
        had to contrive one would be testing the contrivance.
        """
        spec = tune_spec(dataset)

        assert spec.is_better(0.5, 0.6) is True
        assert spec.is_better(0.5, 0.5) is False
        assert spec.is_better(0.6, 0.5) is False

    def test_the_tie_rule_reverses_with_the_direction(self, dataset):
        """One rule, applied once, honouring whichever way the metric runs."""
        spec = tune_spec(dataset, objective="r2", direction="maximise")

        assert spec.is_better(0.6, 0.5) is True
        assert spec.is_better(0.5, 0.5) is False

    def test_the_direction_is_honoured(self, tmp_path, dataset):
        """
        Taken from the specification rather than guessed from the metric.

        ``r2`` is maximised and ``mae`` minimised; a framework that inferred
        the direction from a substring would eventually select the worst
        trial in a search over a custom metric.
        """
        maximised = tune_spec(dataset, objective="r2", direction="maximise")
        result = pipeline_for(tmp_path, maximised).execute()

        assert result.direction == "maximise"
        assert result.objective == "r2"

    def test_the_result_records_which_split_decided(self, tmp_path, dataset):
        """Because a winner chosen on test is not a held-out estimate."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()

        assert result.objective_split == "validation"

    def test_refit_is_refused_rather_than_skipped(self, tmp_path, dataset):
        """
        A user who asked for it must not receive a model trained without it.

        Silently ignoring the flag would hand back a model fitted on the
        training split alone, with no way for the caller to tell.
        """
        spec = tune_spec(dataset, refit=True)

        with pytest.raises(StageError, match=r"refit is not implemented"):
            pipeline_for(tmp_path, spec).execute()


class TestTheSearchSpace:
    """Properties of the space itself, cheap to check without training."""

    def test_a_space_cannot_vary_one_path_twice(self):
        """
        Two axes over one path make the earlier one inert.

        The later would win every merge, so half the budget explores a
        dimension with no effect -- and the report would show it having
        none, which reads as a finding rather than a bug.
        """
        with pytest.raises(SpecError, match=r"must not vary the same path twice"):
            parse_tune_spec(
                {
                    "base": {},
                    "space": {"dimensions": [{"path": "seed", "values": [1]}] * 2},
                }
            )

    def test_an_axis_must_be_an_enumeration_or_a_range(self):
        """Not both, and not neither."""
        with pytest.raises(SpecError, match=r"must give either 'values'"):
            parse_tune_spec(
                {"base": {}, "space": {"seed": {"values": [1], "low": 0.0, "high": 1.0}}}
            )

    def test_an_empty_range_is_rejected(self):
        """Because every trial would propose the same value."""
        with pytest.raises(SpecError, match=r"the range is empty"):
            parse_tune_spec({"base": {}, "space": {"x": {"low": 1.0, "high": 1.0}}})

    def test_a_log_scale_below_zero_is_rejected(self):
        """A log scale is undefined there, and would produce NaN proposals."""
        with pytest.raises(SpecError, match=r"undefined at or below zero"):
            parse_tune_spec({"base": {}, "space": {"x": {"low": 0.0, "high": 1.0, "log": True}}})

    def test_a_grid_cannot_enumerate_a_range(self):
        """
        A grid needs values, not bounds.

        Discretising a range into steps the user did not choose would make
        the search coarser than it reads, and the step count would be a
        framework decision affecting the result.
        """
        with pytest.raises(SpecError, match=r"cannot enumerate the continuous"):
            parse_tune_spec(
                {
                    "base": {},
                    "space": {"x": {"low": 0.1, "high": 1.0}},
                    "sampler": "grid",
                }
            )


class TestProposals:
    """The sampler, tested without training anything."""

    def test_random_proposals_are_reproducible(self):
        """Same seed, same sequence."""
        spec = parse_tune_spec(
            {"base": {}, "space": {"x": {"low": 0.1, "high": 1.0}}, "trials": 8, "seed": 3}
        )

        assert propose(spec) == propose(spec)

    def test_a_different_seed_explores_differently(self):
        """Otherwise the seed would be decoration."""
        space = {"base": {}, "space": {"x": {"low": 0.1, "high": 1.0}}, "trials": 8}

        assert propose(parse_tune_spec({**space, "seed": 1})) != propose(
            parse_tune_spec({**space, "seed": 2})
        )

    def test_continuous_proposals_stay_in_bounds(self):
        """A proposal outside the range is a value the user excluded."""
        spec = parse_tune_spec(
            {"base": {}, "space": {"x": {"low": 0.2, "high": 0.8}}, "trials": 50}
        )

        assert all(0.2 <= proposal["x"] <= 0.8 for proposal in propose(spec))

    def test_log_proposals_stay_in_bounds(self):
        """Including under the exponent transform, which is easy to get wrong."""
        spec = parse_tune_spec(
            {
                "base": {},
                "space": {"x": {"low": 1e-4, "high": 1e-1, "log": True}},
                "trials": 50,
            }
        )

        assert all(1e-4 <= proposal["x"] <= 1e-1 for proposal in propose(spec))

    def test_enumerated_values_keep_their_python_type(self):
        """
        An integer axis must not come back as ``numpy.int64``.

        It would be merged into a specification and fail validation for a
        reason that has nothing to do with what the user wrote.
        """
        spec = parse_tune_spec({"base": {}, "space": {"n": [1, 2, 3]}, "trials": 10})

        assert all(type(proposal["n"]) is int for proposal in propose(spec))

    def test_a_grid_enumerates_the_product(self):
        """Two axes of two and three values give six points."""
        spec = parse_tune_spec(
            {
                "base": {},
                "space": {"a": [1, 2], "b": ["x", "y", "z"]},
                "sampler": "grid",
                "trials": 100,
            }
        )
        proposals = propose(spec)

        assert len(proposals) == 6
        assert len({tuple(sorted(point.items())) for point in proposals}) == 6

    def test_a_grid_is_truncated_to_the_budget(self):
        """And the pipeline logs that the reported best is of a partial grid."""
        spec = parse_tune_spec(
            {
                "base": {},
                "space": {"a": [1, 2], "b": ["x", "y", "z"]},
                "sampler": "grid",
                "trials": 4,
            }
        )

        assert len(propose(spec)) == 4

    def test_dotted_paths_expand_to_nested_mappings(self):
        """Which is the shape a specification is validated in."""
        assert expand({"training.learning_rate": 0.01, "seed": 3}) == {
            "training": {"learning_rate": 0.01},
            "seed": 3,
        }

    def test_a_path_cannot_be_both_a_value_and_a_mapping(self):
        """
        A path cannot be a leaf in one axis and a branch in another.

        One would overwrite the other depending on iteration order, so the
        search would explore a different space on a different day.
        """
        with pytest.raises(SpecError, match=r"Vary one or the other"):
            expand({"training": 1, "training.learning_rate": 0.01})


class TestResolution:
    """Finding the model, and failing usefully when it cannot be found."""

    def test_a_search_can_resolve_its_model_by_name(self, tmp_path, dataset):
        """As a configuration file would, with no definition passed in."""
        pipeline = TunePipeline(
            context=make_run_context(output_directory=tmp_path / "tune"),
            spec=tune_spec(dataset),
        )
        result = pipeline.execute()

        assert len(result.trials) == 4

    def test_an_unnamed_model_is_refused(self, tmp_path, dataset):
        """A search has to know what it is searching over."""
        spec = parse_tune_spec(
            {
                "base": {
                    "source": {"kind": "tabular", "path": str(dataset)},
                    "training": {"engine": ENGINE_TAG},
                },
                "space": {"seed": [1]},
                "trials": 1,
            }
        )
        pipeline = TunePipeline(
            context=make_run_context(output_directory=tmp_path / "tune"), spec=spec
        )

        with pytest.raises(StageError, match=r"names no model"):
            pipeline.execute()

    def test_an_unregistered_model_names_the_problem(self, tmp_path, dataset):
        """So the reader looks at their imports rather than at the search."""
        spec = tune_spec(dataset, model="not_registered")
        pipeline = TunePipeline(
            context=make_run_context(output_directory=tmp_path / "tune"), spec=spec
        )

        with pytest.raises(StageError, match=r"not registered in this process"):
            pipeline.execute()


class TestTheResultContract:
    """Reading a search afterwards."""

    def test_the_summary_names_the_winner_and_the_direction(self, tmp_path, dataset):
        """A column of objective values cannot be read without the direction."""
        result = pipeline_for(tmp_path, tune_spec(dataset)).execute()
        summary = result.describe()

        assert f"#{result.best_trial}" in summary
        assert "minimise" in summary
        assert "4 trial(s)" in summary

    def test_a_best_trial_that_is_not_present_is_refused(self):
        """Rather than returning the first trial as though it had won."""
        from src.rade_qnet.core.contract.result import TrialRecord, TuningResult  # noqa: PLC0415

        result = TuningResult(trials=(TrialRecord(trial=0),), best_trial=7)

        with pytest.raises(ContractError, match=r"none with that number"):
            _ = result.best
