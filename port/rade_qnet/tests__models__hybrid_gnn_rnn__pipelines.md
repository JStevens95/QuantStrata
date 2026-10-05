# `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 13 | 575 | `460b7d1c67ae53f9` |
| 2 | `test_pipelines_eval.py` | 201 | 8064 | `89d17d7ac49b0874` |
| 3 | `test_pipelines_train.py` | 86 | 3568 | `188d734e5e26bca9` |
| 4 | `test_pipelines_tune.py` | 249 | 9351 | `5bd05794926fdea4` |

---

## 1. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/__init__.py`

575 bytes · SHA-256 `460b7d1c67ae53f9`

```python
"""
Tests for the model's pipeline overrides.

The hybrid model overrides one hook on one stage, which is the whole point
of the four customisation tiers: a model whose only quarrel with the base
pipeline is the list of reports it renders should not have to reimplement a
stage to express that.

What these tests mostly assert is therefore an *absence* -- that no stage is
replaced, that the sequence is unchanged, that the user's own selection
survives. An override that grew beyond this is a finding about the
framework's step granularity rather than about this model.
"""
```

---

## 2. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/test_pipelines_eval.py`

8064 bytes · SHA-256 `89d17d7ac49b0874`

```python
"""Tests for the hybrid model's eval-pipeline override."""

from __future__ import annotations

import numpy as np

from tranql.models.rade.rade_qnet.rade_qnet.core.contract.result import EvalResult, EvaluationResult
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.pipelines.eval import (
    HybridEvalPipeline,
    breakdown_notes,
    per_target_errors,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline


def evaluation(**overrides: object) -> EvaluationResult:
    """
    Build a minimal evaluation result to attach notes to.

    Parameters
    ----------
    **overrides
        Fields to set on the result.

    Returns
    -------
    EvaluationResult
        The result.
    """
    return EvaluationResult(
        evaluations={"test": EvalResult(split="test", metrics={"mae": 1.0}, n_samples=4)},
        model_name="hybrid_gnn_rnn",
        bundle_version=1,
        spec_digest="d" * 12,
        source_fingerprint="f" * 12,
        trained_on_fingerprint="f" * 12,
        **overrides,
    )


class TestThePerTargetArithmetic:
    """`per_target_errors`, which is the whole numerical claim."""

    def test_each_target_is_averaged_over_its_own_scenarios(self) -> None:
        """
        Two targets and three scenarios, with a different error per column.

        The first target is out by one everywhere and the second by
        three, so the per-target answer is (1, 3) while the pooled mean
        absolute error is 2 -- a number describing neither of them.
        """
        predictions = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
        targets = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        errors = per_target_errors(predictions, targets, n_targets=2)
        assert list(errors) == [1.0, 3.0]
        assert np.abs(predictions - targets).mean() == 2.0

    def test_the_flat_arrays_are_read_scenario_major(self) -> None:
        """
        Scenario-major, as ``ravel`` produces from a ``(scenario, target)`` matrix.

        Reading them target-major instead would transpose the breakdown
        and attribute every target's error to a different instrument --
        wrong, and completely invisible in the aggregate.
        """
        matrix = np.array([[0.0, 10.0], [0.0, 10.0]])
        errors = per_target_errors(np.ravel(matrix), np.zeros(matrix.size), n_targets=2)
        assert list(errors) == [0.0, 10.0]

    def test_a_single_target_collapses_to_the_pooled_error(self) -> None:
        """The breakdown of one thing is that thing, which is the sanity check."""
        values = np.array([1.0, -3.0, 2.0])
        errors = per_target_errors(values, np.zeros(3), n_targets=1)
        assert errors[0] == np.abs(values).mean()


class TestTheOverride:
    """What the pipeline does and, importantly, does not change."""

    def test_the_stage_sequence_is_untouched(self) -> None:
        """Two stage bodies are extended; the lifecycle is the framework's."""
        assert HybridEvalPipeline.stages == EvaluatePipeline.stages

    def test_the_framework_s_result_is_returned_unaltered(self) -> None:
        """
        The notes are additive.

        A per-target breakdown that changed a metric would make this
        model's evaluations incomparable with every other model's, which
        is the opposite of the point.
        """
        pipeline = HybridEvalPipeline.__new__(HybridEvalPipeline)
        pipeline._breakdown = {"per_target_worst": "EURUSD 1.5"}

        base = evaluation()
        merged = base.model_copy(update={"notes": {**base.notes, **pipeline._breakdown}})
        assert merged.evaluations == base.evaluations
        assert merged.notes["per_target_worst"] == "EURUSD 1.5"

    def test_existing_notes_are_preserved(self) -> None:
        """Merged into, not replaced, so a base note is not lost."""
        pipeline = HybridEvalPipeline.__new__(HybridEvalPipeline)
        pipeline._breakdown = {"per_target_worst": "EURUSD 1.5"}

        base = evaluation(notes={"source": "mock"})
        merged = base.model_copy(update={"notes": {**base.notes, **pipeline._breakdown}})
        assert merged.notes == {"source": "mock", "per_target_worst": "EURUSD 1.5"}


class TestWhenTheBreakdownCannotBeComputed:
    """It is reported as unavailable, never silently dropped."""

    def test_data_without_entity_identifiers_says_so(self) -> None:
        """
        A model with no entity axis has nothing to break down by.

        Returning an empty mapping would read, in the result, exactly
        like a book whose targets all replicated perfectly.
        """
        notes = breakdown_notes((), np.zeros(4), np.zeros(4))
        assert "unavailable" in notes["per_target"]

    def test_a_count_that_does_not_fold_says_so(self) -> None:
        """
        A prediction count need not divide by the target count.

        A sequence model drops the leading rows of a split, so it often
        does not. Folding anyway would reshape across the boundary and report every
        target's error as a mixture of its neighbours'.
        """
        notes = breakdown_notes(("a", "b", "c"), np.zeros(4), np.zeros(4))
        assert "do not fold into 3 target(s)" in notes["per_target"]

    def test_mismatched_predictions_and_targets_say_so(self) -> None:
        """
        Rather than broadcasting.

        Numpy would happily do so for some shapes, and the result would
        compare each target against the wrong row.
        """
        notes = breakdown_notes(("a", "b"), np.zeros(4), np.zeros(6))
        assert "unavailable" in notes["per_target"]

    def test_a_usable_breakdown_names_the_worst_target(self) -> None:
        """Which is the only thing a reader of this note is looking for."""
        predictions = np.array([0.0, 5.0, 0.0, 5.0])
        notes = breakdown_notes(("good", "bad"), predictions, np.zeros(4))

        assert notes["per_target_worst"].startswith("bad 5")
        assert "across 2 target(s)" in notes["per_target_spread"]

    def test_only_the_worst_few_are_named(self) -> None:
        """
        Long enough to show whether the tail is one instrument or a cluster.

        Short enough to read at a glance, which a twenty-target book
        would not be.
        """
        entities = tuple(f"t{i}" for i in range(20))
        notes = breakdown_notes(entities, np.arange(20.0), np.zeros(20))

        assert notes["per_target_worst"].count(",") == 4


class TestTheBreakdownAgreesWithTheAggregate:
    """The arithmetic tie between the override and the framework."""

    def test_the_mean_of_the_per_target_errors_is_the_pooled_error(self) -> None:
        """
        Which is the invariant that catches the inversion bug.

        A raw forward pass returns scaled values while ``collect_targets``
        returns original units, so comparing them without inverting first
        produces a per-target error an order of magnitude too large --
        large enough to read as a broken model rather than a broken
        comparison. The pooled mean absolute error is computed on inverted
        values, so if the two disagree the breakdown is on the wrong scale.
        """
        rng = np.random.default_rng(0)
        predictions = rng.normal(size=12)
        targets = rng.normal(size=12)

        errors = per_target_errors(predictions, targets, n_targets=3)
        assert errors.mean() == np.abs(predictions - targets).mean()

    def test_the_spread_brackets_the_pooled_error(self) -> None:
        """
        Best below, worst above, which is what makes the note readable.

        A breakdown whose best target is worse than the aggregate is
        arithmetically impossible and means the two were computed from
        different numbers.
        """
        rng = np.random.default_rng(1)
        predictions = rng.normal(size=20)
        targets = rng.normal(size=20)
        pooled = np.abs(predictions - targets).mean()

        errors = per_target_errors(predictions, targets, n_targets=4)
        assert errors.min() <= pooled <= errors.max()
```

---

## 3. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/test_pipelines_train.py`

3568 bytes · SHA-256 `188d734e5e26bca9`

```python
"""Tests for the hybrid model's train-pipeline override."""

from __future__ import annotations

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.components import get_report
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.pipelines.train import (
    HYBRID_REPORTS,
    HybridTrainPipeline,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.train import TrainPipeline

from ..test_register import run_spec


class TestTrainOverride:
    """The tier-2 pipeline override."""

    def test_the_stage_sequence_is_untouched(self) -> None:
        """
        The override extends a hook; it does not replace a step.

        If this ever has to change, that is a finding about the
        framework's step granularity rather than about this model, and
        the Phase 3 notes say to raise it rather than absorb it.
        """
        assert HybridTrainPipeline.stages == TrainPipeline.stages
        overridden = set(vars(HybridTrainPipeline)) & set(TrainPipeline.stages)
        assert not overridden

    def test_the_model_report_is_added_to_whatever_the_spec_asked_for(self) -> None:
        """
        A user narrowing the report set still gets the graph diagnostics.

        The graph is the model's largest assumption and the one least
        visible in any metric, so switching it off to save a few seconds
        should not be something a user can do by accident.
        """
        pipeline = HybridTrainPipeline.__new__(HybridTrainPipeline)
        pipeline.spec = run_spec(enabled=("curves",))
        assert pipeline.report_names() == ("curves", "hybrid_graph")

    def test_naming_it_explicitly_does_not_run_it_twice(self) -> None:
        """
        A duplicate would render twice and overwrite its own output.

        Dropped rather than rejected: a user who asks for something
        already guaranteed has made a harmless redundancy, not an error
        worth failing a run over.
        """
        pipeline = HybridTrainPipeline.__new__(HybridTrainPipeline)
        pipeline.spec = run_spec(enabled=("hybrid_graph", "summary"))
        assert pipeline.report_names() == ("hybrid_graph", "summary")

    def test_the_spec_ordering_is_preserved(self) -> None:
        """
        Appended, not prepended.

        Order is occasionally meaningful: a summary report that lists the
        figures produced has to run after them, so reordering the user's
        selection would break it.
        """
        pipeline = HybridTrainPipeline.__new__(HybridTrainPipeline)
        pipeline.spec = run_spec(enabled=("curves", "summary"))
        assert pipeline.report_names()[:2] == ("curves", "summary")

    def test_the_base_hook_returns_the_spec_unchanged(self) -> None:
        """
        A model that does not override it sees exactly the old behaviour.

        The hook was added to the base for this model's sake, so the
        first thing to check is that it changed nothing for anyone else.
        """
        pipeline = TrainPipeline.__new__(TrainPipeline)
        pipeline.spec = run_spec(enabled=("summary", "curves"))
        assert pipeline.report_names() == ("summary", "curves")

    def test_every_name_the_override_adds_resolves(self) -> None:
        """
        Otherwise the run fails at its first stage, after validating cleanly.

        The pipeline's ``resolve`` stage instantiates each report for
        precisely this reason, and this is the same check one import
        earlier.
        """
        for name in HYBRID_REPORTS:
            assert get_report(name)
```

---

## 4. `tranql/models/rade/rade_qnet/tests/models/hybrid_gnn_rnn/pipelines/test_pipelines_tune.py`

9351 bytes · SHA-256 `5bd05794926fdea4`

```python
"""Tests for the hybrid model's tune-pipeline override."""

from __future__ import annotations

import pytest

from tranql.models.rade.rade_qnet.rade_qnet.core.lifecycle.errors import ContractError, SpecError
from tranql.models.rade.rade_qnet.rade_qnet.core.spec.tune import parse_tune_spec
from tranql.models.rade.rade_qnet.rade_qnet.models.hybrid_gnn_rnn.pipelines.tune import (
    HybridTunePipeline,
    is_buildable,
)
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.pipelines.tune import TunePipeline
from tranql.models.rade.rade_qnet.rade_qnet.orchestration.stages.search import expand

from ..test_register import FIXTURE, run_spec


def variant(**params: object):
    """
    Build a run specification with the model parameters overridden.

    Parameters
    ----------
    **params
        Values to set on the model reference's parameter bag.

    Returns
    -------
    SupervisedRunSpec
        The validated specification.
    """
    spec = run_spec()
    reference = spec.model.model_copy(update={"params": {**spec.model.params, **params}})
    return spec.model_copy(update={"model": reference})


def tune_spec(space: dict[str, object], *, trials: int = 64):
    """
    Build a validated search specification over the hybrid model.

    Parameters
    ----------
    space
        Dotted paths to the values to try.
    trials
        The budget, defaulted high enough that a grid is exhausted.

    Returns
    -------
    TuneSpec
        The validated specification.
    """
    return parse_tune_spec(
        {
            "model": "hybrid_gnn_rnn",
            "base": {
                "task": "supervised",
                "model": {"name": "hybrid_gnn_rnn", "params": {"units": 8}},
                "source": {"kind": "model", "params": {"directory": str(FIXTURE)}},
                "training": {"engine": "torch"},
                "reports": {"enabled": []},
            },
            "space": space,
            "trials": trials,
            "sampler": "grid",
        }
    )


def proposals_for(space: dict[str, object], **kwargs):
    """
    Return what the override proposes for a search space.

    Parameters
    ----------
    space
        Dotted paths to the values to try.
    **kwargs
        Passed to :func:`tune_spec`.

    Returns
    -------
    list of dict
        One flat proposal per surviving trial.
    """
    pipeline = HybridTunePipeline.__new__(HybridTunePipeline)
    pipeline.spec = tune_spec(space, **kwargs)
    return pipeline.propose()


class TestTheFeasibilityTest:
    """`is_buildable`, which previews what the layers will say."""

    def test_a_width_divisible_by_its_heads_is_buildable(self) -> None:
        """The ordinary case, and the one a default configuration hits."""
        assert is_buildable(run_spec())

    def test_the_fusion_block_is_checked(self) -> None:
        """
        Because the fusion reshape is the first one a forward pass reaches.

        Eight features over three heads leaves two features that the head
        reshape would drop, which the layer refuses -- correctly, and too
        late to save the trial.
        """
        assert not is_buildable(variant(fusion_heads=3))

    def test_the_attention_block_is_checked_too(self) -> None:
        """
        A proposal can be feasible for one block and not the other.

        Checking only fusion would admit this configuration and spend the
        trial on it, which is the entire failure this file prevents.
        """
        assert not is_buildable(variant(attention_heads=5))

    def test_per_block_widths_are_read_rather_than_the_shared_one(self) -> None:
        """
        ``units`` is only a default; a block may override it.

        Testing the shared knob instead of the resolved width would call a
        configuration feasible whenever its default happened to divide,
        regardless of the width the block will actually build with.
        """
        assert not is_buildable(variant(fusion_units=9, fusion_heads=2))

    def test_another_model_s_spec_is_not_judged(self) -> None:
        """
        Answering True is right: this function knows of no reason it fails.

        Reached when the pipeline is attached to the wrong definition,
        where refusing every trial would be a confusing way to report a
        wiring mistake.
        """
        spec = run_spec()
        foreign = spec.model.model_copy(update={"name": "some_other_model"})
        assert is_buildable(spec.model_copy(update={"model": foreign}))


class TestTheOverride:
    """What the pipeline does with the framework's proposals."""

    def test_the_stage_sequence_is_untouched(self) -> None:
        """
        ``propose`` is extended, not added to or removed from.

        The override replaces a stage's body, which is a tier-3 override
        and the heaviest kind this model uses. What it must not do is
        change the sequence: a model whose lifecycle differs from the
        framework's is a finding about the framework's step granularity
        rather than about this model.
        """
        assert HybridTunePipeline.stages == TunePipeline.stages

    def test_a_wholly_feasible_space_is_passed_through_unchanged(self) -> None:
        """Nothing is filtered when nothing needs to be."""
        space = {"model.params.units": [8, 16, 32]}
        assert len(proposals_for(space)) == 3

    def test_the_infeasible_cells_of_a_grid_are_dropped(self) -> None:
        """
        Three widths crossed with three head counts is nine cells.

        Written as the explicit surviving set rather than as "fewer than
        nine" so that a change to the divisibility rule fails here, rather
        than quietly altering how much of a budget a search spends.
        """
        space = {
            "model.params.units": [8, 16, 24],
            "model.params.fusion_heads": [1, 3, 4],
        }
        proposals = proposals_for(space)

        feasible = {(p["model.params.units"], p["model.params.fusion_heads"]) for p in proposals}
        assert feasible == {(8, 1), (16, 1), (24, 1), (24, 3), (8, 4), (16, 4), (24, 4)}

    def test_every_surviving_proposal_actually_builds(self) -> None:
        """
        The property the filter exists to guarantee, asserted directly.

        Checked against the feasibility test rather than against a
        recomputed expectation, so this still holds if the rule changes.
        """
        space = {
            "model.params.units": [8, 12, 16, 18],
            "model.params.attention_heads": [1, 2, 3],
        }
        pipeline = HybridTunePipeline.__new__(HybridTunePipeline)
        pipeline.spec = tune_spec(space)

        for index, proposal in enumerate(pipeline.propose()):
            assert is_buildable(pipeline.spec.run_spec_for(expand(proposal), trial=index))

    def test_a_path_the_spec_does_not_have_is_the_framework_s_error(self) -> None:
        """
        Validation runs before filtering, so a typo stays a user error.

        Filtering first would turn a mistyped path into a silently
        smaller search -- defect 7 wearing a different hat.
        """
        with pytest.raises(SpecError, match=r"valid run specification"):
            proposals_for({"no_such_block.no_such_field": [1, 2]})

    def test_a_model_parameter_the_schema_rejects_is_caught_here(self) -> None:
        """
        Which the framework's own check cannot do.

        A run spec carries the model as a name and an untyped bag of
        values; the bag meets this model's schema only at ``resolve``, by
        which point the trial is spent. Parsing the params to test
        feasibility means a mistyped knob is found before the search
        starts rather than sixty-four tracebacks into it.
        """
        with pytest.raises(SpecError, match=r"(?s)do not parse.*no_such_knob"):
            proposals_for({"model.params.no_such_knob": [1, 2]})

    def test_a_space_that_can_never_build_is_refused(self) -> None:
        """
        Because a search with no trials has no winner and no error.

        Unlike a partially-infeasible space, this one is a genuine
        contradiction between what was asked for and what exists, and
        silently returning nothing would surface as a confusing failure
        several stages later.
        """
        space = {
            "model.params.units": [8, 16],
            "model.params.fusion_heads": [3, 5],
        }
        with pytest.raises(ContractError, match=r"none of the .* can be built"):
            proposals_for(space)

    def test_the_refusal_names_the_knobs_involved(self) -> None:
        """So the message is actionable without reading this file."""
        space = {"model.params.units": [8], "model.params.fusion_heads": [3]}
        with pytest.raises(ContractError, match=r"fusion_heads.*attention_heads"):
            proposals_for(space)

    def test_the_base_pipeline_filters_nothing(self) -> None:
        """
        A model that does not override this sees the old behaviour.

        The whole override is additive, so the first thing to check is
        that it changed nothing for anybody else.
        """
        pipeline = TunePipeline.__new__(TunePipeline)
        pipeline.spec = tune_spec({"model.params.units": [8, 16], "model.params.fusion_heads": [3]})
        assert len(pipeline.propose()) == 2
```

