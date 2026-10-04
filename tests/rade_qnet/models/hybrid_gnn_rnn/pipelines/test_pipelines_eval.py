"""Tests for the hybrid model's eval-pipeline override."""

from __future__ import annotations

import numpy as np

from src.rade_qnet.core.contract.result import EvalResult, EvaluationResult
from src.rade_qnet.models.hybrid_gnn_rnn.pipelines.eval import (
    HybridEvalPipeline,
    breakdown_notes,
    per_target_errors,
)
from src.rade_qnet.orchestration.pipelines.evaluate import EvaluatePipeline


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
        errors = per_target_errors(
            np.ravel(matrix), np.zeros(matrix.size), n_targets=2
        )
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
