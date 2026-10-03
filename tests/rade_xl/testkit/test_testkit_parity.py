"""
Tests for the parity harness.

The harness decides whether the refactor is correct, so it has to be correct
itself -- and the dangerous failure is the permissive one. A suite that
passes everything provides false assurance and is worse than no suite at all,
because the refactor then ships with a verdict nobody rechecks. Every claim
here is therefore tested in both directions: that a match passes, *and* that
the corresponding mismatch fails.

The reordered-basis test is the one with real history behind it. Basis
selection returns an ordered list and that order fixes column positions in
every array downstream, so a refactor selecting the same instruments in a
different order passes a set comparison and then fails everything after it,
with the symptom nowhere near the cause.

What is deliberately *not* tested here: the fixture's contents. A fixture is
data, not code. Its correctness comes from the capture script having run
against unmodified ``rade_ml_pt``, and is recorded in ``manifest.json`` with
the source commit.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.rade_xl.core.runtime.errors import ContractError
from src.rade_xl.testkit.parity import (
    Comparison,
    ParityReport,
    compare_arrays,
    compare_curve,
    compare_forward,
    compare_job_set,
    compare_state,
    compare_tensors,
    load_golden,
)


@pytest.fixture
def golden(tmp_path):
    """
    Write a miniature fixture with one artifact of each kind.

    Small and synthetic on purpose: these tests are about the comparison
    logic, so a real captured fixture would only make them slower and couple
    them to the model.

    Returns
    -------
    GoldenFixture
        The loaded fixture.
    """
    directory = tmp_path / "toy"
    (directory / "level1_state").mkdir(parents=True)
    (directory / "level2_tensors").mkdir()
    (directory / "level3_forward").mkdir()
    (directory / "level4_training").mkdir()

    (directory / "manifest.json").write_text(
        json.dumps({"captured_at": "2026-01-01T00:00:00Z", "source_commit": "abc123"}),
        encoding="utf-8",
    )
    np.save(directory / "level1_state/scaler_mean.npy", np.array([1.0, 2.0, 3.0]))
    np.save(directory / "level1_state/combined_features.npy", np.eye(3))
    np.save(directory / "level1_state/elementary_idx.npy", np.array([0, 1, 2]))
    (directory / "level1_state/selected_basis.json").write_text(
        json.dumps(["eur", "gbp", "usd"]), encoding="utf-8"
    )
    (directory / "level1_state/universe.json").write_text(
        json.dumps({"elementary_ids": ["a", "b"], "target_ids": ["t1"]}), encoding="utf-8"
    )
    np.savez(
        directory / "level2_tensors/train_batch_000.npz",
        pnl_history=np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        target=np.array([[1.0], [2.0]], dtype=np.float32),
    )
    (directory / "level2_tensors/split_indices.json").write_text(
        json.dumps({"train": [0, 1, 2], "test": [3, 4]}), encoding="utf-8"
    )
    np.save(directory / "level3_forward/outputs.npy", np.array([0.5, -0.25]))
    (directory / "level4_training/curve.json").write_text(
        json.dumps({"train_loss": [1.0, 0.5, 0.25], "val_loss": [1.1, 0.6, 0.3]}),
        encoding="utf-8",
    )
    return load_golden("toy", root=tmp_path)


class TestCompareArrays:
    """The diagnostic that turns a day of bisection into a line of output."""

    def test_identical_arrays_pass(self):
        """
        Comparing an array with itself passes, including at exact tolerance.

        The baseline claim. If this failed, every parity level would fail
        and the harness would be the thing under suspicion.
        """
        values = np.array([1.0, 2.0, 3.0])
        assert compare_arrays(values, values.copy(), name="x").passed

    def test_a_perturbation_above_tolerance_fails(self):
        """
        Because a harness that tolerates everything proves nothing.

        This is the direction that matters: a suite can only be trusted if
        it is known to reject something.
        """
        expected = np.array([1.0, 2.0, 3.0])
        actual = expected + np.array([0.0, 1e-3, 0.0])
        assert not compare_arrays(actual, expected, name="x", atol=1e-6).passed

    def test_a_perturbation_below_tolerance_passes(self):
        """
        So a reassociated but identical computation is not reported as a bug.

        Level 3 exists because a fused kernel perturbs the last bits without
        changing the arithmetic, and a harness that failed on that would be
        unusable.
        """
        expected = np.array([1.0, 2.0, 3.0])
        actual = expected + np.array([0.0, 1e-9, 0.0])
        assert compare_arrays(actual, expected, name="x", atol=1e-6).passed

    def test_exact_is_the_default(self):
        """
        Because levels 1, 2 and 5 have no floating-point excuse available.

        A default tolerance would silently weaken all three, and the weakest
        link in a parity suite is the one nobody specified.
        """
        expected = np.array([1.0])
        assert not compare_arrays(expected + 1e-12, expected, name="x").passed

    def test_a_shape_mismatch_reports_both_shapes(self):
        """
        Reported as a shape difference, not as a value difference.

        A transpose reported as "1,200 elements differ" sends the reader
        hunting for a numerical bug. Naming both shapes gives the cause
        directly.
        """
        result = compare_arrays(np.zeros((3, 2)), np.zeros((2, 3)), name="x")
        assert not result.passed
        assert "(3, 2)" in result.detail
        assert "(2, 3)" in result.detail

    def test_a_dtype_mismatch_is_reported_rather_than_upcast(self):
        """
        Because a narrowed float and a floated index are both real bugs.

        Silently upcasting float32 to float64 would hide a precision loss
        that shows up much later, and an index array that became
        floating-point is a construction error worth failing on.
        """
        result = compare_arrays(
            np.array([1.0], dtype=np.float32), np.array([1.0], dtype=np.float64), name="x"
        )
        assert not result.passed
        assert "float32" in result.detail
        assert "float64" in result.detail

    def test_the_report_locates_the_worst_element(self):
        """
        Name, index, both values and the mismatch count.

        "Arrays differ" wastes a day. "element 2: got 9.0, expected 3.0,
        1 of 4 mismatched" locates the bug immediately, which is the entire
        reason this function is more than ``np.allclose``.
        """
        expected = np.array([1.0, 2.0, 3.0, 4.0])
        actual = np.array([1.0, 2.0, 9.0, 4.0])
        result = compare_arrays(actual, expected, name="combined_features")

        assert "combined_features" in result.detail
        assert "9.0" in result.detail
        assert "3.0" in result.detail
        assert result.n_mismatched == 1
        assert result.n_total == 4

    def test_the_worst_element_is_the_worst_mismatching_one(self):
        """
        Not the largest absolute difference overall.

        Under a relative tolerance a large difference on a large value can
        pass while a small one on a small value fails. Reporting the former
        would point at the element that was fine.
        """
        expected = np.array([1000.0, 1.0])
        actual = np.array([1001.0, 1.1])
        result = compare_arrays(actual, expected, name="x", rtol=1e-2)

        # The 1.0 difference passes (0.1% of 1000); the 0.1 difference fails
        # (10% of 1.0), so index 1 is what must be reported.
        assert not result.passed
        assert "index 1" in result.detail

    def test_two_nans_in_the_same_place_match(self):
        """
        A fixture that recorded NaN and a run that reproduces it have agreed.

        Treating NaN as always-unequal would make any fixture containing one
        permanently unmatchable, and NaN is a legitimate captured value for
        an instrument with no history in a window.
        """
        values = np.array([1.0, np.nan, 3.0])
        assert compare_arrays(values.copy(), values, name="x").passed

    def test_a_nan_on_one_side_only_fails(self):
        """
        Because that is a real difference, and the dangerous direction.

        A refactor that starts producing NaN where the original produced a
        number is exactly what parity should catch.
        """
        expected = np.array([1.0, 2.0])
        actual = np.array([1.0, np.nan])
        assert not compare_arrays(actual, expected, name="x").passed

    def test_empty_arrays_match(self):
        """
        So an empty-but-correct artifact is not a failure.

        A cluster with no targets of a given kind legitimately produces an
        empty array, and the shape check above has already confirmed both
        sides agree about that.
        """
        assert compare_arrays(np.array([]), np.array([]), name="x").passed

    def test_string_arrays_are_compared_exactly(self):
        """
        Identifiers have no tolerance, so the first difference is reported.

        Reporting the "worst" element would be meaningless for strings.
        """
        result = compare_arrays(np.array(["a", "x"]), np.array(["a", "b"]), name="ids")
        assert not result.passed
        assert "index 1" in result.detail


class TestLoadGolden:
    """A missing fixture must never read as a pass."""

    def test_a_fixture_loads_with_its_manifest(self, golden):
        """
        Provenance travels with the data.

        When a long-passing parity test starts failing, the first question
        is whether the baseline moved, and the manifest is what answers it.
        """
        assert golden.manifest["source_commit"] == "abc123"

    def test_a_missing_fixture_raises_clearly(self, tmp_path):
        """
        Naming the path looked at and how to produce it.

        A ``FileNotFoundError`` from inside a comparison tells the reader
        nothing about what to do next.
        """
        with pytest.raises(ContractError, match="no golden fixture"):
            load_golden("absent", root=tmp_path)

    def test_a_fixture_without_a_manifest_is_refused(self, tmp_path):
        """
        Because a fixture without provenance cannot be a baseline.

        It would be impossible to tell which commit it recorded, so a parity
        pass against it would mean nothing.
        """
        (tmp_path / "bare").mkdir()
        with pytest.raises(ContractError, match="manifest"):
            load_golden("bare", root=tmp_path)

    def test_a_missing_artifact_raises_rather_than_returning_none(self, golden):
        """
        The failure mode this guards against is a silent skip.

        An incomplete fixture whose missing files simply skipped their
        comparisons is precisely how a parity suite comes to pass
        everything.
        """
        with pytest.raises(ContractError, match="incomplete"):
            golden.array("level1_state/absent.npy")


class TestCompareState:
    """Level 1, where the ordering trap lives."""

    def _state(self, **overrides):
        """Build a state matching the toy fixture, with optional overrides."""
        state = {
            "selected_basis": ["eur", "gbp", "usd"],
            "scaler_mean": np.array([1.0, 2.0, 3.0]),
            "combined_features": np.eye(3),
            "elementary_idx": np.array([0, 1, 2]),
            "universe": {"elementary_ids": ["a", "b"], "target_ids": ["t1"]},
        }
        state.update(overrides)
        return state

    def test_a_matching_state_passes(self, golden):
        """The baseline, so the failures below mean something."""
        assert compare_state(self._state(), golden).passed

    def test_a_reordered_basis_fails(self, golden):
        """
        The trap this level exists to catch.

        Same instruments, different order. A set comparison passes and then
        every column position downstream is wrong, so the failure surfaces
        somewhere unrelated -- usually as a quietly worse model rather than
        as an error.
        """
        report = compare_state(self._state(selected_basis=["gbp", "eur", "usd"]), golden)
        assert not report.passed

    def test_the_reordering_message_says_it_is_a_reordering(self, golden):
        """
        Because naming the cause is the difference between an hour and a day.

        Membership is identical, so a message about differing values would
        be actively misleading.
        """
        report = compare_state(self._state(selected_basis=["gbp", "eur", "usd"]), golden)
        detail = report.failures[0].detail
        assert "DIFFERENT ORDER" in detail
        assert "position 0" in detail

    def test_a_basis_with_different_members_reports_membership(self, golden):
        """
        Distinguished from a reordering, because the fixes differ entirely.

        A reordering is a sort key; a membership difference is a selection
        bug.
        """
        report = compare_state(self._state(selected_basis=["eur", "gbp", "jpy"]), golden)
        assert "membership differs" in report.failures[0].detail

    def test_the_post_reduction_index_arrays_are_compared(self, golden):
        """
        Because carrying pre-reduction indices forward looks plausible.

        The original recomputes ``elementary_idx`` as ``0..n_e`` *after* the
        basis is selected. A refactor that keeps the original column numbers
        produces an array of the right dtype and a wrong length, and every
        downstream stage indexes the wrong columns.
        """
        report = compare_state(self._state(elementary_idx=np.array([0, 1, 5])), golden)
        assert not report.passed
        assert any("elementary_idx" in failure.name for failure in report.failures)

    def test_the_universe_is_compared_in_order(self, golden):
        """Identifier order positions every row of the feature matrix."""
        state = self._state(universe={"elementary_ids": ["b", "a"], "target_ids": ["t1"]})
        assert not compare_state(state, golden).passed


class TestCompareTensors:
    """Level 2, which confirms the static-input refactor changed nothing."""

    def _tensors(self, **overrides):
        """Build batch contents matching the toy fixture."""
        tensors = {
            "pnl_history": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            "target": np.array([[1.0], [2.0]], dtype=np.float32),
        }
        tensors.update(overrides)
        return {"train": tensors}

    def test_matching_batches_pass(self, golden):
        """The claim that moving static inputs changed nothing."""
        assert compare_tensors(self._tensors(), golden).passed

    def test_a_differing_batch_fails(self, golden):
        """So the level can reject something."""
        changed = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
        changed[0, 0, 0] = 99.0
        assert not compare_tensors(self._tensors(pnl_history=changed), golden).passed

    def test_a_missing_tensor_is_a_failure_not_a_skip(self, golden):
        """
        Because a key that disappeared is a key the model stopped receiving.

        Skipping it would let a refactor that dropped an input pass level 2
        and fail mysteriously at level 3.
        """
        report = compare_tensors(
            {"train": {"target": np.array([[1.0], [2.0]], np.float32)}}, golden
        )
        assert not report.passed
        assert "absent from the produced batch" in report.failures[0].detail

    def test_split_indices_are_compared(self, golden):
        """
        The captured split is what the refactor must reproduce exactly.

        The original's single ``shuffle`` flag drove both the split and the
        batch order, so the only way to reproduce its split is to replay the
        captured indices.
        """
        report = compare_tensors({"train": {"__indices__": [0, 1, 9]}}, golden)
        assert not report.passed


class TestCompareForwardAndCurve:
    """Levels 3 and 4, where the tolerances widen and why."""

    def test_a_matching_forward_pass_passes(self, golden):
        """Weights loaded into the new model reproduce the old output."""
        assert compare_forward(np.array([0.5, -0.25]), golden).passed

    def test_a_last_bit_perturbation_is_tolerated(self, golden):
        """
        Because reassociation is not a behavioural difference.

        A fused kernel or a different reduction order changes the last bits
        without changing the computation, and a level that failed on that
        would be disabled within a week.
        """
        assert compare_forward(np.array([0.5 + 1e-9, -0.25]), golden).passed

    def test_a_real_forward_difference_fails(self, golden):
        """A difference far above the tolerance is structural."""
        assert not compare_forward(np.array([0.6, -0.25]), golden).passed

    def test_a_matching_curve_passes(self, golden):
        """Five epochs at a fixed seed, reproduced."""
        curve = {"train_loss": [1.0, 0.5, 0.25], "val_loss": [1.1, 0.6, 0.3]}
        assert compare_curve(curve, golden).passed

    def test_a_curve_within_relative_tolerance_passes(self, golden):
        """
        Accumulated non-determinism across epochs is expected.

        The tolerance is relative because an absolute one would be far too
        strict on the first epoch and far too loose on the last.
        """
        curve = {"train_loss": [1.0, 0.5004, 0.25], "val_loss": [1.1, 0.6, 0.3]}
        assert compare_curve(curve, golden).passed

    def test_a_diverging_curve_fails(self, golden):
        """A curve that drifted is a model that trains differently."""
        curve = {"train_loss": [1.0, 0.9, 0.25], "val_loss": [1.1, 0.6, 0.3]}
        assert not compare_curve(curve, golden).passed

    def test_a_missing_curve_is_a_failure(self, golden):
        """
        Because a metric that was not produced has not matched.

        Treating absence as a pass would let a refactor that stopped
        computing validation loss sail through level 4.
        """
        report = compare_curve({"train_loss": [1.0, 0.5, 0.25]}, golden)
        assert not report.passed
        assert "val_loss" in report.failures[0].name


class TestCompareJobSet:
    """Level 5, where placement must not change arithmetic."""

    def test_identical_runs_pass(self):
        """Sequential and parallel agreeing is the whole claim."""
        artifacts = {"predictions": np.array([1.0, 2.0])}
        assert compare_job_set(artifacts, {"predictions": np.array([1.0, 2.0])}).passed

    def test_any_difference_fails_because_placement_must_not_matter(self):
        """
        No tolerance at all, deliberately.

        If two placements of the same work disagree even slightly, something
        is sharing state or deriving seeds per worker, and the results
        cannot be reproduced on a differently sized machine.
        """
        sequential = {"predictions": np.array([1.0, 2.0])}
        parallel = {"predictions": np.array([1.0, 2.0 + 1e-12])}
        assert not compare_job_set(sequential, parallel).passed

    def test_a_job_missing_from_one_run_is_reported_as_scheduling(self):
        """
        Because that is a different bug from a numerical difference.

        A job that ran in one placement and not the other points at the
        scheduler, not at the arithmetic.
        """
        report = compare_job_set(
            {"a": np.array([1.0]), "b": np.array([2.0])}, {"a": np.array([1.0])}
        )
        assert not report.passed
        assert "scheduling bug" in report.failures[0].detail


class TestParityReport:
    """The report, which is what a failing test actually prints."""

    def test_a_report_with_no_comparisons_passes(self):
        """Vacuous, but it must not raise."""
        assert ParityReport(level="x").passed

    def test_every_comparison_runs_even_after_one_fails(self, golden):
        """
        One difference is a clue; the set of them is the diagnosis.

        Every array failing points somewhere entirely different from one
        array failing, so stopping at the first would discard the most
        useful signal.
        """
        report = compare_state(
            {
                "selected_basis": ["wrong"],
                "scaler_mean": np.array([9.0, 9.0, 9.0]),
            },
            golden,
        )
        assert len(report.failures) == 2

    def test_the_summary_names_every_failure(self):
        """So the diagnosis is in the test output, not behind a debugger."""
        report = ParityReport(level="1-state")
        report.add(Comparison(name="a", passed=False, detail="a: broke"))
        report.add(Comparison(name="b", passed=True))
        summary = report.summary()
        assert "1 of 2" in summary
        assert "a: broke" in summary

    def test_the_summary_reports_the_worst_deviation_on_a_pass(self):
        """
        Because a level passing at 9e-7 against 1e-6 is worth knowing.

        It is about to start failing, and the warning is free.
        """
        report = ParityReport(level="3-forward")
        report.add(Comparison(name="a", passed=True, worst_deviation=9e-7))
        assert "9.000e-07" in report.summary()
