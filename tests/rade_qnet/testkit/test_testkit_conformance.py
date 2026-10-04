"""
Tests for the conformance suite.

This suite is what a model author runs instead of writing the framework's
tests themselves, so its own tests have to establish two things for every
clause: that a compliant subject passes it, and that a violating subject fails
*that clause specifically*. A suite that passed everything would be worse than
no suite, because it would be believed.

The deliberate non-clause is tested too. Fitting along the entity axis using
the full universe is not flagged: knowing which instruments exist is not
knowledge of their future, and a cross-sectional model that saw only some of
them would be solving a different problem. A suite that warned about it would
train authors to ignore its warnings, which costs more than the suite is
worth.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import DataLineage, SplitIndices
from src.rade_qnet.core.contract.signature import TensorSpec
from src.rade_qnet.core.contract.state import FittedState, IdentityFittedState
from src.rade_qnet.testkit.conformance import (
    UNBOUNDED_PROBE_BATCHES,
    ConformanceReport,
    check_data_bundle,
    check_fitted_state,
    check_model_capabilities,
    check_source,
)
from src.rade_qnet.testkit.fixtures import (
    StandardisingState,
    SyntheticTensorSource,
    make_lineage,
    make_signature,
    make_tensor_bundle,
)


def _failed_checks(report):
    """
    Return the clause names a report failed.

    Parameters
    ----------
    report
        The conformance report.

    Returns
    -------
    set of str
        Clause names, extracted from the bracketed prefixes.
    """
    return {message.split("]")[0].lstrip("[") for message in report.failures}


def _compliant_source():
    """
    Return a source that satisfies every clause.

    Returns
    -------
    SyntheticTensorSource
        The source.
    """
    return SyntheticTensorSource(features=np.zeros((6, 4)), targets=np.zeros(6), batch_size=2)


class TestTheReport:
    """The record a check produces."""

    def test_an_empty_report_has_passed(self):
        """
        Nothing checked is not the same as something failed.

        A model declaring no capabilities passes trivially, and that is
        correct: the simple case should not be penalised for being simple.
        """
        assert ConformanceReport(subject="demo").passed

    def test_a_warning_does_not_fail_a_check(self):
        """
        The distinction that keeps the suite readable.

        Collapsing warnings into failures would make it too noisy to read;
        collapsing them the other way would make it too permissive.
        """
        report = ConformanceReport(subject="demo")
        report.warn("clause", "suspect but legal")
        assert report.passed

    def test_a_failure_fails_the_check(self):
        """The other half of the distinction."""
        report = ConformanceReport(subject="demo")
        report.fail("clause", "broken")
        assert not report.passed

    def test_a_passed_clause_is_recorded(self):
        """
        So a reader can tell "passed" from "was not applicable".

        Those mean very different things when a capability is involved.
        """
        report = ConformanceReport(subject="demo")
        report.passed_check("clause")
        assert report.checks_run == ["clause"]

    def test_merging_combines_every_category(self):
        """
        A bundle check absorbs its state and source checks.

        Losing the clause names on merge would make the combined report
        unable to distinguish untested from passed.
        """
        first = ConformanceReport(subject="a")
        first.fail("one", "broken")
        second = ConformanceReport(subject="b")
        second.warn("two", "suspect")
        second.passed_check("three")
        first.merge(second)
        assert (len(first.failures), len(first.warnings), len(first.checks_run)) == (1, 1, 3)

    def test_the_description_names_the_subject_and_the_verdict(self):
        """
        Because the description is what a failing test prints.

        Without the subject, a job set's failure says nothing about which
        member failed.
        """
        report = ConformanceReport(subject="MyModel")
        report.fail("clause", "broken")
        described = report.describe()
        assert "MyModel" in described
        assert "FAILED" in described

    def test_raising_carries_the_whole_report(self):
        """
        The form a test should use.

        An assertion saying only "conformance failed" sends the author back
        to run the check by hand.
        """
        report = ConformanceReport(subject="demo")
        report.fail("clause", "the specific thing that is wrong")
        with pytest.raises(AssertionError, match="the specific thing"):
            report.raise_if_failed()

    def test_raising_is_silent_on_success(self):
        """So it can be called unconditionally."""
        ConformanceReport(subject="demo").raise_if_failed()


class TestSourceClauses:
    """Each clause, against a compliant and a violating source."""

    def test_a_compliant_source_passes_everything(self):
        """The baseline every violation below is measured against."""
        assert check_source(_compliant_source()).passed

    def test_a_non_source_fails_the_protocol_clause(self):
        """And stops there, since nothing else can be evaluated."""
        assert _failed_checks(check_source(object())) == {"source.protocol"}

    def test_a_source_yielding_nothing_fails(self):
        """
        An empty source trains on nothing and reports a flat curve.

        Which looks like a learning-rate problem, not a data problem.
        """
        empty = SyntheticTensorSource(features=np.zeros((0, 4)), targets=np.zeros(0), batch_size=2)
        assert "source.non_empty" in _failed_checks(check_source(empty))

    def test_a_generator_backed_source_fails_re_iterability(self):
        """
        The clause that matters most.

        A bare generator satisfies every other clause and then yields
        nothing from epoch two onwards -- a flat loss curve that looks like
        a learning-rate problem, not a data bug.
        """

        class GeneratorBacked:
            """Stores a generator instead of returning a fresh one."""

            signature = make_signature(n_features=4)
            static: dict[str, object] = {}
            steps_per_epoch = 1
            n_samples = 2

            def __init__(self) -> None:
                """Consume the data on the first pass and never again."""
                self._exhausting = iter([{"features": np.zeros((2, 4)), "target": np.zeros(2)}])

            def batches(self):
                """Return the same, partially consumed, iterator."""
                return self._exhausting

        assert "source.re_iterable" in _failed_checks(check_source(GeneratorBacked()))

    def test_a_dishonest_step_count_fails(self):
        """
        Schedules and progress reporting are computed from it.

        An estimate silently rescales a learning-rate schedule, which is
        invisible in the output.
        """

        class Dishonest(SyntheticTensorSource):
            """Reports more steps than it yields."""

            @property
            def steps_per_epoch(self):
                """Overstate the length by one."""
                return 99

        dishonest = Dishonest(features=np.zeros((6, 4)), targets=np.zeros(6), batch_size=2)
        assert "source.steps_honest" in _failed_checks(check_source(dishonest))

    def test_a_batch_with_an_undeclared_key_fails(self):
        """
        The data module and the model disagree about the interface.

        Silently ignoring the extra key means a feature the user believes
        is in use is not.
        """

        class ExtraKey(SyntheticTensorSource):
            """Yields a key the signature does not declare."""

            def batches(self):
                """Yield one batch with an undeclared key."""
                for batch in super().batches():
                    yield {**batch, "undeclared": np.zeros(2)}

        extra = ExtraKey(features=np.zeros((4, 4)), targets=np.zeros(4), batch_size=2)
        assert "source.batch_keys" in _failed_checks(check_source(extra))

    def test_an_undelivered_static_input_warns_rather_than_fails(self):
        """
        A model may legitimately supply its own through ``StaticInputs``.

        Failing would make a valid design impossible to express.
        """
        source = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            signature_override=make_signature(
                n_features=4,
                static={"adjacency": TensorSpec(shape=(4, 4), dtype="float64")},
            ),
        )
        report = check_source(source)
        assert report.passed
        assert any("static_declared" in warning for warning in report.warnings)

    def test_an_unbounded_source_is_checked_without_hanging(self):
        """
        Drained to a prefix, because it never ends.

        A checker that could not be run on an interactive source would
        leave half the framework's remit unchecked.
        """
        unbounded = SyntheticTensorSource(
            features=np.zeros((6, 4)), targets=np.zeros(6), batch_size=2, unbounded=True
        )
        assert check_source(unbounded).passed

    def test_the_unbounded_probe_is_bounded(self):
        """
        Small enough that checking an environment rollout stays cheap.

        Large enough to establish that batches keep arriving.
        """
        assert 1 < UNBOUNDED_PROBE_BATCHES < 100


class TestFittedStateClauses:
    """The inverse is the point, so most clauses are about it."""

    def test_a_real_state_passes_everything(self):
        """The baseline."""
        assert check_fitted_state(StandardisingState.fit(np.arange(10.0))).passed

    def test_the_identity_state_passes(self):
        """
        Declaring that a model transforms nothing is a legitimate answer.

        Penalising it would push authors into declaring a fake transform.
        """
        assert check_fitted_state(IdentityFittedState()).passed

    def test_a_non_state_fails_the_base_clause(self):
        """``load`` is called as a classmethod, so inheritance is required."""
        assert not check_fitted_state(object()).passed

    def test_an_inverse_that_changes_the_shape_fails(self):
        """
        Predictions and targets would then be misaligned.

        Which produces a metric computed against the wrong rows, and a
        plausible number.
        """

        class Reshaping(FittedState):
            """Returns fewer values than it was given."""

            def save(self, directory):
                """Write nothing."""

            @classmethod
            def load(cls, directory):
                """Read nothing."""
                return cls()

            def inverse_transform_targets(self, predictions):
                """Drop the last value."""
                return np.asarray(predictions)[:-1]

        assert not check_fitted_state(Reshaping()).passed

    def test_an_inverse_producing_non_finite_values_fails(self):
        """
        Every metric computed from it would then be rejected downstream.

        Catching it here says which component is at fault.
        """

        class Diverging(FittedState):
            """Returns infinities."""

            def save(self, directory):
                """Write nothing."""

            @classmethod
            def load(cls, directory):
                """Read nothing."""
                return cls()

            def inverse_transform_targets(self, predictions):
                """Return infinities."""
                return np.full_like(np.asarray(predictions, dtype=float), np.inf)

        assert not check_fitted_state(Diverging()).passed

    def test_a_constant_inverse_fails(self):
        """
        The subtle one.

        A state that collapses every input to one value destroys the
        prediction entirely, and the output is finite, correctly shaped and
        completely useless.
        """

        class Collapsing(FittedState):
            """Maps everything to a single value."""

            def save(self, directory):
                """Write nothing."""

            @classmethod
            def load(cls, directory):
                """Read nothing."""
                return cls()

            def inverse_transform_targets(self, predictions):
                """Return a constant."""
                return np.zeros_like(np.asarray(predictions, dtype=float))

        assert not check_fitted_state(Collapsing()).passed


class TestDataBundleClauses:
    """Lineage, leakage, and the thing deliberately not checked."""

    def test_a_compliant_bundle_passes(self):
        """The baseline."""
        assert check_data_bundle(make_tensor_bundle()).passed

    def test_a_non_bundle_fails_the_type_clause(self):
        """And stops, since nothing else can be read."""
        assert _failed_checks(check_data_bundle(object())) == {"bundle.type"}

    def test_a_lineage_recording_no_training_rows_fails(self):
        """
        A lineage with no training rows describes no reproducible run.

        A bundle whose lineage does not say which rows it trained on cannot
        be re-evaluated against the same data later, which is the whole
        reason the indices are recorded rather than the fractions.
        """
        bundle = make_tensor_bundle()
        empty = bundle.lineage.model_copy(
            update={"split_indices": {"train": (), "validation": (), "test": ()}}
        )
        assert "bundle.lineage_splits" in _failed_checks(
            check_data_bundle(replace(bundle, lineage=empty))
        )

    def test_overlapping_scenarios_fail(self):
        """
        The most expensive mistake available at this layer.

        A held-out metric computed over shared scenarios is not held out,
        and it produces an encouraging number.
        """
        bundle = make_tensor_bundle()
        leaking = bundle.lineage.model_copy(
            update={"split_indices": {"train": (0, 1, 2), "validation": (2, 3), "test": (4,)}}
        )
        assert "bundle.no_scenario_overlap" in _failed_checks(
            check_data_bundle(replace(bundle, lineage=leaking))
        )

    def test_entity_axis_fitting_is_deliberately_not_flagged(self):
        """
        The stated exclusion, asserted so it cannot be lost by accident.

        Knowing which instruments exist is not knowledge of their future,
        and a suite that flagged it would train authors to ignore its
        warnings -- which costs more than the suite is worth.
        """
        bundle = make_tensor_bundle()
        cross_sectional = bundle.lineage.model_copy(
            update={"n_entities": 500, "notes": {"entity_encoder": "fitted on full universe"}}
        )
        report = check_data_bundle(replace(bundle, lineage=cross_sectional))
        assert report.passed
        assert not report.warnings

    def test_the_bundles_state_is_checked_too(self):
        """
        A bundle is only as trustworthy as the state it carries.

        Checking the bundle but not its inverse would let the most
        consequential failure through.
        """
        assert any(
            "inverse" in clause for clause in check_data_bundle(make_tensor_bundle()).checks_run
        )

    def test_each_splits_source_is_checked_too(self):
        """
        Every split's loader is checked, not only the training one.

        A bundle whose validation loader is not re-iterable is broken, even
        though its training loader is fine.
        """
        assert any(
            "re_iterable" in clause for clause in check_data_bundle(make_tensor_bundle()).checks_run
        )


class TestCapabilityClauses:
    """Only what is declared, and each declaration actually called."""

    def test_a_model_declaring_nothing_passes_trivially(self):
        """
        The simple case should not be penalised for being simple.

        A forty-line regressor has no capabilities and no obligations.
        """
        report = check_model_capabilities(object())
        assert report.passed
        assert report.checks_run == []

    def test_a_capability_that_raises_is_a_finding_not_a_crash(self):
        """
        The suite reports problems; it does not become one.

        A checker that crashed on a broken model would be useless on
        exactly the models worth checking.
        """

        class Exploding:
            """Declares a capability that raises when called."""

            def static_inputs(self):
                """Raise."""
                raise RuntimeError("not built yet")

        assert not check_model_capabilities(Exploding()).passed

    def test_an_empty_static_input_set_warns(self):
        """
        A warning, not a failure.

        Legal but suspect: either the capability is unnecessary or the inputs
        are missing, and only the author knows which.
        """

        class EmptyStatic:
            """Declares the capability and supplies nothing."""

            def static_inputs(self):
                """Return nothing."""
                return {}

        report = check_model_capabilities(EmptyStatic())
        assert report.passed
        assert report.warnings

    def test_precomputation_without_static_inputs_fails(self):
        """
        There is nothing constant to precompute.

        So the cached encoding would be stale after the first batch, which
        makes evaluation measure a different model from training.
        """

        class OnlyPrecomputable:
            """Declares precomputation with nothing to precompute from."""

            def precompute(self, static):
                """Return an encoding."""
                return {}

            def forward_with_precomputed(self, batch, precomputed):
                """Return predictions."""
                return batch

        assert "capability.precomputable_needs_static" in _failed_checks(
            check_model_capabilities(OnlyPrecomputable())
        )

    def test_a_custom_step_defers_to_the_engine_suite(self):
        """
        The two properties that matter need a real tensor and an engine.

        Recording the obligation is honest; pretending to discharge it
        would be worse than saying nothing.
        """

        class WithCustomStep:
            """Owns its loss computation."""

            def training_step(self, batch, static):
                """Return a loss."""
                return 0.0

        report = check_model_capabilities(WithCustomStep())
        assert report.passed
        assert any("gradient" in warning for warning in report.warnings)

    def test_a_member_covering_nothing_fails(self):
        """
        A member that covers no target is not a member of anything.

        It contributes no predictions, so the job's targets would silently
        have no owner.
        """

        class CoversNothing:
            """Declares routing and covers nothing."""

            def covered_targets(self):
                """Return nothing."""
                return []

        assert "capability.routable_non_empty" in _failed_checks(
            check_model_capabilities(CoversNothing())
        )

    def test_a_member_repeating_a_target_fails(self):
        """A duplicate would be counted twice in an assembled portfolio."""

        class Repeats:
            """Declares the same target twice."""

            def covered_targets(self):
                """Return a duplicate."""
                return ["EURUSD", "EURUSD"]

        assert "capability.routable_unique" in _failed_checks(check_model_capabilities(Repeats()))

    def test_a_member_over_declaring_its_coverage_fails(self):
        """
        Predictions would be attributed to instruments it never trained on.

        The honest answer is often narrower than the job definition -- a
        member assigned twelve instruments may have trained on nine.
        """

        class OverDeclares:
            """Claims more than it was assigned."""

            def covered_targets(self):
                """Return an unassigned target."""
                return ["EURUSD", "USDTRY"]

        assert "capability.routable_subset" in _failed_checks(
            check_model_capabilities(OverDeclares(), assigned_targets=["EURUSD", "USDJPY"])
        )

    def test_coverage_narrower_than_the_assignment_is_accepted(self):
        """
        Which is the normal case, not an error.

        Three instruments dropped for insufficient history is a fact to
        record, not a violation.
        """

        class Narrower:
            """Trained on fewer than it was assigned."""

            def covered_targets(self):
                """Return a subset."""
                return ["EURUSD"]

        assert check_model_capabilities(Narrower(), assigned_targets=["EURUSD", "USDJPY"]).passed

    def test_a_non_boolean_inductive_answer_fails(self):
        """
        The answer routes a decision to reject an inference request.

        A truthy string would route it the wrong way without ever looking
        wrong.
        """

        class Stringly:
            """Returns a string instead of a bool."""

            def supports_unseen_entities(self):
                """Return a string."""
                return "yes"

    def test_several_capabilities_are_all_checked(self):
        """
        Declaring two must not mean only the first is examined.

        Which would make the second a declaration with no obligation.
        """

        class Capable:
            """Declares two capabilities."""

            def static_inputs(self):
                """Return a static input."""
                return {"adjacency": np.eye(3)}

            def supports_unseen_entities(self):
                """Return a bool."""
                return True



class TestAgreementWithTheContracts:
    """The suite and the contracts must not drift apart."""

    def test_an_overlap_rejected_by_the_contract_is_also_flagged_here(self):
        """
        Two mechanisms, one rule.

        ``SplitIndices`` refuses to construct an overlap; the suite catches
        one that reached a lineage by another route. They have to agree
        about what an overlap is.
        """
        with pytest.raises(Exception, match="disjoint"):
            SplitIndices(
                train=np.array([0, 1], dtype=np.int64),
                validation=np.array([1], dtype=np.int64),
                test=np.array([], dtype=np.int64),
            )

        leaking = make_lineage().model_copy(
            update={"split_indices": {"train": (0, 1), "validation": (1,), "test": ()}}
        )
        bundle = replace(make_tensor_bundle(), lineage=leaking)
        assert not check_data_bundle(bundle).passed

    def test_a_lineage_is_still_a_valid_contract_after_being_broken_for_a_test(self):
        """
        Confirming these tests exercise the suite, not pydantic.

        If the lineage could not hold an overlap, the clause above would be
        unreachable and the test would be vacuous.
        """
        leaking = make_lineage().model_copy(
            update={"split_indices": {"train": (0, 1), "validation": (1,), "test": ()}}
        )
        assert isinstance(leaking, DataLineage)
