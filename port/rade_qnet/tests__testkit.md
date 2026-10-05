# `tests/rade_qnet/testkit`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 21 | 892 | `4dd05188c31c1115` |
| 2 | `test_testkit_conformance.py` | 685 | 23601 | `e2871cbd5e4fa5f0` |
| 3 | `test_testkit_fixtures.py` | 419 | 16382 | `aa867e4786c01c80` |
| 4 | `test_testkit_golden_fixture.py` | 304 | 12695 | `9f0c11f0e6b72a5e` |
| 5 | `test_testkit_parity.py` | 537 | 21337 | `081eb6c29a54e0ff` |

---

## 1. `tests/rade_qnet/testkit/__init__.py`

892 bytes · SHA-256 `4dd05188c31c1115`

```python
"""
Tests for ``rade_qnet.testkit`` -- the conformance tooling.

The tooling that decides whether a model is correct has to be correct itself.
A conformance suite that passes everything is worse than none, because it
provides false assurance, so these tests check the suite in both directions: it
passes a model built to the contract, and it fails a model deliberately built
to violate each clause.

Planned modules
---------------
``test_testkit_conformance.py``
    Each conformance check verified against a compliant model and against a
    model that breaks exactly that check.  [Phase 1]
``test_testkit_parity.py``
    Comparison against a golden fixture: identical inputs pass, and a
    perturbation larger than the tolerance fails.  [Phase 0]
``test_testkit_fixtures.py``
    The synthetic sources and specs are deterministic, so a test built on them
    cannot flake.  [Phase 1]
"""
```

---

## 2. `tests/rade_qnet/testkit/test_testkit_conformance.py`

23601 bytes · SHA-256 `e2871cbd5e4fa5f0`

```python
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
```

---

## 3. `tests/rade_qnet/testkit/test_testkit_fixtures.py`

16382 bytes · SHA-256 `aa867e4786c01c80`

```python
"""
Tests for the shipped test fixtures.

These are not internal helpers. ``rade_qnet.testkit`` is public API: a model
author imports it to test their own definition against the framework. So the
fixtures need testing for the same reason any public surface does -- but there
is a sharper reason too.

A fixture that models the problem the easy way lets a broken pipeline pass.
``make_tensor_bundle`` fits its standardiser on the training split alone and
splits chronologically; if it fitted on everything, or split randomly, a
leaking pipeline would pass its tests and ship. So the properties asserted
here are the ones that make the fixtures honest, not merely convenient.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.contract.bundle import ModelBundle
from src.rade_qnet.core.contract.data import DataBundle, TensorBatchData
from src.rade_qnet.core.contract.result import TrainingResult
from src.rade_qnet.core.contract.signature import InputSignature
from src.rade_qnet.core.contract.source import BatchSource
from src.rade_qnet.core.contract.state import FittedState
from src.rade_qnet.core.lifecycle.components import MODELS, REPORTS
from src.rade_qnet.core.lifecycle.context import RunContext
from src.rade_qnet.core.lifecycle.hooks import PipelineHook
from src.rade_qnet.core.provenance.hashing import digest_spec
from src.rade_qnet.core.spec.run import SupervisedRunSpec
from src.rade_qnet.testkit.conformance import check_data_bundle, check_fitted_state, check_source
from src.rade_qnet.testkit.fixtures import (
    PLACEHOLDER_DIGEST,
    RecordingHook,
    StandardisingState,
    SyntheticArraySource,
    SyntheticTensorSource,
    isolated_registries,
    make_lineage,
    make_model_bundle,
    make_run_context,
    make_run_spec,
    make_signature,
    make_split_indices,
    make_tensor_bundle,
    make_training_result,
)


class TestRegistryIsolation:
    """Registries are process-global, so a test that registers must clean up."""

    def test_a_registration_is_reverted_on_exit(self):
        """
        Without this, one test's registration changes what another sees.

        And the failure appears in whichever test happens to run second,
        which is the worst possible place for it.
        """
        with isolated_registries():
            MODELS.register("temporary", object)
        assert "temporary" not in MODELS.names()

    def test_pre_existing_registrations_survive(self):
        """
        It restores a snapshot, it does not empty the registries.

        Clearing them would unregister whatever the importing modules had
        already registered -- the summary report, every model in a job set
        -- and break every later test in the process.
        """
        with isolated_registries():
            MODELS.register("registered_before", object)
            with isolated_registries():
                pass
            assert "registered_before" in MODELS.names()

    def test_it_reverts_after_an_exception(self):
        """
        A failing test must not leak its registration into the next one.

        Which is exactly when leakage is hardest to diagnose, because the
        reported failure is in unrelated code.
        """
        with pytest.raises(RuntimeError), isolated_registries():
            MODELS.register("temporary", object)
            raise RuntimeError("the test failed")
        assert "temporary" not in MODELS.names()

    def test_every_registry_is_covered(self):
        """
        Four registries, so missing one leaks silently.

        A model registered inside the block and still present afterwards
        would be found only by a much later, unrelated failure.
        """
        with isolated_registries():
            MODELS.register("temporary_model", object)
            REPORTS.register("temporary_report", object)
        assert "temporary_model" not in MODELS.names()
        assert "temporary_report" not in REPORTS.names()


class TestStandardisingState:
    """A state with a real inverse, which is what makes it useful."""

    def test_it_is_a_fitted_state(self):
        """So it exercises the real interface, not a lookalike."""
        assert isinstance(StandardisingState(), FittedState)

    def test_it_passes_the_conformance_suite(self):
        """
        The fixture has to satisfy the checks it helps authors run.

        A reference implementation that failed its own suite would be
        actively misleading.
        """
        assert check_fitted_state(StandardisingState.fit(np.arange(10.0))).passed

    def test_a_bare_instance_is_the_identity(self):
        """
        So a test that does not care about scaling can ignore it.

        Which keeps the fixture usable without reading its parameters.
        """
        values = np.array([1.0, 2.0])
        assert np.allclose(StandardisingState().inverse_transform_targets(values), values)


class TestSyntheticSources:
    """Re-iterable and honest about their own length."""

    @pytest.fixture
    def source(self):
        """
        Provide a small tensor source.

        Returns
        -------
        SyntheticTensorSource
            Six samples in batches of two.
        """
        return SyntheticTensorSource(features=np.zeros((6, 4)), targets=np.zeros(6), batch_size=2)

    def test_the_tensor_source_passes_conformance(self, source):
        """The reference implementation of the protocol."""
        assert check_source(source).passed

    def test_an_unbounded_source_also_passes(self):
        """
        The interactive case, which must be testable too.

        A conformance suite that could only check datasets would leave half
        the framework's remit unchecked.
        """
        unbounded = SyntheticTensorSource(
            features=np.zeros((6, 4)), targets=np.zeros(6), batch_size=2, unbounded=True
        )
        assert check_source(unbounded).passed

    def test_static_inputs_are_reflected_in_the_signature(self):
        """
        A source delivering an input its signature omits is inconsistent.

        The engine uploads what the signature declares, so the extra input
        would be carried and silently ignored.
        """
        with_static = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            static={"adjacency": np.eye(4)},
        )
        assert "adjacency" in with_static.signature.static

    def test_the_array_source_satisfies_the_protocol_too(self):
        """
        So a one-shot engine's tests use the same shape of fixture.

        Two unrelated fixture styles would make the two engines' tests
        incomparable.
        """
        array_source = SyntheticArraySource(features=np.zeros((4, 3)), targets=np.zeros(4))
        assert isinstance(array_source, BatchSource)

    def test_data_is_deterministic_for_a_given_seed(self):
        """
        The same seed gives the same arrays, every time.

        A fixture that varied between runs would make failures
        irreproducible, which is the one thing a test fixture must not do.
        """
        first = make_tensor_bundle(seed=7).split("train")
        second = make_tensor_bundle(seed=7).split("train")
        assert np.array_equal(
            next(iter(first.loader.batches()))["features"],
            next(iter(second.loader.batches()))["features"],
        )

    def test_different_seeds_give_different_data(self):
        """Otherwise the seed parameter would be decorative."""
        first = next(iter(make_tensor_bundle(seed=1).split("train").loader.batches()))
        second = next(iter(make_tensor_bundle(seed=2).split("train").loader.batches()))
        assert not np.array_equal(first["features"], second["features"])


class TestBundleFixtures:
    """The properties that keep a leaking pipeline from passing."""

    def test_the_tensor_bundle_passes_conformance(self):
        """As the reference bundle, it must."""
        assert check_data_bundle(make_tensor_bundle()).passed

    def test_the_state_is_fitted_on_training_rows_only(self):
        """
        The standardiser sees the training split and nothing else.

        Not incidental: a fixture that fitted on everything would make a
        leaking pipeline pass its tests and ship.
        """
        bundle = make_tensor_bundle()
        train_rows = np.array(bundle.lineage.split_indices["train"])
        test_rows = np.array(bundle.lineage.split_indices["test"])
        assert train_rows.max() < test_rows.min()

    def test_the_splits_are_chronological(self):
        """
        Not random, because a time-series problem is not exchangeable.

        A randomly split fixture would let a model trained on the future
        pass every test here.
        """
        indices = make_split_indices()
        assert indices.train.max() < indices.validation.min()
        assert indices.validation.max() < indices.test.min()

    def test_the_splits_are_disjoint(self):
        """Enforced by the contract, and worth confirming the fixture obeys it."""
        indices = make_split_indices()
        combined = np.concatenate([indices.train, indices.validation, indices.test])
        assert len(set(combined.tolist())) == combined.size

    def test_a_split_carries_the_engine_payload(self):
        """
        Which is what the bundle's type parameter stands for.

        One payload type since Phase 6 retired the second, so there is no
        dispatch left to exercise -- but the parameter remains, because an
        engine written outside this repository may carry something else.
        """
        assert isinstance(make_tensor_bundle().split("train"), TensorBatchData)

    def test_a_narrowed_bundle_omits_the_other_splits(self):
        """
        How a caller tests the no-validation-split path.

        Which is a real configuration and a common source of downstream
        failures, since early stopping then has nothing to monitor.
        """
        bundle = make_tensor_bundle(splits=("train",))
        assert bundle.split_names == ("train",)

    def test_a_bundle_without_a_training_split_is_refused(self):
        """
        Caught in the fixture, so the message points at the call.

        Letting the contract catch it would name the bundle rather than the
        test that asked for the impossible thing.
        """
        with pytest.raises(ValueError, match="train"):
            make_tensor_bundle(splits=("test",))

    def test_an_unknown_split_name_is_refused(self):
        """A typo should not silently produce a train-only bundle."""
        with pytest.raises(ValueError, match="holdout"):
            make_tensor_bundle(splits=("train", "holdout"))

    def test_the_bundle_is_a_data_bundle(self):
        """Which is what lets one pipeline consume what any data build made."""
        assert isinstance(make_tensor_bundle(), DataBundle)


class TestMetadataFixtures:
    """Small builders, each producing something the real validators accept."""

    def test_the_signature_is_valid(self):
        """Built through the real class, so it exercises real validation."""
        assert isinstance(make_signature(), InputSignature)

    def test_the_lineage_is_valid(self):
        """Including the split indices it records."""
        assert make_lineage().split_indices["train"]

    def test_the_run_spec_goes_through_real_parsing(self):
        """
        Rather than constructing the model directly.

        So a fixture exercises the same validation and the same union
        resolution a real configuration file does -- a fixture that bypassed
        them could produce a spec no user could ever write.
        """
        assert isinstance(make_run_spec(), SupervisedRunSpec)

    def test_the_run_spec_accepts_overrides(self):
        """So a test can vary one field without restating the rest."""
        assert make_run_spec({"seed": 99}).seed == 99

    def test_the_training_result_is_plausible(self):
        """
        A falling loss and a best epoch at the end.

        A fixture with a rising loss would make a report's "best epoch"
        rendering look wrong when it is right.
        """
        result = make_training_result()
        assert isinstance(result, TrainingResult)
        assert result.fit.curve("train_loss")[0] > result.fit.curve("train_loss")[-1]

    def test_the_lineage_digest_is_overridable(self):
        """
        Because the lineage is where a bundle's spec digest comes from.

        ``write_bundle`` copies ``lineage.spec_digest`` into the manifest
        rather than recomputing it, so a test proving that link has to be
        able to set a value it can recognise afterwards.
        """
        digest = "a" * 64
        assert make_lineage(spec_digest=digest).spec_digest == digest

    def test_the_lineage_digest_defaults_to_an_obvious_placeholder(self):
        """
        All zeros rather than something that looks real.

        A plausible-looking digest in a synthetic manifest is worse than an
        obviously fake one, because nobody reading it would think to check.
        """
        assert make_lineage().spec_digest == PLACEHOLDER_DIGEST

    def test_the_model_bundle_is_complete(self):
        """Everything a report or a writer needs, with nothing missing."""
        assert isinstance(make_model_bundle(), ModelBundle)

    def test_the_model_bundle_accepts_a_matching_spec_and_lineage(self):
        """
        So a bundle can carry the spec it genuinely came from.

        The default spec and lineage are independent of each other, which is
        fine for a report test and wrong for a writer test: proving that a
        manifest records the right digest needs a bundle whose lineage
        actually refers to its spec.
        """
        spec = make_run_spec({"seed": 11})
        digest = digest_spec(spec)
        bundle = make_model_bundle(spec=spec, lineage=make_lineage(spec_digest=digest))
        assert bundle.spec is spec
        assert bundle.lineage.spec_digest == digest_spec(bundle.spec)

    def test_the_run_context_needs_no_store(self, tmp_path):
        """
        Backed by an in-memory catalog and a null tracker.

        An output directory is still required, because a run that cannot
        say where its output went is not a run -- but no catalog file, lock
        file or tracking backend has to exist.
        """
        context = make_run_context(output_directory=tmp_path)
        assert isinstance(context, RunContext)
        assert not list(tmp_path.iterdir())


class TestRecordingHook:
    """A hook that remembers what it was told, for asserting on order."""

    def test_it_satisfies_the_hook_interface(self):
        """So it can be passed anywhere a real hook can."""
        assert isinstance(RecordingHook(), PipelineHook)

    def test_calls_are_recorded_in_order(self):
        """
        Order is the thing worth asserting about hooks.

        "Stage start before stage end" is the contract, and a set would
        lose exactly that.
        """
        hook = RecordingHook()
        hook.on_stage_start("build_data")
        hook.on_stage_end("build_data", seconds=0.1)
        assert hook.names() == ("stage_start", "stage_end")

    def test_the_recorded_payload_is_available(self):
        """
        So a test can assert which stage, not only that something happened.

        A hook that recorded only event names could not distinguish the
        right stage failing from the wrong one.
        """
        hook = RecordingHook()
        hook.on_stage_start("fit")
        assert hook.events[0] == ("stage_start", "fit")

    @pytest.mark.parametrize(
        ("call", "excluded"),
        [
            (lambda hook: hook.on_stage_end("fit", seconds=1.234), 1.234),
            (lambda hook: hook.on_artifact("curve", "/tmp/pytest-9/curve.png"), "/tmp/pytest-9"),
        ],
    )
    def test_non_deterministic_values_are_deliberately_not_recorded(self, call, excluded):
        """
        Durations and temporary paths vary between runs.

        Recording them would make every assertion against the hook either
        fragile or non-portable, so they are dropped on purpose rather than
        by oversight.
        """
        hook = RecordingHook()
        call(hook)
        assert excluded not in hook.events[0]
```

---

## 4. `tests/rade_qnet/testkit/test_testkit_golden_fixture.py`

12695 bytes · SHA-256 `9f0c11f0e6b72a5e`

```python
"""
Guards on the committed golden fixture itself.

A fixture is data, not code, so its *values* are not asserted here -- their
correctness comes from the capture script having run against unmodified
``rade_ml_pt``, recorded in ``manifest.json`` with the source commit. What is
asserted is that the fixture is still capable of catching the things it was
built to catch.

That distinction matters because a fixture degrades silently. If someone
re-captures with a configuration where basis selection happens to keep every
instrument, every parity test still passes and the suite still looks green --
but two of its sharpest checks have quietly stopped testing anything. The
post-reduction index arrays would equal the pre-reduction ones, so a refactor
carrying the wrong indices forward would pass; and the selected basis would
be in input order, so a reordering would pass too.

These tests are the tripwire for that. They assert the fixture's *shape*: it
reduces, the surviving order is not the input order, and the artifacts the
five levels need are all present.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.rade_qnet.testkit.parity import load_golden

#: Size ceiling, as a definition-of-done item. A fixture that grows past this
#: stops being committed, and a parity suite nobody can run locally is a
#: parity suite nobody runs.
MAX_FIXTURE_BYTES = 10 * 1024 * 1024


@pytest.fixture(scope="module")
def golden():
    """
    Load the committed fixture, skipping the module if it is absent.

    Skipped rather than failed so a fresh clone that has not run the capture
    is not reported as broken code. The capture is a one-off step, and its
    absence is a setup state rather than a defect.

    Returns
    -------
    GoldenFixture
        The loaded fixture.
    """
    try:
        return load_golden("hybrid_gnn_rnn")
    except Exception as exc:
        pytest.skip(f"golden fixture not captured: {exc}")


@pytest.mark.usefixtures("requires_golden")
class TestCompleteness:
    """Every level must have the artifacts it needs."""

    @pytest.mark.parametrize(
        "relative_path",
        [
            "level1_state/scaler_mean.npy",
            "level1_state/scaler_scale.npy",
            "level1_state/selected_basis.json",
            "level1_state/combined_features.npy",
            "level1_state/adjacency_indices.npy",
            "level1_state/adjacency_values.npy",
            "level1_state/adjacency_shape.npy",
            "level1_state/elementary_idx.npy",
            "level1_state/target_idx.npy",
            "level1_state/universe.json",
            "level2_tensors/split_indices.json",
            "level2_tensors/train_batch_000.npz",
            "level3_forward/state_dict.pt",
            "level3_forward/outputs.npy",
            "level4_training/curve.json",
        ],
    )
    def test_the_artifact_is_present(self, golden, relative_path):
        """
        Because a missing artifact makes its comparison a silent skip.

        The harness raises on a missing file rather than skipping, but only
        once something asks for it. This checks the whole set up front.
        """
        assert golden.has(relative_path), relative_path

    def test_the_input_is_stored_without_a_pickle(self, golden):
        """
        A checked-in pickle is code that executes on load.

        The capture materialises the pickles ``rade_ml_pt`` reads into a
        temporary directory instead, so the committed fixture stays
        reviewable in a diff and cannot carry an executable payload.
        """
        pickles = list(golden.directory.rglob("*.pkl"))
        assert not pickles, f"the fixture must not contain pickles: {pickles}"

    def test_the_fixture_is_small_enough_to_commit(self, golden):
        """
        Because a parity suite nobody can run locally is not a gate.

        The limit is generous; the real constraint is that the capture stays
        tiny enough to re-run in seconds.
        """
        total = sum(path.stat().st_size for path in golden.directory.rglob("*") if path.is_file())
        assert total < MAX_FIXTURE_BYTES


class TestProvenance:
    """A baseline without provenance cannot be a baseline."""

    @pytest.mark.parametrize(
        "key", ["captured_at", "source_commit", "seed", "versions", "data_config"]
    )
    def test_the_manifest_records_it(self, golden, key):
        """
        So a parity failure can be attributed.

        When a long-passing test starts failing, the first question is
        whether the baseline moved rather than the code, and only the
        manifest answers it.
        """
        assert key in golden.manifest

    def test_the_preserved_defects_are_named(self, golden):
        """
        Because reproducing a bug on purpose must be written down.

        Two defects change numerical output and are deliberately reproduced.
        Without the record, a later reader finds compatibility flags with no
        explanation and switches them off.
        """
        preserved = golden.manifest["preserved_defects"]
        assert "defect_9_basis_selection_leakage" in preserved
        assert "defect_3_shared_shuffle_flag" in preserved


@pytest.mark.usefixtures("requires_golden")
class TestTheFixtureCanStillFail:
    """
    The tripwire against silent degradation.

    Each test here asserts that the fixture still exercises a trap. If one
    starts failing after a re-capture, the fixture has become weaker and the
    parity suite is passing for less reason than it appears to.
    """

    def test_basis_selection_actually_reduced(self, golden):
        """
        Otherwise the post-reduction index trap is not captured at all.

        With every instrument surviving, ``elementary_idx`` equals the
        pre-reduction indices, so a refactor carrying the wrong ones forward
        passes level 1. This is the specific way the earlier, two-per-group
        input was too weak: selection kept all sixteen.
        """
        basis = golden.json("level1_state/selected_basis.json")
        universe = json.loads(
            (golden.directory / "input" / "universe.json").read_text(encoding="utf-8")
        )
        assert len(basis) < len(universe["elementary_ids"])

    def test_the_index_arrays_are_post_reduction(self, golden):
        """
        They count the *survivors*, not the original columns.

        The original recomputes them as ``0..n_e`` and ``n_e..n_e+n_t``
        after selection. Asserted against the basis length rather than a
        literal, so the test survives a re-capture at a different size.
        """
        basis = golden.json("level1_state/selected_basis.json")
        elementary_idx = golden.array("level1_state/elementary_idx.npy")
        target_idx = golden.array("level1_state/target_idx.npy")

        assert elementary_idx.tolist() == list(range(len(basis)))
        assert target_idx[0] == len(basis)

    def test_the_selected_basis_is_not_in_input_order(self, golden):
        """
        Otherwise a reordering bug would pass the ordered comparison.

        Order fixes column positions in every array downstream. If the
        captured order happened to match the input order, comparing as a
        sequence and comparing as a set would be indistinguishable, and the
        trap level 1 exists for would be untested against real data.
        """
        basis = golden.json("level1_state/selected_basis.json")
        universe = json.loads(
            (golden.directory / "input" / "universe.json").read_text(encoding="utf-8")
        )
        input_order = [tid for tid in universe["elementary_ids"] if tid in set(basis)]
        assert basis != input_order

    def test_the_window_length_is_greater_than_one(self, golden):
        """
        Because at a length of one every boundary question disappears.

        Sequence-aware splitting, the boundary gap and the window-straddling
        half of defect 3 are only exercised when windows actually span
        several scenarios.
        """
        assert golden.manifest["shape"]["seq_length"] > 1

    def test_the_scaler_was_fitted_before_reduction(self, golden):
        """
        Pinning the stage order the refactor has to reproduce.

        The scaler is fitted over the *full* elementary book and the basis is
        selected afterwards, so the captured statistics have one entry per
        original instrument rather than per survivor. A refactor that
        reduces first would produce a shorter array -- caught here as a
        shape difference rather than as a mysterious value difference later.
        """
        basis = golden.json("level1_state/selected_basis.json")
        universe = json.loads(
            (golden.directory / "input" / "universe.json").read_text(encoding="utf-8")
        )
        scaler_mean = golden.array("level1_state/scaler_mean.npy")

        assert scaler_mean.shape == (len(universe["elementary_ids"]),)
        assert scaler_mean.shape[0] > len(basis)


class TestTheCapturedSplit:
    """Defect 3's compatibility mechanism: the exact indices, replayed."""

    def test_the_three_splits_are_recorded(self, golden):
        """So the refactor can replay them rather than reproduce a flag."""
        splits = golden.json("level2_tensors/split_indices.json")
        assert set(splits) == {"train", "validation", "test"}

    def test_the_splits_do_not_overlap(self, golden):
        """
        A captured split that overlapped would bake a leak into the gate.

        Parity would then require the refactor to reproduce the leak, which
        is the one thing the compatibility flags are designed to avoid.
        """
        splits = golden.json("level2_tensors/split_indices.json")
        train, validation, test = (set(splits[name]) for name in ("train", "validation", "test"))
        assert not train & validation
        assert not train & test
        assert not validation & test

    def test_the_split_is_chronological(self, golden):
        """
        Captured with ``shuffle=False``, which is what makes it replayable.

        The original couples the split to the batch order through one flag,
        so capturing with shuffling on would have produced a random split
        that no explicit strategy could reproduce.
        """
        splits = golden.json("level2_tensors/split_indices.json")
        assert max(splits["train"]) < min(splits["validation"])
        assert max(splits["validation"]) < min(splits["test"])


@pytest.mark.usefixtures("requires_golden")
class TestTheCapturedTensors:
    """What the network actually received, which level 2 reproduces."""

    def test_static_inputs_arrive_unbatched(self, golden):
        """
        The claim that moving static inputs out of collation changed nothing.

        The original merged every static tensor into every sample and then
        compared across the batch to recover one copy. The captured
        adjacency has no batch dimension, which is what says the network
        already received exactly one copy -- so delivering it directly is a
        deletion of work, not a change of behaviour.
        """
        with np.load(golden.directory / "level2_tensors/train_batch_000.npz") as stored:
            batch_size = stored["pnl_history"].shape[0]
            assert stored["adjacency_indices"].shape[1] == 2
            assert stored["adjacency_indices"].shape[0] != batch_size
            assert stored["adjacency_dense_shape"].shape == (2,)

    def test_the_dynamic_input_is_windowed(self, golden):
        """
        Shaped (batch, window, instrument), with the window from the manifest.

        A refactor that produced (batch, instrument, window) would be a
        transpose that trains without error and learns nothing useful.
        """
        with np.load(golden.directory / "level2_tensors/train_batch_000.npz") as stored:
            basis = golden.json("level1_state/selected_basis.json")
            assert stored["pnl_history"].shape[1] == golden.manifest["shape"]["seq_length"]
            assert stored["pnl_history"].shape[2] == len(basis)

    def test_the_unused_key_is_present_in_the_baseline(self, golden):
        """
        Recording that the original *declares* ``elementary_indices``.

        Its forward pass never reads it. The refactored signature omits it,
        and this test documents what is being dropped -- so if parity later
        shifts, the first question is whether the key was load-bearing after
        all.
        """
        with np.load(golden.directory / "level2_tensors/train_batch_000.npz") as stored:
            assert "elementary_indices" in stored.files
```

---

## 5. `tests/rade_qnet/testkit/test_testkit_parity.py`

21337 bytes · SHA-256 `081eb6c29a54e0ff`

```python
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

from src.rade_qnet.core.lifecycle.errors import ContractError
from src.rade_qnet.testkit.parity import (
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
```

