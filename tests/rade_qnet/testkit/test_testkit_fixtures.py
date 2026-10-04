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
