"""
Tests for the batch source protocol.

:class:`BatchSource` is the one abstraction that lets a supervised run and an
interactive run share a training loop. A dataset yields batches until the
epoch ends; an environment rollout yields batches until told to stop. The only
difference the loop sees is ``steps_per_epoch``, where ``None`` means unbounded
-- and that single field is what decides which side drives the loop.

The protocol is structural, so a source need not import the framework to
satisfy it. These tests check that this really is the case, because a protocol
that quietly requires inheritance is just an ABC with extra steps.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.contract.source import BatchSource
from src.rade_qnet.testkit.conformance import check_source
from src.rade_qnet.testkit.fixtures import SyntheticTensorSource, make_signature


@pytest.fixture
def source():
    """
    Provide a small compliant source.

    Returns
    -------
    SyntheticTensorSource
        Six samples in batches of two.
    """
    return SyntheticTensorSource(
        features=np.zeros((6, 4)),
        targets=np.zeros(6),
        batch_size=2,
    )


class TestStructuralConformance:
    """Satisfying the protocol must not require importing the framework."""

    def test_a_compliant_source_is_recognised(self, source):
        """The runtime-checkable protocol routes on shape alone."""
        assert isinstance(source, BatchSource)

    def test_an_unrelated_object_is_not_recognised(self):
        """A protocol that matches everything would route nothing."""
        assert not isinstance(object(), BatchSource)

    def test_a_source_defined_without_the_import_is_recognised(self):
        """
        The point of making this structural.

        A user's dataset class, written against no framework type at all,
        plugs in. Requiring a base class would mean every data module in
        every model imports ``core``.
        """

        class Standalone:
            """A source that has never heard of rade_qnet."""

            signature = make_signature()
            static: dict[str, object] = {}
            steps_per_epoch = 1
            n_samples = 2

            def batches(self):
                """Yield one batch."""
                return iter([{"features": np.zeros((2, 4)), "target": np.zeros(2)}])

        assert isinstance(Standalone(), BatchSource)

    def test_a_source_missing_batches_is_not_recognised(self):
        """
        The member that does the work.

        ``isinstance`` is what routes a payload to the batched path, so a
        near-miss has to be rejected rather than routed and then crash.
        """

        class Incomplete:
            """Declares everything but the method that yields data."""

            signature = make_signature()
            static: dict[str, object] = {}
            steps_per_epoch = 1
            n_samples = 2

        assert not isinstance(Incomplete(), BatchSource)


class TestBoundedAndUnbounded:
    """``steps_per_epoch`` is the ML/RL unifier."""

    def test_a_bounded_source_reports_its_length(self, source):
        """
        Six samples in batches of two is three steps.

        The loop uses this for progress reporting and for scheduler steps
        per epoch.
        """
        assert source.steps_per_epoch == 3

    def test_a_partial_final_batch_is_counted(self):
        """
        Seven samples in batches of two is four steps, not three.

        Rounding down would silently drop the tail of every epoch.
        """
        partial = SyntheticTensorSource(
            features=np.zeros((7, 4)), targets=np.zeros(7), batch_size=2
        )
        assert partial.steps_per_epoch == 4

    def test_an_unbounded_source_reports_none(self):
        """
        ``None`` is how an interactive source says "I do not end".

        The loop then takes its stopping condition from the training spec
        rather than from the source, which is exactly the behavioural
        difference between the supervised and interactive cases.
        """
        unbounded = SyntheticTensorSource(
            features=np.zeros((6, 4)),
            targets=np.zeros(6),
            batch_size=2,
            unbounded=True,
        )
        assert unbounded.steps_per_epoch is None

    def test_an_unbounded_source_keeps_yielding(self):
        """
        Demonstrated rather than asserted from the field.

        A source that declares itself unbounded and then stops would hang a
        loop waiting for data that never arrives.
        """
        unbounded = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            unbounded=True,
        )
        produced = 0
        for _ in unbounded.batches():
            produced += 1
            if produced > 5:
                break
        assert produced > 2


class TestIterationContract:
    """Re-iterability is required, and it is easy to get wrong."""

    def test_a_source_can_be_traversed_more_than_once(self, source):
        """
        A training loop traverses the source once per epoch.

        A bare generator satisfies every other clause of the protocol and
        then yields nothing from epoch two onward -- which looks like a model
        that stopped learning, not like a data bug.
        """
        first = [batch["target"].shape[0] for batch in source.batches()]
        second = [batch["target"].shape[0] for batch in source.batches()]
        assert first == second
        assert first

    def test_each_traversal_returns_a_fresh_iterator(self, source):
        """Two concurrent passes must not consume each other's batches."""
        assert source.batches() is not source.batches()

    def test_every_batch_matches_the_declared_signature(self, source):
        """
        The keys the model will look for are the keys that arrive.

        Checked against the source's own signature, so a renamed feature
        fails here rather than as a shape error inside a forward pass.
        """
        for batch in source.batches():
            source.signature.validate_batch_keys(batch, where="test source")

    def test_the_sample_count_matches_what_is_yielded(self, source):
        """
        A dishonest count silently rescales every per-sample metric.

        Which produces a plausible wrong number rather than an error.
        """
        yielded = sum(batch["target"].shape[0] for batch in source.batches())
        assert yielded == source.n_samples


class TestStaticInputs:
    """Static inputs are declared, not discovered."""

    def test_static_inputs_default_to_empty(self, source):
        """Most sources have none."""
        assert source.static == {}

    def test_declared_static_inputs_are_in_the_signature(self):
        """
        A static input absent from the signature would never be placed.

        The engine uploads what the signature declares; anything else is
        carried and ignored.
        """
        adjacency = np.eye(4)
        with_static = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            static={"adjacency": adjacency},
        )
        assert "adjacency" in with_static.signature.static

    def test_static_inputs_are_not_repeated_in_every_batch(self):
        """
        The behaviour the static field exists to produce.

        Collating an adjacency matrix per sample is pure waste, and the
        previous implementation then compared the copies to each other.
        """
        with_static = SyntheticTensorSource(
            features=np.zeros((4, 4)),
            targets=np.zeros(4),
            batch_size=2,
            static={"adjacency": np.eye(4)},
        )
        assert all("adjacency" not in batch for batch in with_static.batches())


class TestConformanceAgreement:
    """The shipped checker agrees with these tests."""

    def test_a_compliant_source_passes_the_conformance_suite(self, source):
        """
        What a model author runs instead of writing the tests above.

        If this disagreed with the tests in this file, one of the two would
        be giving authors the wrong answer.
        """
        assert check_source(source).passed
