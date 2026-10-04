"""
Tests for the data payloads a build hands to an engine.

Two things here are worth more than the rest. :class:`SplitIndices` rejects
overlapping splits on construction, because an overlap produces an encouraging
validation score and a model that fails in production -- the most expensive
mistake available at this layer. And :class:`DataBundle` is generic over the
payload type, which is what lets one pipeline serve a batched Torch run and a
one-shot XGBoost run without branching on the engine.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import (
    SPLIT_NAMES,
    DataBundle,
    DataLineage,
    SplitIndices,
    TensorBatchData,
)
from src.rade_qnet.core.lifecycle.errors import ContractError, SpecError
from src.rade_qnet.testkit.fixtures import (
    make_lineage,
    make_signature,
    make_tensor_bundle,
)


def _indices(train, validation=(), test=()):
    """
    Build split indices from plain sequences.

    Parameters
    ----------
    train, validation, test
        Scenario indices.

    Returns
    -------
    SplitIndices
        The constructed splits.
    """
    return SplitIndices(
        train=np.array(train, dtype=np.int64),
        validation=np.array(validation, dtype=np.int64),
        test=np.array(test, dtype=np.int64),
    )


class TestSplitIndices:
    """Disjointness is enforced where it is cheapest to enforce."""

    def test_disjoint_splits_are_accepted(self):
        """The normal case."""
        assert _indices([0, 1, 2], [3, 4], [5]).sizes == {
            "train": 3,
            "validation": 2,
            "test": 1,
        }

    def test_an_empty_training_split_is_rejected(self):
        """There is nothing to fit."""
        with pytest.raises(SpecError, match="training"):
            _indices([])

    def test_empty_validation_and_test_splits_are_permitted(self):
        """
        A final refit on everything is a legitimate configuration.

        Requiring held-out data would make that inexpressible.
        """
        assert _indices([0, 1]).sizes == {"train": 2, "validation": 0, "test": 0}

    @pytest.mark.parametrize(
        ("first", "second", "third"),
        [
            ([0, 1], [1, 2], [3]),
            ([0, 1], [2], [1]),
            ([0], [1, 2], [2]),
        ],
    )
    def test_any_overlapping_pair_is_rejected(self, first, second, third):
        """
        Every pair is checked, not just train against validation.

        A test index that also appears in validation inflates the final
        number just as effectively.
        """
        with pytest.raises(SpecError, match="disjoint"):
            _indices(first, second, third)

    def test_the_overlap_message_names_the_offending_splits(self):
        """
        So the reader knows which boundary moved.

        A message saying only "splits overlap" sends them to read all three.
        """
        with pytest.raises(SpecError, match="train and validation"):
            _indices([0, 1], [1])

    def test_an_unknown_split_name_is_rejected_on_lookup(self):
        """A typo must not return ``None`` and train on nothing."""
        with pytest.raises(ContractError, match="unknown split"):
            _indices([0])["holdout"]

    def test_lineage_form_is_json_encodable(self):
        """
        Indices are stored, not the fractions that produced them.

        Which is what lets a saved bundle be re-evaluated against exactly the
        data it was trained against, even after the split logic changes.
        """
        rendered = _indices([0, 1], [2]).as_lineage()
        assert rendered["train"] == (0, 1)
        assert all(isinstance(value, int) for value in rendered["validation"])

    def test_lineage_form_covers_every_split(self):
        """An absent split records as empty rather than as missing."""
        assert set(_indices([0]).as_lineage()) == set(SPLIT_NAMES)


class TestTensorBatchData:
    """The batched payload, and the static-input field that earns its place."""

    def test_static_inputs_default_to_empty(self):
        """Most models have none."""
        assert TensorBatchData(loader=[]).static == {}

    def test_static_inputs_are_held_once(self):
        """
        The field that retires a concrete defect.

        The previous implementation merged every static tensor into every
        sample and then compared them across the batch during collation to
        confirm something true by construction. Holding them here delivers
        the same tensors to the same place with no comparison at all.
        """
        adjacency = np.eye(3)
        payload = TensorBatchData(loader=[], static={"adjacency": adjacency})
        assert payload.static["adjacency"] is adjacency

    def test_an_unbounded_source_records_no_batch_count(self):
        """
        ``None`` is the honest answer for a stream.

        An interactive source has no epoch length, and inventing one would
        make the loop stop at an arbitrary point.
        """
        assert TensorBatchData(loader=[], n_batches=None).n_batches is None

    def test_the_payload_is_frozen(self):
        """A stage must not mutate another stage's data."""
        with pytest.raises(AttributeError):
            TensorBatchData(loader=[]).n_samples = 5


class TestDataLineage:
    """Lineage is JSON and must stay JSON."""

    def test_it_round_trips_exactly(self):
        """
        Stored in a bundle, so it has to reload identically.

        A lineage that does not round-trip cannot support the claim that a
        bundle records how its data was built.
        """
        lineage = make_lineage()
        assert DataLineage.model_validate_json(lineage.model_dump_json()) == lineage

    def test_notes_default_to_empty(self):
        """Annotations are optional."""
        assert make_lineage().notes == {}

    def test_notes_carry_free_form_annotations(self):
        """
        Where a reward-shaping change or a parity flag becomes visible.

        Without it, two runs that are not comparable look comparable.
        """
        lineage = make_lineage(notes={"reward_shaping": "v2"})
        assert lineage.notes["reward_shaping"] == "v2"

    def test_a_negative_scenario_count_is_rejected(self):
        """Not a quantity that can be negative."""
        with pytest.raises(Exception, match="greater than or equal"):
            DataLineage(
                source_fingerprint="a",
                spec_digest="b",
                split_indices={},
                n_scenarios=-1,
                framework_version="0.1.0",
                created_at=datetime.now(UTC),
            )

    def test_an_absent_entity_axis_is_expressible(self):
        """Not every problem has one, and zero is not the same as none."""
        assert make_lineage(n_entities=None).n_entities is None


class TestDataBundle:
    """The generic container, and the two invariants it enforces."""

    def test_a_bundle_without_training_data_is_rejected(self):
        """Nothing downstream can proceed, so it fails at the boundary."""
        with pytest.raises(ContractError, match="train"):
            DataBundle(
                splits={"test": object()},
                signature=make_signature(),
                state=make_tensor_bundle().state,
                lineage=make_lineage(),
            )

    def test_an_unknown_split_name_is_rejected(self):
        """
        Fixed names, so reports and metric tables agree on what to call them.

        A ``holdout`` key would silently never be evaluated.
        """
        with pytest.raises(ContractError, match="holdout"):
            DataBundle(
                splits={"train": object(), "holdout": object()},
                signature=make_signature(),
                state=make_tensor_bundle().state,
                lineage=make_lineage(),
            )

    def test_split_names_are_returned_in_canonical_order(self):
        """
        Not insertion order, so a report's column order is stable.

        Two runs whose splits were assembled in different orders should
        produce identically ordered tables.
        """
        bundle = make_tensor_bundle(splits=("test", "train", "validation"))
        assert bundle.split_names == ("train", "validation", "test")

    def test_an_absent_split_lists_what_is_available(self):
        """
        The usual cause is a run with no validation fraction.

        Meeting a callback that monitors validation loss, which is exactly
        the case where listing the present splits answers the question.
        """
        bundle = make_tensor_bundle(splits=("train",))
        with pytest.raises(ContractError, match="available splits"):
            bundle.split("validation")

    def test_presence_can_be_tested_without_raising(self):
        """So a caller can branch rather than catch."""
        bundle = make_tensor_bundle(splits=("train",))
        assert bundle.has_split("train")
        assert not bundle.has_split("test")

    def test_the_same_bundle_type_carries_either_payload(self):
        """
        The property that lets one pipeline serve every engine.

        Both of these are ``DataBundle``; only the payload differs, and no
        pipeline stage has to know which.
        """
        assert isinstance(make_tensor_bundle(), DataBundle)

    def test_the_payload_type_is_preserved(self):
        """The generic parameter is real, not decorative."""
        assert isinstance(make_tensor_bundle().split("train"), TensorBatchData)

    def test_the_bundle_is_frozen(self):
        """A build's output must not be edited by a later stage."""
        with pytest.raises(AttributeError):
            make_tensor_bundle().splits = {}
