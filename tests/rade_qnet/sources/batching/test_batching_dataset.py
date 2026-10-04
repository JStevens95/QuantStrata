"""
Tests for the dataset batch source.

This module carries two of the ten defects and one bug found while building
Phase 2, and each has a test here that fails against the broken behaviour.

**Defect 3 -- one flag drove two decisions.** The previous implementation used
a single ``shuffle`` setting for both the train/validation/test *split* and the
*batch order*. Those are unrelated decisions: shuffling batch order is
standard and desirable, while shuffling a time-series split is leakage. The
test is that turning shuffling on changes the order of the batches and nothing
about which rows are in the split.

**Defect 4 -- static inputs were collated per sample.** A graph adjacency
matrix was merged into every sample and then compared against itself
``batch_size`` times per batch to confirm something true by construction. The
test is that static inputs are delivered once, outside the batch stream, and
that a batch carries only dynamic inputs.

**The alignment bug.** A training source reshuffles between passes, which is
correct for training and wrong for anything that needs two passes to line up.
``ordered()`` is the fix, and the tests assert both halves of its contract:
stable order, and the same samples.
"""

from __future__ import annotations

import csv
from dataclasses import replace

import numpy as np
import pytest

from src.rade_qnet.core.contract.data import TARGET_KEY
from src.rade_qnet.core.contract.source import BatchSource, OrderedSource
from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.core.spec.data import LoaderSpec, SequenceSpec, TabularSourceSpec
from src.rade_qnet.sources.batching.dataset import DatasetSource, sources_for
from src.rade_qnet.sources.dataset.tabular import TabularDataModule


@pytest.fixture
def prepared(tmp_path):
    """
    Build a prepared dataset from a small CSV file.

    Returns
    -------
    tuple
        The prepared dataset and the source spec that produced it.
    """
    rng = np.random.default_rng(3)
    features = rng.normal(size=(400, 3))
    target = features.sum(axis=1)

    path = tmp_path / "data.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "c", "target"])
        writer.writerows(np.column_stack([features, target]).tolist())

    spec = TabularSourceSpec(path=path)
    return TabularDataModule().build(spec, seed=3), spec


def targets_of(source):
    """
    Drain one pass of a source's targets.

    Parameters
    ----------
    source
        A bounded source.

    Returns
    -------
    numpy.ndarray
        Flat target vector, in the order the source yielded it.
    """
    return np.concatenate([np.ravel(np.asarray(batch[TARGET_KEY])) for batch in source.batches()])


class TestTheProtocol:
    """Structural conformance, which is what lets a user's source substitute."""

    def test_a_dataset_source_satisfies_batch_source(self, prepared):
        """Without inheriting from it, which is the point of the protocol."""
        dataset, spec = prepared
        sources = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)
        assert isinstance(sources["train"], BatchSource)

    def test_a_dataset_source_is_also_iterable(self, prepared):
        """
        So one object can be both the source and the loader.

        Without it the pipeline would have to carry the sources alongside the
        data bundle, and the two could drift apart.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert sum(1 for _ in source) == source.steps_per_epoch

    def test_the_batch_count_is_exact_not_estimated(self, prepared):
        """
        Callbacks, schedules and progress all compute from it.

        An estimate would make a cosine schedule anneal over the wrong period,
        which looks like a badly chosen learning rate.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert sum(1 for _ in source.batches()) == source.steps_per_epoch


class TestDefectThreeSplitAndShuffleAreSeparate:
    """One flag drove two unrelated decisions; now it drives one."""

    def test_shuffling_changes_the_batch_order(self, prepared):
        """
        Which is the legitimate purpose of the flag.

        A source that ignored it would train on the same batch order every
        epoch, which correlates consecutive updates and slows convergence.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
            seed=1,
        )
        assert not np.array_equal(targets_of(source), targets_of(source))

    def test_shuffling_does_not_change_which_rows_are_in_the_split(self, prepared):
        """
        The half that was leakage: a shuffled *split*, not a shuffled order.

        The multiset of targets must be identical with shuffling on and off,
        because permuting a split cannot introduce a member of another one.
        """
        dataset = prepared[0]
        shuffled = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
            seed=1,
        )
        ordered = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=False),
            sequence=SequenceSpec(),
            seed=1,
        )
        assert np.allclose(np.sort(targets_of(shuffled)), np.sort(targets_of(ordered)))

    def test_the_held_out_splits_are_never_shuffled(self, prepared):
        """
        Shuffling a split that is only ever read once buys nothing.

        And it costs the alignment that scoring depends on, so the source does
        not do it regardless of the flag.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="test",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
            seed=1,
        )
        assert np.array_equal(targets_of(source), targets_of(source))

    def test_the_batch_order_is_reproducible_under_a_seed(self, prepared):
        """
        Two runs with the same seed must see the same orders in the same epochs.

        Otherwise a reproduced run is not the same run, and a difference in
        its score cannot be attributed.
        """
        dataset = prepared[0]

        def first_pass(seed):
            """Return the first pass's targets for a given seed."""
            source = DatasetSource(
                dataset,
                split="train",
                loader=LoaderSpec(shuffle=True),
                sequence=SequenceSpec(),
                seed=seed,
            )
            return targets_of(source)

        assert np.array_equal(first_pass(7), first_pass(7))
        assert not np.array_equal(first_pass(7), first_pass(8))


class TestDefectFourStaticInputsLeavePerSampleCollation:
    """Delivered once, outside the batch stream."""

    def test_static_inputs_are_exposed_separately(self, prepared):
        """
        So the engine uploads them to the device a single time.

        Empty for a tabular model, which is the common case and the one that
        must not pay for the feature.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert source.static == {}

    def test_a_declared_static_input_is_served_from_the_dataset(self, prepared):
        """
        The last leg of the route from the data module to the engine.

        The empty default above is the common case, but it is also what a
        broken route looks like -- so a test that only ever saw an empty
        mapping would pass against a source that returned ``{}``
        unconditionally, which is exactly the bug this covers. Served by
        reference rather than copied, because the point of a static input
        is that one array is uploaded once.
        """
        dataset, spec = prepared
        adjacency = np.eye(3, dtype=np.float32)
        carrying = replace(dataset, static_inputs={"adjacency": adjacency})
        source = sources_for(carrying, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        assert source.static["adjacency"] is adjacency

    def test_every_split_sees_the_same_static_inputs(self, prepared):
        """
        Because they do not vary by scenario, let alone by split.

        A graph fitted on the training universe is the graph the model was
        trained with, and evaluating against a different one would measure
        a model that never existed.
        """
        dataset, spec = prepared
        carrying = replace(dataset, static_inputs={"adjacency": np.eye(3, dtype=np.float32)})
        sources = sources_for(carrying, loader=spec.loader, sequence=spec.transforms.sequence)
        statics = [source.static["adjacency"] for source in sources.values()]
        assert all(array is statics[0] for array in statics)

    def test_a_batch_carries_only_dynamic_inputs_and_the_target(self, prepared):
        """
        The deleted comparison, asserted as an absence.

        The old implementation merged static tensors into every sample and
        then ran ``torch.equal`` across the batch per static key per batch, to
        confirm what was true by construction.
        """
        dataset, spec = prepared
        source = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)[
            "train"
        ]
        batch = next(source.batches())
        assert set(batch) == {"features", TARGET_KEY}


class TestOrderedView:
    """The fix for the two-pass alignment bug."""

    def test_a_shuffling_source_offers_an_ordered_view(self, prepared):
        """
        Declared as a capability, so ``isinstance`` routes rather than guesses.

        Adding ``ordered`` to ``BatchSource`` itself would have silently
        dropped every existing structural implementation out of the protocol.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        )
        assert isinstance(source, OrderedSource)

    def test_the_ordered_view_is_stable_across_passes(self, prepared):
        """
        The property the whole capability exists for.

        Without it, scoring pairs each prediction with another row's target;
        the counts match, every metric computes, and a correctly trained model
        reports a negative r-squared on its own training split.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        ).ordered()
        assert np.array_equal(targets_of(source), targets_of(source))

    def test_the_ordered_view_has_the_same_samples(self, prepared):
        """
        So the fix for an alignment bug does not introduce a sampling bug.

        A stable order achieved by dropping rows would be harder to notice
        than the problem it replaced.
        """
        dataset = prepared[0]
        shuffled = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        )
        assert np.allclose(np.sort(targets_of(shuffled.ordered())), np.sort(targets_of(shuffled)))

    def test_a_source_that_never_shuffles_returns_itself(self, prepared):
        """
        Nothing is copied when nothing needs to change.

        A held-out split is already stable, so an ordered view of it is the
        same object.
        """
        dataset = prepared[0]
        source = DatasetSource(dataset, split="test", loader=LoaderSpec(), sequence=SequenceSpec())
        assert source.ordered() is source

    def test_the_ordered_view_does_not_disturb_the_original(self, prepared):
        """
        Because a training loop may still be holding it.

        Flipping the flag in place would quietly change what the model is
        learning from, mid-run.
        """
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(shuffle=True),
            sequence=SequenceSpec(),
        )
        source.ordered()
        assert not np.array_equal(targets_of(source), targets_of(source))


class TestSampleCounts:
    """Honest counts, including under ``drop_last``."""

    def test_dropping_the_last_batch_reduces_the_reported_samples(self, prepared):
        """
        Otherwise a metric denominator counts rows the loop never saw.

        The error would be small -- less than one batch in a split -- and so
        would never be large enough to notice.
        """
        dataset = prepared[0]
        kept = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(batch_size=32, drop_last=False),
            sequence=SequenceSpec(),
        )
        dropped = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(batch_size=32, drop_last=True),
            sequence=SequenceSpec(),
        )
        assert dropped.n_samples <= kept.n_samples
        assert dropped.n_samples == dropped.steps_per_epoch * 32

    def test_the_reported_count_matches_what_is_yielded(self, prepared):
        """The claim and the behaviour, checked against each other."""
        dataset = prepared[0]
        source = DatasetSource(
            dataset,
            split="train",
            loader=LoaderSpec(batch_size=32, drop_last=True),
            sequence=SequenceSpec(),
        )
        assert targets_of(source).size == source.n_samples


class TestRefusals:
    """Loud failures where a quiet one would train on nothing."""

    def test_an_unknown_split_is_refused(self, prepared):
        """A typo should not produce an empty source."""
        dataset = prepared[0]
        with pytest.raises(ContractError):
            DatasetSource(
                dataset,
                split="validate",
                loader=LoaderSpec(),
                sequence=SequenceSpec(),
            )

    def test_a_window_wider_than_the_split_is_refused(self, prepared):
        """
        Refused loudly rather than yielding an empty source.

        The alternative is an epoch that completes instantly having trained
        on nothing, and a flat loss curve that reads as a learning-rate fault.
        """
        dataset = prepared[0]
        with pytest.raises(ContractError, match="no usable sample"):
            DatasetSource(
                dataset,
                split="test",
                loader=LoaderSpec(),
                sequence=SequenceSpec(length=10_000),
            )


class TestSourcesFor:
    """One call produces every non-empty split's source."""

    def test_one_source_per_non_empty_split(self, prepared):
        """So a caller never has to know which splits exist."""
        dataset, spec = prepared
        sources = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)
        assert set(sources) == {"train", "validation", "test"}

    def test_every_source_declares_the_dataset_signature(self, prepared):
        """
        A source does not get to invent its own interface.

        Two splits with different signatures would mean the model was served
        differently shaped data at training and scoring time.
        """
        dataset, spec = prepared
        sources = sources_for(dataset, loader=spec.loader, sequence=spec.transforms.sequence)
        assert all(source.signature == dataset.signature for source in sources.values())
