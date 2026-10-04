"""
Tests for the split strategies.

Splitting is where leakage is introduced, and leakage is the failure mode that
produces a *better* result than the truth. Nothing in a metric, a loss curve or
a report says a split leaked; the model simply scores well and keeps scoring
well until it meets real data.

So these tests are mostly about disjointness and ordering, asserted directly on
the indices rather than inferred from a downstream score. An assertion on a
metric cannot distinguish a model that generalises from a split that leaks.

The boundary gap gets the most attention. With a sequence length above one,
every sample is a *window* ending at its label, so a validation window whose
label sits one row after the training split's last row overlaps the training
data by ``length - 1`` rows. Dropping those rows is the whole purpose of the
gap, and it is invisible in the split fractions.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.runtime.errors import SpecError
from src.rade_qnet.core.spec.data import (
    ChronologicalSplitSpec,
    ExplicitSplitSpec,
    GroupedSplitSpec,
    PurgedKFoldSplitSpec,
)
from src.rade_qnet.sources.dataset.splits import (
    boundary_gap,
    split_by_group,
    split_chronologically,
    split_explicitly,
    split_purged_kfold,
    split_scenarios,
)


class TestBoundaryGap:
    """The width that stops a window straddling two splits."""

    def test_a_non_sequential_model_needs_no_gap(self):
        """
        A length of one means each sample is a single row.

        Nothing overlaps, so a gap would discard data for no reason.
        """
        assert boundary_gap(1) == 0

    def test_the_gap_is_one_short_of_the_window(self):
        """
        A window of length ``n`` reaches ``n - 1`` rows back from its label.

        So dropping ``n - 1`` rows is exactly enough, and dropping ``n`` would
        waste one.
        """
        assert boundary_gap(20) == 19

    def test_an_explicit_gap_is_added_on_top(self):
        """
        For a target that looks forward as well as back.

        A five-day forward return observed at the split boundary is not known
        until five days into the next split, so the extra scenarios have to be
        dropped on top of the window.
        """
        assert boundary_gap(20, 5) == 24


class TestChronologicalSplit:
    """Time order preserved, which is the only honest split for a forecast."""

    def test_the_splits_are_in_time_order(self):
        """
        Training rows precede validation rows, which precede test rows.

        A random split of a time series trains on the future to predict the
        past, and scores spectacularly.
        """
        indices = split_chronologically(ChronologicalSplitSpec(), n_scenarios=1000)
        assert indices.train.max() < indices.validation.min()
        assert indices.validation.max() < indices.test.min()

    def test_the_splits_are_disjoint(self):
        """No scenario appears in two splits."""
        indices = split_chronologically(ChronologicalSplitSpec(), n_scenarios=1000)
        combined = np.concatenate([indices.train, indices.validation, indices.test])
        assert combined.size == np.unique(combined).size

    def test_every_scenario_is_used_when_no_gap_is_needed(self):
        """
        At a sequence length of one, nothing is discarded.

        A split that silently dropped rows would make the fractions a lie.
        """
        indices = split_chronologically(ChronologicalSplitSpec(), n_scenarios=1000)
        assert sum(indices.sizes.values()) == 1000

    def test_a_sequence_length_opens_a_gap_between_splits(self):
        """
        The leakage this module exists to prevent.

        With a window of twenty, a validation sample labelled at the first row
        after training would read nineteen training rows. The gap is what makes
        the splits disjoint in *windows* rather than merely in labels.
        """
        indices = split_chronologically(
            ChronologicalSplitSpec(), n_scenarios=1000, sequence_length=20
        )
        assert indices.validation.min() - indices.train.max() > 19
        assert indices.test.min() - indices.validation.max() > 19

    def test_the_gap_costs_scenarios(self):
        """
        Which is the price of the guarantee, and should be visible.

        A reader comparing a run with a sequence length against one without
        should see fewer usable scenarios, not the same number.
        """
        gapped = split_chronologically(
            ChronologicalSplitSpec(), n_scenarios=1000, sequence_length=20
        )
        assert sum(gapped.sizes.values()) < 1000

    def test_a_fraction_that_rounds_to_zero_is_refused(self):
        """
        Found in this implementation, by its own end-to-end test.

        Two groups at a fifteen percent validation fraction rounds to zero, so
        every row went to training and the run trained with no validation and
        no test -- silently, because an empty split is a legitimate
        configuration when the fraction is zero. The distinction is whether
        the caller asked for one.
        """
        # Three scenarios at a fifteen percent fraction: round(0.45) is zero.
        with pytest.raises(SpecError, match="rounds to zero"):
            split_chronologically(ChronologicalSplitSpec(), n_scenarios=3)

    def test_a_deliberately_empty_split_is_allowed(self):
        """
        The other half of the previous rule.

        A fraction of zero means the caller does not want the split, which is
        a different statement from a fraction that was too small to honour.
        """
        indices = split_chronologically(
            ChronologicalSplitSpec(validation_fraction=0.0, test_fraction=0.2),
            n_scenarios=100,
        )
        assert indices.validation.size == 0
        assert indices.train.size > 0

    def test_the_result_is_deterministic(self):
        """A chronological split has nothing to randomise, so it cannot vary."""
        first = split_chronologically(ChronologicalSplitSpec(), n_scenarios=500)
        second = split_chronologically(ChronologicalSplitSpec(), n_scenarios=500)
        assert np.array_equal(first.train, second.train)


class TestPurgedKFold:
    """A fold with its neighbours removed, for overlapping-label problems."""

    def test_the_folds_are_disjoint(self):
        """The basic requirement of any split."""
        indices = split_purged_kfold(PurgedKFoldSplitSpec(), n_scenarios=1000)
        combined = np.concatenate([indices.train, indices.validation, indices.test])
        assert combined.size == np.unique(combined).size

    def test_an_embargo_separates_training_from_the_held_out_fold(self):
        """
        What distinguishes this from plain k-fold.

        Without an embargo, a training label whose window overlaps a
        validation label leaks across the boundary -- and with overlapping
        labels that happens at every fold edge rather than only at the ends.
        """
        embargoed = split_purged_kfold(PurgedKFoldSplitSpec(embargo_scenarios=25), n_scenarios=1000)
        distances = np.abs(embargoed.train[:, None] - embargoed.validation[None, :])
        assert distances.min() > 25

    def test_each_fold_holds_out_a_different_slice(self):
        """
        One fold per run, so a sweep is a job set rather than a loop in here.

        A fold argument that was ignored would make every member of that sweep
        the same run, and the averaged result would look far more stable than
        it is.
        """
        first = split_purged_kfold(PurgedKFoldSplitSpec(), n_scenarios=1000, fold=0)
        second = split_purged_kfold(PurgedKFoldSplitSpec(), n_scenarios=1000, fold=1)
        assert not np.array_equal(first.validation, second.validation)


class TestGroupSplit:
    """Whole groups held out, so no group appears on both sides."""

    def test_no_group_spans_two_splits(self):
        """
        The property the strategy exists for.

        Splitting rows rather than groups puts the same entity in train and
        test, and the model learns that entity rather than the relationship.
        """
        labels = np.repeat(np.arange(20), 50)
        indices = split_by_group(
            GroupedSplitSpec(group_key="entity"), n_scenarios=labels.size, group_labels=labels
        )
        train_groups = set(labels[indices.train].tolist())
        test_groups = set(labels[indices.test].tolist())
        assert not train_groups & test_groups

    def test_a_group_fraction_that_rounds_to_zero_is_refused(self):
        """
        The case that produced the defect this guard was written for.

        Two groups at a fifteen percent fraction rounds to zero groups, so the
        run had no validation and no test and nothing said so.
        """
        labels = np.repeat(np.arange(2), 200)
        with pytest.raises(SpecError, match="rounds to zero"):
            split_by_group(
                GroupedSplitSpec(group_key="entity"), n_scenarios=labels.size, group_labels=labels
            )

    def test_the_assignment_is_reproducible_under_a_seed(self):
        """Group assignment is random, so a seed must pin it."""
        labels = np.repeat(np.arange(20), 50)
        first = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=3,
        )
        second = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=3,
        )
        assert np.array_equal(first.test, second.test)

    def test_a_different_seed_gives_a_different_assignment(self):
        """
        Otherwise the seed is decorative.

        A strategy that ignored its seed would make a repeated experiment look
        robust when it was the same experiment.
        """
        labels = np.repeat(np.arange(20), 50)
        first = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=1,
        )
        second = split_by_group(
            GroupedSplitSpec(group_key="entity"),
            n_scenarios=labels.size,
            group_labels=labels,
            seed=2,
        )
        assert not np.array_equal(first.test, second.test)


class TestExplicitSplit:
    """Caller-supplied indices, passed through and validated."""

    def test_the_supplied_indices_are_used_unchanged(self):
        """
        The point of the strategy: the caller knows something the spec cannot.

        A reproduction of a published result, or a split aligned to a market
        regime, cannot be expressed as a fraction.
        """
        indices = split_explicitly(
            ExplicitSplitSpec(train=(0, 1, 2, 3), validation=(5, 6), test=(8, 9)),
            n_scenarios=10,
        )
        assert indices.train.tolist() == [0, 1, 2, 3]
        assert indices.test.tolist() == [8, 9]

    def test_an_index_past_the_end_is_refused(self):
        """
        Out-of-range indices produce a silent wrap in numpy.

        A negative index reads from the other end of the array, so a mistake
        here trains on the wrong rows rather than failing.
        """
        with pytest.raises(SpecError):
            split_explicitly(
                ExplicitSplitSpec(train=(0, 1), validation=(2,), test=(99,)),
                n_scenarios=10,
            )


class TestTheDispatcher:
    """One entry point, so a caller never matches on the split kind."""

    @pytest.mark.parametrize(
        "spec",
        [
            ChronologicalSplitSpec(),
            PurgedKFoldSplitSpec(),
        ],
    )
    def test_each_strategy_is_reachable_through_the_dispatcher(self, spec):
        """So adding a strategy does not change any caller."""
        indices = split_scenarios(spec, n_scenarios=1000)
        assert indices.train.size > 0

    def test_a_group_split_needs_group_labels(self):
        """
        Refused rather than defaulted.

        Defaulting to one group per row would silently degrade a group split
        to a random one, which is exactly the leakage it exists to prevent.
        """
        with pytest.raises(SpecError):
            split_scenarios(GroupedSplitSpec(group_key="entity"), n_scenarios=100)
