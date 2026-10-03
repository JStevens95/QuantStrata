"""
Tests for rolling-window construction.

Three things go wrong with windows and all three are quiet.

A window that reaches back past the start of the array wraps around in numpy
rather than failing, so the first few samples of every split are built from
the *end* of the history. A window that straddles a split boundary reads
held-out rows into a training sample. And an off-by-one in which row a window
is labelled by shifts every target by one scenario, which degrades a score
rather than breaking a run.

So the tests here assert on exact index arithmetic rather than on shapes.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_xl.core.contract.data import SplitIndices
from src.rade_xl.core.runtime.errors import ContractError
from src.rade_xl.sources.dataset.transforms.sequence import (
    extract_windows,
    usable_labels,
    windows_stay_within,
)


class TestUsableLabels:
    """Which rows can be the last scenario of a complete window."""

    def test_a_non_sequential_model_uses_every_row(self):
        """
        At a length of one, every row is a complete window of itself.

        Dropping anything here would silently shrink the dataset.
        """
        labels = usable_labels(np.arange(100), length=1)
        assert labels.tolist() == list(range(100))

    def test_the_first_rows_of_the_dataset_are_dropped(self):
        """
        Because no complete window ends there.

        A window of twenty ending at scenario five would need scenarios
        minus-fourteen to five, and numpy reads negative indices from the far
        end of the array -- so the sample would be built from the most recent
        data and labelled with the oldest.
        """
        labels = usable_labels(np.arange(0, 100), length=20)
        assert labels.min() == 19
        assert labels.size == 100 - 19

    def test_a_later_split_keeps_every_label(self):
        """
        The index is absolute, not relative to the split.

        A split starting at scenario fifty has fifty scenarios of history
        behind it, so every one of its labels has a complete window. Filtering
        relative to the split would discard nineteen perfectly good samples
        from each of validation and test -- and keeping them is safe precisely
        because the boundary gap already removed the rows that would have
        crossed the split edge.
        """
        labels = usable_labels(np.arange(50, 150), length=20)
        assert labels.min() == 50
        assert labels.size == 100

    def test_a_stride_thins_the_labels(self):
        """
        The reason a stride exists: overlapping windows are nearly duplicates.

        Consecutive windows of length twenty share nineteen rows, so a stride
        trades sample count for independence.
        """
        labels = usable_labels(np.arange(100), length=10, stride=5)
        assert np.all(np.diff(labels) == 5)

    def test_a_window_wider_than_the_split_yields_nothing(self):
        """
        Returned empty rather than raising, so the caller can say which split.

        The caller knows the split's name and the specification that produced
        it, and "the validation split is too narrow for a window of 60" is a
        far better message than this function could write.
        """
        labels = usable_labels(np.arange(10), length=60)
        assert labels.size == 0


class TestConfinement:
    """The alternative to a boundary gap: drop each split's first labels."""

    def test_confining_drops_the_labels_whose_window_reaches_back(self):
        """
        A split starting at fifty loses its first ``length - 1`` labels.

        The two strategies cost the same number of scenarios. The gap
        takes them from between the splits; confinement takes them from
        the front of each. Which one is available depends on whether the
        splits can be moved, and explicit splits cannot.
        """
        labels = usable_labels(np.arange(50, 150), length=20, confine=True)
        assert labels.min() == 69
        assert labels.size == 100 - 19

    def test_not_confining_keeps_them(self):
        """The default, and the behaviour the boundary gap is designed for."""
        labels = usable_labels(np.arange(50, 150), length=20, confine=False)
        assert labels.min() == 50
        assert labels.size == 100

    def test_a_hole_in_the_split_is_respected(self):
        """
        A window may not span an index the split does not contain.

        This is why confinement is a membership test rather than
        arithmetic on the split's first index. A purged fold's indices
        have holes by design -- the purge removed exactly the scenarios
        adjacent to the test period -- and a window spanning one would
        read them straight back in.
        """
        indices = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=np.int64)
        labels = usable_labels(indices, length=4, confine=True)
        # Only 3 and 10 have their three predecessors present.
        assert labels.tolist() == [3, 10]

    def test_the_two_strategies_agree_once_the_gap_is_wide_enough(self):
        """
        With a gap of ``length - 1``, confining changes nothing for a later split.

        Which is the point: a correctly gapped configuration pays for the
        gap once and keeps every label, so turning confinement on costs
        nothing and turning it off is safe.
        """
        indices = np.arange(50, 150)
        gapped = usable_labels(indices, length=1, confine=True)
        ungapped = usable_labels(indices, length=1, confine=False)
        assert gapped.tolist() == ungapped.tolist()


class TestExtractWindows:
    """The windows themselves, and what each row of them contains."""

    def test_a_window_ends_at_its_label(self):
        """
        The convention the whole module depends on.

        Labelling a window by its first row instead of its last shifts every
        target by ``length - 1`` scenarios, which trains the model to predict
        the past.
        """
        values = np.arange(100, dtype=np.float64).reshape(-1, 1)
        windows = extract_windows(values, np.array([30, 50]), length=5)
        assert windows[0, -1, 0] == 30.0
        assert windows[1, -1, 0] == 50.0

    def test_a_window_reaches_back_exactly_its_length(self):
        """So nothing older than ``label - length + 1`` enters the sample."""
        values = np.arange(100, dtype=np.float64).reshape(-1, 1)
        windows = extract_windows(values, np.array([30]), length=5)
        assert windows[0, :, 0].tolist() == [26.0, 27.0, 28.0, 29.0, 30.0]

    def test_the_shape_is_samples_by_window_by_features(self):
        """The axis order the gradient engines and the signature both assume."""
        values = np.arange(300, dtype=np.float64).reshape(100, 3)
        windows = extract_windows(values, np.array([40, 50, 60]), length=7)
        assert windows.shape == (3, 7, 3)

    def test_a_label_too_early_for_a_complete_window_is_refused(self):
        """
        Rather than wrapping, which is what numpy would do.

        A negative start index reads from the end of the array, so the sample
        would be built from the newest data and labelled with the oldest --
        leakage of the most direct possible kind, producing a model that
        scores superbly and predicts nothing.
        """
        values = np.arange(100, dtype=np.float64).reshape(-1, 1)
        with pytest.raises(ContractError):
            extract_windows(values, np.array([2]), length=20)


class TestWindowsStayWithin:
    """The check that a split's windows do not reach into another split."""

    def test_a_gapped_split_passes(self):
        """A gap of at least ``length - 1`` is exactly what makes it safe."""
        splits = SplitIndices(
            train=np.arange(0, 100),
            validation=np.arange(120, 160),
            test=np.arange(180, 200),
        )
        assert windows_stay_within(splits, length=20)

    def test_an_adjacent_split_fails(self):
        """
        The leak the boundary gap exists to prevent.

        With no gap, the validation split's first window reads nineteen
        training rows -- and every metric computed from it is optimistic by an
        amount nobody can estimate after the fact.
        """
        splits = SplitIndices(
            train=np.arange(0, 100),
            validation=np.arange(100, 140),
            test=np.arange(140, 160),
        )
        assert not windows_stay_within(splits, length=20)

    def test_adjacency_is_fine_without_windows(self):
        """
        At a length of one there is nothing to reach back into.

        A check that rejected adjacent splits unconditionally would force a
        gap on every non-sequential model and throw away data for nothing.
        """
        splits = SplitIndices(
            train=np.arange(0, 100),
            validation=np.arange(100, 140),
            test=np.arange(140, 160),
        )
        assert windows_stay_within(splits, length=1)
