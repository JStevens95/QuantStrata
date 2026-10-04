"""
Tests for the instrument universe.

A universe is the contract between a matrix column and a real instrument. A
prediction is a vector of numbers; without a universe it is a vector of
numbers about nothing, and getting the ordering wrong does not raise -- it
produces plausible predictions attributed to the wrong instruments.

So the tests here are mostly about order being preserved and about lookups
failing loudly rather than returning something indexable.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.models.hybrid_gnn_rnn.state import Universe


@pytest.fixture
def universe():
    """
    Provide a small universe with distinguishable identifiers.

    Returns
    -------
    Universe
        Three elementary instruments and two targets.
    """
    return Universe(elementary_ids=("e1", "e2", "e3"), target_ids=("t1", "t2"))


class TestCounts:
    """The sizes every downstream shape is built from."""

    def test_the_two_sides_are_counted_separately(self, universe):
        """A model's input width and output width are different numbers."""
        assert universe.n_elementary == 3
        assert universe.n_targets == 2

    def test_an_empty_side_is_allowed(self):
        """
        A universe with no targets is degenerate but representable.

        Refusing it here would move the error somewhere less informative:
        a data directory with no targets is a problem with the data, and the
        model's ``data.py``, which reads ``universe.json``, is where that
        reads clearly.
        """
        assert Universe(elementary_ids=("e1",), target_ids=()).n_targets == 0


class TestOrdering:
    """Order is the whole contract."""

    def test_identifiers_keep_the_order_they_were_given(self, universe):
        """
        Not sorted.

        Column order is decided by basis selection, and re-sorting here
        would silently reattribute every column to a different instrument.
        """
        assert universe.elementary_ids == ("e1", "e2", "e3")

    def test_the_combined_order_is_elementary_then_target(self, universe):
        """
        The row order of the combined attribute matrix.

        Concatenated here rather than at each call site, because two places
        agreeing by convention is a convention that eventually gets broken
        by someone who did not know it existed.
        """
        assert universe.instrument_ids == ("e1", "e2", "e3", "t1", "t2")

    def test_a_position_matches_the_combined_order(self, universe):
        """Positions are indices into `instrument_ids`, not into one side."""
        assert universe.position_of("e1") == 0
        assert universe.position_of("t1") == 3


class TestLookupFailure:
    """An absent instrument has to raise."""

    def test_an_unknown_instrument_raises(self, universe):
        """
        Rather than returning a sentinel.

        `-1` indexes the last row perfectly happily, and the result would
        be a prediction attributed to whichever instrument happened to be
        there -- wrong rather than absent.
        """
        with pytest.raises(KeyError):
            universe.position_of("absent")

    def test_the_error_says_how_big_the_universe_is(self, universe):
        """
        Context, because the usual cause is a mismatched snapshot.

        An identifier that was valid last week and is not today is a data
        question, and the sizes are the first clue.
        """
        with pytest.raises(KeyError, match="3 elementary and 2 target"):
            universe.position_of("absent")


class TestImmutability:
    """A universe is interpreted by everything downstream."""

    def test_a_universe_cannot_be_reassigned(self, universe):
        """
        Frozen, because reordering after a fit reattributes every index.

        A fitted state, a prediction and a report all read positions out of
        it, and none of them re-check.
        """
        with pytest.raises((AttributeError, TypeError)):
            universe.elementary_ids = ("x",)  # type: ignore[misc]

    def test_two_universes_with_the_same_contents_are_equal(self, universe):
        """
        Value semantics, so parity comparisons can use `==`.

        Two runs producing the same universe should compare equal without
        anyone writing a field-by-field check.
        """
        assert universe == Universe(elementary_ids=("e1", "e2", "e3"), target_ids=("t1", "t2"))

    def test_order_is_part_of_identity(self):
        """
        Two universes differing only in order are not equal.

        They describe different matrices, and anything that treated them as
        interchangeable would be exactly the bug this class guards.
        """
        assert Universe(elementary_ids=("a", "b"), target_ids=()) != Universe(
            elementary_ids=("b", "a"), target_ids=()
        )
