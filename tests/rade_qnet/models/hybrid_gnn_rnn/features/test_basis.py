"""
Tests for basis selection.

The property under test is not "it reduces" -- anything reduces. It is that
the instruments kept *span* the ones dropped, and that the budget is spread
across the book rather than consumed by whichever group happens to be
loudest.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.runtime.errors import ContractError
from src.rade_qnet.models.hybrid_gnn_rnn.features.basis import effective_rank, select_basis

N_SCENARIOS = 200


def redundant_book(
    *, n_groups: int = 2, per_group: int = 5, factors: int = 2, seed: int = 0
) -> tuple[np.ndarray, list[str]]:
    """
    Build a book with fewer underlying drivers than instruments.

    Each group's P&L is a random mix of ``factors`` independent drivers, so
    the group's true dimension is ``factors`` however many instruments it
    holds. That makes the correct answer known in advance, which is what
    lets the tests below assert on it.

    Parameters
    ----------
    n_groups
        How many underlyings.
    per_group
        Instruments per underlying.
    factors
        Independent drivers per group.
    seed
        Generator seed.

    Returns
    -------
    tuple
        The P&L matrix and its instrument identifiers.
    """
    generator = np.random.default_rng(seed)
    blocks, identifiers = [], []
    for group in range(n_groups):
        drivers = generator.normal(size=(N_SCENARIOS, factors))
        loadings = generator.normal(size=(factors, per_group))
        blocks.append(drivers @ loadings)
        identifiers += [f"CCY{group}|forward|{number:03d}" for number in range(per_group)]
    return np.hstack(blocks), identifiers


class TestEffectiveRank:
    """How many components the variance threshold asks for."""

    def test_it_finds_the_true_dimension(self) -> None:
        """
        Three drivers behind ten instruments is a rank of three.

        The whole premise of the reduction is that this number is much
        smaller than the instrument count; a method that could not recover
        it on noiseless data would not be worth running on real data.
        """
        generator = np.random.default_rng(0)
        values = generator.normal(size=(N_SCENARIOS, 3)) @ generator.normal(size=(3, 10))
        assert effective_rank(values, variance_threshold=0.999) == 3

    def test_a_higher_threshold_keeps_at_least_as_many(self) -> None:
        """Monotonicity. A threshold that kept fewer would be unusable as a knob."""
        values, _ = redundant_book(n_groups=1, per_group=8, factors=4)
        ranks = [effective_rank(values, variance_threshold=t) for t in (0.5, 0.9, 0.99)]
        assert ranks == sorted(ranks)

    def test_a_single_instrument_is_its_own_basis(self) -> None:
        """
        A one-instrument group is its own basis.

        The degenerate case, which the SVD would otherwise be asked to
        handle on a one-column matrix.
        """
        assert effective_rank(np.arange(10.0).reshape(-1, 1)) == 1

    def test_a_constant_group_still_keeps_one_instrument(self) -> None:
        """
        A group that never moved has no variance to explain.

        Returning zero would leave that corner of the book with no
        representative at all, so the model could not predict a position in
        it. One is the smallest honest answer.
        """
        assert effective_rank(np.ones((20, 4))) == 1


class TestSelection:
    """Which instruments come back."""

    def test_the_kept_instruments_span_the_dropped_ones(self) -> None:
        """
        The claim that justifies the whole stage.

        If the selected columns really span the book, then regressing a
        dropped instrument onto them leaves almost no residual. A selection
        that merely kept the most volatile instruments would fail this,
        because those tend to be several expressions of one risk.
        """
        pnl, identifiers = redundant_book(n_groups=1, per_group=6, factors=2)
        selected = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.999)
        kept = [identifiers.index(name) for name in selected]
        dropped = [i for i in range(len(identifiers)) if i not in kept]

        basis = pnl[:, kept]
        for column in dropped:
            residual = (
                pnl[:, column] - basis @ np.linalg.lstsq(basis, pnl[:, column], rcond=None)[0]
            )
            assert np.abs(residual).max() < 1e-8 * np.abs(pnl[:, column]).max() + 1e-8

    def test_every_group_keeps_a_representative(self) -> None:
        """
        No underlying may be left without a representative.

        Per-group selection is what stops a large, volatile underlying from
        consuming the whole budget. With a global selection, the loud group below would take every slot
        and the model would have nothing at all to predict a quiet-group
        position from.
        """
        quiet, quiet_ids = redundant_book(n_groups=1, per_group=4, factors=2, seed=1)
        loud, loud_ids = redundant_book(n_groups=1, per_group=4, factors=2, seed=2)
        pnl = np.hstack([quiet * 1e-4, loud * 1e4])
        identifiers = [i.replace("CCY0", "QUIET") for i in quiet_ids] + [
            i.replace("CCY0", "LOUD") for i in loud_ids
        ]

        selected = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.99)
        assert any(name.startswith("QUIET") for name in selected)
        assert any(name.startswith("LOUD") for name in selected)

    def test_the_order_does_not_depend_on_the_input_order(self) -> None:
        """
        Shuffling the input columns must not reorder the result.

        Groups are visited in sorted order. The returned order is the column order of every downstream array,
        so an order that tracked the input would make two runs on the same
        book incomparable.
        """
        pnl, identifiers = redundant_book()
        first = select_basis(pnl, instrument_ids=identifiers)

        permutation = np.random.default_rng(0).permutation(len(identifiers))
        shuffled = select_basis(
            pnl[:, permutation], instrument_ids=[identifiers[i] for i in permutation]
        )
        assert shuffled == first

    def test_the_result_is_a_sequence_not_a_set(self) -> None:
        """
        Pivot order is importance order, and it is load-bearing.

        A selection returned sorted would still be the same instruments but
        a different feature matrix, so parity level 1 compares it in order.
        """
        pnl, identifiers = redundant_book(n_groups=1, per_group=6, factors=4)
        selected = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.99)
        assert list(selected) != sorted(selected)

    def test_tail_weighting_changes_which_instrument_is_chosen(self) -> None:
        """
        Up-weighting the violent days picks a different instrument.

        The book below holds one instrument that is noisy day to day and
        one that is quiet until it is not. Unweighted, the noisy one has
        the larger variance and is chosen. Weighted towards the tail, the
        other is -- and the tail is what a risk model exists for. A knob
        that could not flip this would not be doing anything.
        """
        generator = np.random.default_rng(3)
        tail_days = generator.choice(N_SCENARIOS, 5, replace=False)

        noisy_every_day = generator.normal(scale=1.0, size=N_SCENARIOS)
        quiet_then_violent = generator.normal(scale=0.05, size=N_SCENARIOS)
        # Four sigma on five days out of two hundred. Large enough that the
        # tail detector sees those days, small enough that the instrument's
        # total variance stays below the noisy one's -- which is what makes
        # the unweighted and weighted answers differ.
        quiet_then_violent[tail_days] = 4.0

        pnl = np.column_stack([noisy_every_day, quiet_then_violent])
        identifiers = ["CCY0|forward|000", "CCY0|forward|001"]

        # A threshold of 0.5 keeps one instrument, so the choice is visible.
        unweighted = select_basis(pnl, instrument_ids=identifiers, variance_threshold=0.5)
        weighted = select_basis(
            pnl, instrument_ids=identifiers, variance_threshold=0.5, weight_tail=50.0
        )
        assert unweighted == ("CCY0|forward|000",)
        assert weighted == ("CCY0|forward|001",)

    def test_a_mismatched_identifier_count_is_refused(self) -> None:
        """
        The identifiers name the columns, so a mismatch misnames the basis.

        It would still be the right length, so nothing downstream would
        notice.
        """
        pnl, identifiers = redundant_book()
        with pytest.raises(ContractError, match="identifier"):
            select_basis(pnl, instrument_ids=identifiers[:-1])

    def test_an_unparseable_identifier_is_refused(self) -> None:
        """
        Grouping is parsed out of the identifier, so it must parse.

        An identifier without a product type would land every instrument in
        one group, silently undoing the per-group guarantee above.
        """
        pnl, identifiers = redundant_book()
        with pytest.raises(ContractError, match="does not parse"):
            select_basis(pnl, instrument_ids=["bare_name", *identifiers[1:]])
