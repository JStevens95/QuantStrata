"""
Split strategies for the scenario axis -- every one of them sequence aware.

This is the most consequential module in the framework, and the reason is that
its failure mode is encouraging rather than alarming. A leaky split does not
raise; it produces a validation score better than the model deserves, and the
model then fails in production for reasons nobody can reconstruct.

Three properties every strategy here guarantees
-----------------------------------------------
**Disjoint.** No scenario index appears in two splits. Enforced again by
:class:`~rade_qnet.core.contract.data.SplitIndices`, so a strategy that got it
wrong cannot return.

**Chronologically ordered, where order means anything.** For the
chronological strategy, every training index precedes every validation index,
which precedes every test index.

**Window safe.** This is the property the implementation being replaced did
not have. A model consuming a window of ``sequence_length`` scenarios reads
indices ``[i - length + 1, i]`` to predict at ``i``. If ``i`` is the first
validation index, that window reaches back into training -- so the validation
score is partly a memory of what the model was fitted on.

The fix is a gap of ``sequence_length - 1`` scenarios discarded at each
boundary, plus any additional ``gap_scenarios`` the spec asks for. The two are
additive and both are needed:

- The *sequence* gap handles a window that looks **backwards** from its label.
- The *explicit* gap handles a target computed **forwards** from its index --
  a five-day forward return at scenario ``i`` is a function of ``i + 5``, so
  without a gap the last five training labels encode validation scenarios.

Why the gap is discarded rather than reassigned
-----------------------------------------------
Giving the boundary scenarios to the earlier split would leave the window
problem exactly where it was; giving them to the later one makes its first
windows depend on training data. The only correct answer is that those
scenarios belong to neither split, which costs ``sequence_length - 1`` rows
per boundary and buys a validation score that means what it says.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ...core.contract.data import SplitIndices
from ...core.runtime.errors import SpecError
from ...core.runtime.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...core.spec.data import (
        ChronologicalSplitSpec,
        ExplicitSplitSpec,
        GroupedSplitSpec,
        PurgedKFoldSplitSpec,
        SplitSpec,
    )

__all__ = [
    "boundary_gap",
    "split_by_group",
    "split_chronologically",
    "split_explicitly",
    "split_purged_kfold",
    "split_scenarios",
]

_LOGGER = get_logger(__name__)


def boundary_gap(sequence_length: int, gap_scenarios: int = 0) -> int:
    """
    Return how many scenarios must be discarded at each split boundary.

    Parameters
    ----------
    sequence_length
        Window length the model consumes. ``1`` for a non-sequential model,
        which needs no sequence gap at all.
    gap_scenarios
        Additional scenarios the specification asks to discard, for a target
        computed over a forward window.

    Returns
    -------
    int
        Total gap, the sum of the two contributions.

    Raises
    ------
    SpecError
        If either argument is negative, or ``sequence_length`` is zero. A
        window of zero scenarios is not a shorter window; it is a model with
        no input.
    """
    if sequence_length < 1:
        raise SpecError(
            f"sequence_length must be at least 1, received {sequence_length}; "
            f"use 1 for a non-sequential model"
        )
    if gap_scenarios < 0:
        raise SpecError(f"gap_scenarios cannot be negative, received {gap_scenarios}")
    # A window of length 1 reads only its own scenario, so it cannot straddle
    # anything and contributes no gap.
    return (sequence_length - 1) + gap_scenarios


def split_scenarios(
    spec: SplitSpec,
    *,
    n_scenarios: int,
    sequence_length: int = 1,
    group_labels: NDArray[np.int64] | None = None,
    seed: int = 0,
) -> SplitIndices:
    """
    Split a scenario axis using whichever strategy the specification names.

    The single entry point a data module calls. Dispatching here rather than
    in the data module means a new strategy is a new function and one branch,
    not a change to every data module in existence.

    Parameters
    ----------
    spec
        The split strategy, from the source specification.
    n_scenarios
        Length of the scenario axis.
    sequence_length
        Window length the model consumes, used to size the boundary gap.
    group_labels
        Group identifier per scenario, required by the grouped strategy and
        ignored by the others.
    seed
        Seed for the grouped strategy's group assignment. The other
        strategies are deterministic and ignore it.

    Returns
    -------
    SplitIndices
        Disjoint index arrays.

    Raises
    ------
    SpecError
        If the axis is too short for the requested split, or the strategy
        needs information that was not supplied.
    """
    if n_scenarios < 1:
        raise SpecError(f"cannot split an axis of {n_scenarios} scenarios; at least 1 is required")

    match spec.kind:
        case "chronological":
            return split_chronologically(
                spec, n_scenarios=n_scenarios, sequence_length=sequence_length
            )
        case "purged_kfold":
            return split_purged_kfold(spec, n_scenarios=n_scenarios)
        case "grouped":
            return split_by_group(
                spec, n_scenarios=n_scenarios, group_labels=group_labels, seed=seed
            )
        case "explicit":
            return split_explicitly(spec, n_scenarios=n_scenarios)
        case _:  # pragma: no cover - the discriminated union makes this unreachable
            raise SpecError(f"unknown split kind {spec.kind!r}")


def split_chronologically(
    spec: ChronologicalSplitSpec,
    *,
    n_scenarios: int,
    sequence_length: int = 1,
) -> SplitIndices:
    """
    Split by time order, with a gap at each boundary.

    Test takes the last ``test_fraction`` of the axis and validation the
    ``validation_fraction`` before it. Boundaries are cut on the full axis
    first, then **the gap is removed from the tail of the earlier split** at
    each boundary.

    Which side loses the rows
    -------------------------
    Taking the gap from the earlier split is the established treatment --
    purging training observations whose windows or labels reach into the
    held-out period -- and it is the right way round for two reasons.

    The held-out sets keep the size the configuration asked for, so a test
    score does not quietly become a score over fewer rows when the sequence
    length changes. And the alternative, trimming the *later* split's head,
    would discard the oldest held-out rows while leaving training rows that
    overlap them: it costs the same rows and removes none of the leakage.

    The consequence is that ``train`` and ``validation`` are each one gap
    narrower than their nominal fractions, and ``test`` is exactly its
    fraction. A validation fraction narrower than the gap is rejected rather
    than silently emptied.

    Parameters
    ----------
    spec
        Fractions and any additional gap.
    n_scenarios
        Length of the scenario axis.
    sequence_length
        Window length the model consumes.

    Returns
    -------
    SplitIndices
        Contiguous, ordered, disjoint index arrays.

    Raises
    ------
    SpecError
        If the gap leaves no training scenarios, or is wider than the
        requested validation split. The message reports the arithmetic,
        because the usual cause is a long sequence length on a short history
        and the fix depends on which of the two the user can change.
    """
    gap = boundary_gap(sequence_length, spec.gap_scenarios)

    n_test = round(n_scenarios * spec.test_fraction)
    n_validation = round(n_scenarios * spec.validation_fraction)

    # A non-zero fraction that rounds down to no scenarios is refused rather
    # than honoured.  Requesting no validation data is legitimate and is
    # expressed by setting the fraction to zero; asking for 15% of a
    # three-scenario axis and silently receiving none is not, because the run
    # then reports no held-out score while appearing to have succeeded.
    for name, fraction, count in (
        ("validation_fraction", spec.validation_fraction, n_validation),
        ("test_fraction", spec.test_fraction, n_test),
    ):
        if fraction > 0.0 and count == 0:
            raise SpecError(
                f"{name}={fraction} of {n_scenarios} scenario(s) rounds to zero, so "
                f"that split would be empty; use a longer history, or set {name}=0.0 "
                f"if an empty split is intended"
            )

    test_start = n_scenarios - n_test
    validation_start = test_start - n_validation

    test = np.arange(test_start, n_scenarios, dtype=np.int64)

    # A gap is only needed where two non-empty splits actually meet. With no
    # test split there is no validation/test boundary, so validation runs to
    # the end of the axis and loses nothing.
    validation_end = (test_start - gap) if n_test else n_scenarios
    if n_validation and validation_end <= validation_start:
        raise SpecError(
            f"validation_fraction={spec.validation_fraction} gives "
            f"{n_validation} scenario(s) out of {n_scenarios}, which is not wider "
            f"than the boundary gap of {gap} (sequence_length={sequence_length} + "
            f"gap_scenarios={spec.gap_scenarios}); the whole validation split would "
            f"be discarded. Increase validation_fraction, shorten the sequence, or "
            f"use a longer history"
        )
    validation = (
        np.arange(validation_start, validation_end, dtype=np.int64)
        if n_validation
        else np.empty(0, dtype=np.int64)
    )

    # Training ends one gap before whichever split follows it. With neither
    # held-out split present there is no boundary at all.
    if n_validation:
        train_end = validation_start - gap
    elif n_test:
        train_end = test_start - gap
    else:
        train_end = n_scenarios
    train = np.arange(0, max(0, train_end), dtype=np.int64)

    if train.size == 0:
        raise SpecError(
            f"a chronological split of {n_scenarios} scenarios with "
            f"validation_fraction={spec.validation_fraction}, "
            f"test_fraction={spec.test_fraction} and a boundary gap of {gap} "
            f"(sequence_length={sequence_length} + gap_scenarios="
            f"{spec.gap_scenarios}) leaves no training scenarios; "
            f"reduce the fractions, shorten the sequence, or use a longer history"
        )

    if gap:
        _LOGGER.debug(
            "chronological split discarded %d scenario(s) at each boundary to keep "
            "windows of length %d inside one split",
            gap,
            sequence_length,
        )
    return SplitIndices(train=train, validation=validation, test=test)


def split_purged_kfold(
    spec: PurgedKFoldSplitSpec,
    *,
    n_scenarios: int,
    fold: int = 0,
) -> SplitIndices:
    """
    Hold out one contiguous fold, with an embargo either side of it.

    Cross-validation on serially correlated data leaks in both directions
    across a fold boundary, which plain k-fold does nothing about. The embargo
    discards scenarios either side of the held-out fold so that neither the
    fold's first rows nor the training rows adjacent to it carry the other's
    information.

    One fold at a time
    ------------------
    This returns the split for a *single* fold rather than a list of folds,
    because the framework's unit of work is one run: a pipeline trains one
    model against one train/validation/test division. A cross-validated sweep
    is a job set over folds, which is Phase 4's concern and composes from this
    without change.

    Parameters
    ----------
    spec
        Fold count, embargo width and the final test fraction.
    n_scenarios
        Length of the scenario axis.
    fold
        Which fold to hold out as validation, zero based.

    Returns
    -------
    SplitIndices
        Training indices either side of the embargoed fold, the fold itself as
        validation, and the axis tail as test.

    Raises
    ------
    SpecError
        If ``fold`` is out of range, or the embargo consumes the training set.
    """
    if not 0 <= fold < spec.n_folds:
        raise SpecError(
            f"fold must be in [0, {spec.n_folds}), received {fold}; "
            f"the specification declares n_folds={spec.n_folds}"
        )

    # The test tail is removed from cross-validation entirely, so folds are cut
    # over the remainder. A test set that moved with the fold would make the
    # folds' scores incomparable.
    n_test = round(n_scenarios * spec.test_fraction)
    n_cross_validated = n_scenarios - n_test
    if n_cross_validated < spec.n_folds:
        raise SpecError(
            f"{n_cross_validated} scenarios remain after reserving "
            f"test_fraction={spec.test_fraction} of {n_scenarios}, which cannot be "
            f"divided into {spec.n_folds} folds; reduce n_folds or test_fraction"
        )

    fold_edges = np.linspace(0, n_cross_validated, spec.n_folds + 1).astype(np.int64)
    fold_start, fold_end = int(fold_edges[fold]), int(fold_edges[fold + 1])

    embargo = spec.embargo_scenarios
    validation = np.arange(fold_start, fold_end, dtype=np.int64)
    # Training is everything outside the fold *and* outside its embargo, which
    # is why this is two ranges rather than a slice.
    before = np.arange(0, max(0, fold_start - embargo), dtype=np.int64)
    after = np.arange(min(n_cross_validated, fold_end + embargo), n_cross_validated, dtype=np.int64)
    train = np.concatenate((before, after))
    test = np.arange(n_cross_validated, n_scenarios, dtype=np.int64)

    if train.size == 0:
        raise SpecError(
            f"fold {fold} of {spec.n_folds} with embargo_scenarios={embargo} leaves no "
            f"training scenarios out of {n_cross_validated}; reduce the embargo or "
            f"use more folds so each is smaller"
        )
    return SplitIndices(train=train, validation=validation, test=test)


def split_by_group(
    spec: GroupedSplitSpec,
    *,
    n_scenarios: int,
    group_labels: NDArray[np.int64] | None,
    seed: int = 0,
) -> SplitIndices:
    """
    Assign whole groups to splits, so no group spans two of them.

    The strategy for data where rows within a group are near-duplicates: the
    same trade observed at several tenors, or several scenarios sharing one
    event. Splitting such rows independently puts near-copies on both sides of
    the boundary, which is leakage that no amount of time ordering prevents.

    Groups are assigned by a seeded permutation rather than by label order,
    because label order is frequently meaningful -- group identifiers are
    often allocated chronologically -- and taking the last groups as test
    would silently turn this into a chronological split on a different axis.

    Parameters
    ----------
    spec
        Group key and the fractions, which are fractions *of groups*.
    n_scenarios
        Length of the scenario axis.
    group_labels
        Group identifier per scenario. Required.
    seed
        Seed for the group permutation.

    Returns
    -------
    SplitIndices
        Index arrays whose groups do not overlap.

    Raises
    ------
    SpecError
        If labels are absent, the wrong length, or there are too few groups
        for the requested fractions.
    """
    if group_labels is None:
        raise SpecError(
            f"a grouped split needs one label per scenario for group_key="
            f"{spec.group_key!r}, but none were supplied; the data module must "
            f"pass group_labels"
        )
    if group_labels.shape[0] != n_scenarios:
        raise SpecError(
            f"group_labels has {group_labels.shape[0]} entries but the scenario axis "
            f"has {n_scenarios}; they must correspond one to one"
        )

    groups = np.unique(group_labels)
    n_groups = groups.size
    n_test = round(n_groups * spec.test_fraction)
    n_validation = round(n_groups * spec.validation_fraction)

    # A fraction that rounds down to zero groups is refused rather than
    # honoured.  Asking for no validation data is legitimate and expressed by
    # setting the fraction to zero; *asking* for 15% of two groups and
    # receiving none is a different thing, and accepting it produces a run
    # whose validation split is empty, whose early stopping therefore never
    # fires, and which reports no held-out score while looking successful.
    for name, fraction, count in (
        ("validation_fraction", spec.validation_fraction, n_validation),
        ("test_fraction", spec.test_fraction, n_test),
    ):
        if fraction > 0.0 and count == 0:
            raise SpecError(
                f"{name}={fraction} of {n_groups} group(s) rounds to zero groups, so "
                f"that split would be empty; group more finely, or set {name}=0.0 if "
                f"an empty split is intended"
            )

    if n_groups - n_test - n_validation < 1:
        raise SpecError(
            f"{n_groups} group(s) cannot be divided with validation_fraction="
            f"{spec.validation_fraction} and test_fraction={spec.test_fraction}: "
            f"no training group would remain; reduce the fractions or group more coarsely"
        )

    # Seeded rather than global: a split must be reproducible from the run's
    # seed alone, and must not depend on NumPy's global random state, which
    # something else in the process may have reseeded.
    permuted = np.random.default_rng(seed).permutation(groups)
    test_groups = set(permuted[:n_test].tolist())
    validation_groups = set(permuted[n_test : n_test + n_validation].tolist())

    def indices_for(selected: set[int]) -> NDArray[np.int64]:
        """
        Return the scenario indices whose group is in a selected set.

        Parameters
        ----------
        selected
            Group labels assigned to one split.

        Returns
        -------
        numpy.ndarray
            Scenario indices, ascending.
        """
        if not selected:
            return np.empty(0, dtype=np.int64)
        membership = np.isin(group_labels, np.array(sorted(selected)))
        return np.flatnonzero(membership).astype(np.int64)

    test = indices_for(test_groups)
    validation = indices_for(validation_groups)
    held_out = test_groups | validation_groups
    train = np.flatnonzero(~np.isin(group_labels, np.array(sorted(held_out)))).astype(np.int64)
    if not held_out:
        train = np.arange(n_scenarios, dtype=np.int64)

    return SplitIndices(train=train, validation=validation, test=test)


def split_explicitly(spec: ExplicitSplitSpec, *, n_scenarios: int) -> SplitIndices:
    """
    Use the indices the specification supplies, verbatim.

    The strategy used for refactor parity: a golden fixture records the exact
    indices the previous implementation produced, and reproducing them removes
    the split from the list of things that could explain a difference.

    No gap is inserted. The caller supplied the indices, so the caller owns
    window safety -- which is why
    :class:`~rade_qnet.core.spec.run.SupervisedRunSpec` rejects an explicit
    split combined with a sequence length above one, rather than silently
    leaking here.

    Parameters
    ----------
    spec
        The supplied indices.
    n_scenarios
        Length of the scenario axis, used only to check the indices are in
        range.

    Returns
    -------
    SplitIndices
        The supplied indices, sorted.

    Raises
    ------
    SpecError
        If any index is outside the axis. Checked here because an out-of-range
        index from a stale fixture would otherwise surface as a confusing
        indexing error inside a transform.
    """
    supplied = {"train": spec.train, "validation": spec.validation, "test": spec.test}
    for name, values in supplied.items():
        out_of_range = sorted(value for value in values if not 0 <= value < n_scenarios)
        if out_of_range:
            raise SpecError(
                f"explicit {name} indices {out_of_range[:5]} are outside the scenario "
                f"axis [0, {n_scenarios}); the indices may have been captured from a "
                f"different dataset"
            )

    return SplitIndices(
        # Sorted so that downstream code may rely on ascending order, which the
        # other strategies produce naturally and a hand-written list may not.
        train=np.array(sorted(spec.train), dtype=np.int64),
        validation=np.array(sorted(spec.validation), dtype=np.int64),
        test=np.array(sorted(spec.test), dtype=np.int64),
    )
