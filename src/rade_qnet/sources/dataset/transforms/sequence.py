"""
Rolling-window construction over the scenario axis.

A sequential model predicts at scenario ``i`` from the window
``[i - length + 1, i]``. This module turns a scenario-indexed array into a
window-indexed one, and -- more importantly -- decides which labels have a
window at all.

Nothing here is fitted, so there is no state
--------------------------------------------
Windowing is a deterministic function of the specification and the axis
length. There is no :class:`~rade_qnet.core.contract.state.FittedState` to save,
which is why this module is plain functions while ``scaling`` and
``reduction`` are classes. A window layout can always be rebuilt from the spec
recorded in the bundle.

The two edge cases that cause silent errors
-------------------------------------------
**The first windows do not exist.** Label ``0`` with ``length=20`` would need
scenarios ``-19`` through ``0``. Negative indices are legal in NumPy and wrap
to the *end* of the array, so a naive implementation silently builds a window
from the last nineteen scenarios and the first one -- a window spanning the
whole history, used as training data, with nothing raised.
:func:`usable_labels` drops those labels instead.

**A window may reach across a split boundary.** It must not reach into
*another split's* scenarios, which the boundary gap from
:mod:`rade_qnet.sources.dataset.splits` guarantees. It may legitimately reach
into the discarded gap itself: those scenarios belong to no split, so reading
them leaks nothing. :func:`windows_stay_within` checks the property that
actually matters rather than the stricter one, which would reject every
correct configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....core.lifecycle.errors import ContractError

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ....core.contract.data import SplitIndices

__all__ = ["extract_windows", "usable_labels", "windows_stay_within"]


def usable_labels(
    label_indices: NDArray[np.int64],
    *,
    length: int,
    stride: int = 1,
    confine: bool = False,
) -> NDArray[np.int64]:
    """
    Return the labels that have a complete window behind them.

    Parameters
    ----------
    label_indices
        Candidate scenario indices, normally one split's indices. Need not be
        sorted; the result is.
    length
        Window length. ``1`` makes every label usable.
    stride
        Keep every ``stride``-th usable label. Applied *after* filtering, so
        changing the window length does not change which labels a given
        stride selects among those that remain.
    confine
        Whether a label's whole window must lie inside ``label_indices``.

        False -- the default -- keeps every label whose window exists at
        all, and relies on the boundary gap to stop one split's window
        reaching into another. That is this framework's design: the gap
        belongs to no split, so reading it leaks nothing, and
        :func:`windows_stay_within` is the check that the gap is wide
        enough.

        True is the alternative strategy: drop each split's first
        ``length - 1`` labels instead of inserting a gap between splits.
        It costs the same number of scenarios and needs no gap, which is
        why it is offered -- and it is what the original implementation
        did, so the parity replay sets it.

    Returns
    -------
    numpy.ndarray
        Ascending label indices, each with a complete window behind it.

    Raises
    ------
    ContractError
        If ``length`` or ``stride`` is below one.
    """
    if length < 1:
        raise ContractError(f"window length must be at least 1, received {length}")
    if stride < 1:
        raise ContractError(f"window stride must be at least 1, received {stride}")

    ordered = np.sort(np.asarray(label_indices, dtype=np.int64))
    if confine:
        # Membership rather than arithmetic on the split's first index,
        # because a split need not be contiguous -- a purged k-fold's
        # indices have holes, and a window spanning one would read
        # scenarios the purge deliberately removed.
        complete = ordered[_spans_present(ordered, length)]
    else:
        # A label at index i needs scenarios i-(length-1) .. i, so any label
        # below length-1 has no complete window at all.
        complete = ordered[ordered >= length - 1]
    return complete[::stride]


def _spans_present(ordered: NDArray[np.int64], length: int) -> NDArray[np.bool_]:
    """
    Mark the labels whose whole window lies inside the given indices.

    Parameters
    ----------
    ordered
        Ascending scenario indices.
    length
        Window length.

    Returns
    -------
    numpy.ndarray
        One boolean per label.
    """
    present = set(ordered.tolist())
    return np.array(
        [
            all(step in present for step in range(int(label) - length + 1, int(label) + 1))
            for label in ordered
        ],
        dtype=np.bool_,
    )


def extract_windows(
    values: NDArray[np.generic],
    labels: NDArray[np.int64],
    *,
    length: int,
) -> NDArray[np.generic]:
    """
    Build one window per label, looking backwards from it.

    Parameters
    ----------
    values
        Scenario-indexed array. The first axis is the scenario axis; any
        remaining axes are carried through untouched, so a ``(n_scenarios,
        n_features)`` matrix becomes ``(n_labels, length, n_features)``.
    labels
        Label indices, from :func:`usable_labels`.
    length
        Window length.

    Returns
    -------
    numpy.ndarray
        Windows, with the label's own scenario last in each one. The ordering
        is chronological within a window, which is what a recurrent layer
        expects.

    Raises
    ------
    ContractError
        If any label has no complete window, or any window would run past the
        end of the axis. Checked rather than trusted because the failure is
        otherwise silent -- see the module docstring on negative indices.
    """
    if length < 1:
        raise ContractError(f"window length must be at least 1, received {length}")

    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.size == 0:
        # An empty split is legitimate -- a run configured without validation
        # reaches here -- so this returns an empty array of the right shape
        # rather than failing.
        return np.empty((0, length, *values.shape[1:]), dtype=values.dtype)

    first, last = int(label_array.min()), int(label_array.max())
    if first < length - 1:
        raise ContractError(
            f"label {first} has only {first} scenario(s) before it but the window "
            f"length is {length}; pass labels through usable_labels first"
        )
    if last >= values.shape[0]:
        raise ContractError(
            f"label {last} is outside the scenario axis of length {values.shape[0]}"
        )

    # Offsets run from the oldest scenario in the window to the label itself,
    # so each row of `rows` is one window in chronological order.
    offsets = np.arange(-(length - 1), 1, dtype=np.int64)
    rows = label_array[:, None] + offsets[None, :]
    # Fancy indexing materialises the windows, which costs `n_labels * length`
    # elements. A stride view would avoid the copy, but the result is handed to
    # an engine that will copy it to a device anyway, and an explicit array is
    # far easier to reason about than an overlapping view.
    return values[rows]


def windows_stay_within(
    splits: SplitIndices,
    *,
    length: int,
) -> bool:
    """
    Return whether every window lies clear of the other splits' scenarios.

    The property the boundary gap exists to provide, checked directly rather
    than inferred from the gap arithmetic. A test asserting this is worth more
    than a test asserting the gap is the right width, because it is the
    statement about leakage rather than a statement about an implementation
    detail.

    A window is permitted to reach into the discarded gap between splits --
    those scenarios belong to no split, so reading them leaks nothing. What is
    forbidden is a window whose span includes a scenario assigned to a
    *different* split.

    Parameters
    ----------
    splits
        The split indices to check.
    length
        Window length the model consumes.

    Returns
    -------
    bool
        True if no window crosses into another split.
    """
    assignments = {name: set(splits[name].tolist()) for name in ("train", "validation", "test")}

    for name in assignments:
        others = set().union(*(value for key, value in assignments.items() if key != name))
        # Unconfined deliberately: the question this answers is whether
        # the gaps are wide enough, and confining the windows would make
        # the answer yes by construction.
        for label in usable_labels(splits[name], length=length, confine=False):
            span = range(int(label) - length + 1, int(label) + 1)
            if others.intersection(span):
                return False
    return True
