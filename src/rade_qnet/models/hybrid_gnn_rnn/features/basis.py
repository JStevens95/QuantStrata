"""
Chooses a small set of real instruments that spans the book's risk.

A cluster can hold thousands of elementary instruments whose P&L is driven
by a handful of factors. Feeding all of them to the network wastes capacity
on redundancy and makes the graph quadratically expensive for no gain.

Why real instruments rather than principal components
-----------------------------------------------------
The obvious reduction is PCA: project onto the top few components and train
on those. This does not do that, and the reason is not statistical.

A principal component is a linear combination of every instrument in the
book. A model trained on components produces hedge ratios against synthetic
portfolios that nobody can trade, and attributing a prediction back to a
position means inverting the projection -- which spreads every prediction
across the whole universe and makes the output impossible to explain to a
risk desk.

Selecting *actual* instruments keeps every downstream number tradeable. The
chosen set is a basis in the linear-algebra sense -- the rest of the book is
well approximated by combinations of it -- but each element is a real
position with a real identity.

How
---
Two steps, per instrument group:

1. **How many?** Take the singular values of the group's P&L and keep the
   smallest number of components reaching the variance threshold. This is
   the group's intrinsic dimension: how many independent things are
   actually moving.
2. **Which ones?** Column-pivoted QR. At each step it picks the instrument
   with the largest component orthogonal to everything already chosen, so
   the result is a well-conditioned, non-redundant set rather than simply
   the most volatile instruments -- which would all be the same risk.

Why per group
-------------
The selection runs separately for each ``(underlying, product type)`` pair,
ordered by sorted underlying then sorted product type. A single global
selection would let a large, volatile group consume the entire budget and
leave a smaller one with no representation at all -- so the model would
have nothing to predict a GBP position from. Per-group selection guarantees
every corner of the book keeps a basis proportional to its own complexity.

The scaling the selection sees
------------------------------
The baseline fitted the basis on the P&L **after** scaling, where the
scaler itself was fitted on training rows only. The validation and test
rows therefore influenced which instruments were chosen.

This is defect 9 in ``ARCHITECTURE.md`` §13, and it is deliberately
preserved behind the ``basis_fit_on`` flag rather than fixed, because
fixing it changes which instruments are selected and so changes every
number the model produces. The flag defaults to ``"train"``, the correct
behaviour; parity tests set ``"all"`` explicitly to reproduce the baseline.

The leakage here is mild -- a choice of columns, not a fitted statistic --
but it is leakage, and a backtest that uses it is reporting on a model that
knew which instruments would matter.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import qr

from ....core.lifecycle.errors import ContractError
from ....core.provenance.logging import get_logger

__all__ = ["effective_rank", "select_basis"]

#: Separator in an instrument identifier, as ``underlying|product|number``.
_ID_SEPARATOR = "|"

#: How many leading fields of an identifier define a group.
_GROUP_FIELDS = 2

#: Multiples of a scenario's maximum absolute P&L, above the mean, at which
#: that scenario counts as a tail event. 1.645 is the one-sided 95th
#: percentile of a normal, which is where the baseline drew the line.
_TAIL_SIGMA = 1.645

_LOGGER = get_logger(__name__)


def select_basis(
    pnl: NDArray[np.floating],
    *,
    instrument_ids: Sequence[str],
    variance_threshold: float = 0.99,
    weight_tail: float = 1.0,
) -> tuple[str, ...]:
    """
    Choose a spanning subset of instruments, group by group.

    Parameters
    ----------
    pnl
        P&L history, shape ``(n_scenarios, n_instruments)``, already scaled
        if it is going to be.
    instrument_ids
        One identifier per column, formatted
        ``underlying|product_type|number``. The grouping is parsed from
        this rather than passed separately, matching the baseline.
    variance_threshold
        Fraction of a group's variance the kept components must explain.
    weight_tail
        Emphasis on tail scenarios. One leaves every scenario weighted
        equally.

    Returns
    -------
    tuple of str
        The selected identifiers, **ordered**: groups in sorted order, and
        within a group, the order the pivoted QR chose. That order is the
        column order of every downstream array, so it is a sequence and
        never a set.

    Raises
    ------
    ContractError
        If the identifier count does not match the column count.
    """
    pnl = np.asarray(pnl, dtype=np.float64)
    if len(instrument_ids) != pnl.shape[1]:
        raise ContractError(
            f"basis selection was given {len(instrument_ids)} instrument identifier(s) "
            f"for {pnl.shape[1]} P&L column(s). The identifiers name the columns, so a "
            f"mismatch means the selected basis would name the wrong instruments"
        )

    selected: list[str] = []
    for group, columns in _groups(instrument_ids):
        block = pnl[:, columns]
        rank = effective_rank(block, variance_threshold=variance_threshold)
        chosen = _pivoted_columns(block, k=rank, weight_tail=weight_tail)
        selected.extend(instrument_ids[columns[position]] for position in chosen)
        _LOGGER.debug("group %s: kept %d of %d instrument(s)", group, rank, len(columns))

    _LOGGER.info(
        "basis selection kept %d of %d elementary instrument(s) at a %.3f variance threshold",
        len(selected),
        len(instrument_ids),
        variance_threshold,
    )
    return tuple(selected)


def effective_rank(values: NDArray[np.floating], *, variance_threshold: float = 0.99) -> int:
    """
    Return how many components are needed to explain the variance threshold.

    The variance is measured about the column means, which is what makes
    this a statement about variation rather than about level: an instrument
    with a large constant P&L and no variation contributes nothing, which is
    correct, because a constant is not a risk factor.

    Parameters
    ----------
    values
        Shape ``(n_scenarios, n_instruments)``.
    variance_threshold
        Fraction of total variance to reach, in ``(0, 1]``.

    Returns
    -------
    int
        At least one, and at most the number of columns.
    """
    if values.shape[1] <= 1:
        # A single-instrument group is its own basis. Also guards the
        # degenerate SVD of a one-column matrix.
        return values.shape[1]

    centred = values - values.mean(axis=0)
    singular = np.linalg.svd(centred, compute_uv=False)

    total = float((singular**2).sum())
    if total <= 0.0:
        # Every instrument in the group is constant, so there is nothing to
        # span. One is returned rather than zero because a group with no
        # representative would leave a corner of the book unpredictable.
        return 1

    cumulative = np.cumsum(singular**2 / total)
    # `side="right"` matters and is not interchangeable with the default. At
    # a threshold that a cumulative ratio hits exactly, `"left"` would stop
    # one component short -- and "exactly" is the common case for a group
    # whose last component carries no variance at all.
    rank = int(np.searchsorted(cumulative, variance_threshold, side="right")) + 1
    return min(rank, values.shape[1])


def _groups(instrument_ids: Sequence[str]) -> list[tuple[tuple[str, str], list[int]]]:
    """
    Partition the columns by ``(underlying, product type)``.

    Iterated as the cartesian product of the sorted underlyings and the
    sorted product types, matching the baseline. That means the group order
    -- and therefore the order of the selected basis -- does not depend on
    the order the instruments arrived in.

    Combinations that no instrument occupies are skipped rather than
    treated as empty groups.

    Parameters
    ----------
    instrument_ids
        One identifier per column.

    Returns
    -------
    list
        Each entry is a group key and the column positions it holds.

    Raises
    ------
    ContractError
        If an identifier does not carry an underlying and a product type.
    """
    parsed: list[tuple[str, str]] = []
    for identifier in instrument_ids:
        fields = identifier.split(_ID_SEPARATOR)
        if len(fields) < _GROUP_FIELDS:
            raise ContractError(
                f"instrument identifier {identifier!r} does not parse as "
                f"'underlying{_ID_SEPARATOR}product_type{_ID_SEPARATOR}...'. Basis "
                f"selection groups by underlying and product type, so an unparseable "
                f"identifier would silently land every instrument in one group and let "
                f"a large underlying crowd out a small one"
            )
        parsed.append((fields[0], fields[1]))

    underlyings = sorted({underlying for underlying, _ in parsed})
    product_types = sorted({product_type for _, product_type in parsed})

    groups: list[tuple[tuple[str, str], list[int]]] = []
    for key in product(underlyings, product_types):
        columns = [position for position, entry in enumerate(parsed) if entry == key]
        if columns:
            groups.append((key, columns))
    return groups


def _pivoted_columns(
    values: NDArray[np.float64], *, k: int, weight_tail: float
) -> NDArray[np.intp]:
    """
    Return the ``k`` most linearly independent columns, most important first.

    Column-pivoted QR, which at each step takes the column with the largest
    residual once everything already chosen is projected out. Picking the
    ``k`` most volatile instruments instead would tend to pick ``k``
    expressions of the same risk.

    Parameters
    ----------
    values
        Shape ``(n_scenarios, n_instruments)``.
    k
        How many columns to return.
    weight_tail
        Row emphasis on tail scenarios.

    Returns
    -------
    numpy.ndarray
        Column positions, in pivot order.
    """
    if weight_tail != 1.0:
        # Up-weight the scenarios where the book actually moved. A basis
        # chosen on quiet days spans the quiet-day subspace, which is not
        # the one a risk model is for.
        row_peak = np.max(np.abs(values), axis=1)
        threshold = row_peak.mean() + _TAIL_SIGMA * row_peak.std()
        # The square root is applied because QR works on the matrix while
        # the weighting is meant to apply to the variance.
        weights = np.sqrt(np.where(row_peak > threshold, weight_tail, 1.0))
        values = values * weights[:, None]

    # SciPy rather than NumPy, because NumPy has no pivoting QR. See this
    # model's requirements file for why that dependency is accepted here and
    # refused in the encoder and the graph: this is a computation whose
    # output is a list of identifiers, so nothing third-party reaches the
    # saved state.
    _, _, pivots = qr(values, mode="economic", pivoting=True)
    return pivots[:k]
