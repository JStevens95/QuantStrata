# `src/rade_qnet/models/hybrid_gnn_rnn/features`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 20 | 813 | `011ecbecf59e5538` |
| 2 | `basis.py` | 293 | 11448 | `dd3b20cf7b1ef685` |
| 3 | `encoder.py` | 625 | 22114 | `02df79a5aad79a4d` |
| 4 | `graph.py` | 559 | 20643 | `03c836026a25e1fa` |

---

## 1. `src/rade_qnet/models/hybrid_gnn_rnn/features/__init__.py`

813 bytes · SHA-256 `011ecbecf59e5538`

```python
"""
Feature construction specific to the hybrid graph-temporal network.

These transforms are fitted along the **entity axis**: they look across the
whole instrument universe rather than across time.  That is not leakage --
which instruments exist, and what their attributes are, is known before any
P&L is observed.  The framework's conformance suite distinguishes the two axes
precisely so that these transforms are not wrongly flagged.

Planned modules
---------------
``encoder.py``
    Encodes instrument attributes (currency pair, tenor, product type) into the
    dense feature matrix the graph block consumes.  [Phase 3]
``graph.py``
    Builds the sparse instrument graph from encoded attributes, emitting edge
    indices, edge weights and the dense shape.  [Phase 3]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/models/hybrid_gnn_rnn/features/basis.py`

11448 bytes · SHA-256 `dd3b20cf7b1ef685`

```python
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

from ....core.runtime.errors import ContractError
from ....core.runtime.logging import get_logger

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
```

---

## 3. `src/rade_qnet/models/hybrid_gnn_rnn/features/encoder.py`

22114 bytes · SHA-256 `02df79a5aad79a4d`

```python
"""
Encodes instrument attributes into the static feature matrix the graph uses.

Fitted along the **entity axis** over the whole universe, elementary and
target instruments together. That is legitimate and worth stating plainly,
because the obvious conformance rule would reject it: which instruments exist
and what their attributes are -- currency pair, tenor, product type -- is
known before any P&L is observed. Restricting the encoder to
training-period instruments would not remove information a trader lacks, it
would discard information they have. Only fitting along the *scenario* axis
outside the training rows is leakage.

Why this is implemented here rather than imported
-------------------------------------------------
The original used three scikit-learn transformers. They are reimplemented
here, in about forty lines of arithmetic, for two reasons. The framework's
data path stays free of a heavy dependency that a model-specific encoder
would otherwise force on every host that installs the flagship. And the
fitted state becomes plain arrays, which means the bundle saves as ``.npz``
and ``.json`` rather than as a pickled estimator -- the same objection that
makes a pickled checkpoint unacceptable applies to a pickled scaler.

The port is exact rather than merely equivalent, and the golden fixture is
what proves it: ``combined_features`` is compared element for element at
zero tolerance.

Three details carry the exactness, and each was a place the port could have
drifted silently:

- **Column order.** Numeric attributes first in the order declared, then the
  one-hot blocks in the order declared, then the multi-label blocks. The
  graph measures distance over this matrix, so a different column order
  produces a different graph while every shape still matches.
- **Category order.** One-hot and multi-label categories are sorted, which
  is what scikit-learn does. Insertion order would be stable within a run
  and different across runs whose instruments arrived in a different order.
- **Row normalisation of multi-label blocks.** Each row is divided by its
  norm so an instrument with three risk factors is not three times further
  from the origin than one with a single factor. Without it the graph
  clusters by *how many* risk factors an instrument has rather than by which.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from numpy.typing import NDArray

from ....core.runtime.errors import ContractError
from ..spec import AttributeEncoderSpec

__all__ = ["EncodedAttributes", "EntityEncoderState", "decay_lambdas"]

#: The decay grid's endpoints. A lambda of 10 decays to near zero within a
#: few months, so it reads short-dated structure; a lambda of 0.1 is nearly
#: flat over a decade, so it reads the long end. Spanning them gives the
#: model several views of time to maturity rather than one raw number whose
#: relationship to value is strongly non-linear.
_DECAY_LAMBDA_MAX = 10.0
_DECAY_LAMBDA_MIN = 0.1

#: The default working precision, as a module-level singleton so it can be a
#: parameter default without constructing a dtype on every call.
_FLOAT64 = np.dtype(np.float64)

#: Divisor floor for row normalisation, so an all-zero row stays all-zero
#: instead of becoming NaN. An instrument with no declared risk factors is
#: legitimate and must not poison every distance it participates in.
_NORM_FLOOR = 1e-9


def decay_lambdas(n_terms: int) -> tuple[float, ...]:
    """
    Return the decay rates, on a linear grid from fast to slow.

    Parameters
    ----------
    n_terms
        How many rates to produce. Zero disables the feature.

    Returns
    -------
    tuple of float
        The rates, longest-decay last.
    """
    if n_terms <= 0:
        return ()
    return tuple(
        float(value) for value in np.linspace(_DECAY_LAMBDA_MAX, _DECAY_LAMBDA_MIN, n_terms)
    )


@dataclass(frozen=True, slots=True)
class EncodedAttributes:
    """
    The encoder's output for one set of instruments.

    Parameters
    ----------
    features
        The combined matrix, one row per instrument.
    names
        One name per column, so a feature-importance plot can be labelled
        and a reader can tell which column the graph weighted.
    blocks
        Column index ranges per attribute group, as ``(start, stop)``. The
        graph needs these to apply a per-group weight, and deriving them by
        re-counting categories at the call site is how the two come to
        disagree.
    """

    features: NDArray[np.float32]
    names: tuple[str, ...]
    blocks: Mapping[str, tuple[int, int]]


@dataclass(frozen=True, slots=True)
class EntityEncoderState:
    """
    Everything the encoder fits, as plain arrays.

    Deliberately not a wrapper around estimator objects. Holding arrays
    means the state serialises to ``.npz`` and ``.json``, so reloading a
    bundle never unpickles anything, and a reader can inspect what was
    fitted without constructing the class.

    Parameters
    ----------
    numeric_names
        Numeric attributes in column order, decay terms included.
    numeric_centre
        Per-attribute mean, subtracted on transform.
    numeric_scale
        Per-attribute standard deviation, divided on transform. Zeros are
        replaced by one at fit time: a constant attribute carries no
        information, and dividing by its zero spread would turn it into
        NaN and destroy every distance it takes part in.
    categorical_levels
        Attribute name to its sorted levels. Sorted, not insertion-ordered,
        so two runs whose instruments arrived in a different order encode
        identically.
    multi_label_levels
        The same for list-valued attributes.
    lambdas
        The decay rates fitted with, recorded so transform reproduces the
        same columns even if the spec changed afterwards.
    maturity_key
        Which attribute the decay terms derive from.
    numeric_precision
        Which precision the numeric attributes were rounded to and are
        scaled in. Recorded on the state rather than read from the spec at
        transform time, so a bundle reloaded against an edited spec still
        encodes the way it was fitted.
    """

    numeric_names: tuple[str, ...]
    numeric_centre: NDArray[np.float64]
    numeric_scale: NDArray[np.float64]
    categorical_levels: Mapping[str, tuple[str, ...]]
    multi_label_levels: Mapping[str, tuple[str, ...]]
    lambdas: tuple[float, ...]
    maturity_key: str
    numeric_precision: str = "float64"

    @classmethod
    def fit(
        cls,
        attributes: Mapping[str, Sequence[Any]],
        *,
        spec: AttributeEncoderSpec,
    ) -> Self:
        """
        Fit over the full instrument universe.

        There is deliberately **no parameter restricting which instruments
        are seen**. Entity-axis fitting over the whole universe is correct
        here, and making it structural rather than conventional means a
        future edit cannot quietly introduce a restriction that would change
        the encoding -- and cannot be mistaken for leakage by a reader.

        Parameters
        ----------
        attributes
            Attribute name to one value per instrument, for every
            instrument in the universe.
        spec
            Which attributes to encode and how.

        Returns
        -------
        EntityEncoderState
            The fitted state.

        Raises
        ------
        ContractError
            If a declared attribute is absent, naming every missing one
            rather than failing on the first.
        """
        lambdas = decay_lambdas(spec.n_decay_terms)
        working = np.dtype(spec.numeric_precision)
        enriched = _with_decay_terms(
            attributes, lambdas=lambdas, maturity_key=spec.maturity_key, dtype=working
        )

        numeric_names = tuple(spec.numeric_keys) + tuple(_decay_name(value) for value in lambdas)
        _require(
            enriched, numeric_names + tuple(spec.categorical_keys) + tuple(spec.multi_label_keys)
        )

        # The statistics are always accumulated in float64, even when the
        # values themselves have been rounded to float32. That is what the
        # original did -- the rounding is in the data, not in the summation --
        # and accumulating a mean in float32 over a few hundred rows would
        # lose precision for no reason.
        numeric = _numeric_matrix(enriched, numeric_names, dtype=working).astype(np.float64)
        centre = numeric.mean(axis=0)
        scale = numeric.std(axis=0)
        # A constant attribute has zero spread. Dividing by it yields NaN,
        # which would propagate into every pairwise distance in the graph.
        # Replacing the divisor with one leaves the centred column at zero,
        # which is the honest encoding of an attribute that never varies.
        scale = np.where(scale > 0.0, scale, 1.0)

        return cls(
            numeric_names=numeric_names,
            numeric_centre=centre,
            numeric_scale=scale,
            categorical_levels={
                key: tuple(sorted({str(value) for value in enriched[key]}))
                for key in spec.categorical_keys
            },
            multi_label_levels={
                key: tuple(sorted(_distinct_labels(enriched[key]))) for key in spec.multi_label_keys
            },
            lambdas=lambdas,
            maturity_key=spec.maturity_key,
            numeric_precision=spec.numeric_precision,
        )

    def transform(self, attributes: Mapping[str, Sequence[Any]]) -> EncodedAttributes:
        """
        Encode a set of instruments using the fitted state.

        Parameters
        ----------
        attributes
            Attribute name to one value per instrument.

        Returns
        -------
        EncodedAttributes
            The matrix, its column names and its block ranges.

        Raises
        ------
        ContractError
            If a fitted attribute is absent from the input.
        """
        working = np.dtype(self.numeric_precision)
        enriched = _with_decay_terms(
            attributes, lambdas=self.lambdas, maturity_key=self.maturity_key, dtype=working
        )
        _require(
            enriched,
            self.numeric_names + tuple(self.categorical_levels) + tuple(self.multi_label_levels),
        )

        # Centring and scaling happen *in* the working precision, with the
        # float64 statistics downcast first. Under float64 that is a no-op.
        # Under float32 it reproduces the original: the subtraction and the
        # division both round, and doing them in float64 and rounding once at
        # the end gives a different answer in the sixth decimal place.
        numeric = _numeric_matrix(enriched, self.numeric_names, dtype=working)
        scaled = (numeric - self.numeric_centre.astype(working)) / self.numeric_scale.astype(
            working
        )

        matrices: list[NDArray[np.floating]] = [scaled.astype(np.float64)]
        names: list[str] = list(self.numeric_names)
        blocks: dict[str, tuple[int, int]] = {}

        position = 0
        for name in self.numeric_names:
            blocks[name] = (position, position + 1)
            position += 1

        for key, levels in self.categorical_levels.items():
            block = _one_hot(enriched[key], levels)
            matrices.append(block)
            names.extend(f"{key}={level}" for level in levels)
            blocks[key] = (position, position + block.shape[1])
            position += block.shape[1]

        for key, levels in self.multi_label_levels.items():
            block = _multi_label(enriched[key], levels)
            matrices.append(block)
            names.extend(f"{key}={level}" for level in levels)
            blocks[key] = (position, position + block.shape[1])
            position += block.shape[1]

        combined = np.hstack(matrices).astype(np.float32)
        return EncodedAttributes(features=combined, names=tuple(names), blocks=blocks)

    @property
    def n_features(self) -> int:
        """How many columns the encoding produces."""
        return (
            len(self.numeric_names)
            + sum(len(levels) for levels in self.categorical_levels.values())
            + sum(len(levels) for levels in self.multi_label_levels.values())
        )

    def save(self, directory: Path) -> None:
        """
        Write the fitted state as arrays and JSON.

        Parameters
        ----------
        directory
            Destination, created if absent.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "encoder.npz",
            numeric_centre=self.numeric_centre,
            numeric_scale=self.numeric_scale,
        )
        (directory / "encoder.json").write_text(
            json.dumps(
                {
                    "numeric_names": list(self.numeric_names),
                    # Written as ordered pairs rather than as objects,
                    # because the key order *is* the column order and a JSON
                    # object does not promise to keep it. Serialised as a
                    # mapping, `sort_keys=True` below alphabetises the keys,
                    # so a state fitted with `product_type` before
                    # `product_subtype` reloads with them swapped -- and the
                    # restored encoder then feeds the saved weights a matrix
                    # whose blocks have moved. It runs, and it is wrong.
                    "categorical_levels": [
                        [key, list(levels)] for key, levels in self.categorical_levels.items()
                    ],
                    "multi_label_levels": [
                        [key, list(levels)] for key, levels in self.multi_label_levels.items()
                    ],
                    "lambdas": list(self.lambdas),
                    "maturity_key": self.maturity_key,
                    "numeric_precision": self.numeric_precision,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Read a previously saved state.

        Parameters
        ----------
        directory
            Where :meth:`save` wrote.

        Returns
        -------
        EntityEncoderState
            The restored state.
        """
        payload = json.loads((directory / "encoder.json").read_text(encoding="utf-8"))
        with np.load(directory / "encoder.npz") as arrays:
            centre = arrays["numeric_centre"]
            scale = arrays["numeric_scale"]
        return cls(
            numeric_names=tuple(payload["numeric_names"]),
            numeric_centre=centre,
            numeric_scale=scale,
            categorical_levels={
                key: tuple(levels) for key, levels in payload["categorical_levels"]
            },
            multi_label_levels={
                key: tuple(levels) for key, levels in payload["multi_label_levels"]
            },
            lambdas=tuple(payload["lambdas"]),
            maturity_key=payload["maturity_key"],
            numeric_precision=payload.get("numeric_precision", "float64"),
        )


def _decay_name(value: float) -> str:
    """
    Name a decay column after its rate.

    Parameters
    ----------
    value
        The decay rate.

    Returns
    -------
    str
        The column name.
    """
    return f"ttm_decay_{value}"


def _with_decay_terms(
    attributes: Mapping[str, Sequence[Any]],
    *,
    lambdas: Sequence[float],
    maturity_key: str,
    dtype: np.dtype = _FLOAT64,
) -> dict[str, Sequence[Any]]:
    """
    Return a copy of the attributes with the decay columns added.

    A copy rather than a mutation: the caller's dictionary belongs to the
    data build, and the original's habit of writing derived columns back
    into it meant a second call saw a different input from the first.

    Parameters
    ----------
    attributes
        The raw attributes.
    lambdas
        Decay rates. Empty disables the feature.
    maturity_key
        Which attribute to decay.
    dtype
        Working precision. The exponential is evaluated in it, so under
        float32 the decay columns round exactly as the original's did.

    Returns
    -------
    dict
        Attributes plus one column per rate.

    Raises
    ------
    ContractError
        If decay terms are requested and the maturity attribute is absent.
    """
    enriched = dict(attributes)
    if not lambdas:
        return enriched

    if maturity_key not in enriched:
        raise ContractError(
            f"the encoder needs {maturity_key!r} to compute its {len(lambdas)} decay "
            f"term(s), but the attributes supplied are {sorted(enriched)}"
        )

    # The maturity is rounded to the working precision, but the exponential
    # itself is evaluated in float64 regardless. That asymmetry is not an
    # oversight: the rate is a float64 scalar, so under NumPy's promotion
    # rules the product widens to float64 even when the maturity is float32,
    # and the result is only rounded back when the numeric matrix is
    # assembled. Evaluating the exponential in float32 instead shifts the
    # fast-decaying columns by a last bit, which is enough to fail an exact
    # comparison against the baseline.
    maturity = np.asarray(enriched[maturity_key], dtype=dtype)
    for value in lambdas:
        enriched[_decay_name(value)] = np.exp(-np.float64(value) * maturity)
    return enriched


def _require(attributes: Mapping[str, Sequence[Any]], keys: Sequence[str]) -> None:
    """
    Check every needed attribute is present, naming all that are not.

    Parameters
    ----------
    attributes
        What was supplied.
    keys
        What is needed.

    Raises
    ------
    ContractError
        Listing every missing attribute. Failing on the first would turn
        a mis-specified encoder into several successive runs.
    """
    missing = [key for key in keys if key not in attributes]
    if missing:
        raise ContractError(
            f"the instrument attributes are missing {missing}; the encoder was given "
            f"{sorted(attributes)}"
        )


def _numeric_matrix(
    attributes: Mapping[str, Sequence[Any]],
    names: Sequence[str],
    *,
    dtype: np.dtype = _FLOAT64,
) -> NDArray[np.floating]:
    """
    Stack numeric attributes into a matrix, one column per attribute.

    Parameters
    ----------
    attributes
        The attributes.
    names
        Column order.
    dtype
        Working precision, applied per column before stacking so the
        rounding matches a per-attribute cast rather than a cast of the
        assembled matrix.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_instruments, len(names))``.
    """
    return np.column_stack([np.asarray(attributes[name], dtype=dtype) for name in names])


def _one_hot(column: Sequence[Any], levels: Sequence[str]) -> NDArray[np.float64]:
    """
    Encode a categorical column against a fixed set of levels.

    A value not among the fitted levels encodes as all zeros rather than
    raising. At inference an instrument may legitimately carry a product
    subtype that did not appear in training, and refusing the whole batch
    over it would be worse than encoding it as "none of the known levels".

    Parameters
    ----------
    column
        One value per instrument.
    levels
        The fitted levels, in order.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_instruments, len(levels))``.
    """
    position = {level: index for index, level in enumerate(levels)}
    encoded = np.zeros((len(column), len(levels)), dtype=np.float64)
    for row, value in enumerate(column):
        index = position.get(str(value))
        if index is not None:
            encoded[row, index] = 1.0
    return encoded


def _multi_label(column: Sequence[Any], levels: Sequence[str]) -> NDArray[np.float64]:
    """
    Encode a list-valued column, then normalise each row.

    Normalisation is what keeps the graph measuring *which* labels an
    instrument carries rather than *how many*. Without it, an instrument
    with three risk factors sits further from the origin than one with a
    single factor, and the nearest-neighbour search groups instruments by
    label count.

    Parameters
    ----------
    column
        One value per instrument, each a list or a scalar.
    levels
        The fitted labels, in order.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_instruments, len(levels))``, rows of unit norm.
    """
    position = {level: index for index, level in enumerate(levels)}
    encoded = np.zeros((len(column), len(levels)), dtype=np.float64)
    for row, value in enumerate(column):
        for label in _as_labels(value):
            index = position.get(label)
            if index is not None:
                encoded[row, index] = 1.0

    norms = np.linalg.norm(encoded, axis=1, keepdims=True)
    return encoded / np.maximum(norms, _NORM_FLOOR)


def _as_labels(value: object) -> list[str]:
    """
    Coerce one cell into a list of labels.

    Parameters
    ----------
    value
        A list, tuple, set, or a bare scalar.

    Returns
    -------
    list of str
        The labels.
    """
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value]
    return [str(value)]


def _distinct_labels(column: Sequence[Any]) -> set[str]:
    """
    Collect every label appearing in a list-valued column.

    Parameters
    ----------
    column
        One value per instrument.

    Returns
    -------
    set of str
        The distinct labels.
    """
    return {label for value in column for label in _as_labels(value)}
```

---

## 4. `src/rade_qnet/models/hybrid_gnn_rnn/features/graph.py`

20643 bytes · SHA-256 `03c836026a25e1fa`

```python
"""
Builds the sparse instrument graph the GNN passes messages over.

Like the attribute encoder, this is fitted along the **entity axis** over the
whole universe. Which instruments exist and how similar their attributes are
is known before any P&L is observed, so a graph spanning the full universe
uses no information a trader lacks.

The construction, and why each step is there
--------------------------------------------
1. **Weight the attribute groups.** Each group is scaled by the square root
   of its alpha, so squared Euclidean distance in the weighted space equals
   the alpha-weighted sum of squared differences. Without the square root
   the weights would apply to distance rather than to squared distance, and
   doubling an alpha would not double that attribute's influence.
2. **Divide one-hot blocks by the square root of their width.** A product
   type with eight levels otherwise contributes eight columns against a
   single scaled delta and dominates every distance purely by occupying more
   space.
3. **Find each instrument's nearest neighbours.**
4. **Convert distance to similarity with a Gaussian kernel**, using each
   node's own median neighbour distance as its bandwidth. A single global
   bandwidth would saturate in dense regions and vanish in sparse ones, so a
   thinly populated corner of the universe would end up with no usable edges
   at all.
5. **Add self-loops**, so a node's own features survive aggregation. Without
   them, message passing replaces each instrument with the average of its
   neighbours and the instrument's own attributes are lost after one round.
6. **Row-normalise.** Each row sums to one, which makes aggregation a mean
   rather than a sum and stops high-degree nodes from producing activations
   proportional to their degree.

Why this is implemented here rather than imported
-------------------------------------------------
The original used scikit-learn's neighbour search and SciPy's sparse
matrices. Both are reimplemented here in plain NumPy, for the same reasons
the encoder is: the framework's data path stays free of two heavy
dependencies, and the fitted state becomes plain arrays that serialise
without pickling an estimator.

The search is dense: it materialises the full pairwise distance matrix.
That is deliberate for the sizes this model runs at -- a cluster is tens to
low hundreds of instruments, where a dense matrix is a few hundred kilobytes
and is faster than any tree. It would be the wrong choice at tens of
thousands, and `n_instruments` is logged so that becoming the case is
visible.

The port is exact, and the golden fixture proves it: the edge list and the
edge weights are compared element for element at zero tolerance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from ....core.runtime.errors import ContractError
from ....core.runtime.logging import get_logger
from ..spec import GraphSpec
from .encoder import EncodedAttributes

__all__ = ["SparseGraphState", "build_graph"]

#: The default working precision, as a module-level singleton so it can be a
#: parameter default without constructing a dtype on every call.
_FLOAT64 = np.dtype(np.float64)

#: Floor on a node's kernel bandwidth. A node whose neighbours all sit at
#: distance zero -- duplicate instruments -- would otherwise divide by zero
#: and produce NaN weights on every one of its edges.
_BANDWIDTH_FLOOR = 1e-9

#: Floor on the multi-label row norm, matching the original. Distinct from
#: the encoder's floor only because the two were written separately; the
#: value is immaterial since the rows arriving here are already unit norm.
_NORM_FLOOR = 1e-8

#: Above this many instruments the dense distance matrix stops being a
#: sensible choice. Warned rather than refused: a large universe should
#: still run, just not silently at quadratic cost.
_DENSE_SEARCH_WARNING_THRESHOLD = 5_000

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SparseGraphState:
    """
    The fitted graph, as a coordinate-format sparse matrix.

    Stored as an edge list rather than as a dense matrix because the graph
    is a few neighbours per node: at a hundred instruments and five
    neighbours the dense form is a hundred times larger and is almost
    entirely zeros.

    Parameters
    ----------
    indices
        Edge endpoints, shape ``(n_edges, 2)`` as ``(row, column)`` pairs,
        sorted row-major. The order is part of the state: it is what the
        comparison against the baseline checks, and two graphs with the same
        edges in a different order are the same graph but not the same
        array.
    values
        Edge weights, one per row of ``indices``, each row summing to one.
    n_nodes
        How many instruments. Carried explicitly because an instrument with
        no edges would otherwise be invisible in the edge list and the
        matrix would silently come out too small.
    is_target
        Per-node flag marking target instruments, needed by the neighbour
        quota and by the reports.
    """

    indices: NDArray[np.int64]
    values: NDArray[np.float32]
    n_nodes: int
    is_target: NDArray[np.bool_]

    @property
    def n_edges(self) -> int:
        """How many edges, self-loops included."""
        return int(self.indices.shape[0])

    @property
    def dense_shape(self) -> tuple[int, int]:
        """The shape the sparse matrix would have if densified."""
        return (self.n_nodes, self.n_nodes)

    def degrees(self) -> NDArray[np.int64]:
        """
        Return how many edges leave each node.

        Returns
        -------
        numpy.ndarray
            One count per node, self-loop included.
        """
        return np.bincount(self.indices[:, 0], minlength=self.n_nodes).astype(np.int64)

    def save(self, directory: Path) -> None:
        """
        Write the graph as arrays and JSON.

        Parameters
        ----------
        directory
            Destination, created if absent.
        """
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(
            directory / "graph.npz",
            indices=self.indices,
            values=self.values,
            is_target=self.is_target,
        )
        (directory / "graph.json").write_text(
            json.dumps({"n_nodes": self.n_nodes}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Read a previously saved graph.

        Parameters
        ----------
        directory
            Where :meth:`save` wrote.

        Returns
        -------
        SparseGraphState
            The restored graph.
        """
        payload = json.loads((directory / "graph.json").read_text(encoding="utf-8"))
        with np.load(directory / "graph.npz") as arrays:
            return cls(
                indices=arrays["indices"],
                values=arrays["values"],
                n_nodes=int(payload["n_nodes"]),
                is_target=arrays["is_target"],
            )


def build_graph(
    encoded: EncodedAttributes,
    *,
    spec: GraphSpec,
    is_target: NDArray[np.bool_],
) -> SparseGraphState:
    """
    Build the row-normalised, kernel-weighted neighbour graph.

    Parameters
    ----------
    encoded
        The encoder's output, whose block ranges say which columns belong to
        which attribute group.
    spec
        Neighbour count, metric, quota and the per-group weights.
    is_target
        Per-instrument flag. Needed before the graph exists because the
        quota constrains a target's neighbourhood.

    Returns
    -------
    SparseGraphState
        The fitted graph.

    Raises
    ------
    ContractError
        If every attribute group is weighted to zero, which would leave no
        features to measure distance over. Refused rather than returning a
        graph of arbitrary structure.
    """
    features = weighted_features(encoded, spec=spec)
    n_nodes = int(features.shape[0])

    if n_nodes > _DENSE_SEARCH_WARNING_THRESHOLD:
        _LOGGER.warning(
            "building a graph over %d instruments with a dense distance matrix, which "
            "costs %.1f GB and grows quadratically. This path is tuned for the tens-to-"
            "hundreds a cluster holds; a universe this size wants an approximate index",
            n_nodes,
            n_nodes * n_nodes * 4 / 1e9,
        )

    # Clamped to what exists. Asking for more neighbours than there are
    # other instruments is not an error -- a small cluster is legitimate --
    # but silently returning fewer edges than requested would make the
    # graph's density depend on the universe size in a way nothing reports.
    n_neighbours = min(spec.n_neighbours, max(n_nodes - 1, 1))
    if n_neighbours < spec.n_neighbours:
        _LOGGER.info(
            "reduced n_neighbours from %d to %d: the universe holds %d instrument(s)",
            spec.n_neighbours,
            n_neighbours,
            n_nodes,
        )

    distances, neighbours = _nearest_neighbours(
        features, n_neighbours=n_neighbours, metric=spec.distance_metric
    )
    weights = _kernel_weights(distances)

    rows = np.repeat(np.arange(n_nodes, dtype=np.int64), neighbours.shape[1])
    columns = neighbours.ravel().astype(np.int64)
    data = weights.ravel().astype(np.float32)

    # Self-loops, so aggregation keeps a node's own features. Appended after
    # the neighbour edges and before normalisation, so each node's own
    # weight participates in its row sum.
    diagonal = np.arange(n_nodes, dtype=np.int64)
    rows = np.concatenate([rows, diagonal])
    columns = np.concatenate([columns, diagonal])
    data = np.concatenate([data, np.ones(n_nodes, dtype=np.float32)])

    indices, values = _normalise_rows(
        rows, columns, data, n_nodes=n_nodes, precision=np.dtype(spec.precision)
    )
    _LOGGER.info(
        "built a graph over %d instrument(s) with %d edge(s), %d target(s)",
        n_nodes,
        len(values),
        int(is_target.sum()),
    )
    return SparseGraphState(
        indices=indices, values=values, n_nodes=n_nodes, is_target=np.asarray(is_target, dtype=bool)
    )


def weighted_features(encoded: EncodedAttributes, *, spec: GraphSpec) -> NDArray[np.float32]:
    """
    Scale each attribute group so Euclidean distance becomes alpha-weighted.

    Each group is multiplied by the square root of its weight, so that the
    *squared* distance between two instruments is the weighted sum of
    squared per-attribute differences. Multiplying by the weight itself
    would make the weights apply to distance rather than to squared
    distance, and an alpha of four would quadruple an attribute's influence
    rather than double it.

    One-hot groups are additionally divided by the square root of their
    width, so a categorical attribute's influence depends on its weight
    rather than on how many levels it happens to have.

    Parameters
    ----------
    encoded
        The encoder's output.
    spec
        The per-group weights.

    Returns
    -------
    numpy.ndarray
        The weighted feature matrix.

    Raises
    ------
    ContractError
        If no group survives weighting.
    """
    parts: list[NDArray[np.float32]] = []

    for key, weight in _group_weights(spec).items():
        span = encoded.blocks.get(key)
        if span is None or weight <= 0.0:
            continue
        block = encoded.features[:, span[0] : span[1]].astype(np.float32)

        if key in _MULTI_LABEL_GROUPS:
            # Already unit norm from the encoder, so this is idempotent. Kept
            # because the graph's guarantee -- that label count does not
            # drive distance -- should not depend on a detail of how the
            # encoder happened to normalise.
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            block = block / np.maximum(norms, _NORM_FLOOR)
        elif block.shape[1] > 1:
            block = block / np.sqrt(block.shape[1])

        parts.append(np.sqrt(np.float32(weight)) * block)

    if not parts:
        raise ContractError(
            "every attribute group is either absent from the encoding or weighted to "
            "zero, so there are no features to measure instrument distance over. Check "
            "the graph spec's alpha weights against the encoder's attribute keys"
        )
    return np.hstack(parts).astype(np.float32)


def _group_weights(spec: GraphSpec) -> Mapping[str, float]:
    """
    Map each attribute group to its weight, in column order.

    Ordered, and the order is load-bearing: it fixes the column layout of
    the weighted matrix. Distance is invariant to column order, so a
    reordering would not change the graph -- but it would change the
    captured feature matrix, and a comparison that passes for the wrong
    reason is worth avoiding.

    Parameters
    ----------
    spec
        The graph spec.

    Returns
    -------
    Mapping
        Attribute name to weight.
    """
    return {
        "moneyness": spec.alpha_moneyness,
        "yrs_to_maturity": spec.alpha_maturity,
        "delta": spec.alpha_delta,
        "vega": spec.alpha_vega,
        "product_type": spec.alpha_product_type,
        "product_subtype": spec.alpha_product_subtype,
        "underlying": spec.alpha_underlying,
        "underlying_risk_factors": spec.alpha_underlying_risk_factors,
    }


#: Groups normalised per row rather than per block width.
_MULTI_LABEL_GROUPS = frozenset({"underlying_risk_factors"})


def _nearest_neighbours(
    features: NDArray[np.float32], *, n_neighbours: int, metric: str
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """
    Find each node's nearest neighbours, excluding itself.

    Parameters
    ----------
    features
        The weighted feature matrix.
    n_neighbours
        How many neighbours to return per node.
    metric
        ``euclidean``, ``manhattan`` or ``cosine``.

    Returns
    -------
    tuple
        Distances and indices, each shape ``(n_nodes, n_neighbours)``,
        sorted nearest first.
    """
    distances = _pairwise_distances(features, metric=metric)

    # A node is always its own nearest neighbour at distance zero, so the
    # search asks for one extra and drops the first column. Setting the
    # diagonal to infinity instead is more robust: with duplicate
    # instruments, two rows sit at distance zero from each other and the
    # self column is no longer guaranteed to sort first -- dropping column
    # zero would then drop a genuine neighbour and keep the self-loop twice.
    np.fill_diagonal(distances, np.inf)

    # A stable sort, so that instruments equidistant from a node are taken
    # in index order. With the default quicksort the tie order depends on
    # the array's layout, which would make the graph differ between two runs
    # on identical data.
    order = np.argsort(distances, axis=1, kind="stable")[:, :n_neighbours]
    rows = np.arange(features.shape[0])[:, None]
    return distances[rows, order], order.astype(np.int64)


def _pairwise_distances(features: NDArray[np.float32], *, metric: str) -> NDArray[np.float64]:
    """
    Compute the full distance matrix.

    Parameters
    ----------
    features
        The weighted feature matrix.
    metric
        Which distance to use.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_nodes, n_nodes)``, in float64 so that the subtraction
        below does not lose precision on nearly-identical instruments.

    Raises
    ------
    ContractError
        If the metric is not one the builder implements.
    """
    values = features.astype(np.float64)

    if metric == "euclidean":
        # Computed from the explicit difference rather than from the
        # expanded `|a|^2 - 2ab + |b|^2`. The expansion is faster and
        # cancels catastrophically for near-identical rows, which is
        # precisely the case here: instruments in the same group differ in
        # one attribute out of thirteen, and the expansion can return a
        # small negative number whose square root is NaN.
        difference = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.einsum("ijk,ijk->ij", difference, difference))

    if metric == "manhattan":
        return np.abs(values[:, None, :] - values[None, :, :]).sum(axis=2)

    if metric == "cosine":
        norms = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), _NORM_FLOOR)
        unit = values / norms
        return 1.0 - unit @ unit.T

    raise ContractError(
        f"unknown distance metric {metric!r}; the graph builder implements "
        f"'euclidean', 'manhattan' and 'cosine'"
    )


def _kernel_weights(distances: NDArray[np.float64]) -> NDArray[np.float32]:
    """
    Turn distances into similarities with a per-node Gaussian kernel.

    Each node's bandwidth is the median of its own neighbour distances. A
    single global bandwidth would saturate in dense regions -- every weight
    near one, so the graph carries no information about which neighbour is
    closer -- and vanish in sparse ones, leaving isolated instruments with
    weights that underflow to zero and no usable edges at all.

    Parameters
    ----------
    distances
        Neighbour distances, shape ``(n_nodes, n_neighbours)``.

    Returns
    -------
    numpy.ndarray
        Similarities of the same shape.
    """
    bandwidth = np.median(np.maximum(distances, _BANDWIDTH_FLOOR), axis=1, keepdims=True).astype(
        np.float32
    )
    return np.exp(-(distances.astype(np.float32) ** 2) / (2.0 * bandwidth**2)).astype(np.float32)


def _normalise_rows(
    rows: NDArray[np.int64],
    columns: NDArray[np.int64],
    data: NDArray[np.float32],
    *,
    n_nodes: int,
    precision: np.dtype = _FLOAT64,
) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
    """
    Sum duplicate edges, sort row-major, and scale each row to sum to one.

    Duplicates are summed rather than overwritten, which is what a
    coordinate-format matrix means by convention and what the original's
    sparse library did. They arise when the quota adds an edge the
    neighbour search already found.

    Row normalisation makes aggregation a mean. Without it a node with
    twenty neighbours produces activations roughly twenty times larger than
    one with a single neighbour, and the network spends its capacity
    learning to undo the degree.

    Parameters
    ----------
    rows, columns
        Edge endpoints.
    data
        Edge weights.
    n_nodes
        Matrix order.
    precision
        Which precision the row sums and their reciprocals are computed in.
        ``float32`` reproduces the original; ``float64`` is the default and
        loses nothing.

    Returns
    -------
    tuple
        The ``(n_edges, 2)`` index array and the matching weights.
    """
    # One integer key per cell, so duplicates become equal keys and a single
    # sort puts the whole matrix in row-major order. int64 holds this for
    # any universe up to three billion instruments.
    keys = rows * np.int64(n_nodes) + columns
    order = np.argsort(keys, kind="stable")
    keys, data = keys[order], data[order]

    unique_keys, start = np.unique(keys, return_index=True)
    summed = np.add.reduceat(data, start).astype(np.float32)

    unique_rows = (unique_keys // n_nodes).astype(np.int64)
    unique_columns = (unique_keys % n_nodes).astype(np.int64)

    # Accumulated in float64 and then rounded to the working precision, so
    # that `float32` matches a float32 row sum rather than a float32
    # accumulation -- the original summed through a sparse matrix whose
    # internal accumulation is not reproducible element by element, but
    # whose *result* is a float32 value.
    row_sums = np.bincount(unique_rows, weights=summed, minlength=n_nodes).astype(precision)
    # A node with no edges at all would divide by zero. It cannot happen
    # while self-loops are added, but the guard costs nothing and the
    # alternative is a NaN that propagates through every message-passing
    # round.
    inverse = np.divide(
        precision.type(1.0), row_sums, out=np.zeros_like(row_sums), where=row_sums > 0
    )
    normalised = (summed * inverse[unique_rows].astype(np.float32)).astype(np.float32)

    return np.column_stack([unique_rows, unique_columns]).astype(np.int64), normalised
```

