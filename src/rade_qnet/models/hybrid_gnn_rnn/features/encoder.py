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

from ....core.lifecycle.errors import ContractError
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
