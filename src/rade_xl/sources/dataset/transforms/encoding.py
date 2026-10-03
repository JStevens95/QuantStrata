"""
Categorical encoding along the entity axis.

The transform where the leakage rule is the *opposite* of everywhere else, and
getting that backwards is costly in both directions.

Why the full universe is permitted here
---------------------------------------
Every other fitted transform in this package sees training rows only.
:class:`EncodingState` sees every entity, and that is correct rather than a
relaxation.

An encoder along the entity axis learns which instruments exist and what
category each belongs to. That is not knowledge of the future: the fact that
``USDTRY`` is an emerging-market pair is known before any P&L is observed, and
a cross-sectional model shown only some instruments would be solving a
different problem from the one it is deployed on.

A conformance rule that simply said "fitted transforms may only see training
indices" would wrongly fail the flagship model, whose graph construction spans
the whole instrument universe -- which is why
:mod:`rade_xl.testkit.conformance` distinguishes the two axes rather than
applying one rule to both.

Unseen entities are refused, not guessed
----------------------------------------
An entity absent at fit time has no code. Mapping it to a shared "unknown"
bucket would let a transductive model return a confident prediction for an
instrument it knows nothing about, which is the quiet failure the
:class:`~rade_xl.core.capability.protocols.Inductive` capability exists to
make visible. :meth:`EncodingState.encode` raises instead, and a model that
genuinely generalises to unseen entities declares that capability and uses
entity *features* rather than an identity code.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self

import numpy as np

from ....core.contract.state import FittedState
from ....core.runtime.errors import BundleError, CapabilityError, ContractError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ["EncodingState"]

#: On-disk filename. Part of the format: see the note in ``scaling``.
_CATEGORIES_FILENAME = "encoding.json"

#: How many unknown entities to name in an error message before truncating.
_MAX_REPORTED_UNKNOWN = 5


class EncodingState(FittedState):
    """
    Integer codes for entity identities and their categorical attributes.

    Parameters
    ----------
    entities
        Every known entity identifier, in code order: ``entities[code]`` is
        the identifier that code refers to. A sequence rather than a mapping
        because the order *is* the encoding, and it must survive a JSON round
        trip unchanged -- a model's embedding row three must mean the same
        instrument when the bundle is reloaded.
    attributes
        Attribute name to its ordered category list, encoded the same way.

    Raises
    ------
    ContractError
        If an entity identifier is duplicated. A duplicate makes the
        identifier-to-code direction ambiguous, and whichever occurrence won
        would decide which embedding row an instrument got.
    """

    def __init__(
        self,
        *,
        entities: Sequence[str] = (),
        attributes: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.entities = tuple(entities)
        self.attributes = {
            name: tuple(categories) for name, categories in (attributes or {}).items()
        }

        duplicated = sorted({name for name in self.entities if self.entities.count(name) > 1})
        if duplicated:
            raise ContractError(
                f"entity identifier(s) {duplicated} appear more than once; each "
                f"identifier maps to exactly one code, so duplicates make the "
                f"encoding ambiguous"
            )

        # Built once rather than per lookup: encoding a forty-thousand-row
        # universe through `list.index` would be quadratic.
        self._entity_codes = {name: code for code, name in enumerate(self.entities)}
        self._attribute_codes = {
            name: {category: code for code, category in enumerate(categories)}
            for name, categories in self.attributes.items()
        }

    @classmethod
    def fit(
        cls,
        entity_ids: Sequence[str],
        *,
        attributes: Mapping[str, Sequence[str]] | None = None,
    ) -> Self:
        """
        Learn codes from the full entity universe.

        Takes identifiers directly rather than a matrix and a set of training
        indices, which is the signature that makes the entity-axis rule
        structural: there is no parameter here that *could* restrict the fit to
        training rows, so the question cannot be got wrong by passing the wrong
        argument.

        Categories are sorted before coding, so the encoding depends on the
        *set* of entities rather than the order they happened to arrive in.
        Two runs over the same universe therefore produce the same codes even
        if an upstream query returned the rows differently.

        Parameters
        ----------
        entity_ids
            Identifier per row. Duplicates are expected -- one row per entity
            per scenario -- and are collapsed.
        attributes
            Attribute name to the value observed for each row, aligned with
            ``entity_ids``.

        Returns
        -------
        Self
            The fitted state.

        Raises
        ------
        ContractError
            If an attribute's values do not align with ``entity_ids``.
        """
        for name, values in (attributes or {}).items():
            if len(values) != len(entity_ids):
                raise ContractError(
                    f"attribute {name!r} has {len(values)} value(s) but there are "
                    f"{len(entity_ids)} entity identifier(s); they must correspond "
                    f"one to one"
                )

        return cls(
            entities=sorted(set(entity_ids)),
            attributes={name: sorted(set(values)) for name, values in (attributes or {}).items()},
        )

    @property
    def n_entities(self) -> int:
        """How many distinct entities are known."""
        return len(self.entities)

    def encode(self, entity_ids: Sequence[str]) -> NDArray[np.int64]:
        """
        Map entity identifiers to their codes.

        Parameters
        ----------
        entity_ids
            Identifiers to encode.

        Returns
        -------
        numpy.ndarray
            One code per identifier.

        Raises
        ------
        CapabilityError
            If any identifier was not present at fit time. A
            :class:`~rade_xl.core.runtime.errors.CapabilityError` rather than a
            contract error because the situation is meaningful: the caller
            asked a transductive encoder about an entity it cannot represent,
            which is precisely what the ``Inductive`` capability declares
            whether a model can survive.
        """
        unknown = sorted({name for name in entity_ids if name not in self._entity_codes})
        if unknown:
            shown = unknown[:_MAX_REPORTED_UNKNOWN]
            suffix = f" (and {len(unknown) - len(shown)} more)" if len(unknown) > len(shown) else ""
            raise CapabilityError(
                f"entity identifier(s) {shown}{suffix} were not present when the "
                f"encoder was fitted, so they have no code. This encoding is "
                f"transductive: a model that must predict for new entities should "
                f"consume entity features and declare the Inductive capability"
            )
        return np.array([self._entity_codes[name] for name in entity_ids], dtype=np.int64)

    def encode_attribute(self, name: str, values: Sequence[str]) -> NDArray[np.int64]:
        """
        Map one attribute's values to their codes.

        Parameters
        ----------
        name
            Attribute name.
        values
            Values to encode.

        Returns
        -------
        numpy.ndarray
            One code per value.

        Raises
        ------
        ContractError
            If the attribute is unknown, listing the attributes that were
            fitted -- the usual cause is a renamed column.
        CapabilityError
            If a value was not seen at fit time.
        """
        if name not in self._attribute_codes:
            raise ContractError(
                f"attribute {name!r} was not encoded; fitted attributes are "
                f"{sorted(self._attribute_codes)}"
            )
        codes = self._attribute_codes[name]
        unknown = sorted({value for value in values if value not in codes})
        if unknown:
            raise CapabilityError(
                f"attribute {name!r} received unseen value(s) "
                f"{unknown[:_MAX_REPORTED_UNKNOWN]}; known values are "
                f"{list(self.attributes[name])}"
            )
        return np.array([codes[value] for value in values], dtype=np.int64)

    def one_hot(self, entity_ids: Sequence[str]) -> NDArray[np.float64]:
        """
        Encode entity identifiers as indicator columns.

        Provided for models with no embedding layer -- a linear model or a
        tree -- which need the identity as features rather than as an index.

        Parameters
        ----------
        entity_ids
            Identifiers to encode.

        Returns
        -------
        numpy.ndarray
            Rows by known entities, one indicator per column.

        Raises
        ------
        CapabilityError
            If any identifier is unknown.
        """
        codes = self.encode(entity_ids)
        indicators = np.zeros((codes.size, self.n_entities), dtype=np.float64)
        indicators[np.arange(codes.size), codes] = 1.0
        return indicators

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return the predictions unchanged.

        An encoder acts on entity identity, never on the target. Implemented
        explicitly rather than inherited, for the reason given in
        :class:`~rade_xl.core.contract.state.FittedState`.

        Parameters
        ----------
        predictions
            Model output.

        Returns
        -------
        numpy.ndarray
            The same values.
        """
        return np.asarray(predictions, dtype=np.float64)

    def save(self, directory: Path) -> None:
        """
        Write the category lists beneath a directory.

        JSON rather than ``.npy``, because the content is strings and because
        the mapping from an instrument to an embedding row is exactly the thing
        somebody will want to read by eye when a prediction looks wrong.

        Parameters
        ----------
        directory
            Destination directory, created by the caller.
        """
        payload = {
            "entities": list(self.entities),
            "attributes": {name: list(values) for name, values in self.attributes.items()},
        }
        (directory / _CATEGORIES_FILENAME).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Read the state back from a directory written by :meth:`save`.

        Parameters
        ----------
        directory
            Directory previously written by :meth:`save`.

        Returns
        -------
        Self
            A state equal to the one saved.

        Raises
        ------
        BundleError
            If the file is missing.
        """
        path = directory / _CATEGORIES_FILENAME
        if not path.is_file():
            raise BundleError(
                f"encoding state at {directory} is missing {path.name}; the bundle "
                f"may be incomplete or written by an incompatible version"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(entities=payload["entities"], attributes=payload["attributes"])

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Reports counts rather than the lists themselves: a universe of four
        hundred instruments would make a run summary unreadable, and the full
        lists are in the bundle for anyone who needs them.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "type": type(self).__name__,
            "n_entities": self.n_entities,
            "attributes": {name: len(values) for name, values in self.attributes.items()},
        }

    def __eq__(self, other: object) -> bool:
        """Return whether another state holds the same codes."""
        if not isinstance(other, EncodingState):
            return NotImplemented
        return self.entities == other.entities and self.attributes == other.attributes

    def __hash__(self) -> int:
        """Return a hash over the entity tuple and attribute names."""
        return hash((self.entities, tuple(sorted(self.attributes))))
