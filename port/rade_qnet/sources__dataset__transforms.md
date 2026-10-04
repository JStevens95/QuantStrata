# `src/rade_qnet/sources/dataset/transforms`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 33 | 1514 | `6d3de29fb11b2a0c` |
| 2 | `composite.py` | 331 | 11553 | `f47510e9282e3cc7` |
| 3 | `encoding.py` | 368 | 13222 | `904b518e82471413` |
| 4 | `reduction.py` | 492 | 18021 | `b938b45b7953e380` |
| 5 | `scaling.py` | 446 | 15935 | `a1932b200ec1be9d` |
| 6 | `sequence.py` | 251 | 9184 | `f6955b509b3329f8` |

---

## 1. `src/rade_qnet/sources/dataset/transforms/__init__.py`

1514 bytes · SHA-256 `6d3de29fb11b2a0c`

```python
"""
Fitted transforms, each with an explicit inverse.

Every transform here follows the same contract: it is fitted on training rows,
it reports the ``FittedState`` it produced, and -- if it touched the target --
it can invert a prediction back to the original units.  A transform that cannot
state its inverse cannot be applied to a target, because a metric computed in
transformed space is not the metric anyone asked for.

Modules
-------
``scaling.py``
    Standardisation and robust scaling, fitted on training rows only.  Owns
    the target inverse when it scaled the target.  [Phase 2]
``sequence.py``
    Rolling-window construction, with window boundaries respected at split
    edges.  Plain functions rather than a fitted state: windowing is fully
    determined by the specification, so there is nothing to fit.  [Phase 2]
``reduction.py``
    Dimensionality reduction, including basis selection.  Exposes an explicit
    ``fit_on`` choice (``train`` or ``all``) rather than silently fitting on
    every row, which is a leakage path in the implementation this framework
    replaces.  [Phase 2]
``encoding.py``
    Categorical and attribute encoding along the entity axis.  The one
    transform fitted over the full universe rather than training rows, for the
    reason given in its module docstring.  [Phase 2]
``composite.py``
    Composes the above into the single ``FittedState`` a bundle holds, and
    settles which part owns the target inverse.  [Phase 2]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/sources/dataset/transforms/composite.py`

11553 bytes · SHA-256 `f47510e9282e3cc7`

```python
"""
Composing several fitted transforms into the one state a bundle holds.

:attr:`~rade_qnet.core.contract.data.DataBundle.state` is a single
:class:`~rade_qnet.core.contract.state.FittedState`, but a realistic data build
fits three or four things: a scaler, a reduction, an entity encoder. Something
has to compose them, and that something is here rather than inside
``module.py`` so that a model writing its own data module inherits the
composition instead of reimplementing it.

Exactly one part may own the target
-----------------------------------
:meth:`CompositeState.inverse_transform_targets` delegates to a single part.
Two parts both transforming the target would make the inverse
order-dependent -- and the order is not recorded anywhere, so the composition
would invert correctly today and incorrectly after an unrelated change to the
build. :class:`CompositeState` rejects that at construction rather than
picking an order.

In practice only scaling touches the target. Reduction acts on features and
encoding on entity identity, which is why both of their inverses are the
identity.

How loading works without recording a class path
------------------------------------------------
A subclass declares its parts in :attr:`CompositeState.part_types`, a mapping
from part name to concrete state class. :meth:`CompositeState.load` reads that
declaration rather than anything stored on disk.

The alternative -- recording ``rade_qnet.sources...ScalingState`` in a manifest
and importing it back -- is exactly the fragility
:class:`~rade_qnet.core.contract.state.FittedState` was designed to remove:
renaming or moving the class would invalidate every bundle written before the
rename, and nothing would detect it until a load failed. Declaring the parts in
code means a rename is a refactor the type checker sees.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np

from ....core.contract.state import FittedState
from ....core.lifecycle.errors import BundleError, ContractError
from .encoding import EncodingState
from .reduction import ReductionState
from .scaling import ScalingState

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from numpy.typing import NDArray

__all__ = ["CompositeState", "DatasetState"]

#: Records which parts were actually written, so loading can tell an absent
#: optional part from a failed write.
_PARTS_FILENAME = "parts.json"


class CompositeState(FittedState):
    """
    A fitted state assembled from named parts.

    Parameters
    ----------
    parts
        Part name to fitted state. Names must appear in
        :attr:`part_types`, so a part cannot be saved that could not be
        loaded back.
    target_owner
        Which part inverts the target. ``None`` means no part transforms it,
        and the composite's inverse is the identity.

    Raises
    ------
    ContractError
        If a part name is not declared in :attr:`part_types`, or
        ``target_owner`` names a part that was not supplied.
    """

    #: Part name to concrete state class, declared by each subclass. Empty on
    #: the base, which therefore cannot be loaded -- see the module docstring.
    part_types: ClassVar[Mapping[str, type[FittedState]]] = {}

    def __init__(
        self,
        *,
        parts: Mapping[str, FittedState],
        target_owner: str | None = None,
    ) -> None:
        undeclared = sorted(set(parts) - set(self.part_types))
        if undeclared:
            raise ContractError(
                f"part(s) {undeclared} are not declared in "
                f"{type(self).__name__}.part_types ({sorted(self.part_types)}); "
                f"a part that is not declared could be saved but never loaded back"
            )
        if target_owner is not None and target_owner not in parts:
            raise ContractError(
                f"target_owner={target_owner!r} is not among the supplied parts "
                f"{sorted(parts)}; the target inverse would have nothing to delegate to"
            )

        self.parts = dict(parts)
        self.target_owner = target_owner

    def part(self, name: str) -> FittedState:
        """
        Return one part.

        Parameters
        ----------
        name
            Part name.

        Returns
        -------
        FittedState
            The part.

        Raises
        ------
        ContractError
            If the part is absent, listing what is present. The usual cause is
            reaching for an optional part -- an encoder on a build that had no
            entity axis -- so the message distinguishes that from a typo.
        """
        try:
            return self.parts[name]
        except KeyError:
            raise ContractError(
                f"this state has no part named {name!r}; present parts are {sorted(self.parts)}"
            ) from None

    def has_part(self, name: str) -> bool:
        """
        Return whether a part is present.

        Parameters
        ----------
        name
            Part name.

        Returns
        -------
        bool
            True if present.
        """
        return name in self.parts

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return predictions to the original target units.

        Delegates to the single part that owns the target, or returns the
        input unchanged when no part does.

        Parameters
        ----------
        predictions
            Model output.

        Returns
        -------
        numpy.ndarray
            Predictions in original units.
        """
        if self.target_owner is None:
            return np.asarray(predictions, dtype=np.float64)
        return self.parts[self.target_owner].inverse_transform_targets(predictions)

    def save(self, directory: Path) -> None:
        """
        Write each part into its own subdirectory, plus a parts manifest.

        One subdirectory per part rather than one flat directory, so two parts
        are free to use the same filename -- both the scaler and the reduction
        would otherwise want ``settings.json``.

        Parameters
        ----------
        directory
            Destination directory, created by the caller.
        """
        for name, part in self.parts.items():
            part_directory = directory / name
            part_directory.mkdir(parents=True, exist_ok=True)
            part.save(part_directory)

        manifest = {"parts": sorted(self.parts), "target_owner": self.target_owner}
        (directory / _PARTS_FILENAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
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
            If the manifest is missing, or names a part this class does not
            declare -- which means the bundle was written by a version whose
            composition differed, and loading it would produce a state missing
            a transform without saying so.
        """
        manifest_path = directory / _PARTS_FILENAME
        if not manifest_path.is_file():
            raise BundleError(
                f"composite state at {directory} is missing {_PARTS_FILENAME}; "
                f"the bundle may be incomplete or written by an incompatible version"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        unknown = sorted(set(manifest["parts"]) - set(cls.part_types))
        if unknown:
            raise BundleError(
                f"bundle at {directory} records part(s) {unknown} that "
                f"{cls.__name__} does not declare ({sorted(cls.part_types)}); it was "
                f"written by a version with a different composition, and loading it "
                f"would silently drop a fitted transform"
            )

        parts = {name: cls.part_types[name].load(directory / name) for name in manifest["parts"]}
        return cls(parts=parts, target_owner=manifest["target_owner"])

    def describe(self) -> dict[str, object]:
        """
        Return a summary composed from the parts' own summaries.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "type": type(self).__name__,
            "target_owner": self.target_owner,
            "parts": {name: part.describe() for name, part in sorted(self.parts.items())},
        }

    def __eq__(self, other: object) -> bool:
        """Return whether another composite holds equal parts."""
        if not isinstance(other, CompositeState):
            return NotImplemented
        return (
            type(self) is type(other)
            and self.target_owner == other.target_owner
            and self.parts == other.parts
        )

    def __hash__(self) -> int:
        """Return a hash over the part names and the target owner."""
        return hash((type(self).__name__, tuple(sorted(self.parts)), self.target_owner))


class DatasetState(CompositeState):
    """
    The standard composition produced by :class:`DataModule`.

    Scaling, reduction and encoding, each optional. Scaling owns the target
    inverse because it is the only one of the three that touches the target.

    A model with a data build of its own is free to subclass
    :class:`CompositeState` with a different part list instead; this class is
    the default rather than a requirement.
    """

    part_types: ClassVar[Mapping[str, type[FittedState]]] = {
        "scaling": ScalingState,
        "reduction": ReductionState,
        "encoding": EncodingState,
    }

    @classmethod
    def of(
        cls,
        *,
        scaling: ScalingState | None = None,
        reduction: ReductionState | None = None,
        encoding: EncodingState | None = None,
    ) -> Self:
        """
        Build the composition from whichever parts were fitted.

        A named constructor rather than conditional dictionary building at
        every call site, because "which parts are present" is a question with
        one answer and it should be answered in one place.

        Parameters
        ----------
        scaling
            The fitted scaler, or ``None`` if none was fitted.
        reduction
            The fitted reduction, or ``None``.
        encoding
            The fitted entity encoder, or ``None``.

        Returns
        -------
        Self
            The composed state.
        """
        supplied: dict[str, FittedState] = {}
        if scaling is not None:
            supplied["scaling"] = scaling
        if reduction is not None:
            supplied["reduction"] = reduction
        if encoding is not None:
            supplied["encoding"] = encoding

        # Only a scaler that actually transformed the target owns the inverse.
        # A scaler fitted with `scale_target=False` leaves the target alone, so
        # claiming ownership would be true but misleading in the bundle.
        owns_target = scaling is not None and scaling.scaled_target
        return cls(parts=supplied, target_owner="scaling" if owns_target else None)
```

---

## 3. `src/rade_qnet/sources/dataset/transforms/encoding.py`

13222 bytes · SHA-256 `904b518e82471413`

```python
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
:mod:`rade_qnet.testkit.conformance` distinguishes the two axes rather than
applying one rule to both.

Unseen entities are refused, not guessed
----------------------------------------
An entity absent at fit time has no code. Mapping it to a shared "unknown"
bucket would let a transductive model return a confident prediction for an
instrument it knows nothing about, which is the quiet failure the
:class:`~rade_qnet.core.authoring.capabilities.Inductive` capability exists to
make visible. :meth:`EncodingState.encode` raises instead, and a model that
genuinely generalises to unseen entities declares that capability and uses
entity *features* rather than an identity code.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self

import numpy as np

from ....core.contract.state import FittedState
from ....core.lifecycle.errors import BundleError, CapabilityError, ContractError

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
            :class:`~rade_qnet.core.lifecycle.errors.CapabilityError` rather than a
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
        :class:`~rade_qnet.core.contract.state.FittedState`.

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
```

---

## 4. `src/rade_qnet/sources/dataset/transforms/reduction.py`

18021 bytes · SHA-256 `b938b45b7953e380`

```python
"""
Dimensionality reduction, with an explicit choice of which rows it may see.

This module is where **defect 9** is fixed. Basis selection in the
implementation being replaced ran over the full scaled history -- validation
and test rows included -- so the choice of which series to keep was informed
by the held-out period. The selected features then looked unusually
predictive on exactly the data used to judge the model.

The leak is subtle because selection is not fitting
---------------------------------------------------
It is easy to see why a scaler must not see test rows. Selection feels
different: no parameter is estimated from the held-out data, only a *subset*
is chosen. But the subset is a function of the held-out rows, and the model is
then trained on features that were picked because they work on the test
period. The score is optimistic and nothing in the run reports it.

Why the old behaviour is reachable at all
-----------------------------------------
``fit_on="all"`` reproduces the leak. It exists for one reason: proving a
refactor changed nothing else. A golden fixture captured from the old
implementation can only be reproduced if the old selection can be reproduced,
and fixing the bug in the same change that proves the refactor would mean
neither was verified.

It is not the default, it is logged at warning every time it is used, and the
setting is recorded in the lineage so a bundle produced this way is
identifiable afterwards.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self

import numpy as np

from ....core.contract.state import FittedState
from ....core.lifecycle.errors import BundleError, ContractError
from ....core.provenance.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ....core.spec.data import ReductionSpec

__all__ = ["ReductionState"]

_LOGGER = get_logger(__name__)

#: On-disk filenames. Part of the format: see the note in ``scaling``.
_COMPONENTS_FILENAME = "reduction.npz"
_SETTINGS_FILENAME = "reduction.json"


class ReductionState(FittedState):
    """
    A selected feature subset, or a learned projection.

    One class for both methods because they differ only in how a column of
    output is produced: selection keeps an input column, projection mixes
    several. Keeping them together means the data module has one object to
    hold whichever was configured, and the inverse is the identity for both.

    Parameters
    ----------
    method
        ``none``, ``basis_selection`` or ``pca``.
    selected
        Indices of the kept input columns, in the order they are kept. Order
        is part of the state: a model's first input must mean the same thing
        at inference as it did in training, and a set would not preserve that.
        Empty for ``pca``.
    components
        Projection matrix, inputs by components, for ``pca``. ``None``
        otherwise.
    centre
        Column means subtracted before projecting, for ``pca``. ``None``
        otherwise.
    fit_on
        Which rows the fit observed. Recorded because ``all`` means the output
        is not comparable with a correctly fitted run, and that has to be
        visible in the bundle rather than remembered.
    n_input_features
        How many columns were offered, so a mismatch at inference can name
        both counts.
    """

    def __init__(
        self,
        *,
        method: str = "none",
        selected: NDArray[np.int64] | None = None,
        components: NDArray[np.floating] | None = None,
        centre: NDArray[np.floating] | None = None,
        fit_on: str = "train",
        n_input_features: int = 0,
    ) -> None:
        self.method = method
        self.selected = (
            np.empty(0, dtype=np.int64) if selected is None else np.asarray(selected, np.int64)
        )
        self.components = None if components is None else np.asarray(components, np.float64)
        self.centre = None if centre is None else np.asarray(centre, np.float64)
        self.fit_on = fit_on
        self.n_input_features = n_input_features

    @classmethod
    def fit(
        cls,
        features: NDArray[np.floating],
        target: NDArray[np.floating],
        *,
        spec: ReductionSpec,
        train_indices: NDArray[np.int64],
    ) -> Self:
        """
        Fit the reduction on whichever rows the specification permits.

        Parameters
        ----------
        features
            The full feature matrix, samples by features.
        target
            The full target vector. Used by ``basis_selection``, which ranks
            columns by their association with the target, and ignored by
            ``pca``, which is unsupervised.
        spec
            Method, component count and -- the field that matters --
            ``fit_on``.
        train_indices
            Training rows. Used when ``fit_on="train"``; deliberately ignored
            when ``fit_on="all"``, which is the behaviour being reproduced.

        Returns
        -------
        Self
            The fitted state.

        Raises
        ------
        ContractError
            If the requested component count exceeds the columns available,
            or the training rows are empty.
        """
        n_input_features = features.shape[1]
        if spec.method == "none":
            return cls(method="none", fit_on=spec.fit_on, n_input_features=n_input_features)

        if train_indices.size == 0:
            raise ContractError("a reduction needs at least one training row to fit")

        if spec.fit_on == "all":
            # Reproduces defect 9. Warned every time rather than once, because
            # a forty-job set should produce forty warnings -- one per bundle
            # that is not comparable with a correctly fitted one.
            _LOGGER.warning(
                "reduction method %r is being fitted with fit_on='all', which lets "
                "the selection observe validation and test rows. This reproduces a "
                "known leak in the previous implementation and exists only for "
                "refactor parity; it must not be used for a production run",
                spec.method,
            )
            rows = np.arange(features.shape[0], dtype=np.int64)
        else:
            rows = train_indices

        observed_features = np.asarray(features[rows], dtype=np.float64)
        observed_target = np.asarray(target[rows], dtype=np.float64)

        n_components = spec.n_components or n_input_features
        if n_components > n_input_features:
            raise ContractError(
                f"n_components={n_components} exceeds the {n_input_features} feature "
                f"column(s) available; reduce n_components or supply more features"
            )

        if spec.method == "basis_selection":
            selected = _select_basis(observed_features, observed_target, n_components)
            return cls(
                method="basis_selection",
                selected=selected,
                fit_on=spec.fit_on,
                n_input_features=n_input_features,
            )

        if spec.method == "pca":
            components, centre = _fit_pca(observed_features, n_components)
            return cls(
                method="pca",
                components=components,
                centre=centre,
                fit_on=spec.fit_on,
                n_input_features=n_input_features,
            )

        raise ContractError(
            f"unknown reduction method {spec.method!r}; expected 'none', 'basis_selection' or 'pca'"
        )

    @property
    def n_output_features(self) -> int:
        """How many columns :meth:`transform_features` produces."""
        if self.method == "basis_selection":
            return int(self.selected.size)
        if self.method == "pca" and self.components is not None:
            return int(self.components.shape[1])
        return self.n_input_features

    def transform_features(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Reduce a feature matrix.

        Parameters
        ----------
        features
            Samples by features, with the column count the state was fitted
            on.

        Returns
        -------
        numpy.ndarray
            The reduced matrix, as float64.

        Raises
        ------
        ContractError
            If the column count disagrees with the fit.
        """
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != self.n_input_features:
            raise ContractError(
                f"features have {values.shape[-1]} column(s) but the reduction was "
                f"fitted on {self.n_input_features}; the feature set changed between "
                f"fitting and transforming"
            )
        if self.method == "basis_selection":
            return values[..., self.selected]
        if self.method == "pca" and self.components is not None and self.centre is not None:
            return (values - self.centre) @ self.components
        return values

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return the predictions unchanged.

        A reduction acts on features, never on the target, so its contribution
        to the target inverse is the identity. The method is implemented rather
        than inherited because
        :class:`~rade_qnet.core.contract.state.FittedState` makes it abstract --
        deliberately, so that "this transform does not touch the target" is a
        statement somebody made rather than a default nobody noticed.

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
        Write the components and the settings beneath a directory.

        Parameters
        ----------
        directory
            Destination directory, created by the caller.
        """
        arrays: dict[str, NDArray[np.generic]] = {"selected": self.selected}
        if self.components is not None:
            arrays["components"] = self.components
        if self.centre is not None:
            arrays["centre"] = self.centre
        np.savez(directory / _COMPONENTS_FILENAME, **arrays)

        settings = {
            "method": self.method,
            "fit_on": self.fit_on,
            "n_input_features": self.n_input_features,
        }
        (directory / _SETTINGS_FILENAME).write_text(
            json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8"
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
            If either file is missing.
        """
        components_path = directory / _COMPONENTS_FILENAME
        settings_path = directory / _SETTINGS_FILENAME
        for path in (components_path, settings_path):
            if not path.is_file():
                raise BundleError(
                    f"reduction state at {directory} is missing {path.name}; "
                    f"the bundle may be incomplete or written by an incompatible version"
                )

        with np.load(components_path) as arrays:
            selected = arrays["selected"]
            components = arrays.get("components")
            centre = arrays.get("centre")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

        return cls(
            method=settings["method"],
            selected=selected,
            components=components,
            centre=centre,
            fit_on=settings["fit_on"],
            n_input_features=settings["n_input_features"],
        )

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Returns
        -------
        dict
            JSON-encodable summary. Includes ``fit_on`` unconditionally, so a
            leaking parity run is identifiable from the run summary alone.
        """
        return {
            "type": type(self).__name__,
            "method": self.method,
            "fit_on": self.fit_on,
            "n_input_features": self.n_input_features,
            "n_output_features": self.n_output_features,
            "selected": self.selected.tolist(),
        }

    def __eq__(self, other: object) -> bool:
        """Return whether another state holds the same reduction."""
        if not isinstance(other, ReductionState):
            return NotImplemented
        return (
            self.method == other.method
            and self.fit_on == other.fit_on
            and self.n_input_features == other.n_input_features
            and np.array_equal(self.selected, other.selected)
            and _arrays_equal(self.components, other.components)
            and _arrays_equal(self.centre, other.centre)
        )

    def __hash__(self) -> int:
        """Return a hash over the scalar settings only."""
        return hash((self.method, self.fit_on, self.n_input_features))


def _select_basis(
    features: NDArray[np.float64],
    target: NDArray[np.float64],
    n_components: int,
) -> NDArray[np.int64]:
    """
    Choose the columns most strongly associated with the target.

    Ranks by the absolute Pearson correlation between each column and the
    target, which is the criterion the previous implementation used. Keeping
    the same criterion is deliberate: the defect being fixed is *which rows*
    the ranking observed, not how it ranked. Changing both at once would make
    a parity difference impossible to attribute.

    Parameters
    ----------
    features
        Rows the selection is permitted to observe, samples by features.
    target
        Target values for those rows.
    n_components
        How many columns to keep.

    Returns
    -------
    numpy.ndarray
        Kept column indices, ascending. Ascending rather than
        strongest-first so that the output column order matches the input
        column order, which keeps a feature-importance table readable.
    """
    centred_features = features - features.mean(axis=0)

    # Flattened first. A target supplied as a column vector -- which is the
    # shape the input signature declares, and the shape a batch carries --
    # makes the covariance below two-dimensional, and `argsort` then sorts
    # along an axis of length one and returns the columns unranked. The result
    # is a basis that looks fitted and is in input order, which is the hardest
    # kind of wrong to notice.
    flat_target = np.ravel(np.asarray(target, dtype=np.float64))
    centred_target = flat_target - flat_target.mean()

    feature_norms = np.linalg.norm(centred_features, axis=0)
    target_norm = float(np.linalg.norm(centred_target))

    # A constant column has zero norm, so its correlation is undefined rather
    # than zero. Scoring it zero is the right answer -- it carries no
    # information -- and avoids a divide-by-zero warning on every build.
    denominator = feature_norms * target_norm
    covariance = centred_features.T @ centred_target
    scores = np.where(
        denominator > 0.0, np.abs(covariance) / np.where(denominator > 0.0, denominator, 1.0), 0.0
    )

    # argsort is ascending, so the strongest `n_components` are the tail. The
    # kind is fixed to 'stable' so that tied scores resolve by column order and
    # two runs on the same data select the same basis.
    ranked = np.argsort(scores, kind="stable")
    strongest = ranked[-n_components:]
    return np.sort(strongest).astype(np.int64)


def _fit_pca(
    features: NDArray[np.float64],
    n_components: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Fit a principal-component projection.

    Uses a singular value decomposition of the centred matrix rather than an
    eigendecomposition of the covariance, which is the numerically stable
    choice when the feature count approaches the sample count.

    Parameters
    ----------
    features
        Rows the fit is permitted to observe, samples by features.
    n_components
        How many components to keep.

    Returns
    -------
    tuple of numpy.ndarray
        The projection matrix (inputs by components) and the column means.
    """
    centre = features.mean(axis=0)
    _, _, right_singular_vectors = np.linalg.svd(features - centre, full_matrices=False)
    components = right_singular_vectors[:n_components].T

    # Sign of a singular vector is arbitrary, so two runs on the same data can
    # produce projections differing by a sign per component -- which would make
    # saved state compare unequal and a parity check fail for no real reason.
    # Fixing the sign so each component's largest-magnitude loading is positive
    # makes the result deterministic.
    dominant = np.argmax(np.abs(components), axis=0)
    signs = np.sign(components[dominant, np.arange(components.shape[1])])
    signs[signs == 0.0] = 1.0
    return components * signs, centre


def _arrays_equal(
    first: NDArray[np.floating] | None,
    second: NDArray[np.floating] | None,
) -> bool:
    """
    Return whether two optional arrays are both absent or both equal.

    Parameters
    ----------
    first, second
        Arrays to compare, either of which may be ``None``.

    Returns
    -------
    bool
        True if they agree.
    """
    if first is None or second is None:
        return first is None and second is None
    return np.array_equal(first, second)
```

---

## 5. `src/rade_qnet/sources/dataset/transforms/scaling.py`

15935 bytes · SHA-256 `a1932b200ec1be9d`

```python
"""
Feature and target scaling, fitted on training rows only.

The one transform that is almost always present, and the one whose inverse
matters most: if the target is scaled and the inverse is wrong, every metric a
user reads is wrong by the same factor and nothing says so.

Which rows a scaler may see is not configurable
-----------------------------------------------
:meth:`ScalingState.fit` takes the training rows and nothing else. There is no
``fit_on`` option here, unlike
:class:`~rade_qnet.core.spec.data.ReductionSpec`, because no legitimate reason
exists to fit a scaler on held-out data -- it leaks the test period's mean and
variance into training, and the resulting score flatters the model with no way
to tell by how much.

``ReductionSpec`` has the flag only because the implementation being replaced
leaked there and a refactor has to be able to reproduce the old numbers to
prove it changed nothing else. No equivalent reproduction is needed here.

Why a constant column is scaled by one rather than rejected
-----------------------------------------------------------
A feature with zero variance over the training rows carries no information,
and dividing by its scale would produce infinities. Rejecting the build would
be defensible, but it fails runs for a harmless reason -- a dummy column, an
indicator that happens to be constant in one job of a forty-job set. The scale
is set to one instead, which leaves the column centred and finite, and the
count is logged so it is visible rather than silent.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Self

import numpy as np

from ....core.contract.state import FittedState
from ....core.lifecycle.errors import BundleError, ContractError
from ....core.provenance.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ....core.spec.data import ScalingSpec

__all__ = ["ScalingState"]

_LOGGER = get_logger(__name__)

#: Filenames inside the state directory. Named constants because they are part
#: of the on-disk format: renaming one invalidates every bundle already written.
_STATISTICS_FILENAME = "scaling.npz"
_SETTINGS_FILENAME = "scaling.json"

#: Scale substituted for a feature with no variance over the training rows.
#: See the module docstring for why this is preferred to rejecting the build.
_DEGENERATE_SCALE = 1.0

#: Interquartile range used by the robust method, as percentiles.
_LOWER_QUARTILE = 25.0
_UPPER_QUARTILE = 75.0


class ScalingState(FittedState):
    """
    Centre and scale statistics for features and, optionally, the target.

    Parameters
    ----------
    feature_centre, feature_scale
        One value per feature column. The centre is a mean or a median and the
        scale a standard deviation or an interquartile range, according to the
        method.
    target_centre, target_scale
        Scalars for the target. Both are zero and one respectively when the
        target was not scaled, so the inverse is the identity without needing
        a branch.
    method
        Which statistics were used, recorded so a reader of the bundle can
        tell a mean from a median.
    scaled_target
        Whether the target was transformed. Recorded rather than inferred from
        the statistics, because a target whose mean happens to be zero and
        whose standard deviation happens to be one is indistinguishable from
        an untransformed one.
    n_degenerate_features
        How many columns had no variance and were given a scale of one.
        Carried into :meth:`describe` so it appears in the run summary --
        a sudden jump in this number across runs usually means an upstream
        data problem.
    """

    def __init__(
        self,
        *,
        feature_centre: NDArray[np.floating],
        feature_scale: NDArray[np.floating],
        target_centre: float = 0.0,
        target_scale: float = 1.0,
        method: str = "standard",
        scaled_target: bool = False,
        n_degenerate_features: int = 0,
    ) -> None:
        self.feature_centre = np.asarray(feature_centre, dtype=np.float64)
        self.feature_scale = np.asarray(feature_scale, dtype=np.float64)
        self.target_centre = float(target_centre)
        self.target_scale = float(target_scale)
        self.method = method
        self.scaled_target = scaled_target
        self.n_degenerate_features = n_degenerate_features

    @classmethod
    def fit(
        cls,
        features: NDArray[np.floating],
        target: NDArray[np.floating],
        *,
        spec: ScalingSpec,
        train_indices: NDArray[np.int64],
    ) -> Self:
        """
        Fit the statistics on the training rows.

        Parameters
        ----------
        features
            The full feature matrix, samples by features. Passed whole, with
            the training rows selected here, so that a caller cannot
            accidentally pass a pre-sliced matrix *and* indices into it.
        target
            The full target vector.
        spec
            Which method to use and whether to scale the target.
        train_indices
            Rows the scaler may observe. These and no others.

        Returns
        -------
        Self
            The fitted state.

        Raises
        ------
        ContractError
            If the training rows are empty, or the feature matrix and target
            disagree on sample count.
        """
        if train_indices.size == 0:
            raise ContractError("a scaler needs at least one training row to fit")
        if features.shape[0] != target.shape[0]:
            raise ContractError(
                f"features has {features.shape[0]} rows but target has "
                f"{target.shape[0]}; they must correspond one to one"
            )

        train_features = features[train_indices]
        train_target = target[train_indices]

        if spec.method == "none":
            # An explicit identity rather than a skipped transform, so that
            # every path produces a state with a working inverse.
            n_features = features.shape[1]
            return cls(
                feature_centre=np.zeros(n_features),
                feature_scale=np.ones(n_features),
                method="none",
                scaled_target=False,
            )

        feature_centre, raw_feature_scale = _statistics(train_features, method=spec.method, axis=0)
        feature_scale, n_degenerate = _replace_degenerate(raw_feature_scale)

        if n_degenerate:
            _LOGGER.info(
                "%d of %d feature column(s) have no variance over the training rows; "
                "their scale is set to %g so they remain finite",
                n_degenerate,
                feature_scale.size,
                _DEGENERATE_SCALE,
            )

        target_centre, target_scale = 0.0, 1.0
        if spec.scale_target:
            centre, scale = _statistics(train_target.reshape(-1, 1), method=spec.method, axis=0)
            target_scale_array, _ = _replace_degenerate(scale)
            target_centre, target_scale = float(centre[0]), float(target_scale_array[0])

        return cls(
            feature_centre=feature_centre,
            feature_scale=feature_scale,
            target_centre=target_centre,
            target_scale=target_scale,
            method=spec.method,
            scaled_target=spec.scale_target,
            n_degenerate_features=n_degenerate,
        )

    def transform_features(self, features: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Centre and scale a feature matrix.

        Parameters
        ----------
        features
            Samples by features, with the same column count the state was
            fitted on.

        Returns
        -------
        numpy.ndarray
            The transformed matrix, as float64.

        Raises
        ------
        ContractError
            If the column count disagrees with the fitted statistics. This is
            the earliest point at which a feature set that changed between
            training and inference is detectable, and the message names both
            counts.
        """
        if features.shape[-1] != self.feature_centre.size:
            raise ContractError(
                f"features have {features.shape[-1]} column(s) but the scaler was "
                f"fitted on {self.feature_centre.size}; the feature set changed "
                f"between fitting and transforming"
            )
        return (np.asarray(features, dtype=np.float64) - self.feature_centre) / self.feature_scale

    def transform_targets(self, target: NDArray[np.floating]) -> NDArray[np.float64]:
        """
        Centre and scale a target vector.

        Parameters
        ----------
        target
            Target values.

        Returns
        -------
        numpy.ndarray
            The transformed target, or an unchanged float64 copy when the
            target was not scaled.
        """
        values = np.asarray(target, dtype=np.float64)
        if not self.scaled_target:
            return values
        return (values - self.target_centre) / self.target_scale

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return predictions to the original target units.

        Parameters
        ----------
        predictions
            Model output, in the space the model was trained in.

        Returns
        -------
        numpy.ndarray
            Predictions in original units, same shape as the input.
        """
        values = np.asarray(predictions, dtype=np.float64)
        if not self.scaled_target:
            return values
        return values * self.target_scale + self.target_centre

    def save(self, directory: Path) -> None:
        """
        Write the statistics and the settings beneath a directory.

        Two files rather than one: the arrays go to ``.npz`` and the scalar
        settings to ``.json``, so the settings can be read by a human or by a
        tool that has no NumPy.

        Parameters
        ----------
        directory
            Destination directory, created by the caller.
        """
        np.savez(
            directory / _STATISTICS_FILENAME,
            feature_centre=self.feature_centre,
            feature_scale=self.feature_scale,
        )
        settings = {
            "method": self.method,
            "scaled_target": self.scaled_target,
            "target_centre": self.target_centre,
            "target_scale": self.target_scale,
            "n_degenerate_features": self.n_degenerate_features,
        }
        (directory / _SETTINGS_FILENAME).write_text(
            json.dumps(settings, indent=2, sort_keys=True), encoding="utf-8"
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
            If either file is missing. Named individually, because a missing
            ``.json`` and a missing ``.npz`` have different causes -- the first
            suggests a partial write, the second a bundle from an incompatible
            version.
        """
        statistics_path = directory / _STATISTICS_FILENAME
        settings_path = directory / _SETTINGS_FILENAME
        for path in (statistics_path, settings_path):
            if not path.is_file():
                raise BundleError(
                    f"scaling state at {directory} is missing {path.name}; "
                    f"the bundle may be incomplete or written by an incompatible version"
                )

        with np.load(statistics_path) as statistics:
            feature_centre = statistics["feature_centre"]
            feature_scale = statistics["feature_scale"]
        settings = json.loads(settings_path.read_text(encoding="utf-8"))

        return cls(
            feature_centre=feature_centre,
            feature_scale=feature_scale,
            target_centre=settings["target_centre"],
            target_scale=settings["target_scale"],
            method=settings["method"],
            scaled_target=settings["scaled_target"],
            n_degenerate_features=settings["n_degenerate_features"],
        )

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "type": type(self).__name__,
            "method": self.method,
            "n_features": int(self.feature_centre.size),
            "scaled_target": self.scaled_target,
            "target_centre": self.target_centre,
            "target_scale": self.target_scale,
            "n_degenerate_features": self.n_degenerate_features,
        }

    def __eq__(self, other: object) -> bool:
        """Return whether another state holds the same statistics and settings."""
        if not isinstance(other, ScalingState):
            return NotImplemented
        return (
            np.array_equal(self.feature_centre, other.feature_centre)
            and np.array_equal(self.feature_scale, other.feature_scale)
            and self.target_centre == other.target_centre
            and self.target_scale == other.target_scale
            and self.method == other.method
            and self.scaled_target == other.scaled_target
        )

    def __hash__(self) -> int:
        """Return a hash over the scalar settings only."""
        return hash((self.method, self.scaled_target, self.target_centre, self.target_scale))


def _statistics(
    values: NDArray[np.floating],
    *,
    method: str,
    axis: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Return the centre and scale for one method.

    Parameters
    ----------
    values
        Rows to summarise.
    method
        ``standard`` for mean and standard deviation, ``robust`` for median
        and interquartile range.
    axis
        Axis to reduce over.

    Returns
    -------
    tuple of numpy.ndarray
        Centre and scale, in that order.

    Raises
    ------
    ContractError
        If the method is unrecognised. Reached only through a spec that was
        not validated, so this is a framework bug rather than a user error.
    """
    data = np.asarray(values, dtype=np.float64)
    if method == "standard":
        return np.mean(data, axis=axis), np.std(data, axis=axis)
    if method == "robust":
        centre = np.median(data, axis=axis)
        spread = np.percentile(data, _UPPER_QUARTILE, axis=axis) - np.percentile(
            data, _LOWER_QUARTILE, axis=axis
        )
        return centre, spread
    raise ContractError(f"unknown scaling method {method!r}; expected 'standard' or 'robust'")


def _replace_degenerate(scale: NDArray[np.floating]) -> tuple[NDArray[np.float64], int]:
    """
    Replace non-positive scales with one, and report how many were replaced.

    Parameters
    ----------
    scale
        Fitted scales, one per column.

    Returns
    -------
    tuple
        The corrected scales and the number of replacements.
    """
    values = np.asarray(scale, dtype=np.float64)
    # Not-finite is included deliberately: an all-NaN column produces a NaN
    # scale, and dividing by it would propagate NaN into every downstream
    # metric with nothing identifying the column that caused it.
    degenerate = ~(values > 0.0) | ~np.isfinite(values)
    corrected = np.where(degenerate, _DEGENERATE_SCALE, values)
    return corrected, int(np.count_nonzero(degenerate))
```

---

## 6. `src/rade_qnet/sources/dataset/transforms/sequence.py`

9184 bytes · SHA-256 `f6955b509b3329f8`

```python
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
```

