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
from ....core.runtime.errors import BundleError, ContractError
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
