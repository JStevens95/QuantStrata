"""
Everything a model fits at training time and needs again at inference.

Model weights are not the whole of what training produces. A scaler's mean and
scale, the ordered subset of features a basis selection chose, an entity
encoder's category table, a nearest-neighbour graph -- all of these are fitted
from training data and all are required to make a single prediction later.

The implementation this framework replaces wrote roughly twenty loose sidecar
files for exactly this, with the relationship between them encoded only in the
code that loaded them. The result was predictable: any change to that code
silently invalidated every previously saved model, and nothing detected it.

:class:`FittedState` replaces that with one typed object that knows how to save
itself, load itself, and -- crucially -- put a prediction back into the units
the user cares about.

Why the inverse is abstract
---------------------------
:meth:`FittedState.inverse_transform_targets` has no default implementation.
A model that scales its target and cannot invert the scaling reports a mean
absolute error in standardised space, which is not a quantity anyone can act
on. Making the method abstract means that failure cannot be shipped by
omission -- it has to be chosen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

__all__ = ["FittedState", "IdentityFittedState"]


class FittedState(ABC):
    """
    Abstract base for a model's fitted, non-parameter state.

    Implementations are free to hold whatever they need -- arrays, mappings,
    sparse structures -- provided the three methods below are honest about it.

    Notes
    -----
    Saving is directory-based rather than single-file. A scaler is naturally a
    pair of arrays, a graph is three, and forcing them through one pickle
    reintroduces the fragility this class exists to remove. A directory of
    named ``.npy`` and ``.json`` files can be inspected with standard tools
    and loaded by code that has never imported this framework.
    """

    @abstractmethod
    def save(self, directory: Path) -> None:
        """
        Write the state beneath a directory.

        The directory already exists when this is called. Implementations
        should write named files rather than one opaque blob, and must not
        write outside the directory given.

        Parameters
        ----------
        directory
            Destination directory, created by the caller.
        """

    @classmethod
    @abstractmethod
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
            A state equal to the one that was saved.

        Raises
        ------
        BundleError
            If the directory is missing a required file.
        """

    @abstractmethod
    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return predictions to the original target units.

        Called by the evaluate and infer pipelines *before* any metric is
        computed, so every number a user reads is in the units they think it
        is in.

        A model that does not transform its target should return the input
        unchanged -- see :class:`IdentityFittedState` -- but it must say so
        explicitly rather than inheriting a default.

        Parameters
        ----------
        predictions
            Model output, in whatever space the model produces.

        Returns
        -------
        numpy.ndarray
            Predictions in original target units, with the same shape.
        """

    def describe(self) -> dict[str, object]:
        """
        Return a summary of what was fitted, for reports and logs.

        The default reports only the class name. Implementations should
        override it with the things a reader would want to check at a glance:
        how many features a basis selected, how many categories an encoder
        saw, how many edges a graph has. This ends up in the run summary, so
        it is often the first place an anomaly becomes visible.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {"type": type(self).__name__}


class IdentityFittedState(FittedState):
    """
    Fitted state for a model that fits nothing beyond its parameters.

    Appropriate for a model whose target is untransformed: a tree model on raw
    features, or a network trained on an unscaled target. Using this when the
    target *is* transformed produces metrics in the wrong units, which is why
    it is a named, deliberate choice rather than the base class default.

    Also used by ``rade_qnet.testkit.fixtures`` so that a pipeline test needs no
    bespoke state implementation.
    """

    def save(self, directory: Path) -> None:
        """
        Write a marker file recording that there was nothing to save.

        An empty directory is indistinguishable from a failed write, so a
        marker is written instead. Bundle verification can then confirm the
        state was saved deliberately.

        Parameters
        ----------
        directory
            Destination directory.
        """
        (directory / "identity.marker").write_text(
            "This model fits no state beyond its parameters.\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """
        Return a fresh instance, ignoring the directory's contents.

        Parameters
        ----------
        directory
            Unused; accepted to satisfy the interface.

        Returns
        -------
        Self
            A new instance.
        """
        del directory  # Nothing to read: the state is, by definition, empty.
        return cls()

    def inverse_transform_targets(self, predictions: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Return the predictions unchanged.

        Parameters
        ----------
        predictions
            Model output.

        Returns
        -------
        numpy.ndarray
            The same array.
        """
        return predictions

    def __eq__(self, other: object) -> bool:
        """Return whether the other object is also an identity state."""
        return isinstance(other, IdentityFittedState)

    def __hash__(self) -> int:
        """Return a constant hash, since all instances are equivalent."""
        return hash(IdentityFittedState)
