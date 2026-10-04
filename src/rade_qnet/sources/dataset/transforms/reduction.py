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
from ....core.runtime.errors import BundleError, ContractError
from ....core.runtime.logging import get_logger

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
