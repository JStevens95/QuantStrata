"""
Feature and target scaling, fitted on training rows only.

The one transform that is almost always present, and the one whose inverse
matters most: if the target is scaled and the inverse is wrong, every metric a
user reads is wrong by the same factor and nothing says so.

Which rows a scaler may see is not configurable
-----------------------------------------------
:meth:`ScalingState.fit` takes the training rows and nothing else. There is no
``fit_on`` option here, unlike
:class:`~rade_xl.core.spec.data.ReductionSpec`, because no legitimate reason
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
from ....core.runtime.errors import BundleError, ContractError
from ....core.runtime.logging import get_logger

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
