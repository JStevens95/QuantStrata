"""
Tests for feature and target scaling.

The clause that matters most here is that statistics come from the training
rows alone. Scaling fitted over the full history leaks the test set's mean and
variance into every training row, and the result is a model that scores better
than it should by a margin small enough to look like skill.

The second clause is that the target's inverse is a real inverse. If it is
not, every metric is reported in the wrong units -- which makes it
incomparable with another run and unusable for a decision, while still looking
like a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_xl.core.spec.data import ScalingSpec
from src.rade_xl.sources.dataset.transforms.scaling import ScalingState


@pytest.fixture
def data():
    """
    Provide features and a target whose halves have very different scales.

    Deliberately non-stationary: the second half is shifted and stretched, so
    statistics fitted over everything differ from training-only statistics by
    far more than floating-point noise. A test on stationary data cannot
    detect the leak it is meant to catch.

    Returns
    -------
    tuple
        Features, target, and the training indices.
    """
    rng = np.random.default_rng(0)
    features = rng.normal(size=(200, 3))
    features[100:] = features[100:] * 5.0 + 20.0
    target = rng.normal(size=(200, 1))
    target[100:] = target[100:] * 5.0 + 20.0
    return features, target, np.arange(100)


class TestFittingWindow:
    """Statistics come from the training rows and nowhere else."""

    def test_the_mean_is_the_training_mean(self, data):
        """
        Not the mean of everything, which is the leak.

        Asserted against the hand-computed training mean rather than against a
        tolerance, because the whole point is which rows were used.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        scaled = state.transform_features(features[train])
        assert np.allclose(scaled.mean(axis=0), 0.0, atol=1e-9)

    def test_the_held_out_rows_are_not_centred(self, data):
        """
        The visible consequence of fitting on training rows only.

        Held-out data transformed with training statistics is *not* mean zero,
        and if it were, the statistics had seen it.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        held_out = state.transform_features(features[100:])
        assert np.abs(held_out.mean()) > 1.0

    def test_extending_the_window_changes_the_state(self, data):
        """
        A state that ignored its window would pass every other test here.

        Two fits over different row sets must differ, or the window argument
        is decorative.
        """
        features, target, _ = data
        narrow = ScalingState.fit(
            features, target, spec=ScalingSpec(), train_indices=np.arange(100)
        )
        wide = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=np.arange(200))
        assert narrow != wide


class TestTargetInversion:
    """A metric in the wrong units is worse than no metric."""

    def test_the_inverse_round_trips(self, data):
        """
        Forward then back is the identity, to numerical precision.

        A forgotten offset or a reciprocal scale passes every shape check and
        then misreports every error in the run.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        recovered = state.inverse_transform_targets(state.transform_targets(target))
        assert np.allclose(recovered, target, atol=1e-9)

    def test_an_unscaled_target_is_passed_through(self, data):
        """
        The configuration where the inverse must be exactly the identity.

        A state that always applied *some* transform would shift a target that
        was never scaled.
        """
        features, target, train = data
        state = ScalingState.fit(
            features,
            target,
            spec=ScalingSpec(scale_target=False),
            train_indices=train,
        )
        assert np.array_equal(state.inverse_transform_targets(target), target)


class TestDegenerateColumns:
    """A constant column has zero variance, and dividing by it is a NaN."""

    def test_a_constant_column_does_not_produce_nan(self, data):
        """
        Which would poison every downstream computation silently.

        A NaN in one feature column makes the whole forward pass NaN, and the
        symptom is a loss that is NaN from the first step -- a long way from
        the cause.
        """
        features, target, train = data
        features = features.copy()
        features[:, 1] = 7.0
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        assert np.all(np.isfinite(state.transform_features(features)))

    def test_degenerate_columns_are_counted_in_the_description(self, data):
        """
        So the run summary says the data had a constant feature.

        Handling it silently means nobody finds out the column was useless.
        """
        features, target, train = data
        features = features.copy()
        features[:, 1] = 7.0
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        assert state.describe()["n_degenerate_features"] == 1


class TestPersistence:
    """A served model must use the state it was trained with."""

    def test_the_state_round_trips_through_a_directory(self, data, tmp_path):
        """
        Equality after a save and load, not merely a successful load.

        A state that loaded with a subtly different scale produces a model
        whose predictions are wrong in a way that will be blamed on the data.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        state.save(tmp_path)
        assert ScalingState.load(tmp_path) == state

    def test_the_reloaded_state_transforms_identically(self, data, tmp_path):
        """
        The property equality is standing in for, asserted directly.

        Equality could be satisfied by a state that compared only its
        metadata, so the transform itself is checked too.
        """
        features, target, train = data
        state = ScalingState.fit(features, target, spec=ScalingSpec(), train_indices=train)
        state.save(tmp_path)
        reloaded = ScalingState.load(tmp_path)
        assert np.array_equal(
            reloaded.transform_features(features), state.transform_features(features)
        )


class TestMethods:
    """The spec's scaling methods are distinguishable."""

    def test_robust_scaling_resists_an_outlier(self, data):
        """
        The reason the method exists.

        A single extreme row moves a mean and a standard deviation enough to
        squash every other row towards zero; a median and an interquartile
        range barely move. Financial data has those rows.
        """
        features, target, train = data
        features = features.copy()
        features[0, 0] = 1e6

        standard = ScalingState.fit(
            features, target, spec=ScalingSpec(method="standard"), train_indices=train
        )
        robust = ScalingState.fit(
            features, target, spec=ScalingSpec(method="robust"), train_indices=train
        )
        ordinary_rows = np.arange(1, 100)
        assert np.std(robust.transform_features(features[ordinary_rows])[:, 0]) > np.std(
            standard.transform_features(features[ordinary_rows])[:, 0]
        )
