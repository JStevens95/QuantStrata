"""
Tests for dimensionality reduction, and for defect 9.

Defect 9 is the leak this module was rewritten to fix: the basis -- the
selected feature subset, or the principal components -- was fitted over the
*full scaled history* rather than over the training rows. Selecting which
features matter using the test set's correlations is leakage, and it is a
particularly deniable kind: no row of test data enters the model, only a
decision about which columns to keep. The decision is enough.

The test for it is a positive control. Synthetic data is built so that the
most informative column over the training rows differs from the most
informative column over the whole history, and the two fits are asserted to
select different bases. A test that merely checked "the basis is fitted" would
pass against the defective implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.spec.data import ReductionSpec
from src.rade_qnet.sources.dataset.transforms.reduction import ReductionState


@pytest.fixture
def divergent():
    """
    Build data where the best feature differs by fitting window.

    Column 0 is informative only in the first half, column 2 only in the
    second. So a basis fitted on the training rows picks column 0 and one
    fitted on everything picks column 2 -- which is what makes the leak
    observable rather than merely plausible.

    Returns
    -------
    tuple
        Features, target, and the training indices.
    """
    rng = np.random.default_rng(5)
    n = 400
    target = rng.normal(size=n)

    features = rng.normal(size=(n, 4))
    # Strongly informative in the training half, pure noise afterwards.
    features[:200, 0] = target[:200] * 3.0 + rng.normal(0, 0.2, 200)
    features[200:, 0] = rng.normal(0, 3.0, 200)
    # Weakly informative in the training half, overwhelmingly so afterwards.
    # The asymmetry is what separates the two rankings cleanly: over the
    # training rows column 0 wins at 0.997 against 0.393, and over the whole
    # history column 2 wins at 0.751 against 0.484. Without that margin the
    # two could tie and the test would pass or fail on the tie-break rather
    # than on the leak.
    features[:200, 2] = target[:200] * 0.8 + rng.normal(0, 1.5, 200)
    features[200:, 2] = target[200:] * 8.0 + rng.normal(0, 0.2, 200)
    return features, target.reshape(-1, 1), np.arange(200)


class TestDefectNineTheBasisIsFittedOnTrainingRowsOnly:
    """The leak, and the positive control that detects it."""

    def test_the_fitting_window_changes_the_selected_basis(self, divergent):
        """
        The defect, stated as a difference nothing else would produce.

        If the window were ignored, these two fits would be identical. They
        are not, which is the only direct evidence that the window is
        respected.
        """
        features, target, train = divergent
        spec = ReductionSpec(method="basis_selection", n_components=1)

        correct = ReductionState.fit(features, target, spec=spec, train_indices=train)
        # The leaky configuration is reached through `fit_on='all'`, which the
        # spec documents as existing solely for refactor parity. Reaching it
        # by widening `train_indices` would not exercise the same code path.
        leaky = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=1, fit_on="all"),
            train_indices=train,
        )
        assert list(correct.describe()["selected"]) == [0]
        assert list(leaky.describe()["selected"]) == [2]

    def test_the_training_window_selects_the_training_informative_column(self, divergent):
        """
        Not merely different, but right.

        A fit that differed from the full-history fit by selecting the *wrong*
        column would pass the previous test and still be broken.
        """
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=1),
            train_indices=train,
        )
        assert list(state.describe()["selected"]) == [0]


class TestSelection:
    """Correlation-ranked feature selection."""

    def test_the_output_width_matches_the_request(self, divergent):
        """A reduction that returned the wrong width breaks the signature."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=2),
            train_indices=train,
        )
        assert state.transform_features(features).shape == (400, 2)
        assert state.n_output_features == 2

    def test_requesting_every_column_is_a_no_op(self, divergent):
        """
        The identity case, which should not be a special case.

        A reduction asked for all four of four columns should return them, not
        reorder or rescale them.
        """
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="basis_selection", n_components=4),
            train_indices=train,
        )
        assert np.allclose(state.transform_features(features), features)

    def test_selection_is_deterministic_under_ties(self):
        """
        Two equally correlated columns must resolve the same way every run.

        A tie broken by an unstable sort makes two otherwise identical runs
        select different features, and the difference in their scores looks
        like variance in the model.
        """
        rng = np.random.default_rng(1)
        target = rng.normal(size=100).reshape(-1, 1)
        features = np.column_stack([target.ravel(), target.ravel(), rng.normal(size=100)])
        spec = ReductionSpec(method="basis_selection", n_components=1)
        first = ReductionState.fit(features, target, spec=spec, train_indices=np.arange(100))
        second = ReductionState.fit(features, target, spec=spec, train_indices=np.arange(100))
        assert first.describe()["selected"] == second.describe()["selected"]


class TestPrincipalComponents:
    """An SVD basis, with its signs pinned."""

    def test_the_output_width_matches_the_request(self, divergent):
        """So the signature the model was built against still holds."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=2),
            train_indices=train,
        )
        assert state.transform_features(features).shape == (400, 2)

    def test_the_components_are_sign_stable(self, divergent):
        """
        An SVD's signs are arbitrary, so they have to be fixed deliberately.

        Two runs whose components differ by a sign produce mirrored features,
        a differently signed set of learned weights, and a diff between two
        saved models that looks like a real change.
        """
        features, target, train = divergent
        spec = ReductionSpec(method="pca", n_components=3)
        first = ReductionState.fit(features, target, spec=spec, train_indices=train)
        second = ReductionState.fit(features, target, spec=spec, train_indices=train)
        assert np.array_equal(
            first.transform_features(features), second.transform_features(features)
        )

    def test_the_components_are_orthogonal(self, divergent):
        """The defining property, so a wrong matrix cannot pass unnoticed."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=3),
            train_indices=train,
        )
        reduced = state.transform_features(features[train])
        correlation = np.corrcoef(reduced, rowvar=False)
        off_diagonal = correlation[~np.eye(3, dtype=bool)]
        assert np.allclose(off_diagonal, 0.0, atol=1e-8)


class TestTargetPassThrough:
    """Reduction touches features, never the target."""

    def test_the_target_inverse_is_the_identity(self, divergent):
        """
        Because this transform never scaled the target in the first place.

        A reduction that applied *some* inverse would corrupt the metrics of
        every run that used it.
        """
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=2),
            train_indices=train,
        )
        assert np.array_equal(state.inverse_transform_targets(target), target)


class TestPersistence:
    """The basis has to be the same basis when the model is served."""

    def test_the_state_round_trips(self, divergent, tmp_path):
        """A different basis at serving time is a different model."""
        features, target, train = divergent
        state = ReductionState.fit(
            features,
            target,
            spec=ReductionSpec(method="pca", n_components=2),
            train_indices=train,
        )
        state.save(tmp_path)
        reloaded = ReductionState.load(tmp_path)
        assert reloaded == state
        assert np.array_equal(
            reloaded.transform_features(features), state.transform_features(features)
        )
