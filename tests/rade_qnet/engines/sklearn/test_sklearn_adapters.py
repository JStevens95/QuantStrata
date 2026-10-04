"""
Tests for the scikit-learn adapters.

The adapters are where the framework's streaming view of data meets a library
that wants all of it at once, and nearly every way that can go wrong is
silent. A column order that differs between fit and predict produces a model
that scores well on its own split and badly on anything else. A flatten that
picks the wrong axis produces a matrix of the right size and the wrong
meaning. Neither raises.

So the tests here are mostly about *order* and *shape*, and they assert on
values rather than on sizes wherever a size would also pass by accident.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rade_qnet.core.runtime.errors import EngineError
from src.rade_qnet.engines.sklearn.adapters import drain, flatten, reject_static
from src.rade_qnet.testkit.fixtures import SyntheticTensorSource


def source(n_samples: int = 20, n_features: int = 3, **kwargs: object) -> SyntheticTensorSource:
    """
    Build a small source with recognisable values.

    Feature ``(i, j)`` holds ``i * 10 + j``, so a row, a column or an
    ordering mistake is visible in the number itself rather than only in a
    shape.

    Parameters
    ----------
    n_samples
        Row count.
    n_features
        Column count.
    **kwargs
        Passed through to the fixture.

    Returns
    -------
    SyntheticTensorSource
        A re-iterable source.
    """
    grid = np.arange(n_samples)[:, None] * 10 + np.arange(n_features)[None, :]
    return SyntheticTensorSource(
        features=grid.astype(np.float64),
        targets=np.arange(n_samples, dtype=np.float64)[:, None],
        batch_size=7,
        **kwargs,  # type: ignore[arg-type]
    )


class TestFlattening:
    """A matrix is what scikit-learn takes, whatever arrived."""

    def test_a_matrix_is_returned_unchanged(self) -> None:
        """Two dimensions already are the required shape."""
        array = np.arange(12, dtype=np.float64).reshape(4, 3)
        assert flatten(array) is array

    def test_a_sequence_window_is_flattened_to_one_row_per_sample(self) -> None:
        """Samples are preserved; everything after them collapses."""
        array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert flatten(array).shape == (2, 12)

    def test_flattening_keeps_each_sample_s_values_together(self) -> None:
        """
        The collapse is within a row, not across rows.

        A transposed flatten gives a matrix of exactly the right shape whose
        rows are mixtures of different samples. Nothing downstream can detect
        that, so it is pinned here on values.
        """
        array = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        assert np.array_equal(flatten(array)[0], np.arange(12, dtype=np.float64))

    def test_a_column_vector_is_a_matrix_already(self) -> None:
        """A single feature is still two-dimensional."""
        assert flatten(np.zeros((5, 1))).shape == (5, 1)


class TestDraining:
    """Every batch, concatenated, in the order the source yielded them."""

    def test_the_row_count_is_the_sample_count(self) -> None:
        """Nothing is dropped at a batch boundary."""
        assert drain(source(n_samples=20)).n_samples == 20

    def test_the_rows_arrive_in_the_source_s_order(self) -> None:
        """
        Order is the contract that lets predictions be compared to targets.

        The final batch here is short -- twenty samples in sevens -- which is
        the case where an adapter that pre-allocates gets this wrong.
        """
        drained = drain(source(n_samples=20))
        assert np.array_equal(drained.features[:, 0], np.arange(20) * 10)

    def test_the_target_is_one_dimensional(self) -> None:
        """
        scikit-learn warns on a column vector and silently changes behaviour.

        A ``(n, 1)`` target makes several estimators treat the problem as
        multi-output, which changes the shape of ``predict`` and breaks the
        comparison against targets downstream.
        """
        assert drain(source()).target is not None
        assert drain(source()).target.ndim == 1  # type: ignore[union-attr]

    def test_features_and_target_stay_aligned(self) -> None:
        """Row ``i`` of the matrix belongs to element ``i`` of the target."""
        drained = drain(source(n_samples=20))
        assert drained.target is not None
        assert np.array_equal(drained.features[:, 0] / 10, drained.target)

    def test_a_source_without_targets_is_allowed_when_not_required(self) -> None:
        """Inference has no target and must still drain."""
        batches = source()
        drained = drain(batches, require_target=False)
        assert drained.n_samples == 20

    def test_feature_names_describe_the_flattened_columns(self) -> None:
        """
        One name per column, so a coefficient can be attributed.

        A single wide input is the common case and the one a coefficient
        table is read for, so it must be named rather than skipped.

        Underscores rather than bracket subscripts: XGBoost refuses feature
        names containing brackets, and both engines share these adapters.
        """
        drained = drain(source(n_features=3))
        assert drained.feature_names == ("features_0", "features_1", "features_2")


class TestRefusals:
    """What the adapter will not quietly accept."""

    def test_static_inputs_are_refused(self) -> None:
        """
        A graph cannot be flattened into a design matrix.

        Dropping it would be the dangerous alternative: the run would
        succeed, the model would be missing an input it was specified with,
        and the only evidence would be a worse metric.
        """
        with pytest.raises(EngineError, match="static"):
            reject_static({"adjacency": np.zeros((3, 3))})

    def test_no_static_inputs_is_not_a_refusal(self) -> None:
        """The common case passes through silently."""
        reject_static({})

    def test_an_empty_source_is_refused(self) -> None:
        """
        Zero rows reaches scikit-learn as an obscure error from inside it.

        Raising here names the actual problem, which is a source or a split
        that produced nothing.
        """
        empty = SyntheticTensorSource(
            features=np.zeros((0, 3)), targets=np.zeros((0, 1))
        )
        with pytest.raises(EngineError, match="no batches"):
            drain(empty)

    def test_a_missing_target_is_refused_when_required(self) -> None:
        """Fitting without a target is a specification error, not a crash."""

        class Untargeted(SyntheticTensorSource):
            """A source whose batches carry inputs only."""

            def batches(self):
                for batch in super().batches():
                    yield {k: v for k, v in batch.items() if k != "target"}

        untargeted = Untargeted(
            features=np.zeros((8, 3)), targets=np.zeros((8, 1))
        )
        with pytest.raises(EngineError, match="target"):
            drain(untargeted)
