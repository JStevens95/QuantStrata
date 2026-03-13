"""Unit tests for rade_ml_pt.ensemble.aggregation."""
import numpy as np
import pytest

from src.rade_ml_pt.ensemble.aggregation import (
    concat_aggregate,
    weighted_mean_aggregate,
    get_aggregation_fn,
)


class TestConcatAggregate:
    def test_places_predictions_at_correct_columns(self):
        preds = {
            "c0": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "c1": np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]]),
        }
        indices = {"c0": [0, 1], "c1": [2, 3, 4]}
        result = concat_aggregate(preds, indices, n_total_targets=5)
        assert result.shape == (2, 5)
        np.testing.assert_array_equal(result[:, 0], [1.0, 3.0])
        np.testing.assert_array_equal(result[:, 4], [7.0, 10.0])

    def test_non_contiguous_indices(self):
        preds = {
            "c0": np.array([[10.0], [20.0]]),
            "c1": np.array([[30.0], [40.0]]),
        }
        indices = {"c0": [2], "c1": [0]}
        result = concat_aggregate(preds, indices, n_total_targets=3)
        assert result[0, 2] == 10.0
        assert result[0, 0] == 30.0
        assert result[0, 1] == 0.0  # unfilled

    def test_mismatched_indices_raises(self):
        preds = {"c0": np.array([[1.0, 2.0]])}
        indices = {"c0": [0]}  # 1 index but 2 columns
        with pytest.raises(ValueError, match="index count"):
            concat_aggregate(preds, indices, n_total_targets=2)

    def test_infers_n_scenarios(self):
        preds = {"c0": np.array([[1.0], [2.0], [3.0]])}
        indices = {"c0": [0]}
        result = concat_aggregate(preds, indices, n_total_targets=1)
        assert result.shape == (3, 1)


class TestWeightedMeanAggregate:
    def test_equal_weights(self):
        preds = {
            "c0": np.array([[2.0, 4.0]]),
            "c1": np.array([[6.0, 8.0]]),
        }
        result = weighted_mean_aggregate(preds, weights={"c0": 1.0, "c1": 1.0})
        np.testing.assert_array_almost_equal(result, [[4.0, 6.0]])

    def test_unequal_weights(self):
        preds = {
            "c0": np.array([[0.0]]),
            "c1": np.array([[10.0]]),
        }
        result = weighted_mean_aggregate(preds, weights={"c0": 0.25, "c1": 0.75})
        np.testing.assert_array_almost_equal(result, [[7.5]])

    def test_missing_weight_defaults_to_one(self):
        preds = {
            "c0": np.array([[2.0]]),
            "c1": np.array([[4.0]]),
        }
        result = weighted_mean_aggregate(preds, weights={"c0": 1.0})
        # c1 gets weight 1.0 by default
        np.testing.assert_array_almost_equal(result, [[3.0]])

    def test_zero_total_weight_raises(self):
        preds = {"c0": np.array([[1.0]])}
        with pytest.raises(ValueError, match="Total weight is zero"):
            weighted_mean_aggregate(preds, weights={"c0": 0.0})

    def test_empty_preds_raises(self):
        with pytest.raises(ValueError, match="No member predictions"):
            weighted_mean_aggregate({}, weights={})


class TestGetAggregationFn:
    def test_concat(self):
        fn = get_aggregation_fn("concat")
        assert fn is concat_aggregate

    def test_weighted_mean(self):
        fn = get_aggregation_fn("weighted_mean")
        assert fn is weighted_mean_aggregate

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown aggregation"):
            get_aggregation_fn("stacking")
