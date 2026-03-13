"""Unit tests for rade_ml_pt.ensemble.metrics."""
import numpy as np
import pytest

from src.rade_ml_pt.ensemble.metrics import (
    compute_ensemble_metrics,
    compute_per_member_metrics,
    aggregate_member_metrics,
    build_version_comparison,
    build_trade_to_cluster_mapping,
)


class TestComputeEnsembleMetrics:
    def test_perfect_predictions(self):
        preds = np.array([[1.0, 2.0], [3.0, 4.0]])
        targets = np.array([[1.0, 2.0], [3.0, 4.0]])
        metrics = compute_ensemble_metrics(preds, targets)
        assert metrics["mae"] == 0.0
        assert metrics["mse"] == 0.0
        assert metrics["rmse"] == 0.0

    def test_known_error(self):
        preds = np.array([[1.0]])
        targets = np.array([[0.0]])
        metrics = compute_ensemble_metrics(preds, targets)
        assert metrics["mae"] == 1.0
        assert metrics["mse"] == 1.0
        assert metrics["rmse"] == 1.0
        assert metrics["max_ae"] == 1.0

    def test_percentiles(self):
        np.random.seed(0)
        preds = np.random.randn(100, 5).astype(np.float32)
        targets = np.zeros_like(preds)
        metrics = compute_ensemble_metrics(preds, targets)
        assert metrics["p95_ae"] > metrics["mae"]
        assert metrics["p99_ae"] >= metrics["p95_ae"]

    def test_returns_all_keys(self):
        preds = np.array([[1.0, 2.0]])
        targets = np.array([[1.5, 2.5]])
        metrics = compute_ensemble_metrics(preds, targets)
        expected_keys = {"mae", "mse", "rmse", "max_ae", "p95_ae", "p99_ae"}
        assert set(metrics.keys()) == expected_keys


class TestComputePerMemberMetrics:
    def test_basic(self, member_predictions, member_targets):
        results = compute_per_member_metrics(member_predictions, member_targets)
        assert "cluster_0" in results
        assert "cluster_1" in results
        assert "mae" in results["cluster_0"]
        assert results["cluster_0"]["n_targets"] == 3
        assert results["cluster_1"]["n_targets"] == 2

    def test_missing_targets_skipped(self, member_predictions):
        results = compute_per_member_metrics(
            member_predictions, {"cluster_0": member_predictions["cluster_0"]},
        )
        assert "cluster_0" in results
        assert "cluster_1" not in results


class TestAggregateMemberMetrics:
    def test_rollup(self):
        per_member = {
            "c0": {"mae": 0.04, "mse": 0.002},
            "c1": {"mae": 0.06, "mse": 0.004},
        }
        rollup = aggregate_member_metrics(per_member)
        assert rollup["mean_mae"] == pytest.approx(0.05)
        assert rollup["min_mae"] == pytest.approx(0.04)
        assert rollup["max_mae"] == pytest.approx(0.06)
        assert "per_member" in rollup

    def test_empty_input(self):
        assert aggregate_member_metrics({}) == {}


class TestBuildVersionComparison:
    def test_comparison(self):
        a = {"mae": 0.05, "mse": 0.003}
        b = {"mae": 0.04, "mse": 0.002}
        comp = build_version_comparison(a, b, "v1", "v2")
        assert comp["mae"]["v1"] == 0.05
        assert comp["mae"]["v2"] == 0.04
        assert comp["mae"]["delta"] == pytest.approx(-0.01)
        assert comp["mae"]["improved"] is True

    def test_worsened_metric(self):
        a = {"mae": 0.03}
        b = {"mae": 0.05}
        comp = build_version_comparison(a, b)
        assert comp["mae"]["improved"] is False

    def test_missing_metric_in_one_version(self):
        a = {"mae": 0.03}
        b = {"rmse": 0.05}
        comp = build_version_comparison(a, b)
        assert comp["mae"]["B"] is None
        assert comp["rmse"]["A"] is None


class TestBuildTradeToClusterMapping:
    def test_mapping(self, cluster_mapping):
        tcm = build_trade_to_cluster_mapping(cluster_mapping)
        assert tcm["trade_A"] == "cluster_0"
        assert tcm["trade_D"] == "cluster_1"
        assert len(tcm) == 5
