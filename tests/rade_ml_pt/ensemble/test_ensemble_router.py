"""Unit tests for rade_ml_pt.ensemble.router -- TradeRouter."""
import numpy as np
import pytest

from src.rade_ml_pt.ensemble.router import TradeRouter


@pytest.fixture
def router(cluster_mapping):
    return TradeRouter(cluster_mapping)


@pytest.fixture
def router_with_default(cluster_mapping):
    return TradeRouter(cluster_mapping, default_cluster="cluster_0")


class TestTradeRouterLookup:
    def test_get_cluster_for_known_trade(self, router):
        assert router.get_cluster_for_trade("trade_A") == "cluster_0"
        assert router.get_cluster_for_trade("trade_D") == "cluster_1"

    def test_get_cluster_for_unknown_raises(self, router):
        with pytest.raises(KeyError, match="not assigned"):
            router.get_cluster_for_trade("unknown_trade")

    def test_get_cluster_for_unknown_with_default(self, router_with_default):
        assert router_with_default.get_cluster_for_trade("unknown") == "cluster_0"


class TestTradeRouterRoute:
    def test_route_partitions_correctly(self, router):
        routed = router.route(["trade_A", "trade_D", "trade_B"])
        assert routed["cluster_0"] == ["trade_A", "trade_B"]
        assert routed["cluster_1"] == ["trade_D"]

    def test_route_empty_list(self, router):
        routed = router.route([])
        assert all(len(v) == 0 for v in routed.values())

    def test_route_single_cluster(self, router):
        routed = router.route(["trade_A", "trade_C"])
        assert "cluster_0" in routed
        assert "cluster_1" not in routed


class TestTradeRouterProperties:
    def test_cluster_ids(self, router):
        assert router.cluster_ids == ["cluster_0", "cluster_1"]

    def test_n_trades(self, router):
        assert router.n_trades == 5

    def test_get_trades_for_cluster(self, router):
        trades = router.get_trades_for_cluster("cluster_0")
        assert set(trades) == {"trade_A", "trade_B", "trade_C"}

    def test_get_trades_for_nonexistent_cluster(self, router):
        trades = router.get_trades_for_cluster("nonexistent")
        assert trades == []


class TestTradeRouterTradeClusterMap:
    def test_to_trade_cluster_map(self, router):
        tcm = router.to_trade_cluster_map()
        assert tcm["trade_A"] == "cluster_0"
        assert tcm["trade_D"] == "cluster_1"
        assert len(tcm) == 5


class TestTradeRouterAssignNewTrade:
    def test_assign_with_centroids(self, router):
        centroids = {
            "cluster_0": np.array([1.0, 0.0]),
            "cluster_1": np.array([0.0, 1.0]),
        }
        cid = router.assign_new_trade(
            {"features": np.array([0.9, 0.1])},
            cluster_centroids=centroids,
        )
        assert cid == "cluster_0"

    def test_assign_with_centroids_picks_nearest(self, router):
        centroids = {
            "cluster_0": np.array([1.0, 0.0]),
            "cluster_1": np.array([0.0, 1.0]),
        }
        cid = router.assign_new_trade(
            {"features": np.array([0.1, 0.9])},
            cluster_centroids=centroids,
        )
        assert cid == "cluster_1"

    def test_assign_without_centroids_uses_default(self, router_with_default):
        cid = router_with_default.assign_new_trade({"features": np.array([1.0])})
        assert cid == "cluster_0"

    def test_assign_without_centroids_or_default_raises(self, router):
        with pytest.raises(ValueError, match="Cannot assign"):
            router.assign_new_trade({"features": np.array([1.0])})
