"""Unit tests for rade_ml_pt.ensemble.model -- EnsembleModel."""
import numpy as np
import pytest
import torch

from src.rade_ml_pt.ensemble.model import EnsembleModel
from src.rade_ml_pt.ensemble.router import TradeRouter

from tests.rade_ml_pt.ensemble.conftest import (
    N_SCENARIOS, N_TARGETS_0, N_TARGETS_1, N_TOTAL,
    CLUSTER_0_TRADES, CLUSTER_1_TRADES,
)


@pytest.fixture
def ensemble(member_models, cluster_mapping):
    router = TradeRouter(cluster_mapping)
    return EnsembleModel(
        members=member_models,
        router=router,
        aggregation="concat",
        cluster_trade_indices={
            "cluster_0": [0, 1, 2],
            "cluster_1": [3, 4],
        },
        n_total_targets=N_TOTAL,
    )


class TestEnsembleModelPredict:
    def test_predict_returns_combined_array(self, ensemble, member_inputs):
        result = ensemble.predict(member_inputs)
        assert isinstance(result, np.ndarray)
        assert result.shape == (N_SCENARIOS, N_TOTAL)

    def test_predict_member_returns_numpy(self, ensemble, member_inputs):
        preds = ensemble.predict_member("cluster_0", member_inputs["cluster_0"])
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (N_SCENARIOS, N_TARGETS_0)

    def test_predict_missing_member_skipped(self, ensemble, member_inputs):
        inputs_with_extra = {**member_inputs, "cluster_99": {"features": torch.randn(2, 4)}}
        result = ensemble.predict(inputs_with_extra)
        assert result.shape == (N_SCENARIOS, N_TOTAL)

    def test_predict_no_members_raises(self, ensemble):
        with pytest.raises(RuntimeError, match="No member produced"):
            ensemble.predict({"cluster_99": {"features": torch.randn(2, 4)}})


class TestEnsembleModelMetadata:
    def test_get_member_metadata(self, ensemble):
        meta = ensemble.get_member_metadata()
        assert "cluster_0" in meta
        assert "cluster_1" in meta
        assert meta["cluster_0"]["model_class"] == "SimpleMember"
        assert meta["cluster_0"]["n_trades"] == len(CLUSTER_0_TRADES)
        assert meta["cluster_0"]["n_parameters"] > 0

    def test_member_metadata_param_count(self, ensemble):
        meta = ensemble.get_member_metadata()
        model = ensemble.members["cluster_0"]
        expected = sum(p.numel() for p in model.parameters())
        assert meta["cluster_0"]["n_parameters"] == expected


class TestEnsembleModelEvalMode:
    def test_members_in_eval_mode(self, ensemble):
        for model in ensemble.members.values():
            assert not model.training
