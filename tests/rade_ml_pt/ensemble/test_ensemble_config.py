"""Unit tests for rade_ml_pt.ensemble.config -- EnsembleConfig."""
import json
import pytest

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.pipelines.config import PipelineConfig


class TestEnsembleConfigProperties:
    def test_cluster_ids_sorted(self, ensemble_config):
        assert ensemble_config.cluster_ids == ["cluster_0", "cluster_1"]

    def test_n_members(self, ensemble_config):
        assert ensemble_config.n_members == 2

    def test_all_trade_ids(self, ensemble_config):
        ids = ensemble_config.all_trade_ids
        assert len(ids) == 5
        assert "trade_A" in ids
        assert "trade_E" in ids

    def test_all_trade_ids_ordered_by_cluster(self, ensemble_config):
        ids = ensemble_config.all_trade_ids
        # cluster_0 trades come before cluster_1 (sorted cluster keys)
        assert ids.index("trade_A") < ids.index("trade_D")


class TestEnsembleConfigMemberPipelineConfig:
    def test_returns_pipeline_config(self, ensemble_config):
        pc = ensemble_config.get_member_pipeline_config("cluster_0")
        assert isinstance(pc, PipelineConfig)

    def test_injects_cluster_id(self, ensemble_config):
        pc = ensemble_config.get_member_pipeline_config("cluster_0")
        assert pc.metadata["cluster_id"] == "cluster_0"

    def test_injects_trade_ids(self, ensemble_config):
        pc = ensemble_config.get_member_pipeline_config("cluster_0")
        assert pc.metadata["trade_ids"] == ["trade_A", "trade_B", "trade_C"]

    def test_forwards_training_config(self, ensemble_config):
        pc = ensemble_config.get_member_pipeline_config("cluster_0")
        assert pc.training_config["epochs"] == 2

    def test_missing_cluster_returns_empty_trades(self, ensemble_config):
        pc = ensemble_config.get_member_pipeline_config("nonexistent")
        assert pc.metadata["trade_ids"] == []


class TestEnsembleConfigSerialisation:
    def test_to_dict_roundtrip(self, ensemble_config):
        d = ensemble_config.to_dict()
        restored = EnsembleConfig.from_dict(d)
        assert restored.cluster_ids == ensemble_config.cluster_ids
        assert restored.aggregation == "concat"
        assert restored.n_members == 2

    def test_to_json_roundtrip(self, ensemble_config, tmp_path):
        path = tmp_path / "config.json"
        ensemble_config.to_json(path)
        restored = EnsembleConfig.from_json(path)
        assert restored.cluster_ids == ensemble_config.cluster_ids
        assert restored.all_trade_ids == ensemble_config.all_trade_ids

    def test_json_is_valid(self, ensemble_config, tmp_path):
        path = tmp_path / "config.json"
        ensemble_config.to_json(path)
        with open(path) as f:
            data = json.load(f)
        assert "cluster_mapping" in data
        assert "aggregation" in data


class TestEnsembleConfigDefaults:
    def test_default_aggregation(self):
        config = EnsembleConfig()
        assert config.aggregation == "concat"

    def test_default_weights_none(self):
        config = EnsembleConfig()
        assert config.weights is None

    def test_empty_cluster_mapping(self):
        config = EnsembleConfig()
        assert config.n_members == 0
        assert config.all_trade_ids == []
