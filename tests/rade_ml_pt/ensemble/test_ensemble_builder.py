"""Unit tests for rade_ml_pt.ensemble.builder -- EnsembleBuilder."""
import pytest
import torch.nn as nn

from src.rade_ml_pt.ensemble.builder import EnsembleBuilder
from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.model import EnsembleModel
from src.rade_ml_pt.core.types import TrainingResult
from src.rade_ml_pt.registry.store import ModelRegistry

from tests.rade_ml_pt.ensemble.conftest import SimpleMember


def _register_members(registry, cluster_mapping):
    """Register a SimpleMember for each cluster, return member_versions dict."""
    versions = {}
    for cid, trades in cluster_mapping.items():
        model = SimpleMember(input_dim=4, n_targets=len(trades))
        result = TrainingResult(
            best_val_loss=0.05,
            best_train_loss=0.03,
            final_epoch=10,
            best_epoch=8,
        )
        entry = registry.register(model, result, tags=[f"{cid}_latest"])
        versions[cid] = entry.version
    return versions


class TestEnsembleBuilderBuild:
    def test_build_returns_ensemble_model(self, tmp_path, cluster_mapping):
        registry = ModelRegistry(tmp_path / "reg")
        versions = _register_members(registry, cluster_mapping)

        config = EnsembleConfig(
            cluster_mapping=cluster_mapping,
            aggregation="concat",
        )
        builder = EnsembleBuilder(registry)
        ensemble = builder.build(config, versions)

        assert isinstance(ensemble, EnsembleModel)
        assert len(ensemble.members) == 2

    def test_build_validates_coverage(self, tmp_path, cluster_mapping):
        registry = ModelRegistry(tmp_path / "reg")
        versions = _register_members(registry, cluster_mapping)

        config = EnsembleConfig(
            cluster_mapping=cluster_mapping,
            aggregation="concat",
        )
        builder = EnsembleBuilder(registry)
        # Should succeed without error (disjoint clusters).
        ensemble = builder.build(config, versions)
        assert ensemble.n_total_targets == 5

    def test_build_members_in_eval_mode(self, tmp_path, cluster_mapping):
        registry = ModelRegistry(tmp_path / "reg")
        versions = _register_members(registry, cluster_mapping)

        config = EnsembleConfig(cluster_mapping=cluster_mapping)
        builder = EnsembleBuilder(registry)
        ensemble = builder.build(config, versions)

        for model in ensemble.members.values():
            assert not model.training


class TestEnsembleBuilderValidation:
    def test_overlapping_trades_raises_for_concat(self, tmp_path):
        overlap_mapping = {
            "c0": ["trade_A", "trade_B"],
            "c1": ["trade_B", "trade_C"],
        }
        registry = ModelRegistry(tmp_path / "reg")
        versions = _register_members(registry, overlap_mapping)

        config = EnsembleConfig(
            cluster_mapping=overlap_mapping,
            aggregation="concat",
        )
        builder = EnsembleBuilder(registry)
        with pytest.raises(ValueError, match="disjoint"):
            builder.build(config, versions)

    def test_overlapping_trades_warns_for_weighted_mean(self, tmp_path):
        overlap_mapping = {
            "c0": ["trade_A", "trade_B"],
            "c1": ["trade_B", "trade_C"],
        }
        registry = ModelRegistry(tmp_path / "reg")
        versions = _register_members(registry, overlap_mapping)

        config = EnsembleConfig(
            cluster_mapping=overlap_mapping,
            aggregation="weighted_mean",
        )
        builder = EnsembleBuilder(registry)
        # Should succeed with a warning (not raise).
        ensemble = builder.build(config, versions)
        assert len(ensemble.members) == 2

    def test_empty_mapping_raises(self, tmp_path):
        registry = ModelRegistry(tmp_path / "reg")
        config = EnsembleConfig(cluster_mapping={})
        builder = EnsembleBuilder(registry)
        with pytest.raises(ValueError, match="No trades found"):
            builder.build(config, {})

    def test_missing_member_version_raises(self, tmp_path, cluster_mapping):
        registry = ModelRegistry(tmp_path / "reg")
        config = EnsembleConfig(cluster_mapping=cluster_mapping)
        builder = EnsembleBuilder(registry)
        with pytest.raises(RuntimeError, match="Failed to load"):
            builder.build(config, {"cluster_0": "nonexistent"})


class TestEnsembleBuilderIndices:
    def test_cluster_trade_indices(self, cluster_mapping):
        config = EnsembleConfig(cluster_mapping=cluster_mapping)
        indices = EnsembleBuilder._build_cluster_trade_indices(config)

        all_ids = config.all_trade_ids
        assert len(indices["cluster_0"]) == 3
        assert len(indices["cluster_1"]) == 2

        # Indices should be valid positions in the combined array.
        all_indices = indices["cluster_0"] + indices["cluster_1"]
        assert sorted(all_indices) == list(range(len(all_ids)))
