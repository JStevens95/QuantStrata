"""Unit tests for rade_ml_pt.ensemble.registry -- EnsembleRegistry."""
import json
import pytest

from src.rade_ml_pt.ensemble.config import EnsembleConfig
from src.rade_ml_pt.ensemble.registry import EnsembleRegistry


@pytest.fixture
def ens_registry(tmp_path):
    return EnsembleRegistry(tmp_path / "store")


@pytest.fixture
def sample_config(cluster_mapping):
    return EnsembleConfig(
        cluster_mapping=cluster_mapping,
        aggregation="concat",
        member_configs={
            "cluster_0": {"training_config": {"epochs": 10}},
            "cluster_1": {"training_config": {"epochs": 10}},
        },
    )


@pytest.fixture
def sample_versions():
    return {"cluster_0": "v_abc", "cluster_1": "v_def"}


class TestEnsembleRegistryRegister:
    def test_register_returns_version_string(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        assert isinstance(version, str)
        assert version.startswith("ens_")

    def test_register_creates_directory(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        version_dir = ens_registry.root_dir / version
        assert version_dir.is_dir()

    def test_register_saves_config(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        config_path = ens_registry.root_dir / version / "ensemble_config.json"
        assert config_path.exists()

    def test_register_saves_member_versions(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        mv_path = ens_registry.root_dir / version / "member_versions.json"
        assert mv_path.exists()
        with open(mv_path) as f:
            data = json.load(f)
        assert data == sample_versions

    def test_register_saves_trade_cluster_map(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        tcm_path = ens_registry.root_dir / version / "trade_cluster_map.json"
        assert tcm_path.exists()
        with open(tcm_path) as f:
            tcm = json.load(f)
        assert tcm["trade_A"] == "cluster_0"
        assert tcm["trade_D"] == "cluster_1"

    def test_register_updates_latest(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        assert ens_registry._index["latest"] == version

    def test_register_with_tags(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(
            sample_config, sample_versions, tags=["production"],
        )
        assert ens_registry._index["production"] == version

    def test_register_with_member_summary(self, ens_registry, sample_config, sample_versions):
        summary = {"cluster_0": {"mae": 0.04}, "cluster_1": {"mae": 0.05}}
        version = ens_registry.register(
            sample_config, sample_versions, member_summary=summary,
        )
        ms_path = ens_registry.root_dir / version / "member_summary.json"
        assert ms_path.exists()


class TestEnsembleRegistryLoad:
    def test_load_by_tag(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(
            sample_config, sample_versions, tags=["best"],
        )
        config, mv, resolved = ens_registry.load("best")
        assert resolved == version
        assert isinstance(config, EnsembleConfig)
        assert mv == sample_versions

    def test_load_latest(self, ens_registry, sample_config, sample_versions):
        ens_registry.register(sample_config, sample_versions)
        config, mv, resolved = ens_registry.load("latest")
        assert config.n_members == 2

    def test_load_nonexistent_raises(self, ens_registry):
        with pytest.raises(KeyError, match="not a known tag"):
            ens_registry.load("nonexistent")


class TestEnsembleRegistryTag:
    def test_tag_existing_version(self, ens_registry, sample_config, sample_versions):
        version = ens_registry.register(sample_config, sample_versions)
        ens_registry.tag(version, "staging")
        assert ens_registry._index["staging"] == version

    def test_tag_nonexistent_raises(self, ens_registry):
        with pytest.raises(FileNotFoundError):
            ens_registry.tag("nonexistent", "tag")


class TestEnsembleRegistryList:
    def test_list_versions(self, ens_registry, sample_config, sample_versions):
        ens_registry.register(sample_config, sample_versions)
        versions = ens_registry.list_versions()
        assert len(versions) == 1
        assert versions[0]["n_members"] == 2
        assert versions[0]["n_trades"] == 5

    def test_list_multiple(self, ens_registry, sample_config, sample_versions):
        ens_registry.register(sample_config, sample_versions)
        ens_registry.register(sample_config, sample_versions)
        versions = ens_registry.list_versions()
        assert len(versions) == 2


class TestEnsembleRegistryGetMetadata:
    def test_get_metadata(self, ens_registry, sample_config, sample_versions):
        summary = {"cluster_0": {"mae": 0.04}}
        version = ens_registry.register(
            sample_config, sample_versions, member_summary=summary,
        )
        meta = ens_registry.get_metadata(version)
        assert meta["version"] == version
        assert "trade_cluster_map" in meta
        assert "member_summary" in meta
