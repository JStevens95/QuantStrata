"""
Unit tests for model registry module.

Tests ModelRegistry, ModelVersion, ModelArtifact, and stage transitions.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from src.machine_learning.registry.registry import (
    ModelArtifact,
    ModelRegistry,
    ModelStage,
    ModelVersion,
    RegisteredModel,
    create_registry,
)


class TestModelStage:
    """Tests for ModelStage enum."""
    
    def test_stage_values(self) -> None:
        """Test stage enum values."""
        assert ModelStage.NONE.value == "none"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"
    
    def test_stage_string(self) -> None:
        """Test stage string representation."""
        assert str(ModelStage.PRODUCTION) == "production"


class TestModelVersion:
    """Tests for ModelVersion dataclass."""
    
    def test_version_creation(self) -> None:
        """Test basic version creation."""
        version = ModelVersion(
            version=1,
            created_at=datetime.now(),
            stage=ModelStage.NONE,
            metrics={"mse": 0.001},
            params={"lr": 0.001},
            tags={"model_type": "gnn"},
            description="Test version",
            source_run_id="run_123",
            model_hash="abc123",
            artifact_path="models/v1",
        )
        
        assert version.version == 1
        assert version.stage == ModelStage.NONE
        assert version.metrics["mse"] == 0.001
    
    def test_version_to_dict(self) -> None:
        """Test version serialization."""
        now = datetime.now()
        version = ModelVersion(
            version=1,
            created_at=now,
            stage=ModelStage.STAGING,
            metrics={"mse": 0.001},
            params={"lr": 0.001},
            tags={},
            description="",
            source_run_id=None,
            model_hash="abc123",
            artifact_path="v1",
        )
        
        d = version.to_dict()
        
        assert d["version"] == 1
        assert d["stage"] == "staging"
        assert d["metrics"]["mse"] == 0.001
    
    def test_version_from_dict(self) -> None:
        """Test version deserialization."""
        d = {
            "version": 2,
            "created_at": "2024-01-15T10:00:00",
            "stage": "production",
            "metrics": {"r2": 0.99},
            "params": {},
            "tags": {},
            "description": "Test",
            "source_run_id": None,
            "model_hash": "def456",
            "artifact_path": "v2",
        }
        
        version = ModelVersion.from_dict(d)
        
        assert version.version == 2
        assert version.stage == ModelStage.PRODUCTION


class TestRegisteredModel:
    """Tests for RegisteredModel dataclass."""
    
    def test_model_creation(self) -> None:
        """Test registered model creation."""
        model = RegisteredModel(
            name="test_model",
            description="Test model",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert model.name == "test_model"
        assert model.versions == {}
    
    def test_latest_version_empty(self) -> None:
        """Test latest version when no versions exist."""
        model = RegisteredModel(
            name="test",
            description="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        assert model.latest_version is None
    
    def test_latest_version(self) -> None:
        """Test getting latest version."""
        model = RegisteredModel(
            name="test",
            description="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        v1 = ModelVersion(
            version=1, created_at=datetime.now(), stage=ModelStage.NONE,
            metrics={}, params={}, tags={}, description="",
            source_run_id=None, model_hash="a", artifact_path="v1",
        )
        v2 = ModelVersion(
            version=2, created_at=datetime.now(), stage=ModelStage.NONE,
            metrics={}, params={}, tags={}, description="",
            source_run_id=None, model_hash="b", artifact_path="v2",
        )
        
        model.versions = {1: v1, 2: v2}
        
        assert model.latest_version.version == 2
    
    def test_get_version_by_stage(self) -> None:
        """Test getting version by stage."""
        model = RegisteredModel(
            name="test",
            description="",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        
        v1 = ModelVersion(
            version=1, created_at=datetime.now(), stage=ModelStage.ARCHIVED,
            metrics={}, params={}, tags={}, description="",
            source_run_id=None, model_hash="a", artifact_path="v1",
        )
        v2 = ModelVersion(
            version=2, created_at=datetime.now(), stage=ModelStage.PRODUCTION,
            metrics={}, params={}, tags={}, description="",
            source_run_id=None, model_hash="b", artifact_path="v2",
        )
        
        model.versions = {1: v1, 2: v2}
        
        prod = model.get_version_by_stage(ModelStage.PRODUCTION)
        assert prod is not None
        assert prod.version == 2


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_registry_creation(self) -> None:
        """Test registry initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            assert registry._storage_path == Path(tmpdir)
            assert registry._models == {}
    
    def test_register_model(self) -> None:
        """Test registering a new model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            # Create a simple model artifact
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test model data")
            
            version = registry.register_model(
                name="test_pricer",
                model_path=model_path,
                metrics={"mse": 0.001},
                params={"lr": 0.001},
                description="Test model",
            )
            
            assert version.version == 1
            assert version.metrics["mse"] == 0.001
            assert "test_pricer" in registry.list_models()
    
    def test_register_multiple_versions(self) -> None:
        """Test registering multiple versions of same model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("v1")
            
            v1 = registry.register_model("model", model_path, metrics={"mse": 0.5})
            
            (model_path / "model.txt").write_text("v2")
            v2 = registry.register_model("model", model_path, metrics={"mse": 0.3})
            
            assert v1.version == 1
            assert v2.version == 2
            
            versions = registry.list_versions("model")
            assert len(versions) == 2
    
    def test_register_model_file_not_found(self) -> None:
        """Test that registering non-existent model raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            with pytest.raises(FileNotFoundError):
                registry.register_model("model", "/nonexistent/path")
    
    def test_get_model_by_version(self) -> None:
        """Test getting model by version number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            registry.register_model("model", model_path)
            registry.register_model("model", model_path)
            
            artifact = registry.get_model("model", version=1)
            
            assert artifact.version.version == 1
    
    def test_get_model_by_stage(self) -> None:
        """Test getting model by stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            v1 = registry.register_model("model", model_path)
            registry.transition_stage("model", v1.version, ModelStage.PRODUCTION)
            
            artifact = registry.get_model("model", stage=ModelStage.PRODUCTION)
            
            assert artifact.version.version == 1
            assert artifact.version.stage == ModelStage.PRODUCTION
    
    def test_get_model_not_found(self) -> None:
        """Test getting non-existent model raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            with pytest.raises(KeyError, match="Model not found"):
                registry.get_model("nonexistent")
    
    def test_transition_stage(self) -> None:
        """Test transitioning model stage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            v = registry.register_model("model", model_path)
            
            assert v.stage == ModelStage.NONE
            
            registry.transition_stage("model", v.version, ModelStage.STAGING)
            
            artifact = registry.get_model("model", version=v.version)
            assert artifact.version.stage == ModelStage.STAGING
    
    def test_transition_to_production_archives_existing(self) -> None:
        """Test that transitioning to production archives existing production."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            v1 = registry.register_model("model", model_path)
            registry.transition_stage("model", v1.version, ModelStage.PRODUCTION)
            
            v2 = registry.register_model("model", model_path)
            registry.transition_stage("model", v2.version, ModelStage.PRODUCTION)
            
            # v1 should now be archived
            artifact_v1 = registry.get_model("model", version=1)
            assert artifact_v1.version.stage == ModelStage.ARCHIVED
    
    def test_delete_version(self) -> None:
        """Test deleting a model version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            registry.register_model("model", model_path)
            registry.register_model("model", model_path)
            
            registry.delete_version("model", 1)
            
            versions = registry.list_versions("model")
            assert len(versions) == 1
            assert versions[0].version == 2
    
    def test_update_tags(self) -> None:
        """Test updating tags on a version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            v = registry.register_model("model", model_path)
            
            registry.update_tags("model", v.version, {"env": "prod", "owner": "team_a"})
            
            artifact = registry.get_model("model", version=v.version)
            assert artifact.version.tags["env"] == "prod"
            assert artifact.version.tags["owner"] == "team_a"
    
    def test_search_models(self) -> None:
        """Test searching models."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            registry.register_model(
                "gnn_pricer",
                model_path,
                tags={"asset_class": "fx"},
            )
            registry.register_model(
                "lstm_pricer",
                model_path,
                tags={"asset_class": "equity"},
            )
            
            # Search by name
            results = registry.search_models(name_contains="gnn")
            assert len(results) == 1
            assert results[0].name == "gnn_pricer"
            
            # Search by tags
            results = registry.search_models(tags={"asset_class": "fx"})
            assert len(results) == 1
    
    def test_get_model_info(self) -> None:
        """Test getting model information."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = ModelRegistry(tmpdir)
            
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            registry.register_model("model", model_path)
            registry.register_model("model", model_path)
            
            info = registry.get_model_info("model")
            
            assert info["name"] == "model"
            assert info["n_versions"] == 2
            assert info["latest_version"] == 2
    
    def test_persistence(self) -> None:
        """Test that registry persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "source_model"
            model_path.mkdir()
            (model_path / "model.txt").write_text("test")
            
            # Create and register
            registry1 = ModelRegistry(tmpdir)
            registry1.register_model("model", model_path, metrics={"mse": 0.1})
            
            # Create new instance
            registry2 = ModelRegistry(tmpdir)
            
            # Should find the model
            assert "model" in registry2.list_models()
            artifact = registry2.get_model("model")
            assert artifact.version.metrics["mse"] == 0.1


class TestModelArtifact:
    """Tests for ModelArtifact."""
    
    def test_artifact_dir(self) -> None:
        """Test artifact directory property."""
        version = ModelVersion(
            version=1,
            created_at=datetime.now(),
            stage=ModelStage.NONE,
            metrics={},
            params={},
            tags={},
            description="",
            source_run_id=None,
            model_hash="abc",
            artifact_path="model/v1",
        )
        
        artifact = ModelArtifact(
            name="model",
            version=version,
            registry_path=Path("/registry"),
        )
        
        assert artifact.artifact_dir == Path("/registry/model/v1")


class TestCreateRegistry:
    """Tests for create_registry factory function."""
    
    def test_create_registry(self) -> None:
        """Test creating registry with factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = create_registry(tmpdir)
            
            assert isinstance(registry, ModelRegistry)
