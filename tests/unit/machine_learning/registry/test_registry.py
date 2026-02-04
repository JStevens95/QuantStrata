"""
Unit tests for model registry module.

Tests ModelRegistry, ModelArtifact, ModelVersion, and related functionality.
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Add project root to path for direct import
sys.path.insert(0, str(Path(__file__).parents[4]))

# Import directly from module file to avoid triggering tensorflow
from src.machine_learning.registry.registry import (
    ModelRegistry,
    ModelArtifact,
    ModelVersion,
    ModelStage,
    create_registry,
)


class TestModelStage:
    """Tests for ModelStage enum."""
    
    def test_stage_values(self):
        """Test stage enum values."""
        assert ModelStage.NONE.value == "none"
        assert ModelStage.STAGING.value == "staging"
        assert ModelStage.PRODUCTION.value == "production"
        assert ModelStage.ARCHIVED.value == "archived"
    
    def test_stage_str(self):
        """Test stage string representation."""
        assert str(ModelStage.PRODUCTION) == "production"


class TestModelVersion:
    """Tests for ModelVersion dataclass."""
    
    def test_version_creation(self):
        """Test creating model version."""
        version = ModelVersion(
            version=1,
            created_at=datetime.now(),
            stage=ModelStage.NONE,
            metrics={"mse": 0.001, "mae": 0.02},
            params={"lr": 0.001, "epochs": 100},
            tags={"asset_class": "fx"},
            description="Initial version",
            source_run_id="run_123",
            model_hash="abc123",
            artifact_path="model_v1/v1",
        )
        
        assert version.version == 1
        assert version.stage == ModelStage.NONE
        assert version.metrics["mse"] == 0.001
        assert version.params["lr"] == 0.001
        assert version.tags["asset_class"] == "fx"
    
    def test_version_to_dict(self):
        """Test converting version to dictionary."""
        now = datetime.now()
        version = ModelVersion(
            version=1,
            created_at=now,
            stage=ModelStage.STAGING,
            metrics={"mse": 0.001},
            params={"lr": 0.001},
            tags={},
            description="Test",
            source_run_id=None,
            model_hash="abc123",
            artifact_path="test/v1",
        )
        
        d = version.to_dict()
        
        assert d["version"] == 1
        assert d["stage"] == "staging"
        assert d["metrics"] == {"mse": 0.001}
        assert d["created_at"] == now.isoformat()
    
    def test_version_from_dict(self):
        """Test creating version from dictionary."""
        d = {
            "version": 2,
            "created_at": "2024-01-15T10:30:00",
            "stage": "production",
            "metrics": {"mse": 0.001},
            "params": {"lr": 0.001},
            "tags": {"env": "prod"},
            "description": "Production model",
            "source_run_id": "run_456",
            "model_hash": "def456",
            "artifact_path": "model/v2",
        }
        
        version = ModelVersion.from_dict(d)
        
        assert version.version == 2
        assert version.stage == ModelStage.PRODUCTION
        assert version.source_run_id == "run_456"


class TestModelRegistry:
    """Tests for ModelRegistry class."""
    
    @pytest.fixture
    def registry(self):
        """Create a temporary registry for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield ModelRegistry(tmpdir)
    
    @pytest.fixture
    def sample_model_dir(self):
        """Create a sample model directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "sample_model"
            model_dir.mkdir()
            
            # Create a simple model file
            (model_dir / "model.json").write_text('{"architecture": "mlp"}')
            (model_dir / "weights.h5").write_bytes(b"fake weights")
            
            yield model_dir
    
    def test_registry_initialization(self, registry):
        """Test registry initialization."""
        assert len(registry.list_models()) == 0
        assert registry._storage_path.exists()
    
    def test_register_model(self, registry, sample_model_dir):
        """Test registering a model."""
        version = registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.001},
            params={"lr": 0.001},
            tags={"env": "test"},
            description="Test model",
        )
        
        assert version.version == 1
        assert version.metrics == {"mse": 0.001}
        assert version.params == {"lr": 0.001}
        assert version.stage == ModelStage.NONE
        assert "test_model" in registry.list_models()
    
    def test_register_multiple_versions(self, registry, sample_model_dir):
        """Test registering multiple versions of same model."""
        v1 = registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.1},
        )
        
        v2 = registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.05},
        )
        
        v3 = registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.01},
        )
        
        assert v1.version == 1
        assert v2.version == 2
        assert v3.version == 3
        
        versions = registry.list_versions("test_model")
        assert len(versions) == 3
    
    def test_get_model_by_version(self, registry, sample_model_dir):
        """Test getting model by specific version."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.1},
        )
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.05},
        )
        
        artifact = registry.get_model("test_model", version=1)
        
        assert artifact.name == "test_model"
        assert artifact.version.version == 1
        assert artifact.version.metrics["mse"] == 0.1
    
    def test_get_model_latest(self, registry, sample_model_dir):
        """Test getting latest model version."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.1},
        )
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.05},
        )
        
        artifact = registry.get_model("test_model")  # No version specified
        
        assert artifact.version.version == 2
        assert artifact.version.metrics["mse"] == 0.05
    
    def test_get_model_by_stage(self, registry, sample_model_dir):
        """Test getting model by stage."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.1},
        )
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            metrics={"mse": 0.05},
        )
        
        # Promote version 1 to production
        registry.transition_stage(
            name="test_model",
            version=1,
            stage=ModelStage.PRODUCTION,
        )
        
        artifact = registry.get_model("test_model", stage=ModelStage.PRODUCTION)
        
        assert artifact.version.version == 1
        assert artifact.version.stage == ModelStage.PRODUCTION
    
    def test_transition_stage(self, registry, sample_model_dir):
        """Test transitioning model stage."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        
        # None -> Staging
        registry.transition_stage(
            name="test_model",
            version=1,
            stage=ModelStage.STAGING,
        )
        
        artifact = registry.get_model("test_model", version=1)
        assert artifact.version.stage == ModelStage.STAGING
        
        # Staging -> Production
        registry.transition_stage(
            name="test_model",
            version=1,
            stage=ModelStage.PRODUCTION,
        )
        
        artifact = registry.get_model("test_model", version=1)
        assert artifact.version.stage == ModelStage.PRODUCTION
    
    def test_transition_archives_existing_production(self, registry, sample_model_dir):
        """Test that transitioning to production archives existing production version."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        
        # Make v1 production
        registry.transition_stage("test_model", 1, ModelStage.PRODUCTION)
        
        # Make v2 production (should archive v1)
        registry.transition_stage("test_model", 2, ModelStage.PRODUCTION)
        
        v1 = registry.get_model("test_model", version=1)
        v2 = registry.get_model("test_model", version=2)
        
        assert v1.version.stage == ModelStage.ARCHIVED
        assert v2.version.stage == ModelStage.PRODUCTION
    
    def test_list_versions_by_stage(self, registry, sample_model_dir):
        """Test listing versions filtered by stage."""
        for _ in range(3):
            registry.register_model(
                name="test_model",
                model_path=sample_model_dir,
            )
        
        registry.transition_stage("test_model", 1, ModelStage.ARCHIVED)
        registry.transition_stage("test_model", 2, ModelStage.STAGING)
        registry.transition_stage("test_model", 3, ModelStage.PRODUCTION)
        
        archived = registry.list_versions("test_model", stage=ModelStage.ARCHIVED)
        staging = registry.list_versions("test_model", stage=ModelStage.STAGING)
        production = registry.list_versions("test_model", stage=ModelStage.PRODUCTION)
        
        assert len(archived) == 1
        assert len(staging) == 1
        assert len(production) == 1
    
    def test_delete_version(self, registry, sample_model_dir):
        """Test deleting a model version."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        
        registry.delete_version("test_model", 1)
        
        versions = registry.list_versions("test_model")
        assert len(versions) == 1
        assert versions[0].version == 2
        
        with pytest.raises(KeyError):
            registry.get_model("test_model", version=1)
    
    def test_update_tags(self, registry, sample_model_dir):
        """Test updating version tags."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
            tags={"env": "dev"},
        )
        
        registry.update_tags("test_model", 1, {"env": "prod", "validated": "true"})
        
        artifact = registry.get_model("test_model", version=1)
        assert artifact.version.tags["env"] == "prod"
        assert artifact.version.tags["validated"] == "true"
    
    def test_search_models_by_name(self, registry, sample_model_dir):
        """Test searching models by name."""
        registry.register_model(
            name="fx_pricer",
            model_path=sample_model_dir,
        )
        registry.register_model(
            name="equity_pricer",
            model_path=sample_model_dir,
        )
        registry.register_model(
            name="fx_calibrator",
            model_path=sample_model_dir,
        )
        
        results = registry.search_models(name_contains="fx")
        
        assert len(results) == 2
        names = [r.name for r in results]
        assert "fx_pricer" in names
        assert "fx_calibrator" in names
    
    def test_search_models_by_tags(self, registry, sample_model_dir):
        """Test searching models by tags."""
        registry.register_model(
            name="model1",
            model_path=sample_model_dir,
            tags={"asset_class": "fx", "type": "pricer"},
        )
        registry.register_model(
            name="model2",
            model_path=sample_model_dir,
            tags={"asset_class": "equity", "type": "pricer"},
        )
        registry.register_model(
            name="model3",
            model_path=sample_model_dir,
            tags={"asset_class": "fx", "type": "calibrator"},
        )
        
        results = registry.search_models(tags={"asset_class": "fx"})
        
        assert len(results) == 2
    
    def test_search_models_by_metric(self, registry, sample_model_dir):
        """Test searching models by metric filter."""
        registry.register_model(
            name="model1",
            model_path=sample_model_dir,
            metrics={"mse": 0.1},
        )
        registry.register_model(
            name="model2",
            model_path=sample_model_dir,
            metrics={"mse": 0.01},
        )
        registry.register_model(
            name="model3",
            model_path=sample_model_dir,
            metrics={"mse": 0.05},
        )
        
        # Find models with mse < 0.05
        results = registry.search_models(
            metric_filter=lambda m: m.get("mse", float("inf")) < 0.05
        )
        
        assert len(results) == 1
        assert results[0].name == "model2"
    
    def test_get_model_info(self, registry, sample_model_dir):
        """Test getting model info."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        registry.transition_stage("test_model", 2, ModelStage.PRODUCTION)
        
        info = registry.get_model_info("test_model")
        
        assert info["name"] == "test_model"
        assert info["n_versions"] == 2
        assert info["latest_version"] == 2
        assert info["production_version"] == 2
    
    def test_model_not_found(self, registry):
        """Test error when model not found."""
        with pytest.raises(KeyError, match="Model not found"):
            registry.get_model("nonexistent")
        
        with pytest.raises(KeyError, match="Model not found"):
            registry.list_versions("nonexistent")
    
    def test_version_not_found(self, registry, sample_model_dir):
        """Test error when version not found."""
        registry.register_model(
            name="test_model",
            model_path=sample_model_dir,
        )
        
        with pytest.raises(KeyError, match="Version 99 not found"):
            registry.get_model("test_model", version=99)
    
    def test_model_path_not_found(self, registry):
        """Test error when model path doesn't exist."""
        with pytest.raises(FileNotFoundError):
            registry.register_model(
                name="test_model",
                model_path="/nonexistent/path",
            )
    
    def test_persistence(self, sample_model_dir):
        """Test that registry persists across instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create registry and register model
            registry1 = ModelRegistry(tmpdir)
            registry1.register_model(
                name="test_model",
                model_path=sample_model_dir,
                metrics={"mse": 0.001},
            )
            registry1.transition_stage("test_model", 1, ModelStage.PRODUCTION)
            
            # Create new registry instance pointing to same storage
            registry2 = ModelRegistry(tmpdir)
            
            assert "test_model" in registry2.list_models()
            artifact = registry2.get_model("test_model", stage=ModelStage.PRODUCTION)
            assert artifact.version.metrics["mse"] == 0.001


class TestModelArtifact:
    """Tests for ModelArtifact class."""
    
    @pytest.fixture
    def artifact_with_model(self):
        """Create an artifact with a loadable model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model files
            model_dir = Path(tmpdir) / "models" / "test/v1"
            model_dir.mkdir(parents=True)
            
            # Create a simple model.json + weights
            (model_dir / "model.json").write_text('{}')
            
            version = ModelVersion(
                version=1,
                created_at=datetime.now(),
                stage=ModelStage.NONE,
                metrics={},
                params={},
                tags={},
                description="",
                source_run_id=None,
                model_hash="abc123",
                artifact_path="test/v1",
            )
            
            artifact = ModelArtifact(
                name="test_model",
                version=version,
                registry_path=Path(tmpdir) / "models",
            )
            
            yield artifact
    
    def test_artifact_dir(self, artifact_with_model):
        """Test getting artifact directory."""
        artifact_dir = artifact_with_model.artifact_dir
        assert artifact_dir.exists()


class TestCreateRegistry:
    """Tests for create_registry factory function."""
    
    def test_create_registry(self):
        """Test creating registry with factory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = create_registry(tmpdir)
            
            assert isinstance(registry, ModelRegistry)
            assert registry._storage_path == Path(tmpdir)
