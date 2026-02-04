"""
Model Registry for versioning and managing ML models.

Provides production-grade model management with:
- Version tracking and metadata storage
- Stage transitions (staging, production, archived)
- Model artifact storage
- Query and retrieval by name, version, or stage

Example:
    from src.machine_learning.registry import ModelRegistry, ModelStage
    
    # Initialize registry
    registry = ModelRegistry("./model_registry")
    
    # Register a trained model
    version = registry.register_model(
        name="gnn_pricer",
        model_path="./trained_models/gnn_v1",
        metrics={"mse": 0.001, "r2": 0.99},
        params={"hidden_units": 128, "learning_rate": 0.001},
        tags={"asset_class": "fx", "model_type": "gnn_rnn"},
    )
    print(f"Registered version: {version.version}")
    
    # Transition to production
    registry.transition_stage(
        name="gnn_pricer",
        version=version.version,
        stage=ModelStage.PRODUCTION,
    )
    
    # Load production model for inference
    artifact = registry.get_model("gnn_pricer", stage=ModelStage.PRODUCTION)
    model = artifact.load()
    predictions = model.predict(features)
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class ModelStage(Enum):
    """Lifecycle stages for registered models."""
    
    NONE = "none"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    
    def __str__(self) -> str:
        return self.value


@dataclass
class ModelVersion:
    """
    Metadata for a specific model version.
    
    Attributes
    ----------
    version : int
        Version number (auto-incremented).
    created_at : datetime
        When this version was registered.
    stage : ModelStage
        Current lifecycle stage.
    metrics : dict
        Evaluation metrics at registration time.
    params : dict
        Hyperparameters used for training.
    tags : dict
        User-defined tags.
    description : str
        Human-readable description.
    source_run_id : str, optional
        ID of the experiment tracking run that produced this model.
    model_hash : str
        Hash of model artifacts for integrity verification.
    artifact_path : str
        Path to stored artifacts relative to registry root.
    """
    
    version: int
    created_at: datetime
    stage: ModelStage
    metrics: Dict[str, float]
    params: Dict[str, Any]
    tags: Dict[str, str]
    description: str
    source_run_id: Optional[str]
    model_hash: str
    artifact_path: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "stage": self.stage.value,
            "metrics": self.metrics,
            "params": self.params,
            "tags": self.tags,
            "description": self.description,
            "source_run_id": self.source_run_id,
            "model_hash": self.model_hash,
            "artifact_path": self.artifact_path,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelVersion":
        """Create from dictionary."""
        return cls(
            version=d["version"],
            created_at=datetime.fromisoformat(d["created_at"]),
            stage=ModelStage(d["stage"]),
            metrics=d["metrics"],
            params=d["params"],
            tags=d["tags"],
            description=d["description"],
            source_run_id=d.get("source_run_id"),
            model_hash=d["model_hash"],
            artifact_path=d["artifact_path"],
        )


@dataclass
class ModelArtifact:
    """
    A registered model artifact with loading capabilities.
    
    Attributes
    ----------
    name : str
        Model name.
    version : ModelVersion
        Version metadata.
    registry_path : Path
        Path to the registry root.
    """
    
    name: str
    version: ModelVersion
    registry_path: Path
    
    @property
    def artifact_dir(self) -> Path:
        """Get the full path to model artifacts."""
        return self.registry_path / self.version.artifact_path
    
    def load(
        self,
        custom_objects: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Load the model from artifacts.
        
        Parameters
        ----------
        custom_objects : dict, optional
            Custom objects for Keras model loading.
            
        Returns
        -------
        model
            The loaded model.
        """
        # Check for different model formats
        artifact_dir = self.artifact_dir
        
        # TensorFlow SavedModel format
        if (artifact_dir / "saved_model.pb").exists():
            import tensorflow as tf
            return tf.keras.models.load_model(
                str(artifact_dir),
                custom_objects=custom_objects,
            )
        
        # Keras .keras format
        keras_files = list(artifact_dir.glob("*.keras"))
        if keras_files:
            import tensorflow as tf
            return tf.keras.models.load_model(
                str(keras_files[0]),
                custom_objects=custom_objects,
            )
        
        # HDF5 format
        h5_files = list(artifact_dir.glob("*.h5"))
        if h5_files:
            import tensorflow as tf
            return tf.keras.models.load_model(
                str(h5_files[0]),
                custom_objects=custom_objects,
            )
        
        # PyTorch format
        pt_files = list(artifact_dir.glob("*.pt")) + list(artifact_dir.glob("*.pth"))
        if pt_files:
            import torch
            return torch.load(str(pt_files[0]))
        
        # Pickle format
        pkl_files = list(artifact_dir.glob("*.pkl"))
        if pkl_files:
            import pickle
            with open(pkl_files[0], "rb") as f:
                return pickle.load(f)
        
        # Check for model.json + weights
        if (artifact_dir / "model.json").exists():
            import tensorflow as tf
            with open(artifact_dir / "model.json") as f:
                model = tf.keras.models.model_from_json(
                    f.read(),
                    custom_objects=custom_objects,
                )
            weights_file = artifact_dir / "weights.h5"
            if weights_file.exists():
                model.load_weights(str(weights_file))
            return model
        
        raise ValueError(
            f"Could not find a loadable model format in {artifact_dir}. "
            f"Supported formats: SavedModel, .keras, .h5, .pt, .pth, .pkl"
        )
    
    def verify_integrity(self) -> bool:
        """Verify model integrity against stored hash."""
        computed_hash = _compute_directory_hash(self.artifact_dir)
        return computed_hash == self.version.model_hash


@dataclass
class RegisteredModel:
    """
    A registered model with all its versions.
    
    Internal class for managing model metadata.
    """
    
    name: str
    description: str
    created_at: datetime
    updated_at: datetime
    versions: Dict[int, ModelVersion] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    @property
    def latest_version(self) -> Optional[ModelVersion]:
        """Get the latest version."""
        if not self.versions:
            return None
        max_version = max(self.versions.keys())
        return self.versions[max_version]
    
    def get_version_by_stage(self, stage: ModelStage) -> Optional[ModelVersion]:
        """Get the version in a specific stage."""
        for v in sorted(self.versions.values(), key=lambda x: x.version, reverse=True):
            if v.stage == stage:
                return v
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "versions": {
                str(k): v.to_dict() for k, v in self.versions.items()
            },
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RegisteredModel":
        """Create from dictionary."""
        return cls(
            name=d["name"],
            description=d["description"],
            created_at=datetime.fromisoformat(d["created_at"]),
            updated_at=datetime.fromisoformat(d["updated_at"]),
            versions={
                int(k): ModelVersion.from_dict(v) 
                for k, v in d.get("versions", {}).items()
            },
            tags=d.get("tags", {}),
        )


# =============================================================================
# Helper Functions
# =============================================================================


def _compute_directory_hash(path: Path) -> str:
    """Compute a hash of all files in a directory."""
    if not path.exists():
        return ""
    
    hasher = hashlib.sha256()
    
    if path.is_file():
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
    else:
        for file_path in sorted(path.rglob("*")):
            if file_path.is_file():
                rel_path = file_path.relative_to(path)
                hasher.update(str(rel_path).encode())
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        hasher.update(chunk)
    
    return hasher.hexdigest()


def _copy_model_artifacts(source: Path, dest: Path) -> None:
    """Copy model artifacts to registry storage."""
    dest.mkdir(parents=True, exist_ok=True)
    
    if source.is_file():
        shutil.copy2(source, dest / source.name)
    elif source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
    else:
        raise ValueError(f"Source path does not exist: {source}")


# =============================================================================
# Model Registry
# =============================================================================


class ModelRegistry:
    """
    Central registry for versioning and managing ML models.
    
    Provides model lifecycle management with:
    - Version tracking with auto-increment
    - Stage transitions (staging → production → archived)
    - Artifact storage with integrity verification
    - Metadata and tag management
    
    Example:
        registry = ModelRegistry("./model_registry")
        
        # Register a new model version
        version = registry.register_model(
            name="option_pricer",
            model_path="./trained_model",
            metrics={"mse": 0.001},
        )
        
        # Promote to production
        registry.transition_stage(
            name="option_pricer",
            version=version.version,
            stage=ModelStage.PRODUCTION,
        )
        
        # Load production model
        artifact = registry.get_model("option_pricer", stage=ModelStage.PRODUCTION)
        model = artifact.load()
    """
    
    def __init__(self, storage_path: Union[str, Path]) -> None:
        """
        Initialize model registry.
        
        Parameters
        ----------
        storage_path : str or Path
            Root directory for registry storage.
        """
        self._storage_path = Path(storage_path)
        self._storage_path.mkdir(parents=True, exist_ok=True)
        
        self._models_dir = self._storage_path / "models"
        self._models_dir.mkdir(exist_ok=True)
        
        self._metadata_file = self._storage_path / "registry.json"
        self._models: Dict[str, RegisteredModel] = {}
        
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load registry metadata from disk."""
        if self._metadata_file.exists():
            with open(self._metadata_file) as f:
                data = json.load(f)
            
            self._models = {
                name: RegisteredModel.from_dict(model_data)
                for name, model_data in data.get("models", {}).items()
            }
            logger.info("Loaded registry with %d models", len(self._models))
    
    def _save_metadata(self) -> None:
        """Save registry metadata to disk."""
        data = {
            "models": {
                name: model.to_dict()
                for name, model in self._models.items()
            },
            "updated_at": datetime.now().isoformat(),
        }
        
        with open(self._metadata_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def register_model(
        self,
        name: str,
        model_path: Union[str, Path],
        metrics: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
        source_run_id: Optional[str] = None,
    ) -> ModelVersion:
        """
        Register a new model version.
        
        Parameters
        ----------
        name : str
            Model name. Creates new registered model if doesn't exist.
        model_path : str or Path
            Path to model artifacts (file or directory).
        metrics : dict, optional
            Evaluation metrics.
        params : dict, optional
            Training hyperparameters.
        tags : dict, optional
            User-defined tags.
        description : str
            Version description.
        source_run_id : str, optional
            ID of the experiment run that produced this model.
            
        Returns
        -------
        ModelVersion
            The registered version metadata.
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Get or create registered model
        if name not in self._models:
            self._models[name] = RegisteredModel(
                name=name,
                description=description,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            logger.info("Created new registered model: %s", name)
        
        registered_model = self._models[name]
        
        # Determine next version number
        next_version = max(registered_model.versions.keys(), default=0) + 1
        
        # Create artifact storage path
        artifact_subpath = f"{name}/v{next_version}"
        artifact_dest = self._models_dir / artifact_subpath
        
        # Copy artifacts
        _copy_model_artifacts(model_path, artifact_dest)
        
        # Compute hash for integrity
        model_hash = _compute_directory_hash(artifact_dest)
        
        # Create version
        version = ModelVersion(
            version=next_version,
            created_at=datetime.now(),
            stage=ModelStage.NONE,
            metrics=metrics or {},
            params=params or {},
            tags=tags or {},
            description=description,
            source_run_id=source_run_id,
            model_hash=model_hash,
            artifact_path=artifact_subpath,
        )
        
        # Register version
        registered_model.versions[next_version] = version
        registered_model.updated_at = datetime.now()
        
        # Save metadata
        self._save_metadata()
        
        logger.info(
            "Registered model %s version %d (hash=%s)",
            name,
            next_version,
            model_hash[:12],
        )
        
        return version
    
    def get_model(
        self,
        name: str,
        version: Optional[int] = None,
        stage: Optional[ModelStage] = None,
    ) -> ModelArtifact:
        """
        Get a registered model.
        
        Parameters
        ----------
        name : str
            Model name.
        version : int, optional
            Specific version number. If None, uses latest or stage.
        stage : ModelStage, optional
            Get model in this stage. Ignored if version specified.
            
        Returns
        -------
        ModelArtifact
            The model artifact with loading capabilities.
            
        Raises
        ------
        KeyError
            If model or version not found.
        """
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        
        registered_model = self._models[name]
        
        if version is not None:
            if version not in registered_model.versions:
                raise KeyError(f"Version {version} not found for model {name}")
            model_version = registered_model.versions[version]
        elif stage is not None:
            model_version = registered_model.get_version_by_stage(stage)
            if model_version is None:
                raise KeyError(f"No version in stage {stage} for model {name}")
        else:
            model_version = registered_model.latest_version
            if model_version is None:
                raise KeyError(f"No versions registered for model {name}")
        
        return ModelArtifact(
            name=name,
            version=model_version,
            registry_path=self._models_dir,
        )
    
    def transition_stage(
        self,
        name: str,
        version: int,
        stage: ModelStage,
        archive_existing: bool = True,
    ) -> None:
        """
        Transition a model version to a new stage.
        
        Parameters
        ----------
        name : str
            Model name.
        version : int
            Version number to transition.
        stage : ModelStage
            Target stage.
        archive_existing : bool
            If True and transitioning to PRODUCTION, archive any existing
            production version.
        """
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        
        registered_model = self._models[name]
        
        if version not in registered_model.versions:
            raise KeyError(f"Version {version} not found for model {name}")
        
        # Archive existing production version if needed
        if archive_existing and stage == ModelStage.PRODUCTION:
            for v in registered_model.versions.values():
                if v.stage == ModelStage.PRODUCTION and v.version != version:
                    v.stage = ModelStage.ARCHIVED
                    logger.info(
                        "Archived previous production version %d of %s",
                        v.version,
                        name,
                    )
        
        # Transition
        old_stage = registered_model.versions[version].stage
        registered_model.versions[version].stage = stage
        registered_model.updated_at = datetime.now()
        
        self._save_metadata()
        
        logger.info(
            "Transitioned %s version %d: %s → %s",
            name,
            version,
            old_stage.value,
            stage.value,
        )
    
    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self._models.keys())
    
    def list_versions(
        self,
        name: str,
        stage: Optional[ModelStage] = None,
    ) -> List[ModelVersion]:
        """
        List versions of a model.
        
        Parameters
        ----------
        name : str
            Model name.
        stage : ModelStage, optional
            Filter by stage.
            
        Returns
        -------
        list
            List of ModelVersion objects.
        """
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        
        versions = list(self._models[name].versions.values())
        
        if stage is not None:
            versions = [v for v in versions if v.stage == stage]
        
        return sorted(versions, key=lambda v: v.version)
    
    def delete_version(self, name: str, version: int) -> None:
        """
        Delete a model version.
        
        Parameters
        ----------
        name : str
            Model name.
        version : int
            Version to delete.
            
        Note
        ----
        Deletes both metadata and artifacts.
        """
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        
        registered_model = self._models[name]
        
        if version not in registered_model.versions:
            raise KeyError(f"Version {version} not found for model {name}")
        
        # Remove artifacts
        artifact_path = self._models_dir / registered_model.versions[version].artifact_path
        if artifact_path.exists():
            shutil.rmtree(artifact_path)
        
        # Remove metadata
        del registered_model.versions[version]
        registered_model.updated_at = datetime.now()
        
        self._save_metadata()
        
        logger.info("Deleted version %d of %s", version, name)
    
    def update_tags(
        self,
        name: str,
        version: int,
        tags: Dict[str, str],
    ) -> None:
        """Update tags on a model version."""
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        
        registered_model = self._models[name]
        
        if version not in registered_model.versions:
            raise KeyError(f"Version {version} not found for model {name}")
        
        registered_model.versions[version].tags.update(tags)
        registered_model.updated_at = datetime.now()
        
        self._save_metadata()
    
    def search_models(
        self,
        name_contains: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        stage: Optional[ModelStage] = None,
        metric_filter: Optional[Callable[[Dict[str, float]], bool]] = None,
    ) -> List[ModelArtifact]:
        """
        Search for models matching criteria.
        
        Parameters
        ----------
        name_contains : str, optional
            Filter by name substring.
        tags : dict, optional
            Filter by tags (all must match).
        stage : ModelStage, optional
            Filter by stage.
        metric_filter : callable, optional
            Function (metrics_dict) -> bool for custom filtering.
            
        Returns
        -------
        list
            List of matching ModelArtifact objects.
        """
        results = []
        
        for model_name, registered_model in self._models.items():
            # Name filter
            if name_contains and name_contains not in model_name:
                continue
            
            for version in registered_model.versions.values():
                # Stage filter
                if stage is not None and version.stage != stage:
                    continue
                
                # Tags filter
                if tags:
                    if not all(
                        version.tags.get(k) == v
                        for k, v in tags.items()
                    ):
                        continue
                
                # Metric filter
                if metric_filter and not metric_filter(version.metrics):
                    continue
                
                results.append(ModelArtifact(
                    name=model_name,
                    version=version,
                    registry_path=self._models_dir,
                ))
        
        return results
    
    def get_model_info(self, name: str) -> Dict[str, Any]:
        """
        Get complete information about a registered model.
        
        Parameters
        ----------
        name : str
            Model name.
            
        Returns
        -------
        dict
            Model information including all versions.
        """
        if name not in self._models:
            raise KeyError(f"Model not found: {name}")
        
        registered_model = self._models[name]
        
        return {
            "name": name,
            "description": registered_model.description,
            "created_at": registered_model.created_at.isoformat(),
            "updated_at": registered_model.updated_at.isoformat(),
            "n_versions": len(registered_model.versions),
            "latest_version": (
                registered_model.latest_version.version
                if registered_model.latest_version
                else None
            ),
            "production_version": (
                registered_model.get_version_by_stage(ModelStage.PRODUCTION).version
                if registered_model.get_version_by_stage(ModelStage.PRODUCTION)
                else None
            ),
            "tags": registered_model.tags,
        }


# =============================================================================
# Factory Function
# =============================================================================


def create_registry(
    storage_path: Union[str, Path] = "./model_registry",
) -> ModelRegistry:
    """
    Create a model registry.
    
    Parameters
    ----------
    storage_path : str or Path
        Root directory for registry storage.
        
    Returns
    -------
    ModelRegistry
        Configured registry instance.
    """
    return ModelRegistry(storage_path)


__all__ = [
    "ModelRegistry",
    "ModelArtifact",
    "ModelVersion",
    "ModelStage",
    "RegisteredModel",
    "create_registry",
]
