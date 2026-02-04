"""
Model registry for versioning and managing ML models.

Provides:
- ModelRegistry: Central registry for model versions
- ModelArtifact: Represents a registered model version
- ModelVersion: Version metadata and lifecycle stages

Usage:
    from src.machine_learning.registry import (
        ModelRegistry, ModelArtifact, ModelStage
    )
    
    # Create registry
    registry = ModelRegistry(storage_path="./model_registry")
    
    # Register a model
    version = registry.register_model(
        name="option_pricer",
        model_path="./trained_model",
        metrics={"mse": 0.001, "mae": 0.02},
        tags={"asset_class": "fx"},
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

from src.machine_learning.registry.registry import (
    ModelRegistry,
    ModelArtifact,
    ModelVersion,
    ModelStage,
    create_registry,
)

__all__ = [
    "ModelRegistry",
    "ModelArtifact",
    "ModelVersion",
    "ModelStage",
    "create_registry",
]
