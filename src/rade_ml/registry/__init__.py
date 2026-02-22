"""
Model registry: versioned checkpoint storage with tag-based retrieval.
"""
from src.rade_ml.registry.entry import RegistryEntry
from src.rade_ml.registry.store import ModelRegistry

__all__ = ["RegistryEntry", "ModelRegistry"]
