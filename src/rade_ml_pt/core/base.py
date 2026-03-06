"""
Base model classes for rade_ml_pt models.

This module provides abstract base classes that all ML models in the library should inherit from. These ensure
consistent interfaces for training, evaluation, serialization and inference.

Architecture:
    BaseModel (torch.nn.Module)

Usage:
    from rade_ml_pt import BaseModel

"""
from __future__ import annotations

import json
import torch
import torch.nn as nn

from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union


class BaseModel(nn.Module):
    """
    Abstract base class for all ML models in the library.

    Provides:
        - Consistent interface for training and inference via forward().
        - Built-in model metadata tracking.
        - Configuration-based serialization (save config JSON alongside state_dict).
        - Summary utility for architecture inspection.

    All models should inherit from this class to ensure compatibility with the library's training,
    evaluation and inference pipelines.
    """

    def __init__(self, name: str = "base_model", metadata: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialise BaseModel instance.

        :param name: human-readable model name used for logging and registry.
        :param metadata: optional dictionary of extra metadata to attach to the model.
        """
        super().__init__()

        # unique model name used by registry and logging
        self._model_name = name

        # metadata dictionary tracks provenance, framework info, and user-supplied tags
        self._model_metadata: Dict[str, Any] = {
            "model_name": name,
            "model_class": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "framework": "pytorch",
            "framework_version": torch.__version__,
        }
        if metadata is not None:
            self._model_metadata.update(metadata)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Return the human-readable model name."""
        return self._model_name

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return a copy of model metadata dictionary."""
        return self._model_metadata.copy()

    def update_metadata(self, **kwargs) -> None:
        """Update model metadata with additional key-value pairs."""
        self._model_metadata.update(kwargs)

    # ------------------------------------------------------------------
    # Forward pass (subclasses must implement)
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(self, inputs: Any, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.

        :param inputs: model inputs (tensor or dict of tensors).
        :returns: model output tensor.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    # ------------------------------------------------------------------
    # Configuration-based serialization
    # ------------------------------------------------------------------

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization.

        Subclasses should override this to include architecture-specific parameters
        so that a model can be reconstructed from config alone.
        """
        return {"name": self._model_name, "metadata": self._model_metadata}

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "BaseModel":
        """Create model instance from a configuration dictionary.

        :param config: dictionary produced by get_config().
        """
        metadata = config.pop("metadata", {})
        model = cls(**config, **kwargs)
        model._model_metadata.update(metadata)
        return model

    def save_config(self, path: Union[str, Path]) -> None:
        """Save model configuration to a JSON file alongside the state_dict."""
        with open(path, "w") as f:
            json.dump(self.get_config(), f, indent=2, default=str)

    @classmethod
    def load_config(cls, path: Union[str, Path]) -> Dict[str, Any]:
        """Load model configuration from a JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Architecture summary
    # ------------------------------------------------------------------

    def summary_dict(self) -> Dict[str, Any]:
        """Return a dictionary summary of the model architecture."""
        # count trainable vs non-trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)

        # list immediate child modules by name
        child_names = [name for name, _ in self.named_children()]

        return {
            "name": self._model_name,
            "class": self.__class__.__name__,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "modules": child_names,
            "metadata": self._model_metadata,
        }
