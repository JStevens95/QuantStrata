"""
Base model classes for rade ML models.

This module provides abstract base classes that all ML models in the library should inherit from. These ensure
consistent interfaces for training, evaluation, serialization and inference.

Architecture:
    BaseModel (tf.keras.Model)

Usage:
    from rade_ml import BaseModel

"""
from __future__ import annotations

import tensorflow as tf

from abc import abstractmethod
from datetime import datetime
from typing import Dict, Any
try:
    from keras.saving import register_keras_serializable
except ImportError:
    register_keras_serializable = tf.keras.saving.register_keras_serializable

#
_REGISTER_PACKAGE = "Tranql.RadeMl"


@register_keras_serializable(package=_REGISTER_PACKAGE)
class BaseModel(tf.keras.Model):
    """
    Abstract base class for all ML models in the libraru.

    Provides:
        - Consistent interface for training and inference.
        - Built-in model metadata tracking.
        - Standardised save/load with SaveModel format.
        - Configuration Management.

    All models should inherit from this class (or its subclasses) to ensure compatibility with the library's training,
    evaluation and inference pipelines.
    """

    def __init__(self, name: str = "base_model", **kwargs):
        """Initiate BaseModel instance."""
        metadata = kwargs.pop("metadata", None)
        super().__init__(name=name, **kwargs)
        self._model_metadata = {
            "model_name": name,
            "model_class": self.__class__.__name__,
            "created_at": datetime.now().isoformat(),
            "framework": "tensorflow",
            "framework_version": tf.__version__,
        }
        if metadata is not None:
            self._model_metadata.update(metadata)
        self._is_built = False

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return model metadata dictionary."""
        return self._model_metadata.copy()

    def update_metadata(self, **kwargs) -> None:
        """Update model metadata with additional key-value pairs."""
        self._model_metadata.update(kwargs)

    @abstractmethod
    def call(self, inputs: Any, training: bool = False) -> tf.Tensor:
        """
        Forward pass of the model.

        :param inputs: model inputs (tensor or dict of tensors)
        :param training: whether in training mode.
        """
        raise NotImplementedError("Subclasses must implement call()")

    def get_config(self) -> Dict[str, Any]:
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({"metadata": self._model_metadata})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any], **kwargs) -> "BaseModel":
        """Create model from configuration dictionary.
        """
        metadata = config.pop("metadata", {})
        model = cls(**config)
        model._model_metadata.update(metadata)
        return model

    def summary_dict(self) -> Dict[str, Any]:
        """Return a dictionary summary of the model architecture."""
        return {
            "name": self.name,
            "class": self.__class__.__name__,
            "trainable_params": int(sum(tf.reduce_prod(w.shape) for w in self.trainable_weights)),
            "non_trainable_params": int(sum(tf.reduce_prod(w.shape) for w in self.non_trainable_weights)),
            "layers": [layer.name for layer in self.layers],
            "metadata": self._model_metadata,
        }
