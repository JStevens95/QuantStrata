"""
TensorFlow distribution strategy utilities for training.

Provides get_training_strategy() to create the appropriate tf.distribute.Strategy
from TrainingConfig, enabling GPU detection and multi-GPU training.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import tensorflow as tf

if TYPE_CHECKING:
    from src.rade_ml.core.config import TrainingConfig

logger = logging.getLogger(__name__)


def get_training_strategy(
    config: Optional[Union["TrainingConfig", str]] = None,
) -> Optional[tf.distribute.Strategy]:
    """
    Create a tf.distribute.Strategy from TrainingConfig or strategy name.

    Call this before building the model, then build the model inside
    strategy.scope() so variables are placed correctly.

    :param config: TrainingConfig instance, or a strategy string
        ("auto" | "mirrored" | "one_device_gpu" | "one_device_cpu"),
        or None (returns None, no strategy scope).
    :return: Strategy instance, or None if strategy is disabled/unspecified.

    Strategy modes
    --------------
    - "auto": Use GPU(s) if available (MirroredStrategy), else CPU
      (OneDeviceStrategy).
    - "mirrored": MirroredStrategy — replicates model across all visible GPUs.
      Fails if no GPU and TF requires one.
    - "one_device_gpu": OneDeviceStrategy("/GPU:0") — single GPU. Falls back
      to CPU if no GPU.
    - "one_device_cpu": OneDeviceStrategy("/CPU:0") — CPU only.
    - None: No strategy. Model trains on default device (typically single GPU
      or CPU if TF auto-places).

    Example
    -------
    >>> config = TrainingConfig(strategy="auto")
    >>> strategy = get_training_strategy(config)
    >>> if strategy:
    ...     with strategy.scope():
    ...         model = build_model(...)
    >>> else:
    ...     model = build_model(...)
    """
    strategy_name: Optional[str] = None
    if config is not None:
        if hasattr(config, "strategy"):
            strategy_name = getattr(config, "strategy", None)
        elif isinstance(config, str):
            strategy_name = config if config else None
        else:
            strategy_name = None

    if not strategy_name:
        return None

    name = str(strategy_name).strip().lower()
    if name == "auto":
        return _strategy_auto()
    if name == "mirrored":
        return _strategy_mirrored()
    if name == "one_device_gpu":
        return _strategy_one_device_gpu()
    if name == "one_device_cpu":
        return _strategy_one_device_cpu()

    logger.warning(
        "Unknown strategy %r; expected 'auto', 'mirrored', "
        "'one_device_gpu', 'one_device_cpu'. Using None.",
        strategy_name,
    )
    return None


def _strategy_auto() -> tf.distribute.Strategy:
    """Use GPU(s) if available, else CPU."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("Detected %d GPU(s); using MirroredStrategy.", len(gpus))
        return tf.distribute.MirroredStrategy()
    logger.info("No GPU detected; using OneDeviceStrategy('/CPU:0').")
    return tf.distribute.OneDeviceStrategy("/CPU:0")


def _strategy_mirrored() -> tf.distribute.Strategy:
    """MirroredStrategy across all visible GPUs."""
    return tf.distribute.MirroredStrategy()


def _strategy_one_device_gpu() -> tf.distribute.Strategy:
    """Single GPU, with fallback to CPU if none available."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        return tf.distribute.OneDeviceStrategy("/GPU:0")
    logger.warning(
        "strategy='one_device_gpu' but no GPU found; falling back to CPU."
    )
    return tf.distribute.OneDeviceStrategy("/CPU:0")


def _strategy_one_device_cpu() -> tf.distribute.Strategy:
    """CPU only."""
    return tf.distribute.OneDeviceStrategy("/CPU:0")
