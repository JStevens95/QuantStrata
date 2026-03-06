"""
Device / strategy selection for PyTorch training.

Replaces TensorFlow's ``tf.distribute.Strategy`` with a simple
``torch.device`` selector.  The function inspects CUDA and MPS availability
and returns the best device matching the requested strategy name.

Usage::

    device = get_training_strategy(config)       # from TrainingConfig
    device = get_training_strategy("auto")        # from a plain string
    model.to(device)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from src.rade_ml_pt.core.config import TrainingConfig

logger = logging.getLogger(__name__)


def get_training_strategy(
    config: Optional[Union["TrainingConfig", str]] = None,
) -> torch.device:
    """Select the best available ``torch.device`` from *config*.

    Strategy modes
    --------------
    * ``"auto"``  – CUDA GPU if available, else MPS (Apple Silicon), else CPU.
    * ``"cuda"`` / ``"gpu"`` – force CUDA GPU.
    * ``"mps"``  – force Apple Metal Performance Shaders.
    * ``"cpu"`` / ``"one_device_cpu"`` – force CPU.
    * ``"ddp"``  – placeholder for DistributedDataParallel (returns CUDA if available).
    * ``None``   – returns CPU.

    :param config: a ``TrainingConfig`` instance or a bare strategy string.
    :returns: resolved ``torch.device``.
    """
    strategy_name: Optional[str] = None

    if config is not None:
        if hasattr(config, "strategy"):
            strategy_name = getattr(config, "strategy", None)
        elif isinstance(config, str):
            strategy_name = config if config else None

    if not strategy_name:
        return torch.device("cpu")

    name = str(strategy_name).strip().lower()

    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if name in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("MPS requested but not available; falling back to CPU.")
        return torch.device("cpu")

    if name in ("cpu", "one_device_cpu"):
        return torch.device("cpu")

    if name == "ddp":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("DDP requested but CUDA not available; falling back to CPU.")
        return torch.device("cpu")

    logger.warning("Unknown strategy %r; using CPU.", strategy_name)
    return torch.device("cpu")
