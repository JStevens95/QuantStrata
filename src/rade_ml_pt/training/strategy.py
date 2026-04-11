"""
Device / strategy selection for PyTorch training.

Replaces TensorFlow's ``tf.distribute.Strategy`` with a simple
``torch.device`` selector.  The function inspects CUDA and MPS availability
and returns the best device matching the requested strategy name.

For DDP (DistributedDataParallel), this module also provides helpers to
initialise / tear down the process group and query rank information.
DDP processes are expected to be launched via ``torchrun`` which sets
the ``RANK``, ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, and
``MASTER_PORT`` environment variables automatically.

Usage::

    device = get_training_strategy(config)       # from TrainingConfig
    device = get_training_strategy("auto")        # from a plain string
    model.to(device)
"""
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, Union

import torch

if TYPE_CHECKING:
    from src.rade_ml_pt.core.config import TrainingConfig

logger = logging.getLogger(__name__)


# ======================================================================
# DDP helpers
# ======================================================================

def setup_ddp(backend: str = "nccl") -> None:
    """Initialise the distributed process group for DDP training.

    Expects ``torchrun`` (or equivalent) to have set ``RANK``,
    ``LOCAL_RANK``, ``WORLD_SIZE``, ``MASTER_ADDR``, ``MASTER_PORT``.
    Safe to call multiple times — returns immediately if already initialised.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    dist.init_process_group(backend=backend)
    logger.info(
        "DDP initialised: rank=%d, local_rank=%d, world_size=%d",
        get_ddp_rank(), local_rank, get_ddp_world_size(),
    )


def cleanup_ddp() -> None:
    """Destroy the distributed process group."""
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def get_ddp_rank() -> int:
    """Global rank of this process (0 on single-process runs)."""
    return int(os.environ.get("RANK", 0))


def get_ddp_local_rank() -> int:
    """Local (per-node) rank — maps to the GPU index on this machine."""
    return int(os.environ.get("LOCAL_RANK", 0))


def get_ddp_world_size() -> int:
    """Total number of DDP processes."""
    return int(os.environ.get("WORLD_SIZE", 1))


def is_ddp_active() -> bool:
    """True if a distributed process group is currently initialised."""
    import torch.distributed as dist

    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    """True if this is rank 0 (or a non-DDP single-process run)."""
    return get_ddp_rank() == 0


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
    * ``"ddp"``  – DistributedDataParallel; returns ``cuda:<LOCAL_RANK>``.
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

    if name in ("cuda", "gpu") or name.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(name if name.startswith("cuda") else "cuda")
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
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            return torch.device(f"cuda:{local_rank}")
        logger.warning("DDP requested but CUDA not available; falling back to CPU.")
        return torch.device("cpu")

    logger.warning("Unknown strategy %r; using CPU.", strategy_name)
    return torch.device("cpu")
