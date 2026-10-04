"""
Distributed data-parallel setup and teardown.

Small, because the hard part of distributed training in this framework is an
*ordering* guarantee rather than a configuration one, and that guarantee is
enforced in :mod:`.materialise` and in the pipeline's stage order.

The ordering, and what goes wrong without it
--------------------------------------------
**Defect 6, the other half.** ``DistributedDataParallel`` records the
parameters it must synchronise when it wraps a model. A model with lazily
shaped parameters has none yet, so wrapping before materialisation hands the
wrapper an empty set.

The two outcomes are a crash inside the distributed library, or -- the reason
this is a defect rather than an inconvenience -- a wrapper that synchronises
nothing. Each rank then trains an independent copy on its own shard, the loss
curves look entirely normal, and the "distributed" run produces N unrelated
models of which one is saved. Nothing in the logs distinguishes that from a
correct run.

:func:`distribute` therefore refuses to wrap an unmaterialised model. It is
the last line of defence rather than the mechanism: the pipeline already runs
``materialise`` as its own stage before ``prepare_hardware``, so reaching this
check means something was reordered.

Single-process runs are the normal case
---------------------------------------
``distributed='none'`` returns the model untouched, and that is what almost
every run does. The framework's answer to "train forty models" is a job set
across processes, not one model across devices -- so distribution here is for
the single model too large for one device, which is a different and rarer
problem.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as distributed_backend

from ....core.lifecycle.errors import EngineError
from ....core.provenance.logging import get_logger
from ..materialise import has_lazy_parameters

if TYPE_CHECKING:
    from ....core.spec.hardware import HardwareSpec

__all__ = [
    "distribute",
    "is_distributed_run",
    "local_rank",
    "shutdown_distributed",
    "undistribute",
    "world_size",
]

_LOGGER = get_logger(__name__)

#: Environment variables a launcher sets. Read rather than configured, because
#: the launcher is the authority on how many processes exist -- a spec field
#: duplicating it would be a second source of truth that could disagree.
_WORLD_SIZE_VARIABLE = "WORLD_SIZE"
_LOCAL_RANK_VARIABLE = "LOCAL_RANK"


def _read_count(variable: str, *, default: int) -> int:
    """
    Read a non-negative integer from the environment, tolerating rubbish.

    These variables are set by an external launcher, so their contents are not
    this framework's to guarantee. An unparseable value is treated as absent
    and warned about, because the alternative is an ``int()`` traceback at
    import-adjacent code with no indication that the cause is a stray shell
    export rather than the model.

    Parameters
    ----------
    variable
        Environment variable name.
    default
        Value to use when the variable is absent or unreadable.

    Returns
    -------
    int
        The parsed value, or ``default``.
    """
    raw = os.environ.get(variable)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        _LOGGER.warning(
            "%s=%r is not an integer; treating it as %d. This variable is set "
            "by the launcher, so an unreadable value usually means a stray "
            "shell export rather than a problem with the job",
            variable,
            raw,
            default,
        )
        return default


def world_size() -> int:
    """
    Return how many processes are taking part.

    Returns
    -------
    int
        The process count, or one when this is not a distributed launch.
    """
    if distributed_backend.is_available() and distributed_backend.is_initialized():
        return int(distributed_backend.get_world_size())
    return _read_count(_WORLD_SIZE_VARIABLE, default=1)


def local_rank() -> int:
    """
    Return this process's index on its own machine.

    Returns
    -------
    int
        The local rank, or zero when this is not a distributed launch.
    """
    return _read_count(_LOCAL_RANK_VARIABLE, default=0)


def is_distributed_run() -> bool:
    """
    Return whether more than one process is taking part.

    Returns
    -------
    bool
        True when a launcher has started several processes.
    """
    return world_size() > 1


def distribute(
    model: torch.nn.Module, *, spec: HardwareSpec, device: torch.device
) -> tuple[torch.nn.Module, bool]:
    """
    Wrap a model for distributed training, if that was asked for and is possible.

    Parameters
    ----------
    model
        The **materialised** model.
    spec
        The hardware request, naming the strategy.
    device
        The device this process owns.

    Returns
    -------
    tuple
        The model -- wrapped or not -- and whether it was wrapped. The flag
        travels into :class:`~rade_qnet.engines.base.ModelHandle` so a report
        can state whether a run that asked to be distributed actually was.

    Raises
    ------
    EngineError
        If the model still has lazy parameters. See the module docstring: this
        is the one condition here that must not degrade, because degrading
        would mean training N unsynchronised models that look like one.
    """
    if spec.distributed == "none":
        return model, False

    if has_lazy_parameters(model):
        raise EngineError(
            "cannot wrap a model with uninitialised parameters for distributed "
            "training: the wrapper would record an empty parameter set and "
            "synchronise nothing, so each rank would train its own independent "
            "copy while reporting a normal loss curve. Run materialise() first -- "
            "the train pipeline does this as its own stage, so reaching this "
            "error means the stage order was changed"
        )

    if not is_distributed_run():
        # Warn and degrade: a single-process run of a distributed
        # configuration is correct, just not parallel.  This is the same
        # policy as an absent accelerator, and for the same reason.
        _LOGGER.warning(
            "distributed=%r was requested but %s reports a world size of 1; "
            "training in a single process. Launch with torchrun to distribute",
            spec.distributed,
            _WORLD_SIZE_VARIABLE,
        )
        return model, False

    if not distributed_backend.is_available():
        _LOGGER.warning(
            "distributed=%r was requested but this Torch build has no "
            "distributed support; training in a single process",
            spec.distributed,
        )
        return model, False

    if not distributed_backend.is_initialized():
        # NCCL for CUDA, Gloo for everything else.  Chosen here rather than
        # configured, because picking the wrong one is never what anybody
        # wanted and the right one follows from the device.
        backend = "nccl" if device.type == "cuda" else "gloo"
        _LOGGER.info(
            "initialising the %s process group: rank %d of %d",
            backend,
            local_rank(),
            world_size(),
        )
        distributed_backend.init_process_group(backend=backend)

    wrapped = torch.nn.parallel.DistributedDataParallel(
        model,
        # Only meaningful for CUDA.  Passing a CPU device here is an error in
        # Torch rather than a no-op, so it is set conditionally.
        device_ids=[device.index if device.index is not None else local_rank()]
        if device.type == "cuda"
        else None,
    )
    _LOGGER.info("wrapped model for distributed training across %d process(es)", world_size())
    return wrapped, True


def undistribute(model: torch.nn.Module) -> torch.nn.Module:
    """
    Return the underlying model from inside any wrapper.

    Unwraps a distributed wrapper and a compiled one, and both together.
    Used to populate :attr:`~rade_qnet.engines.base.ModelHandle.unwrapped`, so
    that a checkpoint carries plain parameter names and loads into a
    single-process model -- see :mod:`.checkpoint`.

    Parameters
    ----------
    model
        A model, possibly wrapped.

    Returns
    -------
    torch.nn.Module
        The innermost model.
    """
    inner = model
    # Looped because the wrappers nest: `torch.compile` applied to a
    # distributed model leaves both layers in place.
    while True:
        if isinstance(inner, torch.nn.parallel.DistributedDataParallel):
            inner = inner.module
            continue
        original = getattr(inner, "_orig_mod", None)
        if isinstance(original, torch.nn.Module):
            inner = original
            continue
        return inner


def shutdown_distributed() -> None:
    """
    Tear the process group down, if one was started.

    Called at the end of a fit. Without it a process that finishes early
    leaves its peers blocked on a collective operation until they time out,
    which presents as a job set that hangs rather than as an error.
    """
    if distributed_backend.is_available() and distributed_backend.is_initialized():
        _LOGGER.info("destroying the distributed process group")
        distributed_backend.destroy_process_group()
