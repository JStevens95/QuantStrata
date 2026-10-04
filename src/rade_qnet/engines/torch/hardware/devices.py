"""
Resolving a ``HardwareSpec`` into the Torch objects a fit actually uses.

One module turns a declarative request -- "cuda, bf16, compiled" -- into a
device, an autocast context and a gradient scaler. Everything that inspects
``torch.cuda`` or ``torch.backends`` lives here, so the rest of the engine
never asks what machine it is on.

A named device is a request, not a guarantee
--------------------------------------------
:func:`resolve_device` warns and degrades when the requested accelerator is
absent, rather than raising. That is a deliberate asymmetry with the rest of
the framework, which refuses almost everything it cannot honour exactly.

The reasoning is that an absent accelerator does not make a run *wrong*, only
slower -- whereas the configurations this framework does refuse (a leaky
split, an unsupported precision) make the result incorrect or meaningless. A
job set submitted against a GPU host and rerun on a laptop should produce the
same numbers more slowly, not fail. The warning is what keeps it from being
silent, and the chosen device is recorded in the model handle so a report can
show that a run intended for CUDA ran on the CPU.

``fp16`` is the one exception, and it is handled by the spec rather than here:
``HardwareSpec`` already refuses ``fp16`` on a CPU at load time, because that
combination is not slow-but-correct -- most CPU kernels have no ``fp16`` path.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING

import torch

from ....core.runtime.errors import EngineError
from ....core.runtime.logging import get_logger

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from ....core.spec.hardware import HardwareSpec

__all__ = [
    "ResolvedHardware",
    "apply_thread_budget",
    "autocast_for",
    "available_accelerators",
    "compile_if_requested",
    "resolve_device",
    "resolve_hardware",
]

_LOGGER = get_logger(__name__)

#: Spec precision names mapped to Torch dtypes. ``fp32`` is absent on purpose:
#: full precision means "do not autocast at all", which is a different
#: decision from "autocast to float32" and is handled by returning a null
#: context rather than by looking up a dtype.
_AUTOCAST_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

#: Device types that support autocast. Autocast on an MPS device is not
#: implemented in Torch, so requesting reduced precision there degrades to
#: full precision with a warning rather than erroring.
_AUTOCAST_DEVICES = frozenset({"cpu", "cuda"})


class ResolvedHardware:
    """
    The concrete Torch objects one fit runs against.

    Parameters
    ----------
    device
        Where tensors and parameters live.
    precision
        The precision actually in effect, which may differ from the one
        requested if the chosen device could not honour it.
    requested_device
        What the specification asked for, kept so a report can show that a
        run intended for an accelerator ran on the CPU.
    scaler
        Gradient scaler, present only for ``fp16``. ``float16`` has roughly
        five exponent bits fewer than ``float32``, so small gradients
        underflow to zero and the affected parameters simply stop learning --
        quietly, and only for the layers with the smallest gradients.
        ``bfloat16`` keeps ``float32``'s exponent range and needs no scaler.
    """

    def __init__(
        self,
        *,
        device: torch.device,
        precision: str,
        requested_device: str,
        scaler: torch.amp.GradScaler | None = None,
    ) -> None:
        self.device = device
        self.precision = precision
        self.requested_device = requested_device
        self.scaler = scaler

    @property
    def degraded(self) -> bool:
        """Whether the resolved device differs from the one requested."""
        return self.requested_device not in ("auto", self.device.type)

    def autocast(self) -> AbstractContextManager[None]:
        """
        Return the context a forward pass runs inside.

        Returns
        -------
        contextlib.AbstractContextManager
            An autocast context, or a null context under full precision.
        """
        return autocast_for(self.device, self.precision)

    def describe(self) -> dict[str, object]:
        """
        Return a summary for reports and logs.

        Returns
        -------
        dict
            JSON-encodable summary.
        """
        return {
            "device": str(self.device),
            "precision": self.precision,
            "requested_device": self.requested_device,
            "degraded": self.degraded,
            "gradient_scaler": self.scaler is not None,
        }


def available_accelerators() -> tuple[str, ...]:
    """
    Return the accelerator types this machine can actually use.

    Returns
    -------
    tuple of str
        Device types, best first, always ending with ``cpu``.
    """
    accelerators: list[str] = []
    if torch.cuda.is_available():
        accelerators.append("cuda")
    # `mps.is_available()` is false on non-Apple builds, and the attribute
    # itself is absent on sufficiently old Torch, so both are checked.
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        accelerators.append("mps")
    accelerators.append("cpu")
    return tuple(accelerators)


def resolve_device(spec: HardwareSpec) -> torch.device:
    """
    Return the device a fit will run on.

    Parameters
    ----------
    spec
        The hardware request.

    Returns
    -------
    torch.device
        The chosen device, with an index when one was requested.
    """
    accelerators = available_accelerators()

    if spec.device == "auto":
        chosen = accelerators[0]
        _LOGGER.info("device='auto' resolved to %s (available: %s)", chosen, accelerators)
        return torch.device(chosen)

    if spec.device not in accelerators:
        # Warn and degrade rather than raise: see the module docstring.
        _LOGGER.warning(
            "device=%r was requested but is not available on this machine "
            "(available: %s); running on cpu instead. Results will match, "
            "throughput will not",
            spec.device,
            accelerators,
        )
        return torch.device("cpu")

    if spec.device_index is None:
        return torch.device(spec.device)
    return torch.device(f"{spec.device}:{spec.device_index}")


def autocast_for(device: torch.device, precision: str) -> AbstractContextManager[None]:
    """
    Return the autocast context for a device and precision.

    Parameters
    ----------
    device
        The resolved device.
    precision
        ``fp32``, ``fp16`` or ``bf16``.

    Returns
    -------
    contextlib.AbstractContextManager
        An autocast context, or a null context when no casting applies.

    Raises
    ------
    EngineError
        If the precision name is not recognised. Unlike an absent
        accelerator, an unrecognised precision is a programming error rather
        than a property of the machine, so it raises.
    """
    if precision == "fp32":
        return nullcontext()
    if precision not in _AUTOCAST_DTYPES:
        raise EngineError(
            f"unknown precision {precision!r}; expected one of "
            f"{['fp32', *sorted(_AUTOCAST_DTYPES)]}"
        )
    if device.type not in _AUTOCAST_DEVICES:
        _LOGGER.warning(
            "precision=%r is not supported on a %s device; running in fp32",
            precision,
            device.type,
        )
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=_AUTOCAST_DTYPES[precision])


def apply_thread_budget(spec: HardwareSpec) -> int:
    """
    Apply ``threads_per_worker`` to this process's intra-op thread pool.

    **Why this is not left to the environment.** A worker process has its
    thread budget set by environment variable before it imports anything,
    which works because the worker is a fresh interpreter. The process that
    launched it has already imported Torch, so the same variable does
    nothing there. The budget would then apply to pooled jobs and not to
    sequential ones -- and because the number of threads decides the order
    in which a reduction accumulates, the same job would score differently
    depending on where it ran.

    That difference is small and entirely real: measured on the flagship
    model it moves the test :math:`R^2` in the seventh significant figure,
    reproducibly, with each value stable for its own thread count. Setting
    the budget from the specification in both paths is what makes placement
    an operational choice rather than a modelling one.

    Parameters
    ----------
    spec
        The hardware request. ``threads_per_worker=None`` means "whatever
        this machine chose", which is left untouched.

    Returns
    -------
    int
        The thread count in effect afterwards, whether or not it changed.
    """
    if spec.threads_per_worker is None:
        return torch.get_num_threads()

    # Idempotent, and cheap enough to call per job. Torch tolerates being
    # set repeatedly in a process; it is resizing a pool, not creating one.
    torch.set_num_threads(spec.threads_per_worker)
    _LOGGER.debug("set torch intra-op threads to %d", spec.threads_per_worker)
    return torch.get_num_threads()


def resolve_hardware(spec: HardwareSpec) -> ResolvedHardware:
    """
    Resolve a hardware request into the objects a fit uses.

    Parameters
    ----------
    spec
        The hardware request.

    Returns
    -------
    ResolvedHardware
        Device, effective precision and gradient scaler.
    """
    apply_thread_budget(spec)
    device = resolve_device(spec)

    precision = spec.precision
    if precision != "fp32" and device.type not in _AUTOCAST_DEVICES:
        _LOGGER.warning(
            "precision=%r cannot be honoured on a %s device; using fp32",
            precision,
            device.type,
        )
        precision = "fp32"

    # `HardwareSpec` refuses fp16-on-CPU at load time, but degrading an absent
    # accelerator can recreate the combination it refused: `device='cuda',
    # precision='fp16'` on a machine with no CUDA lands on the CPU still
    # asking for fp16.  The same rule has to hold however the pairing arose,
    # or the spec's guarantee would depend on which machine ran the job.
    if precision == "fp16" and device.type == "cpu":
        _LOGGER.warning(
            "precision='fp16' is not usable on a cpu device, which this run "
            "degraded to; using fp32. Specify precision='bf16' for reduced "
            "precision that works on both"
        )
        precision = "fp32"

    # A scaler is only meaningful for fp16, and only on a device where
    # autocast runs.  See `ResolvedHardware.scaler` for why bf16 needs none.
    scaler = (
        torch.amp.GradScaler(device=device.type)
        if precision == "fp16" and device.type in _AUTOCAST_DEVICES
        else None
    )

    resolved = ResolvedHardware(
        device=device,
        precision=precision,
        requested_device=spec.device,
        scaler=scaler,
    )
    _LOGGER.info("resolved hardware: %s", resolved.describe())
    return resolved


def compile_if_requested(model: torch.nn.Module, *, spec: HardwareSpec) -> torch.nn.Module:
    """
    Apply Torch's graph compiler when the specification asks for it.

    Must be called *after* lazy parameters are materialised. Compiling an
    unmaterialised module traces a graph whose shapes are not yet known, which
    is the same ordering mistake as wrapping for distributed training too
    early -- defect 6.

    Parameters
    ----------
    model
        The materialised model.
    spec
        The hardware request.

    Returns
    -------
    torch.nn.Module
        The compiled model, or the original if compilation was not requested
        or is unavailable. Compilation is a throughput optimisation and never
        changes what the model computes, so an unavailable compiler warns and
        returns the original rather than failing the run.
    """
    if not spec.compile_model:
        return model

    compiler = getattr(torch, "compile", None)
    if compiler is None:
        _LOGGER.warning(
            "compile_model=True but this Torch build has no torch.compile; running uncompiled"
        )
        return model

    _LOGGER.info("compiling model with torch.compile")
    # The return type is a callable wrapper rather than a Module as far as the
    # type checker is concerned, but it behaves as one -- including
    # `state_dict` -- so it is narrowed here once instead of at each use.
    compiled: torch.nn.Module = compiler(model)
    return compiled
