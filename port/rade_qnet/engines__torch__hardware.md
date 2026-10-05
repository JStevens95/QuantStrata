# `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 31 | 1388 | `0f5c6d3c17c4d67f` |
| 2 | `determinism.py` | 124 | 5690 | `0e3900bc3906f0c3` |
| 3 | `devices.py` | 369 | 12856 | `f5fbc3699eba7e1b` |
| 4 | `distributed.py` | 276 | 9488 | `9addadb978642a89` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/__init__.py`

1388 bytes · SHA-256 `0f5c6d3c17c4d67f`

```python
"""
Where the computation runs, and whether it runs the same way twice.

Nothing in this package knows what is being trained.  It answers three
questions a model author should never have to ask: which device is available,
how work is split across more than one of them, and what has to be pinned so
that the same specification produces the same numbers tomorrow.

Why determinism belongs with hardware
--------------------------------------
``determinism.py`` looks like it should sit beside the framework's own
seeding, and it does not, because what it seeds is the hardware: CUDA
generators, cuDNN algorithm selection, the choice between a fast
non-deterministic kernel and a slower reproducible one.  Those are device
facts, not framework facts.  :mod:`rade_qnet.core.provenance.seeding` holds
the library-agnostic half and calls into this one through a registered
callback, which is how ``core`` manages to seed PyTorch without importing it.

Modules
-------
``devices.py``
    Resolving a hardware specification against what is actually present:
    CUDA, MPS or CPU, with mixed precision and memory settings.
``distributed.py``
    Multi-process training, and putting the model back into a shape that can
    be saved once the processes are done with it.
``determinism.py``
    Seeding PyTorch, and the registration that lets ``core`` ask for it.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/determinism.py`

5690 bytes · SHA-256 `0e3900bc3906f0c3`

```python
"""
Seeds Torch's random number generators, and makes ``strict`` mean something.

``core.provenance.seeding`` seeds Python and NumPy and then calls out to whatever
seeders have registered. It cannot seed Torch itself: ``core`` has an empty
dependency set, so it cannot import a training library. The registration
therefore belongs here, in the package that owns the library -- which is the
same dependency inversion the engine registry uses, applied to a different
concern.

Without this module, ``seed_everything`` leaves Torch unseeded. Every weight
initialisation, every dropout mask and every shuffled batch order would differ
between two runs of the same configuration, and the framework would be
reporting a seed that it had not actually applied to the one library that
matters most. Nothing in the output would say so: both runs complete, both
report the seed they were given, and the scores differ.

**What ``strict`` does and does not buy.** Seeding makes the *sampling*
reproducible. It does not make the *arithmetic* reproducible: a GPU reduction
sums in an order that depends on how the work was scheduled, and cuDNN selects
among algorithms by benchmarking them. ``strict`` therefore also asks Torch to
use deterministic algorithm implementations and turns the benchmark selector
off. That costs throughput, which is exactly why it is not the default.

Some operations have no deterministic implementation at all. Under ``strict``
Torch raises when one is reached, and that error propagates rather than being
caught: a user who asked for reproducibility must be told it is unavailable
for their model rather than receiving results that quietly are not
reproducible. Under ``warn`` the same situation logs and continues, which is
the level to use when reproducibility is desirable but not worth refusing a
run over.
"""

from __future__ import annotations

import os

import torch

from ....core.provenance.logging import get_logger
from ....core.provenance.seeding import Determinism, register_seeder

__all__ = ["SEEDER_NAME", "seed_torch"]

#: Name this seeder registers under. Exported so a test can unregister it and
#: so a user replacing it does not have to guess the string.
SEEDER_NAME = "torch"

#: Environment variable cuBLAS reads to select a deterministic reduction
#: workspace. It is read by the library at its first allocation, so setting it
#: later in a process has no effect -- which is why it is set here, at seeding
#: time, rather than inside the fit.
_CUBLAS_WORKSPACE_VARIABLE = "CUBLAS_WORKSPACE_CONFIG"

#: The configuration cuBLAS requires for deterministic reductions. ``:4096:8``
#: is the larger of the two documented settings; it costs a little more memory
#: than ``:16:8`` and does not restrict the stream count.
_CUBLAS_WORKSPACE_VALUE = ":4096:8"

_LOGGER = get_logger(__name__)


def seed_torch(seed: int, determinism: Determinism = "off") -> None:
    """
    Seed every Torch generator and apply the requested determinism level.

    Parameters
    ----------
    seed
        A non-negative integer below ``2 ** 32``.
    determinism
        ``off`` seeds and nothing more. ``warn`` additionally asks for
        deterministic algorithms but tolerates an operation that has none.
        ``strict`` refuses such an operation, so that a run claiming
        reproducibility has it.

    Raises
    ------
    RuntimeError
        Propagated from Torch under ``strict`` when an operation reached
        during the run has no deterministic implementation. Deliberately not
        caught: ``seed_everything`` turns it into a specification error naming
        this seeder, and the alternative is a run that reports a seed it could
        not honour.
    """
    # `manual_seed` covers the CPU generator and every visible CUDA device, so
    # a per-device loop is unnecessary and would miss a device that appears
    # later.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # Seeds all devices, including ones not yet initialised.
        torch.cuda.manual_seed_all(seed)

    if determinism == "off":
        return

    # cuDNN benchmarks its convolution algorithms on the first call and caches
    # the winner. The benchmark is timing-dependent, so the chosen algorithm --
    # and therefore the arithmetic -- can differ between two runs on the same
    # machine. Turning it off is a throughput cost and a prerequisite for
    # reproducibility.
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    if determinism == "strict":
        # Must be set before cuBLAS allocates its workspace, which happens on
        # the first matrix multiply. Setting it here means seeding is the last
        # moment it can still take effect.
        os.environ.setdefault(_CUBLAS_WORKSPACE_VARIABLE, _CUBLAS_WORKSPACE_VALUE)

    # `warn_only=True` logs and proceeds where no deterministic implementation
    # exists; under `strict` the same situation raises. That is the whole
    # difference between the two levels, and it is a difference in what the
    # user is promised rather than in what the framework attempts.
    torch.use_deterministic_algorithms(True, warn_only=determinism == "warn")
    _LOGGER.debug("seeded torch with %d at determinism=%r", seed, determinism)


# Registered at import, so importing the engine is enough to make
# `seed_everything` cover Torch. The train pipeline imports the engine before
# it seeds, which is the ordering this relies on -- and the ordering is tested,
# because a seeder registered after seeding is a seeder that did nothing.
register_seeder(SEEDER_NAME, seed_torch, replace=True)
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/devices.py`

12856 bytes · SHA-256 `f5fbc3699eba7e1b`

```python
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

from ....core.lifecycle.errors import EngineError
from ....core.provenance.logging import get_logger

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
```

---

## 4. `tranql/models/rade/rade_qnet/rade_qnet/engines/torch/hardware/distributed.py`

9488 bytes · SHA-256 `9addadb978642a89`

```python
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
```

