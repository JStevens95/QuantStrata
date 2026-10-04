"""
One worker per accelerator, each pinned to its own device.

Deliberately thin. :class:`GpuExecutor` is
:class:`~.processes.ProcessExecutor` with the device list filled in and the
worker count tied to it -- it adds no execution machinery of its own.

That is a decision about testability rather than about elegance. Everything
here that could be wrong in an interesting way is the environment variable's
*name*, its *value*, and *when* it is set, and all three can be asserted on a
machine with no accelerator at all, by reading them back from inside a worker.
What genuinely needs a GPU is only the question of whether the library then
honours them, which is one assertion and is skipped rather than omitted.

The alternative -- a separate implementation with its own pool handling,
its own ordering and its own failure capture -- would put the parts that
*cannot* be tested here next to parts that are merely untested, and nobody
would be able to tell afterwards which was which.

Why visibility and not a device index
--------------------------------------
A job could in principle be told "use device two" through
:class:`~rade_qnet.core.spec.hardware.HardwareSpec`. Pinning visibility instead
is better for two reasons.

The first is timing. A GPU library caches its device list when it
initialises, so visibility set afterwards has no effect -- and the symptom is
not an error but every worker quietly sharing device zero, which presents as
a memory problem rather than a configuration one. The pool initialiser runs
before the worker imports anything, which is the only reliable moment.

The second is isolation. A worker that can see one device cannot accidentally
allocate on another, whatever the job's own configuration says. An index is a
request; visibility is a boundary.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

from ...core.provenance.logging import get_logger
from .processes import CUDA_VISIBILITY_VARIABLE, ProcessExecutor

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["GpuExecutor", "visible_device_ids"]

_LOGGER = get_logger(__name__)


def visible_device_ids() -> tuple[int, ...]:
    """
    Return the accelerators this process can see, without importing one.

    Reads ``CUDA_VISIBLE_DEVICES`` rather than asking a training library,
    which matters for two reasons. Importing a GPU library in the *parent*
    process to count devices would initialise a context there, and a parent
    holding a context is the classic cause of a worker that hangs on startup
    -- the exact failure the spawn start method exists to avoid. It also
    keeps this module free of any dependency on a particular engine.

    The cost is that this cannot discover devices nobody has declared. An
    unset variable yields an empty tuple, and the placement policy then
    treats the machine as having no accelerator, which is the safe way round:
    a job set that runs on the CPU when it could have used a GPU is slow, and
    one that assumes four GPUs it does not have fails outright.

    Returns
    -------
    tuple of int
        Device ordinals, in the order they are declared. Empty when none are
        declared, or when the declaration is not a list of ordinals --
        ``CUDA_VISIBLE_DEVICES`` also accepts GPU UUIDs, which are device
        identifiers this function cannot order or count against indices.
    """
    declared = os.environ.get(CUDA_VISIBILITY_VARIABLE, "").strip()
    if not declared:
        return ()

    entries = [entry.strip() for entry in declared.split(",") if entry.strip()]
    if not all(entry.isdigit() for entry in entries):
        _LOGGER.warning(
            "%s is set to %r, which is not a list of ordinals; treating this "
            "machine as having no declared accelerator",
            CUDA_VISIBILITY_VARIABLE,
            declared,
        )
        return ()
    return tuple(int(entry) for entry in entries)


class GpuExecutor(ProcessExecutor):
    """
    A process pool with one worker per device, each pinned to its own.

    Parameters
    ----------
    device_ids
        Which devices to use. ``None`` discovers them with
        :func:`visible_device_ids`.
    workers
        How many processes. ``None`` means one per device, which is the
        normal arrangement. A larger number packs several jobs onto each
        card and is a legitimate choice for small models -- the devices are
        then handed out cyclically.
    threads_per_worker
        Intra-op thread budget per worker. A GPU job still runs its data
        loading and its host-side work on the CPU, so the budget matters
        here for the same reason it does in a CPU pool.
    start_method
        How workers are created. ``spawn`` is required in practice: a forked
        child cannot use a CUDA context inherited from its parent, and the
        failure is a hang rather than an error.
    visibility_variable
        Name of the environment variable to pin visibility with.

    Raises
    ------
    ValueError
        If no devices are given and none can be discovered. Failing here is
        deliberate: silently running a job set on the CPU because the GPUs
        were not visible would turn a four-hour run into a four-day one, and
        the only evidence would be the wall time.
    """

    def __init__(
        self,
        *,
        device_ids: Sequence[int | str] | None = None,
        workers: int | None = None,
        threads_per_worker: int | None = None,
        start_method: Literal["spawn", "forkserver"] = "spawn",
        visibility_variable: str = CUDA_VISIBILITY_VARIABLE,
    ) -> None:
        resolved = tuple(device_ids) if device_ids is not None else visible_device_ids()
        if not resolved:
            message = (
                f"the gpus executor needs at least one device, and none are declared "
                f"in {visibility_variable}. Set it, pass device_ids explicitly, or use "
                f"the processes executor -- falling back to the CPU silently would turn "
                f"a four-hour run into a four-day one with nothing to show why"
            )
            raise ValueError(message)

        super().__init__(
            workers=workers if workers is not None else len(resolved),
            threads_per_worker=threads_per_worker,
            start_method=start_method,
            device_ids=resolved,
            visibility_variable=visibility_variable,
        )

    @property
    def description(self) -> str:
        """
        Describe where work runs.

        Returns
        -------
        str
            For example ``gpus (4 workers over devices 0, 1, 2, 3)``.
        """
        devices = ", ".join(str(device) for device in self.device_ids or ())
        return f"gpus ({self.workers} workers over devices {devices})"
