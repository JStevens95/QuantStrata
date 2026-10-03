"""
Choosing where a job set runs, when the user has not said.

A sensible choice made unaided is worth more than a hand-tuned one, because a
hand-tuned one gets copied between machines and stops being sensible. So
``placement.executor: auto`` is the default, and this module is what it means.

Every decision is logged with its reason. A set that ran more slowly than
expected is otherwise a mystery: the configuration says ``auto`` and the
output says nothing, so there is no way to tell whether the policy chose
badly or the jobs were simply slow.

What the policy will not do
---------------------------
It does not measure the machine's memory. Capping the worker count by
available memory needs two figures: how much the machine has, and how much a
job will use. The second cannot be known before running the job, and the
first cannot be obtained portably without a dependency. A policy that guessed
either would cap confidently and wrongly.

Instead the per-job figure is an input --
:attr:`~rade_xl.core.spec.jobs.PlacementSpec.memory_per_job_gb` -- and the
machine's total is read only where the platform offers it for free. A user
who knows their job's footprint gets the cap; one who does not is no worse
off than with no policy at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ...core.runtime.logging import get_logger
from .gpus import GpuExecutor, visible_device_ids
from .local import LocalExecutor
from .processes import ProcessExecutor

if TYPE_CHECKING:
    from ...core.spec.jobs import ExecutorName, PlacementSpec
    from .base import Executor

__all__ = ["Placement", "available_memory_gb", "choose_placement", "describe_machine"]

_LOGGER = get_logger(__name__)

#: Bytes in a gibibyte, named so the conversion below is not a magic number.
_BYTES_PER_GIB = 1024**3

#: Upper bound on automatically chosen workers. A pool far wider than this is
#: almost always a mistake on a single machine: the jobs contend for memory
#: and for the filesystem, and the per-worker thread budget falls to one
#: before the worker count is reached. A user who genuinely wants more says
#: so explicitly, and is not capped.
_MAX_AUTOMATIC_WORKERS = 16


@dataclass(frozen=True, slots=True)
class Placement:
    """
    A chosen executor, and why it was chosen.

    The reason is carried alongside the executor rather than only logged,
    so that it can be recorded in the job-set manifest. A set that chose
    its own placement and a set that was told one are not the same
    experiment, and six months later the manifest is the only thing that
    remembers which this was.

    Parameters
    ----------
    executor
        The executor to run with.
    name
        Which kind was chosen, as it would be written in a specification.
    reason
        One sentence explaining the choice, for a log line and the manifest.
    """

    executor: Executor
    name: ExecutorName
    reason: str


def describe_machine() -> str:
    """
    Return a one-line description of the hardware the policy can see.

    Returns
    -------
    str
        For example ``10 core(s), 32.0 GiB, 0 declared accelerator(s)``.
    """
    memory = available_memory_gb()
    rendered = f"{memory:.1f} GiB" if memory is not None else "memory unknown"
    return (
        f"{os.cpu_count() or 1} core(s), {rendered}, "
        f"{len(visible_device_ids())} declared accelerator(s)"
    )


def available_memory_gb() -> float | None:
    """
    Return the machine's total memory, where the platform offers it cheaply.

    Uses ``os.sysconf``, which is present on Linux and macOS and absent on
    Windows. ``None`` is returned rather than a guess when it is
    unavailable, because a wrong figure here produces a confidently wrong
    worker cap -- which is worse than no cap, since no cap at least fails
    visibly and loudly.

    Returns
    -------
    float or None
        Total physical memory in gibibytes, or ``None`` if it cannot be
        determined without a dependency.
    """
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    # ValueError when the name is unrecognised, AttributeError on Windows
    # where `os.sysconf` does not exist at all.
    except (ValueError, OSError, AttributeError):
        return None

    if pages <= 0 or page_size <= 0:
        return None
    return pages * page_size / _BYTES_PER_GIB


def choose_placement(spec: PlacementSpec, *, n_jobs: int) -> Placement:
    """
    Resolve a placement specification into an executor.

    Parameters
    ----------
    spec
        What the user asked for. ``executor='auto'`` delegates the choice
        here; anything else is honoured as written.
    n_jobs
        How many jobs are in the set. A decision input rather than a detail:
        a pool for one job is pure overhead, and a pool wider than the set
        starts workers with nothing to do.

    Returns
    -------
    Placement
        The executor, the name it would be given in a specification, and the
        reason for the choice.
    """
    name = spec.executor if spec.executor != "auto" else _automatic_name(spec, n_jobs=n_jobs)
    workers = _resolve_workers(spec, n_jobs=n_jobs, name=name)
    reason = _reason(spec, name=name, workers=workers, n_jobs=n_jobs)

    placement = Placement(
        executor=_build(spec, name=name, workers=workers), name=name, reason=reason
    )
    _LOGGER.info("placement: %s -- %s", placement.executor.description, reason)
    return placement


def _automatic_name(spec: PlacementSpec, *, n_jobs: int) -> ExecutorName:
    """
    Choose an executor kind for ``auto``.

    Parameters
    ----------
    spec
        The placement specification.
    n_jobs
        How many jobs are in the set.

    Returns
    -------
    ExecutorName
        The chosen kind.
    """
    # One job has nothing to parallelise, and a pool would add a process
    # launch, a pickle round trip and an unusable debugger to a run that
    # gains nothing from any of them.
    if n_jobs <= 1:
        return "local"

    # An explicit worker count of one says the same thing, in the user's own
    # words, so it is honoured rather than second-guessed into a pool of one.
    if spec.workers == 1:
        return "local"

    if len(visible_device_ids()) > 1:
        return "gpus"
    return "processes"


def _resolve_workers(spec: PlacementSpec, *, n_jobs: int, name: ExecutorName) -> int | None:
    """
    Decide the worker count, honouring an explicit one.

    Parameters
    ----------
    spec
        The placement specification.
    n_jobs
        How many jobs are in the set.
    name
        The chosen executor kind.

    Returns
    -------
    int or None
        The worker count, or ``None`` where the executor has no use for one.
    """
    if name == "local":
        return None
    if spec.workers is not None:
        return spec.workers

    if name == "gpus":
        # One per device is the normal arrangement; the executor itself
        # defaults to it, so there is nothing to decide here.
        return None

    cores = os.cpu_count() or 1
    workers = min(cores, n_jobs, _MAX_AUTOMATIC_WORKERS)
    return max(1, _capped_by_memory(workers, memory_per_job_gb=spec.memory_per_job_gb))


def _capped_by_memory(workers: int, *, memory_per_job_gb: float | None) -> int:
    """
    Reduce a worker count to what the machine's memory supports.

    Applied only when the caller supplied a per-job figure, for the reason in
    this module's docstring. A quarter of the machine is held back: a job's
    peak is an estimate, the operating system needs headroom, and a worker
    killed for memory costs its whole run -- so erring low is much cheaper
    than erring high.

    Parameters
    ----------
    workers
        The count before capping.
    memory_per_job_gb
        Estimated peak memory for one job, or ``None`` to skip the cap.

    Returns
    -------
    int
        The capped count, at least one. One worker that might be killed is
        still better than zero workers, which could not run the set at all.
    """
    total = available_memory_gb()
    if memory_per_job_gb is None or total is None:
        return workers

    affordable = int(total * 0.75 / memory_per_job_gb)
    if affordable >= workers:
        return workers

    _LOGGER.info(
        "reducing workers from %d to %d: %.1f GiB of memory at an estimated %.1f GiB per job",
        workers,
        max(1, affordable),
        total,
        memory_per_job_gb,
    )
    return max(1, affordable)


def _build(spec: PlacementSpec, *, name: ExecutorName, workers: int | None) -> Executor:
    """
    Construct the chosen executor.

    Parameters
    ----------
    spec
        The placement specification, for the settings an executor needs.
    name
        Which kind to build.
    workers
        The resolved worker count, or ``None`` to let the executor decide.

    Returns
    -------
    Executor
        The executor.
    """
    if name == "local":
        return LocalExecutor()
    if name == "gpus":
        # The devices are passed in rather than rediscovered by the executor.
        # Two independent observations of the same environment variable can
        # disagree -- the policy counts four devices and decides on a GPU
        # pool, then the executor looks again and finds none -- and the
        # failure lands in the executor, several decisions away from the
        # observation that was actually wrong.
        return GpuExecutor(
            device_ids=visible_device_ids(),
            workers=workers,
            threads_per_worker=spec.threads_per_worker,
            start_method=spec.start_method,
        )
    return ProcessExecutor(
        workers=workers,
        threads_per_worker=spec.threads_per_worker,
        start_method=spec.start_method,
    )


def _reason(spec: PlacementSpec, *, name: ExecutorName, workers: int | None, n_jobs: int) -> str:
    """
    Explain the choice in one sentence.

    Parameters
    ----------
    spec
        The placement specification.
    name
        The chosen kind.
    workers
        The resolved worker count.
    n_jobs
        How many jobs are in the set.

    Returns
    -------
    str
        The explanation, recorded in the manifest.
    """
    if spec.executor != "auto":
        return f"requested explicitly in the specification ({describe_machine()})"
    if name == "local":
        plural = "" if n_jobs == 1 else "s"
        return f"{n_jobs} job{plural}: a pool would cost more than it saves"
    if name == "gpus":
        return f"{len(visible_device_ids())} declared accelerator(s) and {n_jobs} jobs"
    return f"{n_jobs} jobs over {workers} worker(s) ({describe_machine()})"
