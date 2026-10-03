"""
Parallel execution in a pool of worker processes.

Three things break here and nowhere else, and all three work perfectly in a
sequential run. That asymmetry is why each is handled explicitly rather than
discovered.

Picklability
------------
Under the spawn start method a worker is a fresh interpreter, so everything
it needs arrives by pickle. A closure or a bound method cannot make the trip,
and the error names the pickle protocol rather than the design mistake.
:class:`~.base.WorkItem` is split into a module-level function and a payload
for exactly this reason.

Thread oversubscription
-----------------------
Numerical libraries default to one thread per core. Eight workers each
defaulting to every core produces eight times the core count in threads, the
operating system spends its time switching between them, and a parallel run
comes out *slower than sequential* -- with nothing in any log to say why.

The budget has to be set before the library is imported, because the thread
pool is sized at import. The only place in a worker's life that is reliably
before every import is the pool initialiser, which is where this sets it.

Device visibility
-----------------
The same timing argument, more sharply. A GPU library caches the set of
visible devices when it initialises, so ``CUDA_VISIBLE_DEVICES`` set after
import has no effect at all -- and the symptom is not an error but every
worker quietly sharing device zero, which looks like a memory problem rather
than a configuration one.

A pool initialiser receives identical arguments for every worker, so there is
no index to assign a device from. The identifiers are handed out through a
shared queue that each initialiser pops from once.
"""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import TYPE_CHECKING, Literal

from ...core.runtime.logging import get_logger
from .base import WorkFailure, WorkResult, execute_item

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.sharedctypes import Synchronized

    from .base import WorkItem

__all__ = ["THREAD_VARIABLES", "ProcessExecutor", "configure_worker"]

_LOGGER = get_logger(__name__)

#: Environment variables that cap the intra-op thread pools of the numerical
#: libraries this framework sits on. All of them are set to the same figure:
#: a budget that constrained some of the stack and not the rest would be no
#: budget at all, since whichever library was left uncapped would claim the
#: machine on its own.
#:
#: Declared as a constant so the test that reads them back from inside a
#: worker checks the same list the executor sets, rather than a copy of it
#: that can drift.
THREAD_VARIABLES: tuple[str, ...] = (
    "OMP_NUM_THREADS",  # OpenMP: PyTorch's CPU kernels, and much of SciPy.
    "MKL_NUM_THREADS",  # Intel MKL, when NumPy is linked against it.
    "OPENBLAS_NUM_THREADS",  # OpenBLAS, when it is not.
    "NUMEXPR_NUM_THREADS",  # numexpr, reached through pandas.
    "VECLIB_MAXIMUM_THREADS",  # Apple Accelerate.
)

#: Default name of the device-visibility variable. A parameter rather than a
#: hardcoded string so a non-CUDA accelerator can use the same machinery.
CUDA_VISIBILITY_VARIABLE = "CUDA_VISIBLE_DEVICES"


def configure_worker(
    threads: int | None,
    devices: tuple[int | str, ...] | None = None,
    counter: Synchronized | None = None,  # type: ignore[type-arg]
    visibility_variable: str = CUDA_VISIBILITY_VARIABLE,
) -> None:
    """
    Prepare one worker process, before it imports anything heavy.

    Runs as the pool's initialiser, which is the only point in a worker's
    life that is reliably before every import of a training library.

    Module-level, because an initialiser crosses the process boundary under
    the same rules as the work itself.

    Parameters
    ----------
    threads
        Intra-op thread budget, or ``None`` to leave the libraries' defaults
        alone. ``None`` is correct for a single worker and wrong for a pool;
        the executor resolves which applies before calling this.
    devices
        Device identifiers to distribute, or ``None`` when visibility is not
        being managed.
    counter
        A shared counter the workers draw their position from. Every worker
        receives identical initialiser arguments, so this is the only thing
        distinguishing one from another.
    visibility_variable
        Name of the environment variable to pin visibility with.
    """
    if threads is not None:
        for variable in THREAD_VARIABLES:
            os.environ[variable] = str(threads)

    if not devices or counter is None:
        return

    # A shared counter rather than a queue of identifiers. A queue looks like
    # the natural fit and is subtly wrong: `multiprocessing.Queue.put` hands
    # the item to a background feeder thread and returns immediately, so a
    # worker that starts quickly can call `get_nowait` before anything has
    # arrived and find the queue empty. The failure is intermittent, depends
    # on how fast the workers start, and leaves the worker unpinned -- which
    # presents as a memory problem on device zero rather than as a race.
    #
    # The counter is shared memory with a lock, so there is nothing to race.
    # It also handles a replaced worker gracefully: it draws the next
    # position and cycles, rather than finding nothing left.
    with counter.get_lock():
        position = counter.value
        counter.value = position + 1

    device = devices[position % len(devices)]
    os.environ[visibility_variable] = str(device)
    _LOGGER.debug("worker %d pinned to %s=%s", position, visibility_variable, device)


class ProcessExecutor:
    """
    Runs work items across a pool of processes.

    Parameters
    ----------
    workers
        How many processes. Defaults to the machine's core count, capped at
        the number of items -- there is nothing for a surplus worker to do
        but consume memory.
    threads_per_worker
        Intra-op thread budget per worker. ``None`` divides the machine's
        cores between the workers, which is the setting that decides whether
        a parallel run is faster or slower than a sequential one.
    start_method
        ``spawn`` by default and the only one tested. ``fork`` is unsafe in a
        process that has already initialised a threaded numerical library or
        a GPU context, and the resulting hangs are intermittent and very hard
        to attribute to their cause.
    device_ids
        Device identifiers to distribute across the workers, one each, or
        ``None`` to leave visibility alone. Used by
        :class:`~.gpus.GpuExecutor`; see this module's docstring for why the
        distribution goes through a queue.
    visibility_variable
        Name of the environment variable that pins device visibility.
    """

    def __init__(
        self,
        *,
        workers: int | None = None,
        threads_per_worker: int | None = None,
        start_method: Literal["spawn", "forkserver"] = "spawn",
        device_ids: Sequence[int | str] | None = None,
        visibility_variable: str = CUDA_VISIBILITY_VARIABLE,
    ) -> None:
        self.workers = workers
        self.threads_per_worker = threads_per_worker
        self.start_method = start_method
        self.device_ids = tuple(device_ids) if device_ids is not None else None
        self.visibility_variable = visibility_variable

    def map(self, items: Sequence[WorkItem[object, object]]) -> list[WorkResult[object]]:
        """
        Run every item across the pool and return results in input order.

        Parameters
        ----------
        items
            The work to do.

        Returns
        -------
        list of WorkResult
            One result per item, in the order the items were given. Never in
            completion order: a manifest ordered by which job happened to
            finish first would differ between two identical runs.
        """
        if not items:
            return []

        workers = self._resolve_workers(len(items))
        threads = self._resolve_threads(workers)
        context = multiprocessing.get_context(self.start_method)
        # Fresh per call, so a reused executor does not carry one set's
        # device assignment into the next.
        counter = context.Value("i", 0) if self.device_ids else None

        _LOGGER.info(
            "running %d item(s) across %d %s worker(s), %s thread(s) each",
            len(items),
            workers,
            self.start_method,
            threads if threads is not None else "default",
        )

        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=configure_worker,
            initargs=(threads, self.device_ids, counter, self.visibility_variable),
        ) as pool:
            # Submitted in order and collected in the same order, so the
            # result list is positional. `pool.map` would also preserve
            # order, but it re-raises the first exception rather than
            # returning it, which is the opposite of what a job set needs.
            futures = [pool.submit(execute_item, item) for item in items]
            return [
                self._collect(item, future) for item, future in zip(items, futures, strict=True)
            ]

    @property
    def description(self) -> str:
        """
        Describe where work runs.

        Returns
        -------
        str
            For example ``processes (spawn, 4 workers, 2 threads each)``.
        """
        workers = self.workers if self.workers is not None else "auto"
        threads = self.threads_per_worker if self.threads_per_worker is not None else "auto"
        return f"processes ({self.start_method}, {workers} workers, {threads} threads each)"

    @staticmethod
    def _collect(item: WorkItem[object, object], future: object) -> WorkResult[object]:
        """
        Turn one future into a result, surviving a dead worker.

        ``execute_item`` already captures anything the user's code raises, so
        a failure reaching this point means the *worker* died rather than the
        work: an out-of-memory kill, a segmentation fault in a native
        library, an operator terminating the process. Those arrive as a
        broken pool rather than as an exception from the job, and they must
        still be recorded as that job's failure -- a set that crashed outright
        because one job was killed would discard every other job's work.

        Parameters
        ----------
        item
            The item the future was submitted for, for its key.
        future
            The future.

        Returns
        -------
        WorkResult
            The result, or a failure describing the worker's death.
        """
        try:
            return future.result()  # type: ignore[attr-defined, no-any-return]
        except BrokenProcessPool as error:
            _LOGGER.error(
                "the worker running %s died; it was most likely killed for "
                "using too much memory. Reduce 'placement.workers', or set "
                "'placement.memory_per_job_gb' so the policy caps them",
                item.key,
            )
            return WorkResult(key=item.key, failure=WorkFailure.of(error))
        # Deliberately broad, for the same reason as in `execute_item`: the
        # alternative to recording this failure is losing every other result.
        except Exception as error:
            return WorkResult(key=item.key, failure=WorkFailure.of(error))

    def _resolve_workers(self, n_items: int) -> int:
        """
        Decide how many processes to start.

        Parameters
        ----------
        n_items
            How many items there are.

        Returns
        -------
        int
            At least one, never more than the number of items.
        """
        requested = self.workers if self.workers is not None else (os.cpu_count() or 1)
        return max(1, min(requested, n_items))

    def _resolve_threads(self, workers: int) -> int | None:
        """
        Decide each worker's intra-op thread budget.

        Dividing the cores between the workers is the default rather than
        leaving the libraries alone, because "leave it alone" means every
        worker claims every core. One thread each is the floor: a budget of
        zero would be read by some libraries as "use the default", which is
        the behaviour being prevented.

        Parameters
        ----------
        workers
            How many processes will run.

        Returns
        -------
        int or None
            The budget, or ``None`` if one was explicitly not wanted.
        """
        if self.threads_per_worker is not None:
            return self.threads_per_worker
        return max(1, (os.cpu_count() or 1) // workers)

    def device_assignment(self, workers: int) -> tuple[int | str, ...]:
        """
        Return the device each worker position will be pinned to.

        What :func:`configure_worker` computes, exposed so the distribution
        can be checked without starting a pool. Observing it through a
        running pool is not reliable: a pool creates workers on demand, so
        with cheap work the first worker may finish everything before a
        second exists, and the test would be racing the scheduler.

        Parameters
        ----------
        workers
            How many processes will run.

        Returns
        -------
        tuple
            One identifier per worker position. Empty when visibility is not
            being managed.

        Notes
        -----
        Cycled rather than truncated when there are more workers than
        devices. Several small jobs per card is a legitimate configuration
        and is the caller's decision to make, not this method's to refuse --
        but they are packed evenly rather than piled onto device zero.
        """
        if not self.device_ids:
            return ()
        return tuple(self.device_ids[index % len(self.device_ids)] for index in range(workers))
