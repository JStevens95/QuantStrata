# `src/rade_qnet/orchestration/compute`

6 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 67 | 2561 | `4c51e1a4f4256585` |
| 2 | `base.py` | 354 | 11327 | `06675d4d776bb767` |
| 3 | `gpus.py` | 166 | 6856 | `355ebcbfadec27d6` |
| 4 | `local.py` | 68 | 1911 | `875720e4a6d53480` |
| 5 | `policy.py` | 334 | 10831 | `144dd00b30eed227` |
| 6 | `processes.py` | 358 | 14297 | `8ad395c7655c5e83` |

---

## 1. `src/rade_qnet/orchestration/compute/__init__.py`

2561 bytes · SHA-256 `4c51e1a4f4256585`

```python
"""
Where a unit of work executes.

An executor takes a list of work items and runs them.  That is the whole
interface, and its narrowness is what keeps parallelism from leaking into
pipeline logic.  A pipeline cannot tell which executor is running it, so
switching from sequential to eight processes is a configuration change that
cannot alter results -- which parity level 5 verifies by running both and
comparing what they produced.

Modules
-------
``base.py``
    ``WorkItem``, ``WorkResult``, ``WorkFailure`` and the ``Executor``
    protocol.  Failures are returned, not raised, so one bad job cannot abort
    a set; results come back in input order, so a manifest never depends on
    which job happened to finish first.
``local.py``
    In-process sequential execution.  The reference implementation that all
    others must reproduce exactly, and the only sane way to debug.
``processes.py``
    A process pool using the spawn start method, with per-worker thread
    budgets set in the pool initialiser so N workers do not each claim every
    core and collapse throughput through oversubscription.
``gpus.py``
    One worker per visible device, each pinned by device-visibility
    environment variable before the training library is imported -- the only
    point at which such pinning reliably takes effect.
``policy.py``
    Chooses an executor and worker count from the hardware actually present
    and the size of the job set, so placement need not be hand-tuned, and
    records why it chose what it did.

Registration
------------
Nothing here is registered by name.  An executor is constructed by the
placement policy from a specification, not resolved through the component
registry, because there are four of them and they are framework-owned.  A
user supplying their own satisfies the ``Executor`` protocol and passes it to
the job-set runner directly.
"""

from __future__ import annotations

from .base import Executor, ResultSummary, WorkFailure, WorkItem, WorkResult, execute_item
from .gpus import GpuExecutor, visible_device_ids
from .local import LocalExecutor
from .policy import Placement, choose_placement, describe_machine
from .processes import THREAD_VARIABLES, ProcessExecutor, configure_worker

__all__ = [
    "THREAD_VARIABLES",
    "Executor",
    "GpuExecutor",
    "LocalExecutor",
    "Placement",
    "ProcessExecutor",
    "ResultSummary",
    "WorkFailure",
    "WorkItem",
    "WorkResult",
    "choose_placement",
    "configure_worker",
    "describe_machine",
    "execute_item",
    "visible_device_ids",
]
```

---

## 2. `src/rade_qnet/orchestration/compute/base.py`

11327 bytes · SHA-256 `06675d4d776bb767`

```python
"""
The executor protocol, and the values that cross a process boundary.

An executor takes a list of work items and runs them. That is the entire
interface. Its narrowness is the point: a pipeline cannot observe which
executor is running it, so moving from sequential to eight processes is a
configuration change that *cannot* alter results -- and parity level 5
verifies exactly that by running both and comparing what they produced.

Two properties are fixed here rather than left to each implementation,
because a manifest that depended on either would not be reproducible.

**Results come back in input order.** Never in completion order. A parallel
run finishes its jobs in whatever order the scheduler and the data happen to
produce, and a manifest ordered by that would differ between two identical
runs.

**Failures are returned, not raised.** A set of forty clusters where one has
insufficient history returns thirty-nine trained models and one recorded
failure with its reason. Raising would discard the other thirty-nine, which
is the behaviour that makes a framework untrustworthy at scale -- and the
workaround people adopt is running jobs one at a time by hand.

Why a function and a payload rather than a callable
---------------------------------------------------
:class:`WorkItem` holds a *module-level function* and a *payload*, separately,
rather than one opaque zero-argument callable.

Under the spawn start method a closure or a bound method cannot cross the
process boundary, and the error it produces names the pickle protocol rather
than the design mistake. Splitting the item in two turns "is this picklable?"
-- which is unanswerable about a closure -- into two mechanical checks on two
named things, each of which a test can make directly. It also makes the
constraint visible in the type: a reader can see that the work is a function
of its data, and that both halves have to survive serialisation.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ...core.runtime.logging import get_logger

__all__ = [
    "Executor",
    "ResultSummary",
    "WorkFailure",
    "WorkItem",
    "WorkResult",
    "execute_item",
]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class WorkFailure:
    """
    A failure, captured as text at the point it happened.

    Deliberately not an exception object. An exception need not be picklable
    -- a custom ``__init__`` that takes more than a message is enough to
    break it -- and the part that actually matters, the traceback, does not
    survive pickling at all. By the time an exception reaches the parent
    process its traceback has been discarded, so the only way the reason
    reaches a manifest is to render it in the worker.

    Parameters
    ----------
    kind
        The exception's class name. Separate from the message so a set can
        be grouped by failure type -- "nineteen jobs failed, all with
        ``InsufficientHistory``" is a different situation from nineteen
        unrelated failures, and only one of them has a single fix.
    message
        ``str(exception)``.
    traceback_text
        The formatted traceback, for a human.
    """

    kind: str
    message: str
    traceback_text: str

    @classmethod
    def of(cls, error: BaseException) -> WorkFailure:
        """
        Capture an exception.

        Parameters
        ----------
        error
            The exception, caught where it was raised.

        Returns
        -------
        WorkFailure
            The captured failure.
        """
        return cls(
            kind=type(error).__name__,
            message=str(error),
            traceback_text="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )

    def summary(self) -> str:
        """
        Return a one-line description, for a log or a manifest row.

        Returns
        -------
        str
            For example ``SpecError: no job named 'EURUSD'``.
        """
        return f"{self.kind}: {self.message}" if self.message else self.kind


@dataclass(frozen=True, slots=True)
class WorkItem[P, T]:
    """
    One unit of work: a key, a function, and the function's argument.

    Parameters
    ----------
    key
        Identifies this item in results, logs and the manifest. A job
        identifier, in practice.
    function
        A module-level function. Not a closure, a lambda or a bound method
        -- see this module's docstring.
    payload
        The single argument the function is called with. Must be picklable
        for any executor that crosses a process boundary.
    """

    key: str
    function: Callable[[P], T]
    payload: P


@dataclass(frozen=True, slots=True)
class WorkResult[T]:
    """
    What one work item produced, or why it did not.

    Exactly one of :attr:`value` and :attr:`failure` is set, which
    :attr:`succeeded` is the readable way to ask about.

    Parameters
    ----------
    key
        The key of the item that produced this.
    value
        The function's return value, or ``None`` on failure.
    failure
        The captured failure, or ``None`` on success.
    wall_seconds
        How long the item took, measured around the call. Recorded even for
        a failure, because "it failed after four hours" and "it failed
        immediately" call for different responses.
    """

    key: str
    value: T | None = None
    failure: WorkFailure | None = None
    wall_seconds: float = 0.0

    @property
    def succeeded(self) -> bool:
        """
        Whether the item completed.

        Returns
        -------
        bool
            True if a value is present.
        """
        return self.failure is None

    def unwrap(self) -> T:
        """
        Return the value, raising if the item failed.

        For a caller that genuinely cannot continue without the value --
        a single-job run, or a test. A job set does not use it, because
        tolerating failure is the whole point.

        Returns
        -------
        T
            The value.

        Raises
        ------
        RuntimeError
            If the item failed, carrying the worker's traceback as text.
            The original exception cannot be re-raised: it was captured in
            another process and only its rendering survived.
        """
        if self.failure is None:
            # The cast is safe by the invariant above: no failure means a
            # value, and only a function annotated as returning `T | None`
            # could put a genuine None here.
            return self.value  # type: ignore[return-value]
        raise RuntimeError(
            f"work item {self.key!r} failed with {self.failure.summary()}\n"
            f"{self.failure.traceback_text}"
        )


@runtime_checkable
class Executor(Protocol):
    """
    Runs work items somewhere, and returns one result per item.

    A protocol rather than a base class, so a user's own executor -- a Dask
    cluster, a job scheduler, a hosted runner -- substitutes without
    inheriting anything from this framework.

    Implementations must satisfy three rules, each of which a conformance
    test checks:

    1. One result per item, in the **order the items were given**.
    2. A failing item yields a failed :class:`WorkResult`; it does not raise
       and does not prevent other items from running.
    3. Running the same items twice produces the same values. Placement is
       not allowed to change results.
    """

    def map(self, items: Sequence[WorkItem[object, object]]) -> list[WorkResult[object]]:
        """
        Run every item and return the results in input order.

        Parameters
        ----------
        items
            The work to do.

        Returns
        -------
        list of WorkResult
            One result per item, in input order.
        """
        ...

    @property
    def description(self) -> str:
        """
        A short description of where work runs, for logs and the manifest.

        Recorded against a job set so that a set which ran slowly, or
        produced an unexpected result, can be compared against the placement
        it actually used rather than the one that was configured -- those
        differ whenever the policy chose.
        """
        ...


def execute_item[P, T](item: WorkItem[P, T]) -> WorkResult[T]:
    """
    Run one item, capturing timing and any failure.

    The single place an item is actually called. Every executor funnels
    through it, in whichever process the work runs, which is what makes the
    capture rules identical everywhere: the sequential executor cannot
    accidentally let an exception escape where the pool would have caught it,
    because neither of them implements the catching.

    Parameters
    ----------
    item
        The work item.

    Returns
    -------
    WorkResult
        The value or the captured failure, with elapsed wall time.
    """
    started = time.perf_counter()
    try:
        value = item.function(item.payload)
    # Deliberately broad. Anything a user's model can raise is a job failure
    # and must not reach the executor, including the exception types a
    # narrower clause would miss -- which, in a framework that runs arbitrary
    # user code, is all of them.
    except Exception as error:
        elapsed = time.perf_counter() - started
        _LOGGER.warning("work item %s failed after %.2fs: %s", item.key, elapsed, error)
        return WorkResult(key=item.key, failure=WorkFailure.of(error), wall_seconds=elapsed)

    elapsed = time.perf_counter() - started
    _LOGGER.debug("work item %s completed in %.2fs", item.key, elapsed)
    return WorkResult(key=item.key, value=value, wall_seconds=elapsed)


@dataclass(frozen=True, slots=True)
class ResultSummary:
    """
    An aggregate view of a set of results.

    Parameters
    ----------
    results
        Every result, in input order.
    """

    results: tuple[WorkResult[object], ...] = field(default_factory=tuple)

    @property
    def succeeded(self) -> tuple[WorkResult[object], ...]:
        """
        The results that produced a value.

        Returns
        -------
        tuple of WorkResult
            In input order.
        """
        return tuple(result for result in self.results if result.succeeded)

    @property
    def failed(self) -> tuple[WorkResult[object], ...]:
        """
        The results that failed.

        Returns
        -------
        tuple of WorkResult
            In input order.
        """
        return tuple(result for result in self.results if not result.succeeded)

    @property
    def wall_seconds(self) -> float:
        """
        Total time across every item.

        The sum of the items' own durations, not the elapsed time of the
        set: under a pool those differ by the parallel speed-up, and this is
        the figure that is comparable between a sequential and a parallel
        run of the same work.

        Returns
        -------
        float
            Seconds.
        """
        return sum(result.wall_seconds for result in self.results)
```

---

## 3. `src/rade_qnet/orchestration/compute/gpus.py`

6856 bytes · SHA-256 `355ebcbfadec27d6`

```python
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

from ...core.runtime.logging import get_logger
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
```

---

## 4. `src/rade_qnet/orchestration/compute/local.py`

1911 bytes · SHA-256 `875720e4a6d53480`

```python
"""
Sequential, in-process execution.

The reference implementation. Every other executor is held to reproducing it
exactly, which is what parity level 5 asserts, so this one is written for
obviousness rather than for speed: it is a loop.

That is also why it is the right thing to debug with. A failure under a
process pool arrives as text, in a traceback from another process, with no
way to attach a debugger. The same job under this executor fails where it
happened, in the caller's process, with the caller's breakpoints intact --
and because placement cannot change results, a failure here is the same
failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...core.runtime.logging import get_logger
from .base import WorkResult, execute_item

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import WorkItem

__all__ = ["LocalExecutor"]

_LOGGER = get_logger(__name__)


class LocalExecutor:
    """
    Runs every item in the calling process, one after another.

    Holds no state, so one instance may be reused across sets.
    """

    def map(self, items: Sequence[WorkItem[object, object]]) -> list[WorkResult[object]]:
        """
        Run every item in order.

        Parameters
        ----------
        items
            The work to do.

        Returns
        -------
        list of WorkResult
            One result per item, in input order -- which here is also
            completion order, since there is nothing to reorder.
        """
        _LOGGER.info("running %d item(s) sequentially in this process", len(items))
        return [execute_item(item) for item in items]

    @property
    def description(self) -> str:
        """
        Describe where work runs.

        Returns
        -------
        str
            ``local (sequential, in-process)``.
        """
        return "local (sequential, in-process)"
```

---

## 5. `src/rade_qnet/orchestration/compute/policy.py`

10831 bytes · SHA-256 `144dd00b30eed227`

```python
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
:attr:`~rade_qnet.core.spec.jobs.PlacementSpec.memory_per_job_gb` -- and the
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
```

---

## 6. `src/rade_qnet/orchestration/compute/processes.py`

14297 bytes · SHA-256 `8ad395c7655c5e83`

```python
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
```

