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

from ...core.provenance.logging import get_logger

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
