"""
An exclusive file lock that works on every platform the framework runs on.

Why this module exists
----------------------
Version assignment in :mod:`rade_qnet.storage.runs.catalog` must be exclusive: two
workers in a job set that both compute ``max(existing) + 1`` are given the
same number, and the second bundle write makes the first invisible. The lock
is what makes that section single-writer.

The obvious implementation, ``fcntl.flock``, is POSIX-only. Importing it at
module scope made the whole catalog -- and therefore
:mod:`rade_qnet.api`, which constructs a catalog -- fail to import on Windows
with ``ModuleNotFoundError: No module named 'fcntl'``. Not a degraded feature:
the library did not load at all.

So the platform-specific call is isolated here, behind two functions, and
every caller takes the lock through :func:`exclusive_lock` without knowing
which one it got.

How each platform locks
-----------------------
POSIX uses ``fcntl.flock`` with ``LOCK_EX``, which blocks in the kernel until
the lock is free. There is nothing to retry and no timeout to choose.

Windows has no ``flock``. ``msvcrt.locking`` locks a byte range rather than a
whole file, and its blocking mode (``LK_LOCK``) gives up after roughly ten
seconds with no way to extend it -- too short for a job set whose workers
queue behind one another. So Windows uses the non-blocking mode
(``LK_NBLCK``) in a retry loop this module controls, with a timeout long
enough for a realistic queue and an error when it is exceeded.

Because the byte range is what is locked, Windows callers lock the same single
byte at offset zero. Two processes agreeing on which byte to lock is what
makes the lock mutually exclusive; the byte's contents are never read.

Testing the part that can go wrong
----------------------------------
The retry loop is the only non-trivial logic here, and it is the half that
cannot run on the developer's machine. So it is factored into
:func:`retry_until_acquired`, which takes the attempt as a callable and is
therefore exercised on every platform -- including its timeout path -- rather
than only on the one where it is used.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import IO, TYPE_CHECKING

from ..core.lifecycle.errors import BundleError
from ..core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = [
    "LOCK_RETRY_INTERVAL_SECONDS",
    "LOCK_TIMEOUT_SECONDS",
    "acquire_exclusive",
    "exclusive_lock",
    "release_exclusive",
    "retry_until_acquired",
]

_LOGGER = get_logger(__name__)

#: How long to keep retrying a contended lock before giving up, in seconds.
#:
#: Generous on purpose. The critical section it guards does one read and one
#: append, so a worker's turn lasts milliseconds; a minute of waiting means
#: something is genuinely wrong -- a crashed process holding the lock, or a
#: filesystem that is not honouring it -- and failing with that diagnosis is
#: more useful than waiting forever.
LOCK_TIMEOUT_SECONDS = 60.0

#: How long to sleep between attempts on a contended lock, in seconds.
#:
#: Short enough that an uncontended handover is imperceptible, long enough
#: that two hundred queued workers do not spin a core between them.
LOCK_RETRY_INTERVAL_SECONDS = 0.01

#: The byte range Windows locks, as an offset and a length.
#:
#: One byte at the start of the file. The region is never read or written --
#: only its being locked matters -- but every process must name the same
#: region, or two of them hold "the lock" simultaneously.
_WINDOWS_LOCK_OFFSET = 0
_WINDOWS_LOCK_BYTES = 1


def retry_until_acquired(
    attempt: Callable[[], None],
    *,
    subject: Path,
    timeout: float = LOCK_TIMEOUT_SECONDS,
    interval: float = LOCK_RETRY_INTERVAL_SECONDS,
) -> None:
    """
    Call ``attempt`` until it succeeds, or raise once ``timeout`` has passed.

    Used to build a blocking lock out of a non-blocking one. Separated from
    the platform call so that this -- the only logic here with a branch in it
    -- is testable on a platform that never runs it.

    The deadline is checked *before* sleeping rather than after, so a timeout
    of zero makes exactly one attempt. That is what lets a caller ask "is this
    lock free right now?" without a special case.

    Parameters
    ----------
    attempt
        Tries to take the lock. Returns ``None`` on success and raises
        :exc:`OSError` when the lock is held elsewhere. Any other exception
        is a real fault and propagates untouched -- retrying it would turn a
        bug into a hang.
    subject
        The lock file, for the error message. Taken as a parameter rather than
        read from ``attempt`` because a closure's target is not inspectable
        and an error naming no file is an error nobody can act on.
    timeout
        How long to keep trying, in seconds.
    interval
        How long to sleep between attempts, in seconds.

    Raises
    ------
    BundleError
        If the lock was still held when the deadline passed. Carries how long
        it waited, because "waited 60s" and "waited 0.01s" point at different
        causes.
    """
    deadline = time.monotonic() + timeout
    attempts = 0
    while True:
        attempts += 1
        try:
            attempt()
        except OSError as error:
            # The lock is held by someone else. Every other failure mode --
            # a bad descriptor, a read-only filesystem -- also arrives as
            # OSError, which is why the deadline exists: a permanent failure
            # retries briefly and then reports, rather than looping forever.
            if time.monotonic() >= deadline:
                raise BundleError(
                    f"could not acquire the lock at {subject} after "
                    f"{timeout:.1f}s and {attempts} attempt(s): {error}"
                ) from error
            time.sleep(interval)
        else:
            if attempts > 1:
                _LOGGER.debug("acquired the lock at %s after %d attempt(s)", subject, attempts)
            return


if sys.platform == "win32":  # pragma: no cover - selected by platform
    import msvcrt

    def acquire_exclusive(handle: IO[str], *, subject: Path) -> None:
        """
        Take an exclusive lock on ``handle``, waiting for it if necessary.

        Parameters
        ----------
        handle
            An open file. Its position is moved to the locked region, so a
            caller that also writes through this handle must seek first --
            which :mod:`rade_qnet.storage.runs.catalog` does not, holding the lock
            on a file separate from the data it guards.
        subject
            The lock file, for the error message.

        Raises
        ------
        BundleError
            If the lock is still held after :data:`LOCK_TIMEOUT_SECONDS`.
        """
        handle.seek(_WINDOWS_LOCK_OFFSET)
        retry_until_acquired(
            lambda: msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, _WINDOWS_LOCK_BYTES),
            subject=subject,
        )

    def release_exclusive(handle: IO[str]) -> None:
        """
        Release the lock held on ``handle``.

        Parameters
        ----------
        handle
            The file locked by :func:`acquire_exclusive`.
        """
        handle.seek(_WINDOWS_LOCK_OFFSET)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, _WINDOWS_LOCK_BYTES)

else:
    import fcntl

    def acquire_exclusive(handle: IO[str], *, subject: Path) -> None:
        """
        Take an exclusive lock on ``handle``, waiting for it if necessary.

        Blocks in the kernel, so there is no polling and no timeout: a worker
        queues until its turn. The ``subject`` argument is unused here and
        kept so the two platform implementations are substitutable.

        Parameters
        ----------
        handle
            An open file.
        subject
            The lock file. Unused: ``flock`` cannot time out, so there is no
            error for it to name.
        """
        del subject
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)

    def release_exclusive(handle: IO[str]) -> None:
        """
        Release the lock held on ``handle``.

        Parameters
        ----------
        handle
            The file locked by :func:`acquire_exclusive`.
        """
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@contextmanager
def exclusive_lock(path: Path) -> Iterator[IO[str]]:
    """
    Hold an exclusive lock on ``path`` for the duration of a block.

    The lock file is opened in append mode so that taking a lock never
    truncates it and never requires it to exist already.

    Parameters
    ----------
    path
        The lock file. Created if absent. Conventionally a file of its own
        rather than the data being guarded, so that locking does not depend on
        the data existing yet.

    Yields
    ------
    IO
        The open lock file, yielded rather than discarded so a reader can see
        that the lock is held for exactly the block's extent.

    Raises
    ------
    BundleError
        If the lock file cannot be opened, or the lock cannot be acquired
        within :data:`LOCK_TIMEOUT_SECONDS`.
    """
    try:
        handle = path.open("a+", encoding="utf-8")
    except OSError as error:
        raise BundleError(f"could not open the lock at {path}: {error}") from error
    try:
        acquire_exclusive(handle, subject=path)
        try:
            yield handle
        finally:
            # Released before the handle closes, in its own `finally`, so that
            # an exception inside the block still unlocks. Closing would
            # release it on POSIX anyway; on Windows it would not, and a lock
            # that outlives its block deadlocks the next worker.
            release_exclusive(handle)
    finally:
        handle.close()
