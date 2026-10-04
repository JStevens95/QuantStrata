"""
Tests for the exclusive file lock.

Two things are under test, and they are tested very differently.

The **lock itself** is exercised through its real implementation on whatever
platform the suite is running on: a second process must wait, the lock must
survive an exception inside the block, and a lock file must not need to exist
beforehand. These tests say nothing about *how* the lock is taken, because
that is the half that changes per platform.

The **retry loop** is the half that cannot run here. It is only reached on
Windows, where there is no blocking lock to call. So it is tested directly,
through :func:`~rade_qnet.storage.locking.retry_until_acquired`, with a
callable standing in for the platform call. That is the reason the loop was
factored out of the Windows branch at all: a timeout path that has never been
executed is a timeout path that does not work, and the developer's machine
would never execute it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from src.rade_qnet.core.runtime.errors import BundleError
from src.rade_qnet.storage.locking import exclusive_lock, retry_until_acquired


class TestTheRetryLoop:
    """
    The blocking-lock-from-a-non-blocking-one logic, tested off Windows.

    ``attempt`` stands in for the platform call, so every branch here runs on
    every platform.
    """

    def test_an_uncontended_lock_is_taken_on_the_first_attempt(self, tmp_path):
        """One call, no sleeping, when the lock is free."""
        attempts = []

        def attempt():
            attempts.append(1)

        retry_until_acquired(attempt, subject=tmp_path / "lock")

        assert len(attempts) == 1

    def test_a_contended_lock_is_retried_until_it_frees(self, tmp_path):
        """
        An ``OSError`` means "held elsewhere", so the loop tries again.

        Three failures then a success, which is the normal shape of waiting
        for another worker rather than an error.
        """
        attempts = []

        def attempt():
            attempts.append(1)
            if len(attempts) < 4:
                raise OSError("held elsewhere")

        retry_until_acquired(attempt, subject=tmp_path / "lock", interval=0.0)

        assert len(attempts) == 4

    def test_a_permanently_held_lock_raises_rather_than_hanging(self, tmp_path):
        """
        The deadline is what separates waiting from hanging.

        Without it, a crashed process holding the lock would stop a job set
        forever with no diagnosis.
        """

        def attempt():
            raise OSError("held forever")

        with pytest.raises(BundleError, match="could not acquire the lock"):
            retry_until_acquired(attempt, subject=tmp_path / "lock", timeout=0.0, interval=0.0)

    def test_the_error_names_the_lock_file_and_the_wait(self, tmp_path):
        """
        A message identifying neither is a message nobody can act on.

        "Waited 60s" and "waited 0s" point at different causes, so the
        duration and the attempt count are both reported.
        """

        def attempt():
            raise OSError("held forever")

        with pytest.raises(BundleError) as caught:
            retry_until_acquired(
                attempt, subject=tmp_path / "catalog.lock", timeout=0.0, interval=0.0
            )

        message = str(caught.value)
        assert "catalog.lock" in message
        assert "0.0s" in message
        assert "1 attempt" in message

    def test_a_timeout_of_zero_still_makes_one_attempt(self, tmp_path):
        """
        The deadline is checked after attempting, not before.

        So "is this free right now?" needs no special case, and a zero
        timeout can still succeed.
        """
        attempts = []

        def attempt():
            attempts.append(1)

        retry_until_acquired(attempt, subject=tmp_path / "lock", timeout=0.0)

        assert len(attempts) == 1

    def test_a_non_oserror_is_not_retried(self, tmp_path):
        """
        Only ``OSError`` means contention.

        Retrying anything else would turn a bug -- a typo in the platform
        call, a closed handle -- into a sixty-second hang followed by a
        misleading "could not acquire" message.
        """
        attempts = []

        def attempt():
            attempts.append(1)
            raise ValueError("a real fault")

        with pytest.raises(ValueError, match="a real fault"):
            retry_until_acquired(attempt, subject=tmp_path / "lock", interval=0.0)

        assert len(attempts) == 1


class TestTheLockFile:
    """``exclusive_lock`` as callers use it, on the real platform."""

    def test_the_lock_file_need_not_exist_beforehand(self, tmp_path):
        """
        Taking a lock creates it.

        Which is why the lock is conventionally a file of its own: locking
        must not depend on the data being guarded existing yet, and the first
        run against a fresh catalog has no catalog file.
        """
        path = tmp_path / "catalog.lock"
        assert not path.exists()

        with exclusive_lock(path):
            pass

        assert path.exists()

    def test_taking_a_lock_does_not_truncate_the_file(self, tmp_path):
        """
        Opened for append, so a lock file carrying content keeps it.

        Nothing in the framework stores anything in a lock file, but a lock
        that destroys data is a lock that cannot be pointed at an existing
        file by a future caller.
        """
        path = tmp_path / "catalog.lock"
        path.write_text("existing", encoding="utf-8")

        with exclusive_lock(path):
            pass

        assert path.read_text(encoding="utf-8") == "existing"

    def test_the_lock_is_released_when_the_block_raises(self, tmp_path):
        """
        Release happens in a ``finally``, so an exception still unlocks.

        On POSIX closing the handle would release it anyway; on Windows it
        would not, and a lock outliving its block deadlocks the next worker.
        Tested by taking the lock again afterwards, which is the only
        observable difference.
        """
        path = tmp_path / "catalog.lock"

        with pytest.raises(RuntimeError, match="deliberate"), exclusive_lock(path):
            raise RuntimeError("deliberate")

        with exclusive_lock(path):
            pass

    def test_the_lock_can_be_taken_repeatedly_in_sequence(self, tmp_path):
        """
        Sequential acquisition is the uncontended case, and must not block.

        A lock that is never released passes the single-use test and fails
        here.
        """
        path = tmp_path / "catalog.lock"

        for _ in range(5):
            with exclusive_lock(path):
                pass

    def test_an_unopenable_lock_path_is_reported_as_a_bundle_error(self, tmp_path):
        """
        A directory where a file is expected cannot be opened for append.

        Reported as a ``BundleError`` naming the path rather than surfacing a
        bare ``IsADirectoryError``, so the caller learns which lock failed.
        """
        path = tmp_path / "catalog.lock"
        path.mkdir()

        with pytest.raises(BundleError, match="could not open the lock"), exclusive_lock(path):
            pass


# Held by the child process for long enough that the parent is certainly
# waiting rather than merely slow, and short enough not to lengthen the suite
# noticeably.
_HOLD_SECONDS = 0.5

# What the child process runs. Spawned rather than threaded on purpose: a
# thread in this process would test nothing, because both POSIX and Windows
# locks are held per *process* and a second acquisition from the same one
# either succeeds immediately or deadlocks, neither of which is the behaviour
# under test.
_CHILD = """
import sys, time
from pathlib import Path

sys.path.insert(0, {root!r})
from src.rade_qnet.storage.locking import exclusive_lock

with exclusive_lock(Path({lock!r})):
    print("held", flush=True)
    time.sleep({hold})
"""


class TestMutualExclusion:
    """
    The property the lock exists for: two processes cannot both hold it.

    This is what prevents the defect ``catalog.py`` was written to fix -- two
    workers reserving the same version number, and the second bundle write
    making the first invisible.
    """

    def test_a_second_process_waits_for_the_first_to_finish(self, tmp_path):
        """
        The parent cannot take the lock until the child drops it.

        Measured by elapsed time rather than by observing the lock directly,
        because "was made to wait" is the only externally visible consequence
        of the lock working.
        """
        lock = tmp_path / "catalog.lock"
        root = str(Path(__file__).resolve().parents[3])
        script = _CHILD.format(root=root, lock=str(lock), hold=_HOLD_SECONDS)

        child = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONPATH": root},
        )
        try:
            # Wait for the child to confirm it holds the lock, so the
            # measurement below starts from a known state rather than racing
            # the interpreter's startup.
            assert child.stdout is not None
            assert child.stdout.readline().strip() == "held"

            started = time.monotonic()
            with exclusive_lock(lock):
                waited = time.monotonic() - started
        finally:
            child.wait(timeout=30)

        # A generous fraction of the hold rather than the whole of it: the
        # child's sleep began slightly before the parent started timing, so
        # requiring the full duration would make this flaky for no gain.
        assert waited > _HOLD_SECONDS * 0.5
