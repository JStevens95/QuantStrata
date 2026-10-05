# `tests/rade_qnet/storage`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 31 | 1382 | `3ccdb89a57a83475` |
| 2 | `test_storage_bundle.py` | 411 | 14400 | `235c3977d0ba70ca` |
| 3 | `test_storage_locking.py` | 288 | 9850 | `d2ca71c71c19d047` |
| 4 | `test_storage_manifest.py` | 299 | 11217 | `b655369174dc00c0` |

---

## 1. `tests/rade_qnet/storage/__init__.py`

1382 bytes · SHA-256 `3ccdb89a57a83475`

```python
"""
Tests for ``rade_qnet.storage`` -- bundles, catalog and tracking.

Storage is tested for its failure behaviour as much as its success behaviour,
because the failures are silent. An interrupted write must leave no bundle that
looks valid; a concurrent catalog update must lose no entry; a corrupted file
must be detected at load rather than producing quiet nonsense.

Planned modules
---------------
``test_storage_bundle.py``
    Write and read round-trips, and that an interrupted write leaves nothing
    loadable -- verified by writing to a temporary location and failing before
    the rename.  [Phase 1]
``test_storage_manifest.py``
    Manifest schema, content hashes, and a modified file failing verification.
    [Phase 1]
``test_storage_catalog.py``
    Registration, query by model and job, and concurrent writes from several
    processes losing no entry.  [Phase 1]
``test_storage_registry.py``
    Selecting runs by tag, metric and alias; tags and promotions recorded
    without rewriting a bundle; and concurrent decisions from several
    processes losing none.
``test_storage_locking.py``
    The cross-platform lock: retry deadline, lock-file errors, and mutual
    exclusion between processes.
``test_storage_tracker.py``
    The tracking interface, and that an unreachable backend degrades to a
    warning instead of failing a completed run.  [Phase 1]
"""
```

---

## 2. `tests/rade_qnet/storage/test_storage_bundle.py`

14400 bytes · SHA-256 `235c3977d0ba70ca`

```python
"""
Tests for writing and reading a bundle on disk.

Two disciplines are under test, and both exist because of specific ways model
stores go wrong.

Writes are atomic: a crash, a cancellation or a full disk leaves either no
bundle or a complete one, never a directory holding weights but no manifest --
which looks loadable and is not. The implementation being replaced wrote files
in place, so an interrupted run left behind a bundle that failed at inference
time rather than at write time.

And weights are serialised by the engine, not here. ``storage`` never imports
a training library, which is what lets one bundle format hold a network's
tensors and a boosted-tree dump.
"""

from __future__ import annotations

import json
import re
import shutil

import pytest

from src.rade_qnet.core.contract.bundle import (
    FITTED_STATE_DIRNAME,
    MANIFEST_FILENAME,
    SPEC_FILENAME,
    WEIGHTS_FILENAME,
    Manifest,
)
from src.rade_qnet.core.lifecycle.errors import BundleError
from src.rade_qnet.storage.bundle import (
    TEMPORARY_PREFIX,
    bundle_directory,
    load_fitted_state,
    load_lineage,
    load_manifest,
    load_result,
    load_signature,
    open_bundle,
    write_bundle,
)
from src.rade_qnet.testkit.fixtures import StandardisingState, make_model_bundle


def _write_weights(path):
    """
    Stand in for an engine's parameter serialiser.

    Parameters
    ----------
    path
        Where to write.
    """
    path.write_bytes(b"weights")


def _write(root, bundle=None, **overrides):
    """
    Write a bundle with plausible metadata.

    Parameters
    ----------
    root
        The model store root.
    bundle
        The in-memory bundle, defaulting to the standard fixture.
    overrides
        Arguments to override.

    Returns
    -------
    SavedBundle
        The written bundle.
    """
    arguments = {
        "root": root,
        "version": 1,
        "framework_version": "0.1.0",
        "engine": "sklearn",
        "model_name": "demo",
        "write_weights": _write_weights,
    }
    return write_bundle(bundle or make_model_bundle(), **{**arguments, **overrides})


class TestLayout:
    """Where a bundle lives, and why it lives there."""

    def test_a_single_run_omits_the_job_level(self, tmp_path):
        """No empty path segment for a run that has no job."""
        path = bundle_directory(tmp_path, model_name="demo", version=3, job_id=None)
        assert path == tmp_path / "demo" / "v3"

    def test_a_job_run_nests_under_its_job(self, tmp_path):
        """
        Nested rather than flattened to ``model-job-vN``.

        A job set with three hundred members stays navigable with ``ls``,
        which matters more than it sounds when something has gone wrong.
        """
        path = bundle_directory(tmp_path, model_name="demo", version=3, job_id="EURUSD")
        assert path == tmp_path / "demo" / "EURUSD" / "v3"


class TestWriting:
    """What a write produces, and what it refuses to do."""

    def test_every_expected_file_is_present(self, tmp_path):
        """
        The complete set, stated in one place in the writer.

        A bundle missing one of these loads partially, which is worse than
        not loading at all.
        """
        saved = _write(tmp_path)
        present = {entry.relative_path for entry in saved.manifest.files}
        assert present == {
            "spec.json",
            "signature.json",
            "lineage.json",
            "result.json",
            f"{FITTED_STATE_DIRNAME}/standardiser.npy",
            WEIGHTS_FILENAME,
        }

    def test_the_manifest_is_written_but_not_self_listed(self, tmp_path):
        """
        It cannot list itself: its digest would precede its own contents.

        But it must exist on disk, or the bundle cannot be verified at all.
        """
        saved = _write(tmp_path)
        assert (saved.directory / MANIFEST_FILENAME).is_file()
        assert MANIFEST_FILENAME not in saved.manifest.file_paths

    def test_the_engine_supplies_the_weights(self, tmp_path):
        """
        ``storage`` writes the bytes it is handed and nothing more.

        Which is what keeps a training library out of this layer, and what
        the layering test enforces independently.
        """
        saved = _write(tmp_path)
        assert (saved.directory / WEIGHTS_FILENAME).read_bytes() == b"weights"

    def test_headline_metrics_reach_the_manifest(self, tmp_path):
        """So a catalog listing need not open every result file."""
        assert "mae" in _write(tmp_path).manifest.metrics

    def test_an_existing_version_is_never_overwritten(self, tmp_path):
        """
        A version is final once it exists.

        A bundle that has been catalogued, evaluated and perhaps promoted
        must not change underneath those decisions.
        """
        _write(tmp_path)
        with pytest.raises(BundleError, match="refusing to overwrite"):
            _write(tmp_path)

    def test_distinct_versions_coexist(self, tmp_path):
        """The normal case for a model that is retrained."""
        first = _write(tmp_path, version=1)
        second = _write(tmp_path, version=2)
        assert first.directory != second.directory
        assert first.directory.is_dir()

    def test_tags_are_recorded(self, tmp_path):
        """Free-form labels, used for querying a store later."""
        assert _write(tmp_path, tags=("nightly",)).manifest.tags == ("nightly",)


class TestAtomicity:
    """A failed write leaves nothing behind."""

    def test_a_failure_leaves_no_bundle(self, tmp_path):
        """
        Either no bundle or a complete one, never a half-written one.

        A directory holding weights but no manifest looks loadable and is
        not -- the failure then surfaces at inference time, long after the
        run that caused it.
        """

        def exploding(path):
            """Fail partway through writing."""
            raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            _write(tmp_path, write_weights=exploding)
        assert not bundle_directory(tmp_path, model_name="demo", version=1, job_id=None).exists()

    def test_a_failure_leaves_no_staging_directory(self, tmp_path):
        """
        Otherwise abandoned partial writes accumulate in the store.

        Which eventually fills the disk that caused the first failure.
        """

        def exploding(path):
            """Fail partway through writing."""
            raise OSError("disk full")

        with pytest.raises(OSError):
            _write(tmp_path, write_weights=exploding)
        assert not list((tmp_path / "demo").glob(f"{TEMPORARY_PREFIX}*"))

    def test_an_interruption_is_cleaned_up_too(self, tmp_path):
        """
        ``KeyboardInterrupt`` does not derive from ``Exception``.

        Catching only ``Exception`` would mean a cancelled run is exactly the
        case that litters the store, and cancelling runs is common.
        """

        def interrupting(path):
            """Simulate the user pressing control-C mid-write."""
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            _write(tmp_path, write_weights=interrupting)
        assert not list((tmp_path / "demo").glob(f"{TEMPORARY_PREFIX}*"))

    def test_a_retry_after_a_failure_succeeds(self, tmp_path):
        """
        The practical consequence of cleaning up.

        If the failed attempt left anything behind, the retry would hit the
        overwrite guard and the version would be unusable forever.
        """

        def exploding(path):
            """Fail the first attempt."""
            raise OSError("disk full")

        with pytest.raises(OSError):
            _write(tmp_path, write_weights=exploding)
        assert _write(tmp_path).directory.is_dir()


class TestReading:
    """Everything written comes back unchanged."""

    @pytest.fixture
    def saved(self, tmp_path):
        """
        Provide a written bundle.

        Returns
        -------
        SavedBundle
            The located bundle.
        """
        return _write(tmp_path)

    def test_a_written_bundle_verifies_on_open(self, saved):
        """
        Verification is on by default, and must pass on a fresh write.

        A writer and a verifier that disagreed would make every bundle
        unloadable, so this is the single most important test here.
        """
        assert open_bundle(saved.directory).manifest == saved.manifest

    def test_the_signature_round_trips(self, saved):
        """
        Signature plus spec is what rebuilds the model.

        Without the data build, which is the property that makes a
        six-month-old bundle usable rather than archaeological.
        """
        assert load_signature(saved) == make_model_bundle().signature

    def test_the_lineage_round_trips(self, saved):
        """
        Including the exact split indices.

        Which is what lets a saved model be re-evaluated against the data it
        was actually trained against, even after the split logic changed.
        """
        assert load_lineage(saved) == make_model_bundle().lineage

    def test_the_result_round_trips(self, saved):
        """Metrics and the full training history."""
        assert load_result(saved) == make_model_bundle().result

    def test_the_fitted_state_round_trips(self, saved):
        """
        The state that makes the model's output interpretable.

        Loaded into a type the caller names, because recording a class path
        would mean renaming the class breaks every bundle written before.
        """
        assert load_fitted_state(saved, StandardisingState) == make_model_bundle().state

    def test_verification_can_be_skipped(self, saved):
        """
        Verification is a choice, because it is not free.

        Hashing a multi-gigabyte checkpoint to list bundles in a UI would
        make the UI unusable, so the caller decides.
        """
        (saved.directory / SPEC_FILENAME).write_text("tampered", encoding="utf-8")
        assert open_bundle(saved.directory, verify=False) is not None

    def test_verification_catches_tampering(self, saved):
        """And the default is the safe one."""
        (saved.directory / SPEC_FILENAME).write_text("tampered", encoding="utf-8")
        with pytest.raises(BundleError):
            open_bundle(saved.directory)


class TestReadFailures:
    """Every failure names the file, because a bundle holds five of them."""

    def test_a_missing_manifest_is_reported(self, tmp_path):
        """The most common cause is pointing at the wrong directory."""
        with pytest.raises(BundleError, match=re.escape(MANIFEST_FILENAME)):
            load_manifest(tmp_path)

    def test_a_missing_contract_file_names_it(self, tmp_path):
        """
        "Validation failed" without a path is nearly useless here.

        There are five JSON files, and the reader has to know which.
        """
        saved = _write(tmp_path)
        saved.signature_path.unlink()
        with pytest.raises(BundleError, match=r"signature\.json"):
            load_signature(saved)

    def test_malformed_json_names_the_file_and_the_type(self, tmp_path):
        """
        Both, because the type says what the file was supposed to be.

        A truncated signature and a truncated lineage look identical
        otherwise.
        """
        saved = _write(tmp_path)
        saved.result_path.write_text("{not json", encoding="utf-8")
        with pytest.raises(BundleError, match="TrainingResult"):
            load_result(saved)

    def test_valid_json_of_the_wrong_shape_is_rejected(self, tmp_path):
        """
        Parsing is not the same as validating.

        A file that happens to be JSON but is not a result would otherwise
        load as an empty result and report no metrics at all.
        """
        saved = _write(tmp_path)
        saved.result_path.write_text('{"unexpected": 1}', encoding="utf-8")
        with pytest.raises(BundleError, match="not a valid"):
            load_result(saved)

    def test_a_missing_fitted_state_directory_is_reported(self, tmp_path):
        """
        Distinguished from an empty one, and reported as a version problem.

        A bundle written by an incompatible version is the likely cause, and
        saying so saves the reader a filesystem investigation.
        """
        saved = _write(tmp_path)
        shutil.rmtree(saved.fitted_state_directory)
        with pytest.raises(BundleError, match="fitted state"):
            load_fitted_state(saved, StandardisingState)

    def test_a_manifest_from_a_future_layout_is_refused(self, tmp_path):
        """
        Refused outright rather than loaded partially.

        A partially loaded bundle looks like a working model, which is the
        failure this schema version exists to prevent.
        """
        saved = _write(tmp_path)
        future = saved.manifest.model_copy(
            update={"schema_version": saved.manifest.schema_version + 1}
        )
        (saved.directory / MANIFEST_FILENAME).write_text(future.model_dump_json(), encoding="utf-8")
        with pytest.raises(BundleError, match="schema version"):
            open_bundle(saved.directory)


class TestNoPickling:
    """The format stores parameters, never objects."""

    def test_the_stored_spec_is_readable_json(self, tmp_path):
        """
        Not a pickle, so it survives the model's code being reorganised.

        Pickling embeds the class's import path and source semantics into the
        file, so a refactor breaks every bundle ever written -- and loading
        one executes whatever the file says.
        """
        saved = _write(tmp_path)
        assert saved.spec_path.read_text(encoding="utf-8").lstrip().startswith("{")

    def test_the_manifest_is_readable_without_the_framework(self, tmp_path):
        """
        Plain JSON, so a bundle can be inspected by anything.

        Which matters when the thing inspecting it is a monitoring script
        that has no reason to install a training library.
        """
        saved = _write(tmp_path)
        payload = json.loads((saved.directory / MANIFEST_FILENAME).read_text(encoding="utf-8"))
        assert payload["model_name"] == "demo"
        assert Manifest.model_validate(payload) == saved.manifest
```

---

## 3. `tests/rade_qnet/storage/test_storage_locking.py`

9850 bytes · SHA-256 `d2ca71c71c19d047`

```python
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

from src.rade_qnet.core.lifecycle.errors import BundleError
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
```

---

## 4. `tests/rade_qnet/storage/test_storage_manifest.py`

11217 bytes · SHA-256 `b655369174dc00c0`

```python
"""
Tests for manifest construction and verification.

A manifest is what turns a directory of files into a bundle that can be
trusted. The failures it catches -- a file truncated by a full disk, a bundle
half-copied off a file share, an edit from someone's debugging session six
weeks ago -- all produce a model that loads and predicts, just not the model
anyone believes it is.

Verification is deliberately not automatic, so these tests also pin the three
discrepancy kinds apart. "Missing", "changed" and "extra" are different
problems with different causes, and collapsing them into one message sends the
reader after the wrong explanation.
"""

from __future__ import annotations

import pytest

from src.rade_qnet.core.contract.bundle import BUNDLE_SCHEMA_VERSION
from src.rade_qnet.core.lifecycle.errors import BundleError
from src.rade_qnet.storage.manifest import (
    EXCLUDED_DIRECTORIES,
    EXCLUDED_NAMES,
    build_manifest,
    collect_entries,
    verify_manifest,
)


def _populate(directory, files):
    """
    Write files into a directory, creating parents as needed.

    Parameters
    ----------
    directory
        Target directory.
    files
        Mapping of relative POSIX path to text content.

    Returns
    -------
    pathlib.Path
        The directory.
    """
    for name, content in files.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return directory


@pytest.fixture
def bundle_directory(tmp_path):
    """
    Provide a small written-out bundle directory.

    Returns
    -------
    pathlib.Path
        A directory with two files, one of them nested.
    """
    return _populate(tmp_path, {"spec.json": "{}", "fitted_state/scaler.npy": "data"})


def _manifest_for(directory, **overrides):
    """
    Build a manifest for a directory with plausible metadata.

    Parameters
    ----------
    directory
        The bundle directory.
    overrides
        Fields to override.

    Returns
    -------
    Manifest
        The manifest.
    """
    fields = {
        "model_name": "demo",
        "engine": "sklearn",
        "version": 1,
        "framework_version": "0.1.0",
        "spec_digest": "deadbeef",
    }
    return build_manifest(directory, **{**fields, **overrides})


class TestCollectingEntries:
    """What goes into a manifest, and what deliberately does not."""

    def test_every_file_is_listed(self, bundle_directory):
        """Including nested ones, since a fitted state is a directory."""
        paths = [entry.relative_path for entry in collect_entries(bundle_directory)]
        assert paths == ["fitted_state/scaler.npy", "spec.json"]

    def test_paths_are_posix_form(self, bundle_directory):
        """
        So a bundle written on Windows verifies on Linux.

        A backslash in a recorded path would make every nested file appear
        missing on the other platform.
        """
        assert all("\\" not in entry.relative_path for entry in collect_entries(bundle_directory))

    def test_entries_are_sorted_by_path(self, tmp_path):
        """
        Not in filesystem order, which is arbitrary.

        The manifest's own digest is computed over its JSON, so a stable
        order is what lets two bundles with identical contents produce
        identical manifests.
        """
        _populate(tmp_path, {"z.json": "1", "a.json": "2", "m.json": "3"})
        paths = [entry.relative_path for entry in collect_entries(tmp_path)]
        assert paths == sorted(paths)

    def test_the_manifest_itself_is_excluded(self, bundle_directory):
        """
        It cannot list itself.

        Its own digest would have to be known before it was written.
        """
        _populate(bundle_directory, {"manifest.json": "{}"})
        assert "manifest.json" not in [e.relative_path for e in collect_entries(bundle_directory)]

    @pytest.mark.parametrize("debris", sorted(EXCLUDED_NAMES - {"manifest.json"}))
    def test_filesystem_debris_is_excluded(self, bundle_directory, debris):
        """
        Opening a bundle directory in Finder must not invalidate it.

        A bundle that fails verification because someone looked at it would
        train everyone to ignore verification failures.
        """
        _populate(bundle_directory, {debris: "junk"})
        assert debris not in [e.relative_path for e in collect_entries(bundle_directory)]

    @pytest.mark.parametrize("ignored", sorted(EXCLUDED_DIRECTORIES))
    def test_excluded_directories_are_skipped_entirely(self, bundle_directory, ignored):
        """Their contents are debris too, however they are named."""
        _populate(bundle_directory, {f"{ignored}/cached": "junk"})
        paths = [entry.relative_path for entry in collect_entries(bundle_directory)]
        assert all(ignored not in path for path in paths)

    def test_the_recorded_size_matches_the_file(self, tmp_path):
        """
        Recorded so a truncated file is detectable without hashing it.

        Which makes the common corruption case cheap to catch.
        """
        _populate(tmp_path, {"a.txt": "12345"})
        assert collect_entries(tmp_path)[0].size_bytes == 5

    def test_an_empty_directory_yields_no_entries(self, tmp_path):
        """Empty is a valid state, not an error."""
        assert collect_entries(tmp_path) == ()

    def test_a_missing_directory_is_rejected(self, tmp_path):
        """
        Distinguished from an empty one.

        An empty manifest for a path that does not exist would verify
        happily and describe nothing.
        """
        with pytest.raises(BundleError, match="not a directory"):
            collect_entries(tmp_path / "absent")

    def test_identical_contents_produce_identical_entries(self, tmp_path):
        """
        Two bundles written from the same data are comparable.

        Which is what makes "did anything actually change?" answerable.
        """
        first = _populate(tmp_path / "one", {"a.txt": "x", "b.txt": "y"})
        second = _populate(tmp_path / "two", {"b.txt": "y", "a.txt": "x"})
        assert collect_entries(first) == collect_entries(second)


class TestBuildingAManifest:
    """Metadata, plus whatever is on disk at the moment it is called."""

    def test_the_current_schema_version_is_recorded(self, bundle_directory):
        """So a future reader can refuse a layout it does not understand."""
        assert _manifest_for(bundle_directory).schema_version == BUNDLE_SCHEMA_VERSION

    def test_every_file_present_is_listed(self, bundle_directory):
        """
        It hashes what it finds, which is why it is called last.

        A file written afterwards produces a manifest that no longer
        describes the bundle.
        """
        assert len(_manifest_for(bundle_directory).files) == 2

    def test_headline_metrics_are_carried(self, bundle_directory):
        """
        So a catalog listing need not open every result file.

        Listing a hundred bundles should not cost a hundred extra reads.
        """
        manifest = _manifest_for(bundle_directory, metrics={"mae": 0.25})
        assert manifest.metrics["mae"] == 0.25

    def test_a_freshly_built_manifest_verifies(self, bundle_directory):
        """The round trip, which everything else here is a variation on."""
        verify_manifest(bundle_directory, _manifest_for(bundle_directory))


class TestVerification:
    """Three discrepancy kinds, kept distinct."""

    def test_a_missing_file_is_reported(self, bundle_directory):
        """An incomplete or partially copied bundle."""
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").unlink()
        with pytest.raises(BundleError, match="not present"):
            verify_manifest(bundle_directory, manifest)

    def test_changed_contents_are_reported(self, bundle_directory):
        """
        Corruption, or an edit.

        The case the digests exist for: the file is the right size and the
        right name, and the wrong model.
        """
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").write_text("{}" + " ", encoding="utf-8")
        with pytest.raises(BundleError):
            verify_manifest(bundle_directory, manifest)

    def test_contents_changed_without_a_size_change_are_still_caught(self, bundle_directory):
        """
        The size check is a shortcut, not the guarantee.

        A single flipped byte keeps the size identical, so the digest has to
        be what finally decides.
        """
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").write_text("[]", encoding="utf-8")
        with pytest.raises(BundleError, match="contents have changed"):
            verify_manifest(bundle_directory, manifest)

    def test_an_extra_file_is_reported(self, bundle_directory):
        """
        Something wrote into the bundle after it was sealed.

        Every listed file is intact, and the bundle is still no longer the
        thing the manifest describes.
        """
        manifest = _manifest_for(bundle_directory)
        _populate(bundle_directory, {"stray.txt": "written later"})
        with pytest.raises(BundleError, match="not listed"):
            verify_manifest(bundle_directory, manifest)

    def test_every_discrepancy_is_collected_before_raising(self, bundle_directory):
        """
        All three discrepancy kinds appear in one message.

        Being told one file of two hundred is wrong, when eighty are, sends
        the reader after the wrong explanation entirely.
        """
        manifest = _manifest_for(bundle_directory)
        (bundle_directory / "spec.json").unlink()
        (bundle_directory / "fitted_state" / "scaler.npy").write_text("x", encoding="utf-8")
        _populate(bundle_directory, {"stray.txt": "later"})
        with pytest.raises(BundleError) as caught:
            verify_manifest(bundle_directory, manifest)
        message = str(caught.value)
        assert "spec.json" in message
        assert "scaler.npy" in message
        assert "stray.txt" in message

    def test_an_unknown_schema_version_is_refused_outright(self, bundle_directory):
        """
        Checked before any file is read.

        Loading a future layout partially is worse than refusing it, because
        a partially loaded bundle looks like a working model.
        """
        manifest = _manifest_for(bundle_directory).model_copy(
            update={"schema_version": BUNDLE_SCHEMA_VERSION + 1}
        )
        with pytest.raises(BundleError, match="schema version"):
            verify_manifest(bundle_directory, manifest)

    def test_debris_added_after_sealing_does_not_fail_verification(self, bundle_directory):
        """
        The exclusion list has to apply on both sides.

        Otherwise a bundle verified on a Mac fails because Finder wrote a
        ``.DS_Store`` into it, and the exclusion would be pointless.
        """
        manifest = _manifest_for(bundle_directory)
        _populate(bundle_directory, {".DS_Store": "junk"})
        verify_manifest(bundle_directory, manifest)
```

