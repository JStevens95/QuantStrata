# `tranql/models/rade/rade_qnet/rade_qnet/storage`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 67 | 2948 | `5ba06be2a276cadf` |
| 2 | `bundle.py` | 529 | 16121 | `fcaf136ba9d57f84` |
| 3 | `locking.py` | 275 | 10008 | `a6dcb2a486d49d91` |
| 4 | `manifest.py` | 213 | 7319 | `593c50e7d6217c29` |

---

## 1. `tranql/models/rade/rade_qnet/rade_qnet/storage/__init__.py`

2948 bytes · SHA-256 `5ba06be2a276cadf`

```python
"""
The system of record.

Everything a run produces lands here, and nothing else writes to the model
store.  A single writer is what makes concurrent job sets safe: a catalog
updated by eight worker processes through read-modify-write will lose entries,
which is a defect this package exists to prevent.

A bundle is self-describing.  Given only a bundle directory, the framework can
state which model produced it, from which spec, against which data
fingerprint, at which code version, and can rebuild the model and invert its
target transforms without consulting the original run.

One run, or all of them
-----------------------
That is the seam this package is organised on, and it was invisible while
six modules sat flat.  ``bundle.py`` and ``manifest.py`` are about a single
directory: what is in it, whether it is intact, how to write it without ever
leaving a half-finished one that looks valid.  ``runs/`` is about the
collection: which runs happened, which are blessed, which is running now.

The two halves are read by different people at different times.  A pipeline
writes a bundle once and never looks at the index; a quant choosing a model
for Monday queries the index and never opens a bundle by hand.

Modules
-------
``bundle.py``
    Writing and reading a versioned bundle: weights, fitted state, spec,
    signature, metrics and manifest.  Written to a staging directory and
    renamed into place, so a crash can never leave a half-written bundle that
    looks valid.  Parameters are stored, never pickled model objects.
    [Phase 1, delivered]
``manifest.py``
    The manifest schema plus content hashes for every file, making corruption
    and silent drift detectable at load time.  Verification is an explicit step
    rather than automatic, because hashing a large checkpoint to populate a
    listing would make the listing unusable.  [Phase 1, delivered]
``locking.py``
    The exclusive file lock that makes the catalog's single-writer discipline
    real, with one implementation per platform behind one function.  It sits
    at this level rather than inside ``runs/`` because it is infrastructure
    both halves may take, and because ``fcntl`` is POSIX-only: imported from
    the catalog directly, it made the entire library fail to load on Windows
    rather than lose a feature.  [Phase 1, delivered]

Sub-packages
------------
``runs/``
    The index across runs: ``catalog.py`` records every run that happened,
    ``registry.py`` records which ones are blessed and under what tag, and
    ``tracker.py`` follows one that is still going.

Planned modules
---------------
``predictions.py``
    Writing a prediction set alongside its bundle reference, for the inference
    pipeline.  [Phase 5]

Dependency rule
---------------
May import: ``core``.
May not import: ``engines``.  Serialising weights is the engine's job; this
package stores the bytes the engine hands it.
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `tranql/models/rade/rade_qnet/rade_qnet/storage/bundle.py`

16121 bytes · SHA-256 `fcaf136ba9d57f84`

```python
"""
Writing and reading a versioned bundle on disk.

Two disciplines govern this module, and both exist because of specific ways
model stores go wrong in production.

**Writes are atomic.** A bundle is written into a temporary directory beside
its destination and then renamed into place in one filesystem operation. A
crash, a cancellation or a full disk therefore leaves either no bundle or a
complete one -- never a directory containing weights but no manifest, which
looks loadable and is not. The implementation this replaces wrote files
in place, so an interrupted run left behind a bundle that failed at inference
time rather than at write time.

**Weights are serialised by the engine, not here.** This module writes bytes
it is handed. It never imports a training library, which is what the layering
test enforces and what lets the same bundle format hold a network's tensors
and a boosted-tree dump.

A note on pickled modules
-------------------------
The format deliberately stores *parameters*, not objects. Pickling a model
object embeds the class's import path and its source semantics into the file,
so refactoring the class breaks every bundle ever written -- and loading one
with ``weights_only=False`` executes whatever the file says. Writing
parameters plus a signature means a bundle can be loaded after the model's
code has been reorganised, which is the normal course of events over the life
of a model.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel

from ..core.contract.bundle import (
    FITTED_STATE_DIRNAME,
    LINEAGE_FILENAME,
    MANIFEST_FILENAME,
    RESULT_FILENAME,
    SIGNATURE_FILENAME,
    SPEC_FILENAME,
    WEIGHTS_FILENAME,
    Manifest,
    ModelBundle,
    SavedBundle,
)
from ..core.contract.data import DataLineage
from ..core.contract.result import TrainingResult
from ..core.contract.signature import InputSignature, PolicySignature
from ..core.lifecycle.errors import BundleError, SpecError
from ..core.provenance.logging import get_logger
from ..core.spec.run import RunSpec, parse_run_spec
from .manifest import build_manifest, verify_manifest

if TYPE_CHECKING:
    from ..core.contract.state import FittedState

__all__ = [
    "bundle_directory",
    "load_fitted_state",
    "load_lineage",
    "load_manifest",
    "load_policy_signature",
    "load_result",
    "load_signature",
    "load_spec",
    "open_bundle",
    "write_bundle",
]

_LOGGER = get_logger(__name__)

#: Prefix for in-progress bundle directories. Recognisable so a cleanup task
#: can distinguish abandoned partial writes from real bundles.
TEMPORARY_PREFIX = ".writing-"


def bundle_directory(root: Path, *, model_name: str, version: int, job_id: str | None) -> Path:
    """
    Return the canonical directory for one bundle.

    The layout is ``<root>/<model>/<job>/v<N>``, with the job level omitted for
    a single run. Nesting by job rather than flattening to ``model-job-vN``
    keeps a job set's output navigable with ordinary tools when it has three
    hundred members.

    Parameters
    ----------
    root
        The model store root.
    model_name
        Registered model name.
    version
        Version number.
    job_id
        Job identifier, or ``None``.

    Returns
    -------
    Path
        The bundle directory, which may not exist yet.
    """
    base = root / model_name
    if job_id is not None:
        base = base / job_id
    return base / f"v{version}"


def write_bundle(
    bundle: ModelBundle,
    *,
    root: Path,
    version: int,
    framework_version: str,
    engine: str,
    model_name: str,
    write_weights: Callable[[Path], None],
    job_id: str | None = None,
    tags: tuple[str, ...] = (),
) -> SavedBundle:
    """
    Write a bundle atomically and return its location.

    The sequence matters. Everything is written into a temporary directory;
    the manifest is built last, over the files actually present; and only then
    is the directory renamed into place. Building the manifest before the last
    file was written would produce a manifest that does not describe the
    bundle, and renaming before the manifest exists would briefly expose a
    bundle that cannot be verified.

    Parameters
    ----------
    bundle
        The finished run, in memory.
    root
        The model store root.
    version
        Version number, obtained from the catalog so it is unique.
    framework_version
        Version of ``rade_qnet`` writing this bundle.
    engine
        Registered engine name.
    model_name
        Registered model name.
    write_weights
        Callback that writes the model's parameters to the path it is given.
        Supplied by the engine, because only the engine knows how to
        serialise its own model -- and because ``storage`` must not import a
        training library.
    job_id
        Job identifier, or ``None`` for a single run.
    tags
        Free-form labels recorded in the manifest.

    Returns
    -------
    SavedBundle
        The written bundle's directory and manifest.

    Raises
    ------
    BundleError
        If the destination already exists. A version is never overwritten:
        a bundle that has been recorded in the catalog, evaluated, and perhaps
        promoted must not change underneath those decisions.
    """
    destination = bundle_directory(root, model_name=model_name, version=version, job_id=job_id)
    if destination.exists():
        raise BundleError(
            f"refusing to overwrite the existing bundle at {destination}; "
            f"request a new version from the catalog instead"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    # Beside the destination, not in the system temporary directory, so the
    # final rename stays on one filesystem and therefore stays atomic.
    staging = destination.parent / f"{TEMPORARY_PREFIX}{destination.name}-{uuid4().hex[:8]}"
    staging.mkdir(parents=True)

    try:
        _write_payload(bundle, staging, write_weights=write_weights)
        manifest = build_manifest(
            staging,
            model_name=model_name,
            engine=engine,
            version=version,
            framework_version=framework_version,
            spec_digest=bundle.lineage.spec_digest,
            job_id=job_id,
            metrics=dict(bundle.result.headline()),
            tags=tags,
        )
        (staging / MANIFEST_FILENAME).write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        # The one operation that makes the bundle visible. Everything above
        # this line is discardable; everything below it is committed.
        staging.rename(destination)
    except BaseException:
        # Covers KeyboardInterrupt and SystemExit too: an interrupted write
        # should not leave staging directories accumulating in the store.
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _LOGGER.info("wrote bundle %s to %s", manifest.identifier, destination)
    return SavedBundle(directory=destination, manifest=manifest)


def _write_payload(
    bundle: ModelBundle,
    directory: Path,
    *,
    write_weights: Callable[[Path], None],
) -> None:
    """
    Write every part of a bundle except the manifest.

    Separated from :func:`write_bundle` so the atomic-rename logic reads as
    one short sequence, and so the set of files a bundle contains is stated in
    one place.

    Parameters
    ----------
    bundle
        The finished run.
    directory
        The staging directory.
    write_weights
        The engine's parameter-serialisation callback.
    """
    (directory / SPEC_FILENAME).write_text(bundle.spec.model_dump_json(indent=2), encoding="utf-8")
    (directory / SIGNATURE_FILENAME).write_text(
        bundle.signature.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / LINEAGE_FILENAME).write_text(
        bundle.lineage.model_dump_json(indent=2), encoding="utf-8"
    )
    (directory / RESULT_FILENAME).write_text(
        bundle.result.model_dump_json(indent=2), encoding="utf-8"
    )

    state_directory = directory / FITTED_STATE_DIRNAME
    state_directory.mkdir()
    bundle.state.save(state_directory)

    write_weights(directory / WEIGHTS_FILENAME)


def open_bundle(directory: Path, *, verify: bool = True) -> SavedBundle:
    """
    Locate a bundle on disk and read its manifest.

    Parameters
    ----------
    directory
        The bundle directory.
    verify
        Whether to re-hash every file and check it against the manifest.
        Defaults to true: the cost is worth paying by default, and a caller
        that is only listing bundles can opt out explicitly.

    Returns
    -------
    SavedBundle
        The bundle's directory and manifest.

    Raises
    ------
    BundleError
        If the directory or its manifest is missing, unreadable, or -- when
        verifying -- does not match what is on disk.
    """
    manifest = load_manifest(directory)
    if verify:
        verify_manifest(directory, manifest)
    return SavedBundle(directory=directory, manifest=manifest)


def load_manifest(directory: Path) -> Manifest:
    """
    Read a bundle's manifest.

    Parameters
    ----------
    directory
        The bundle directory.

    Returns
    -------
    Manifest
        The manifest.

    Raises
    ------
    BundleError
        If the manifest is missing or invalid.
    """
    return _read_contract(directory / MANIFEST_FILENAME, Manifest)


def load_signature(saved: SavedBundle) -> InputSignature:
    """
    Read a supervised bundle's input signature.

    This is what makes a model rebuildable: the signature plus the spec is
    enough to reconstruct the model object without re-running the data build.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    InputSignature
        The saved signature.

    Raises
    ------
    BundleError
        If the file is missing or invalid -- which includes the case of an
        interactive bundle, whose signature file holds a
        :class:`~rade_qnet.core.contract.signature.PolicySignature` with no
        target. Call :func:`load_policy_signature` for one of those.
    """
    return _read_contract(saved.signature_path, InputSignature)


def load_policy_signature(saved: SavedBundle) -> PolicySignature:
    """
    Read an interactive bundle's policy signature.

    The counterpart of :func:`load_signature`, reading the same file into the
    other member of the union
    :attr:`~rade_qnet.core.contract.bundle.ModelBundle.signature` permits.

    Two functions rather than one that sniffs the file, because the caller
    always knows which it wants: the bundle's spec names the task, and a
    reader that guessed would turn a mismatch -- a supervised pipeline handed
    an interactive bundle -- into a successful parse of the wrong thing
    instead of an error.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    PolicySignature
        The saved observation and action spaces, which together with the spec
        are enough to rebuild the policy with no environment present.

    Raises
    ------
    BundleError
        If the file is missing or invalid, which includes the case of a
        supervised bundle.
    """
    return _read_contract(saved.signature_path, PolicySignature)


def load_lineage(saved: SavedBundle) -> DataLineage:
    """
    Read a bundle's data lineage.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    DataLineage
        Where the training data came from, including the exact split indices.

    Raises
    ------
    BundleError
        If the file is missing or invalid.
    """
    return _read_contract(saved.lineage_path, DataLineage)


def load_result(saved: SavedBundle) -> TrainingResult:
    """
    Read a bundle's training result.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    TrainingResult
        Metrics and training history.

    Raises
    ------
    BundleError
        If the file is missing or invalid.
    """
    return _read_contract(saved.result_path, TrainingResult)


def load_spec(saved: SavedBundle) -> RunSpec:
    """
    Read the run specification a bundle was produced from.

    The counterpart to writing ``spec.json``, which Phase 1 did without
    providing a way to read it back -- invisible until something re-loaded a
    bundle, which nothing did until evaluation existed.

    Validated through the same loader a configuration file goes through, not
    merely parsed as JSON. A bundle that was written by a version whose
    schema has since changed should fail here, naming the field, rather than
    produce a specification object with a field quietly missing.

    Parameters
    ----------
    saved
        The located bundle.

    Returns
    -------
    RunSpec
        The validated specification.

    Raises
    ------
    BundleError
        If the file is missing, unreadable, or no longer validates.
    """
    path = saved.spec_path
    if not path.is_file():
        raise BundleError(f"expected {path.name} in the bundle at {path.parent}, but it is absent")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BundleError(f"could not read {path}: {error}") from error
    try:
        return parse_run_spec(payload, origin=str(path))
    except SpecError as error:
        raise BundleError(f"{path} no longer validates as a run specification: {error}") from error


def load_fitted_state[StateT: FittedState](
    saved: SavedBundle,
    state_type: type[StateT],
) -> StateT:
    """
    Read a bundle's fitted state.

    The state type is supplied by the caller rather than recorded in the
    bundle, deliberately. Recording a class path would mean a bundle names a
    Python import, and renaming the class would break every bundle written
    before the rename -- the exact fragility this format avoids. The model
    definition knows its own state type, so it passes it in.

    Parameters
    ----------
    saved
        The located bundle.
    state_type
        The concrete state class to load into.

    Returns
    -------
    StateT
        The loaded state.

    Raises
    ------
    BundleError
        If the state directory is missing.
    """
    directory = saved.fitted_state_directory
    if not directory.is_dir():
        raise BundleError(
            f"bundle at {saved.directory} has no fitted state directory "
            f"({FITTED_STATE_DIRNAME}/); it may have been written by an "
            f"incompatible version"
        )
    return state_type.load(directory)


def _read_contract[ContractT: BaseModel](path: Path, contract_type: type[ContractT]) -> ContractT:
    """
    Read and validate one JSON contract file.

    One helper for every contract read, so the error message is uniform and
    always names the file -- a validation failure reported without the path is
    nearly useless when a bundle holds five JSON files.

    Parameters
    ----------
    path
        The file to read.
    contract_type
        The pydantic model to validate against.

    Returns
    -------
    ContractT
        The validated contract.

    Raises
    ------
    BundleError
        If the file is missing, unreadable, or fails validation.
    """
    if not path.is_file():
        raise BundleError(f"expected {path.name} in the bundle at {path.parent}, but it is absent")
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as error:
        raise BundleError(f"could not read {path}: {error}") from error
    try:
        return contract_type.model_validate_json(payload)
    except ValueError as error:
        raise BundleError(f"{path} is not a valid {contract_type.__name__}: {error}") from error
```

---

## 3. `tranql/models/rade/rade_qnet/rade_qnet/storage/locking.py`

10008 bytes · SHA-256 `a6dcb2a486d49d91`

```python
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
```

---

## 4. `tranql/models/rade/rade_qnet/rade_qnet/storage/manifest.py`

7319 bytes · SHA-256 `593c50e7d6217c29`

```python
"""
Building and verifying a bundle's file manifest.

The manifest is what turns a directory of files into a bundle that can be
trusted. Every file is listed with its SHA-256 digest and its size, so loading
a bundle can establish that it is the same bundle that was written -- rather
than one that was truncated by a full disk, partially synchronised by a file
share, or edited by someone's debugging session six weeks ago.

Why verification is a separate, explicit step
---------------------------------------------
:func:`verify_manifest` is not called automatically on every read. Hashing a
multi-gigabyte checkpoint takes real time, and paying it to list bundles in a
UI would make the UI unusable. The caller chooses: cheap metadata reads skip
it, and anything that is about to produce predictions does not.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from ..core.contract.bundle import BUNDLE_SCHEMA_VERSION, Manifest, ManifestEntry
from ..core.lifecycle.errors import BundleError
from ..core.provenance.hashing import digest_file
from ..core.provenance.logging import get_logger

__all__ = ["build_manifest", "collect_entries", "verify_manifest"]

_LOGGER = get_logger(__name__)

#: Files excluded from a manifest. The manifest cannot list itself -- its own
#: digest would have to be known before it was written -- and editor or
#: filesystem debris must not invalidate an otherwise sound bundle.
EXCLUDED_NAMES = frozenset({"manifest.json", ".DS_Store", "Thumbs.db"})

#: Directories skipped entirely while walking a bundle.
EXCLUDED_DIRECTORIES = frozenset({"__pycache__", ".ipynb_checkpoints"})


def collect_entries(directory: Path) -> tuple[ManifestEntry, ...]:
    """
    Hash every file beneath a directory.

    Entries are sorted by path, which matters for more than tidiness: a
    manifest's own digest is computed over its JSON, so a stable file order is
    what lets two bundles with identical contents produce identical
    manifests.

    Parameters
    ----------
    directory
        The bundle directory.

    Returns
    -------
    tuple of ManifestEntry
        One entry per file, sorted by relative path.

    Raises
    ------
    BundleError
        If the directory does not exist.
    """
    if not directory.is_dir():
        raise BundleError(f"cannot build a manifest: {directory} is not a directory")

    entries: list[ManifestEntry] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        if path.name in EXCLUDED_NAMES:
            continue
        if EXCLUDED_DIRECTORIES & set(path.relative_to(directory).parts):
            continue
        entries.append(
            ManifestEntry(
                # POSIX form so a bundle written on Windows verifies on Linux.
                relative_path=path.relative_to(directory).as_posix(),
                sha256=digest_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    return tuple(entries)


def build_manifest(
    directory: Path,
    *,
    model_name: str,
    engine: str,
    version: int,
    framework_version: str,
    spec_digest: str,
    job_id: str | None = None,
    metrics: dict[str, float] | None = None,
    tags: tuple[str, ...] = (),
) -> Manifest:
    """
    Build a manifest describing a bundle directory as it currently stands.

    Called after every other file has been written, because it hashes what it
    finds. Writing a file into a bundle afterwards produces a manifest that no
    longer describes it, and :func:`verify_manifest` will say so.

    Parameters
    ----------
    directory
        The bundle directory, fully written.
    model_name
        Registered model name.
    engine
        Registered engine name.
    version
        Version assigned by the catalog.
    framework_version
        Version of ``rade_qnet`` writing this bundle.
    spec_digest
        Digest of the run specification.
    job_id
        Job identifier, or ``None`` for a single run.
    metrics
        Headline metrics, duplicated into the manifest so a catalog listing
        need not open every result file.
    tags
        Free-form labels.

    Returns
    -------
    Manifest
        A manifest listing every file with its digest.
    """
    entries = collect_entries(directory)
    _LOGGER.debug("manifest for %s lists %d file(s)", directory, len(entries))
    return Manifest(
        schema_version=BUNDLE_SCHEMA_VERSION,
        model_name=model_name,
        engine=engine,
        version=version,
        created_at=datetime.now(UTC),
        framework_version=framework_version,
        spec_digest=spec_digest,
        job_id=job_id,
        files=entries,
        metrics=metrics or {},
        tags=tags,
    )


def verify_manifest(directory: Path, manifest: Manifest) -> None:
    """
    Check that a directory still matches its manifest.

    All three failure modes are checked, and they are genuinely different
    problems:

    - A **missing** file means an incomplete or partially copied bundle.
    - A **changed** digest means corruption, or an edit.
    - An **extra** file means something wrote into the bundle after it was
      sealed, which makes the bundle no longer the thing the manifest
      describes -- even though every listed file is intact.

    Every discrepancy is collected before raising, rather than failing on the
    first. Being told that one file of two hundred is wrong, when in fact
    eighty are, sends the reader after the wrong explanation.

    Parameters
    ----------
    directory
        The bundle directory.
    manifest
        The manifest it should match.

    Raises
    ------
    BundleError
        If the directory and the manifest disagree, with every discrepancy
        listed.
    """
    if manifest.schema_version != BUNDLE_SCHEMA_VERSION:
        raise BundleError(
            f"bundle at {directory} has schema version {manifest.schema_version}, "
            f"but this version of rade_qnet reads version {BUNDLE_SCHEMA_VERSION}; "
            f"refusing to load it partially"
        )

    problems: list[str] = []
    for entry in manifest.files:
        path = directory / entry.relative_path
        if not path.is_file():
            problems.append(f"{entry.relative_path}: listed in the manifest but not present")
            continue
        actual_size = path.stat().st_size
        if actual_size != entry.size_bytes:
            problems.append(
                f"{entry.relative_path}: size is {actual_size} bytes, "
                f"manifest records {entry.size_bytes}"
            )
            continue
        actual_digest = digest_file(path)
        if actual_digest != entry.sha256:
            problems.append(
                f"{entry.relative_path}: contents have changed "
                f"(digest {actual_digest[:12]}, manifest records {entry.sha256[:12]})"
            )

    present = {entry.relative_path for entry in collect_entries(directory)}
    for extra in sorted(present - set(manifest.file_paths)):
        problems.append(f"{extra}: present but not listed in the manifest")

    if problems:
        detail = "\n  ".join(problems)
        raise BundleError(f"bundle at {directory} does not match its manifest:\n  {detail}")
```

