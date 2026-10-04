# `src/rade_qnet/storage`

5 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 50 | 2111 | `254809fcd93a9f5a` |
| 2 | `bundle.py` | 493 | 14739 | `aa4fc03feda237ed` |
| 3 | `catalog.py` | 437 | 14956 | `27ca69a90dac9cfb` |
| 4 | `manifest.py` | 213 | 7311 | `aedb0848af6c4401` |
| 5 | `tracker.py` | 186 | 5608 | `49262e26b707ae17` |

---

## 1. `src/rade_qnet/storage/__init__.py`

2111 bytes · SHA-256 `254809fcd93a9f5a`

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

Modules
-------
``manifest.py``
    The manifest schema plus content hashes for every file, making corruption
    and silent drift detectable at load time.  Verification is an explicit step
    rather than automatic, because hashing a large checkpoint to populate a
    listing would make the listing unusable.  [Phase 1, delivered]
``bundle.py``
    Writing and reading a versioned bundle: weights, fitted state, spec,
    signature, metrics and manifest.  Written to a staging directory and
    renamed into place, so a crash can never leave a half-written bundle that
    looks valid.  Parameters are stored, never pickled model objects.
    [Phase 1, delivered]
``catalog.py``
    The index of bundles, queryable by model, job and tag, with a single-writer
    discipline and an append-oriented log rather than whole-file rewrites.
    ``JsonlCatalog`` for real runs, ``InMemoryCatalog`` for tests and
    notebooks.  [Phase 1, delivered]
``tracker.py``
    Experiment tracking behind one interface, with a no-op default.  Tracking
    is optional infrastructure; a run must never fail because a tracking server
    is unreachable.  [Phase 1, delivered]

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

## 2. `src/rade_qnet/storage/bundle.py`

14739 bytes · SHA-256 `aa4fc03feda237ed`

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
from ..core.contract.signature import InputSignature
from ..core.runtime.errors import BundleError, SpecError
from ..core.runtime.logging import get_logger
from ..core.spec.run import RunSpec, parse_run_spec
from .manifest import build_manifest, verify_manifest

if TYPE_CHECKING:
    from ..core.contract.state import FittedState

__all__ = [
    "bundle_directory",
    "load_fitted_state",
    "load_lineage",
    "load_manifest",
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
    Read a bundle's input signature.

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
        If the file is missing or invalid.
    """
    return _read_contract(saved.signature_path, InputSignature)


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
        raise BundleError(
            f"{path} no longer validates as a run specification: {error}"
        ) from error


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

## 3. `src/rade_qnet/storage/catalog.py`

14956 bytes · SHA-256 `27ca69a90dac9cfb`

```python
"""
The index of what has been trained, and where it was put.

This module exists to fix a specific, reproducible defect in the
implementation being replaced. There, assigning a version number meant:

1. read the catalog file,
2. compute ``max(existing_versions) + 1``,
3. write the whole file back.

With one run at a time that works. With a job set of two hundred members
running eight at a time -- which is the normal case, not an edge case -- two
workers read the same state, both compute the same version, and the second
write erases the first worker's entry. The bundle is still on disk; it has
simply become invisible, and nothing reports an error. The failure surfaces
weeks later as "we definitely trained that model".

How this implementation avoids it
---------------------------------
**Version assignment is exclusive.** :meth:`JsonlCatalog.next_version` takes an
``flock`` on a lock file, reserves the number, and releases. Two processes
cannot be inside that section at once, so they cannot be given the same
number.

**Recording is append-only.** An entry is one line appended to a JSON Lines
file under the same lock. There is no read-modify-write of existing content,
so no concurrent writer can overwrite another's entry -- the structural fix,
beyond the lock itself.

**A corrupt line does not destroy the catalog.** Reads skip unparseable lines
with a warning. One torn write (from a process killed mid-append) costs one
entry, not the whole index.

Two caveats worth stating plainly
---------------------------------
``flock`` is advisory and is unreliable on some network filesystems -- notably
NFS without a lock daemon. On a shared filesystem, point the catalog at local
storage, or substitute an implementation backed by a database.

``fcntl`` is POSIX-only, so this module does not import on Windows. Both
limitations are contained by the
:class:`~rade_qnet.core.runtime.context.Catalog` protocol: an alternative
implementation -- including :class:`InMemoryCatalog` below -- substitutes with
no change anywhere else, which is the reason that protocol is declared in
``core`` rather than this class being used directly.
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import IO

from ..core.contract.bundle import Manifest
from ..core.runtime.errors import BundleError
from ..core.runtime.logging import get_logger

__all__ = ["InMemoryCatalog", "JsonlCatalog"]

_LOGGER = get_logger(__name__)

#: The append-only index.
CATALOG_FILENAME = "catalog.jsonl"

#: The lock file guarding version assignment and appends. Separate from the
#: catalog itself so locking never depends on the catalog existing.
LOCK_FILENAME = "catalog.lock"


class JsonlCatalog:
    """
    A single-writer catalog backed by an append-only JSON Lines file.

    Satisfies :class:`~rade_qnet.core.runtime.context.Catalog`.

    Parameters
    ----------
    root
        Directory holding the catalog and its lock. Created if absent.

    Notes
    -----
    Suitable for one machine, including many processes on that machine. For a
    shared store across machines, see the caveat in the module docstring.
    """

    def __init__(self, root: Path) -> None:
        """
        Prepare the catalog directory.

        Parameters
        ----------
        root
            Directory for the catalog and its lock file.
        """
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / CATALOG_FILENAME
        self.lock_path = self.root / LOCK_FILENAME

    @contextmanager
    def _locked(self) -> Iterator[IO[str]]:
        """
        Hold an exclusive lock on the catalog for the duration of a block.

        Yields
        ------
        IO
            The open lock file. Yielded rather than discarded so the caller
            can see that the lock is held for exactly the block's extent.

        Raises
        ------
        BundleError
            If the lock cannot be acquired.
        """
        try:
            handle = self.lock_path.open("a+", encoding="utf-8")
        except OSError as error:
            raise BundleError(
                f"could not open the catalog lock at {self.lock_path}: {error}"
            ) from error
        try:
            # Blocking: a job set's workers should queue for a few
            # milliseconds, not fail. The critical section does no I/O beyond
            # one append, so the wait is bounded in practice.
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            yield handle
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def next_version(self, model_name: str, *, job_id: str | None = None) -> int:
        """
        Reserve the next version number for a model and job, atomically.

        The read and the reservation happen inside one lock, which is the
        whole point: see the module docstring for what happens when they do
        not.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None`` for a single run. Versions are
            numbered per job, so each member of a job set has its own
            sequence starting at one.

        Returns
        -------
        int
            A version number no concurrent caller will be given.
        """
        with self._locked():
            recorded = [
                manifest.version
                for manifest in self._read_all()
                if manifest.model_name == model_name and manifest.job_id == job_id
            ]
            # Reserved numbers count as used, so two callers that reserve
            # before either records still get different numbers. Consulting
            # only recorded bundles would hand out the same version twice
            # whenever a reservation is followed by a long data build.
            reserved = self._reserved_versions(model_name, job_id)
            version = max([*recorded, *reserved], default=0) + 1
            # The marker is appended inside the lock, so the reservation is
            # durable before any other process can read it.
            self._append(
                {
                    "reserved": True,
                    "model_name": model_name,
                    "job_id": job_id,
                    "version": version,
                }
            )
        _LOGGER.debug("reserved version %d for %s/%s", version, model_name, job_id or "-")
        return version

    def record(self, manifest: Manifest) -> None:
        """
        Record a written bundle by appending its manifest.

        Parameters
        ----------
        manifest
            The manifest of a bundle already on disk. Recording a bundle that
            has not been written would leave the catalog pointing at nothing,
            which breaks every reader -- so this is always called last.
        """
        with self._locked():
            self._append(json.loads(manifest.model_dump_json()))
        _LOGGER.info("recorded %s in the catalog", manifest.identifier)

    def latest(self, model_name: str, *, job_id: str | None = None) -> Manifest | None:
        """
        Return the highest-versioned recorded manifest.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.

        Returns
        -------
        Manifest or None
            The latest manifest, or ``None`` if nothing matches.
        """
        matching = [
            manifest
            for manifest in self._read_all()
            if manifest.model_name == model_name and manifest.job_id == job_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda manifest: manifest.version)

    def entries(
        self,
        *,
        model_name: str | None = None,
        job_id: str | None = None,
        tag: str | None = None,
    ) -> Sequence[Manifest]:
        """
        Return recorded manifests, optionally filtered.

        Parameters
        ----------
        model_name
            Restrict to one model.
        job_id
            Restrict to one job.
        tag
            Restrict to manifests carrying a tag.

        Returns
        -------
        Sequence of Manifest
            Matching manifests, ordered by model, job and version.
        """
        manifests = self._read_all()
        if model_name is not None:
            manifests = [m for m in manifests if m.model_name == model_name]
        if job_id is not None:
            manifests = [m for m in manifests if m.job_id == job_id]
        if tag is not None:
            manifests = [m for m in manifests if tag in m.tags]
        return tuple(sorted(manifests, key=lambda m: (m.model_name, m.job_id or "", m.version)))

    def _append(self, payload: dict[str, object]) -> None:
        """
        Append one record. The caller must hold the lock.

        Parameters
        ----------
        payload
            A JSON-encodable record.
        """
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            # Flushed and synced inside the lock so the next holder reads a
            # complete line. Without the fsync, a crash can leave a partial
            # line that the reader must then skip.
            handle.flush()
            os.fsync(handle.fileno())

    def _read_all(self) -> list[Manifest]:
        """
        Read every recorded manifest, skipping reservations and bad lines.

        Returns
        -------
        list of Manifest
            Every valid entry. A line that cannot be parsed is logged and
            skipped rather than raised: one torn write should cost one entry,
            not the entire index.
        """
        if not self.path.is_file():
            return []
        manifests: list[Manifest] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    _LOGGER.warning("skipping unparseable catalog line %d in %s", number, self.path)
                    continue
                if payload.get("reserved"):
                    # A reservation marker, not a bundle. It occupies a version
                    # number so it is read by `next_version`, but it describes
                    # no bundle so it is not a catalog entry.
                    continue
                try:
                    manifests.append(Manifest.model_validate(payload))
                except ValueError:
                    _LOGGER.warning(
                        "skipping invalid catalog entry on line %d in %s", number, self.path
                    )
        return manifests

    def _reserved_versions(self, model_name: str, job_id: str | None) -> list[int]:
        """
        Return version numbers reserved but not yet recorded.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.

        Returns
        -------
        list of int
            Reserved version numbers.
        """
        if not self.path.is_file():
            return []
        reserved: list[int] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if (
                    payload.get("reserved")
                    and payload.get("model_name") == model_name
                    and payload.get("job_id") == job_id
                ):
                    reserved.append(int(payload["version"]))
        return reserved


class InMemoryCatalog:
    """
    A catalog that keeps everything in memory.

    Satisfies :class:`~rade_qnet.core.runtime.context.Catalog`. Used by
    ``rade_qnet.testkit`` so a pipeline test needs no filesystem and no locking,
    and usable in a notebook where nothing should be persisted.

    Not safe across processes, by construction rather than by oversight: a
    single-process catalog that pretended otherwise would be the more
    dangerous object.
    """

    def __init__(self) -> None:
        """Start with an empty catalog."""
        self._manifests: list[Manifest] = []
        self._reserved: dict[tuple[str, str | None], int] = {}

    def next_version(self, model_name: str, *, job_id: str | None = None) -> int:
        """
        Return the next version number for a model and job.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.

        Returns
        -------
        int
            The next version.
        """
        key = (model_name, job_id)
        recorded = [
            manifest.version
            for manifest in self._manifests
            if manifest.model_name == model_name and manifest.job_id == job_id
        ]
        version = max([*recorded, self._reserved.get(key, 0)], default=0) + 1
        self._reserved[key] = version
        return version

    def record(self, manifest: Manifest) -> None:
        """
        Record a manifest.

        Parameters
        ----------
        manifest
            The manifest to record.
        """
        self._manifests.append(manifest)

    def latest(self, model_name: str, *, job_id: str | None = None) -> Manifest | None:
        """
        Return the highest-versioned recorded manifest.

        Parameters
        ----------
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.

        Returns
        -------
        Manifest or None
            The latest manifest, or ``None``.
        """
        matching = [
            manifest
            for manifest in self._manifests
            if manifest.model_name == model_name and manifest.job_id == job_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda manifest: manifest.version)

    def entries(self) -> Sequence[Manifest]:
        """
        Return every recorded manifest.

        Returns
        -------
        Sequence of Manifest
            Recorded manifests, in the order they were recorded.
        """
        return tuple(self._manifests)
```

---

## 4. `src/rade_qnet/storage/manifest.py`

7311 bytes · SHA-256 `aedb0848af6c4401`

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
from ..core.runtime.errors import BundleError
from ..core.runtime.hashing import digest_file
from ..core.runtime.logging import get_logger

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

---

## 5. `src/rade_qnet/storage/tracker.py`

5608 bytes · SHA-256 `49262e26b707ae17`

```python
"""
Experiment tracking, behind one interface, with a no-op default.

The governing rule: **tracking is never load-bearing.** A model that trained
successfully but could not reach a tracking server has still trained
successfully, and the bundle on disk is the system of record -- not the
tracker. So every implementation here swallows its own failures and warns,
and the default implementation does nothing at all.

That default matters more than it sounds. If the framework's baseline
behaviour required a tracking backend, every test, every notebook and every
quick experiment would need one configured. Making
:class:`NullTracker` the default means tracking is something you add, not
something you disable.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from ..core.runtime.hashing import canonical_json
from ..core.runtime.logging import get_logger

__all__ = ["JsonlTracker", "NullTracker"]

_LOGGER = get_logger(__name__)


class NullTracker:
    """
    A tracker that records nothing.

    Satisfies :class:`~rade_qnet.core.runtime.context.Tracker`. The default, so
    that a run needs no tracking infrastructure to proceed.

    Every method is empty rather than raising ``NotImplementedError``: this is
    a working implementation of "do not track", not an incomplete one.
    """

    def log_params(self, params: Mapping[str, object]) -> None:
        """
        Discard the run's configuration.

        Parameters
        ----------
        params
            Flattened specification values.
        """

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Discard metrics.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None``.
        """

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Discard an artifact reference.

        Parameters
        ----------
        path
            The file that was written.
        name
            Logical name.
        """

    def finish(self, *, succeeded: bool) -> None:
        """
        Do nothing on completion.

        Parameters
        ----------
        succeeded
            Whether the run completed.
        """


class JsonlTracker:
    """
    A tracker that appends events to a local JSON Lines file.

    Satisfies :class:`~rade_qnet.core.runtime.context.Tracker`. Useful when a
    hosted tracker is unavailable or inappropriate, and as a reference for
    what an implementation has to do.

    Parameters
    ----------
    path
        File to append to. Its parent directory is created if absent.

    Notes
    -----
    One line per event, so a reader can follow a run in progress with ``tail
    -f`` and a crashed run still leaves everything written before the crash.
    Writes are appends with no locking, which is safe for the one-process-per
    -run case this is intended for; a job set should give each job its own
    file, which is what a per-job
    :class:`~rade_qnet.core.runtime.context.RunContext` naturally produces.
    """

    def __init__(self, path: Path) -> None:
        """
        Prepare the event file.

        Parameters
        ----------
        path
            File to append events to.
        """
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_params(self, params: Mapping[str, object]) -> None:
        """
        Append the run's configuration.

        Parameters
        ----------
        params
            Flattened specification values.
        """
        self._write({"event": "params", "params": json.loads(canonical_json(dict(params)))})

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """
        Append metrics.

        Parameters
        ----------
        metrics
            Metric name to value.
        step
            Epoch or boosting round, or ``None`` for a final metric.
        """
        self._write({"event": "metrics", "step": step, "metrics": dict(metrics)})

    def log_artifact(self, path: Path, *, name: str | None = None) -> None:
        """
        Append an artifact reference.

        Parameters
        ----------
        path
            The file that was written.
        name
            Logical name, defaulting to the file name.
        """
        self._write({"event": "artifact", "name": name or path.name, "path": str(path)})

    def finish(self, *, succeeded: bool) -> None:
        """
        Append the terminal event.

        Parameters
        ----------
        succeeded
            Whether the run completed.
        """
        self._write({"event": "finish", "succeeded": succeeded})

    def _write(self, payload: Mapping[str, object]) -> None:
        """
        Append one event, swallowing I/O failures.

        Parameters
        ----------
        payload
            A JSON-encodable event.
        """
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, separators=(",", ":"), default=str) + "\n")
        except OSError:
            # Caught here as well as in RunContext, because this class may be
            # used directly. See the module docstring: tracking never fails a
            # run, and that has to hold wherever the tracker is called from.
            _LOGGER.warning("could not append a tracking event to %s", self.path, exc_info=True)
```

