# `src/rade_qnet/storage`

7 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 61 | 2748 | `39b4f86a55beec16` |
| 2 | `bundle.py` | 491 | 14717 | `2bcc86c5342517ed` |
| 3 | `catalog.py` | 664 | 22110 | `49cd75d16f6407da` |
| 4 | `locking.py` | 275 | 9993 | `e4de3cbef37a2525` |
| 5 | `manifest.py` | 213 | 7311 | `aedb0848af6c4401` |
| 6 | `registry.py` | 806 | 26755 | `bd900ec7476ad023` |
| 7 | `tracker.py` | 186 | 5608 | `49262e26b707ae17` |

---

## 1. `src/rade_qnet/storage/__init__.py`

2748 bytes · SHA-256 `39b4f86a55beec16`

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
``registry.py``
    ``RunRegistry``: choose a trained run by tag, by best metric or by an
    alias such as ``production``, and record promotions and tags added after
    training as append-only events beside the catalog -- never by rewriting a
    bundle.  [Delivered after Phase 6]
``locking.py``
    The exclusive file lock that makes the catalog's single-writer discipline
    real, with one implementation per platform behind one function.  Separate
    from ``catalog.py`` because ``fcntl`` is POSIX-only: imported there, it
    made the entire library fail to load on Windows rather than lose a
    feature.  [Phase 1, delivered]
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

14717 bytes · SHA-256 `2bcc86c5342517ed`

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

## 3. `src/rade_qnet/storage/catalog.py`

22110 bytes · SHA-256 `49cd75d16f6407da`

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

One caveat worth stating plainly
--------------------------------
A file lock is advisory and is unreliable on some network filesystems --
notably NFS without a lock daemon. On a shared filesystem, point the catalog
at local storage, or substitute an implementation backed by a database. That
limitation is contained by the
:class:`~rade_qnet.core.runtime.context.Catalog` protocol: an alternative
implementation -- including :class:`InMemoryCatalog` below -- substitutes with
no change anywhere else, which is the reason that protocol is declared in
``core`` rather than this class being used directly.

Where each bundle lives
-----------------------
An entry records the bundle's directory alongside its manifest. A reader
selecting a run -- by tag, by alias, by best metric, through
:mod:`rade_qnet.storage.registry` -- needs to open it, and cannot derive the
directory: a single run and a job set lay bundles out differently, and the
catalog may sit somewhere else entirely. The location is stored relative to
the catalog when the bundle is beneath it, so a model store copied to another
machine, or from macOS to Windows, still resolves; an entry written before
locations were recorded reads back with none.

The lock itself lives in :mod:`rade_qnet.storage.locking`, which is what keeps
this module importable on Windows: ``fcntl`` is POSIX-only, and importing it
here once made the whole library -- including :mod:`rade_qnet.api`, which
constructs a catalog -- fail to import rather than merely lose a feature.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from ..core.contract.bundle import Manifest
from ..core.runtime.logging import get_logger
from .locking import exclusive_lock

__all__ = ["CatalogEntry", "InMemoryCatalog", "JsonlCatalog"]

_LOGGER = get_logger(__name__)

#: The append-only index.
CATALOG_FILENAME = "catalog.jsonl"

#: The lock file guarding version assignment and appends. Separate from the
#: catalog itself so locking never depends on the catalog existing.
LOCK_FILENAME = "catalog.lock"

#: The key an entry's bundle directory is stored under, beside the manifest's
#: own fields. Stripped before the manifest is validated, because a manifest
#: is a strict contract and a location is a fact about the store, not the
#: bundle -- the same bundle copied elsewhere is the same bundle.
LOCATION_KEY = "location"


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """
    One recorded bundle: what it is, and where it is.

    Parameters
    ----------
    manifest
        The bundle's manifest, exactly as written.
    location
        The bundle's directory, or ``None`` when the entry was recorded
        without one -- including every entry written before locations were
        recorded.
    """

    manifest: Manifest
    location: Path | None = None


def _sort_key(entry: CatalogEntry) -> tuple[str, str, int]:
    """
    Order entries by model, then job, then version.

    Parameters
    ----------
    entry
        A catalog entry.

    Returns
    -------
    tuple
        The sort key. A single run's ``None`` job sorts before any named job.
    """
    manifest = entry.manifest
    return (manifest.model_name, manifest.job_id or "", manifest.version)


def _filtered(
    entries: Iterable[CatalogEntry],
    *,
    model_name: str | None,
    job_id: str | None,
    tag: str | None,
) -> tuple[CatalogEntry, ...]:
    """
    Apply the shared filters, so both catalogs answer a query identically.

    Parameters
    ----------
    entries
        Every entry.
    model_name
        Restrict to one model, or ``None`` for all.
    job_id
        Restrict to one job, or ``None`` for all.
    tag
        Restrict to entries whose manifest carries this tag, or ``None``.

    Returns
    -------
    tuple of CatalogEntry
        Matching entries, ordered by model, job and version.
    """
    matching = [
        entry
        for entry in entries
        if (model_name is None or entry.manifest.model_name == model_name)
        and (job_id is None or entry.manifest.job_id == job_id)
        and (tag is None or tag in entry.manifest.tags)
    ]
    return tuple(sorted(matching, key=_sort_key))


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

        Blocking, by design: a job set's workers should queue for a few
        milliseconds, not fail. The critical section does no I/O beyond one
        append, so the wait is bounded in practice.

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
        with exclusive_lock(self.lock_path) as handle:
            yield handle

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

    def record(self, manifest: Manifest, *, location: Path | None = None) -> None:
        """
        Record a written bundle by appending its manifest.

        Parameters
        ----------
        manifest
            The manifest of a bundle already on disk. Recording a bundle that
            has not been written would leave the catalog pointing at nothing,
            which breaks every reader -- so this is always called last.
        location
            The bundle's directory. Stored relative to the catalog when it
            is beneath it, so the store can be moved as a whole.
        """
        payload = json.loads(manifest.model_dump_json())
        if location is not None:
            payload[LOCATION_KEY] = self._portable(Path(location))
        with self._locked():
            self._append(payload)
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

    def records(
        self,
        *,
        model_name: str | None = None,
        job_id: str | None = None,
        tag: str | None = None,
    ) -> Sequence[CatalogEntry]:
        """
        Return recorded entries -- manifest and location -- optionally filtered.

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
        Sequence of CatalogEntry
            Matching entries, ordered by model, job and version.
        """
        return _filtered(self._read_entries(), model_name=model_name, job_id=job_id, tag=tag)

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
        records = self.records(model_name=model_name, job_id=job_id, tag=tag)
        return tuple(entry.manifest for entry in records)

    def _portable(self, location: Path) -> str:
        """
        Return a location in the form it is stored: relative where possible.

        Parameters
        ----------
        location
            The bundle directory.

        Returns
        -------
        str
            The path relative to the catalog root when the bundle is beneath
            it, otherwise absolute -- in POSIX form either way, which every
            platform's ``Path`` reads back correctly.
        """
        absolute = location.resolve()
        try:
            return absolute.relative_to(self.root.resolve()).as_posix()
        except ValueError:
            # Outside the store -- or, on Windows, on another drive, where no
            # relative path exists at all.
            return absolute.as_posix()

    def _resolved(self, stored: object) -> Path | None:
        """
        Turn a stored location back into a usable path.

        Parameters
        ----------
        stored
            The value read from the catalog line, or ``None``.

        Returns
        -------
        Path or None
            An absolute path, or ``None`` if the entry has no location.
        """
        if not isinstance(stored, str) or not stored:
            return None
        path = Path(stored)
        return path if path.is_absolute() else self.root / path

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
            Every valid entry's manifest.
        """
        return [entry.manifest for entry in self._read_entries()]

    def _read_entries(self) -> list[CatalogEntry]:
        """
        Read every recorded entry, skipping reservations and bad lines.

        Returns
        -------
        list of CatalogEntry
            Every valid entry, in the order recorded. A line that cannot be
            parsed is logged and skipped rather than raised: one torn write
            should cost one entry, not the entire index.
        """
        if not self.path.is_file():
            return []
        entries: list[CatalogEntry] = []
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
                location = self._resolved(payload.pop(LOCATION_KEY, None))
                try:
                    manifest = Manifest.model_validate(payload)
                except ValueError:
                    _LOGGER.warning(
                        "skipping invalid catalog entry on line %d in %s", number, self.path
                    )
                    continue
                entries.append(CatalogEntry(manifest=manifest, location=location))
        return entries

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
        self._entries: list[CatalogEntry] = []
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
            for manifest in self._manifests()
            if manifest.model_name == model_name and manifest.job_id == job_id
        ]
        version = max([*recorded, self._reserved.get(key, 0)], default=0) + 1
        self._reserved[key] = version
        return version

    def record(self, manifest: Manifest, *, location: Path | None = None) -> None:
        """
        Record a manifest.

        Parameters
        ----------
        manifest
            The manifest to record.
        location
            The bundle's directory, kept as given.
        """
        stored = Path(location) if location is not None else None
        self._entries.append(CatalogEntry(manifest=manifest, location=stored))

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
            for manifest in self._manifests()
            if manifest.model_name == model_name and manifest.job_id == job_id
        ]
        if not matching:
            return None
        return max(matching, key=lambda manifest: manifest.version)

    def records(
        self,
        *,
        model_name: str | None = None,
        job_id: str | None = None,
        tag: str | None = None,
    ) -> Sequence[CatalogEntry]:
        """
        Return recorded entries, filtered exactly as :class:`JsonlCatalog` does.

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
        Sequence of CatalogEntry
            Matching entries, ordered by model, job and version.
        """
        return _filtered(self._entries, model_name=model_name, job_id=job_id, tag=tag)

    def entries(
        self,
        *,
        model_name: str | None = None,
        job_id: str | None = None,
        tag: str | None = None,
    ) -> Sequence[Manifest]:
        """
        Return recorded manifests, filtered exactly as :class:`JsonlCatalog` does.

        The same signature and the same ordering as the persistent catalog,
        so a test written against this one exercises the query a production
        caller makes rather than a simpler one that happens to pass.

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
        records = self.records(model_name=model_name, job_id=job_id, tag=tag)
        return tuple(entry.manifest for entry in records)

    def _manifests(self) -> list[Manifest]:
        """
        Return every recorded manifest, in recording order.

        Returns
        -------
        list of Manifest
            The manifests.
        """
        return [entry.manifest for entry in self._entries]
```

---

## 4. `src/rade_qnet/storage/locking.py`

9993 bytes · SHA-256 `e4de3cbef37a2525`

```python
"""
An exclusive file lock that works on every platform the framework runs on.

Why this module exists
----------------------
Version assignment in :mod:`rade_qnet.storage.catalog` must be exclusive: two
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

from ..core.runtime.errors import BundleError
from ..core.runtime.logging import get_logger

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
            which :mod:`rade_qnet.storage.catalog` does not, holding the lock
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

## 5. `src/rade_qnet/storage/manifest.py`

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

## 6. `src/rade_qnet/storage/registry.py`

26755 bytes · SHA-256 `bd900ec7476ad023`

```python
"""
Choosing a trained run: by tag, by metric, or by a name a person gave it.

Training produces many runs -- a hyperparameter sweep, a nightly retrain, a
group set of forty members -- and the questions that follow are always the
same. *Which runs belong to this experiment? Which was best on the test set?
Which one is in production, and since when?* The catalog can list what was
trained; this module answers those questions, and records the answers people
give to the last one.

Two kinds of label
------------------
**Tags** describe what a run *was*: ``"lr-sweep"``, ``"baseline"``,
``"data_fingerprint:3fa2..."``. Most are set at training time, through the run
specification's ``tags`` field, and are written into the bundle's manifest. A
tag can also be added afterwards -- ``"reviewed"``, ``"rejected"`` -- because
what a run turned out to be is often only known later. A run carries any
number of tags, and many runs share a tag.

**Aliases** say what a run is *for*: ``"production"``, ``"champion"``,
``"candidate"``. An alias points at exactly one version of a model within a
job, and moving it to another version is a promotion. Scoped per job because
in a group set each group has its own production model; a single alias across
forty groups could only ever point at one of them.

Why nothing in a bundle is rewritten
------------------------------------
A bundle is immutable and its manifest is verified by hash; editing one to
add a tag would make the record of what was trained depend on what was
decided later. So tags added afterwards and every alias move are *events*,
appended to a log beside the catalog under the same cross-process lock the
catalog uses. Replaying the log gives the current state; the log itself is the
audit trail -- who promoted what, and when -- which a mutable
``tag -> version`` index cannot give and which a production review will ask
for.

The alternative this replaces read an index file, changed it and wrote it
back. Two people promoting at once -- or a scheduled job promoting while
someone tags by hand -- lost one of the two updates, the same failure the
catalog's append-only design exists to prevent.

What is deliberately absent
---------------------------
**Deleting a run.** Removing bundles is a retention policy -- how long, which
ones, who may -- and a one-line ``delete`` makes the irreversible case the
easy one. Untagging and moving aliases cover the everyday need to stop a run
being selected; reclaiming disk is a separate, deliberate act on the bundle
directories.

**Guessing a metric's direction.** :meth:`RunRegistry.best` requires one,
for the reason recorded at :data:`~rade_qnet.core.spec.tune.Direction`.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ..core.runtime.errors import BundleError, SpecError
from ..core.runtime.logging import get_logger
from .catalog import CatalogEntry, JsonlCatalog
from .locking import exclusive_lock

if TYPE_CHECKING:
    from ..core.contract.bundle import Manifest
    from ..core.spec.tune import Direction

__all__ = ["RegisteredRun", "RegistryEvent", "RunRegistry"]

_LOGGER = get_logger(__name__)

#: The append-only log of tags added afterwards and alias moves.
REGISTRY_FILENAME = "registry.jsonl"

#: Its lock, separate from the catalog's so selecting a run never waits on a
#: training run recording one.
REGISTRY_LOCK_FILENAME = "registry.lock"

#: Not storable as an alias: ``latest`` is computed from the catalog, and an
#: alias that shadowed it would make ``get(alias="latest")`` mean two things.
RESERVED_ALIASES = frozenset({"latest"})

#: The shape of a version reference, which an alias may not imitate.
_VERSION_PATTERN = re.compile(r"^v\d+$")

#: The four things that can happen to a label.
EventKind = Literal["tag", "untag", "promote", "demote"]


@dataclass(frozen=True, slots=True)
class RegistryEvent:
    """
    One recorded decision about a run.

    Parameters
    ----------
    kind
        ``"tag"`` or ``"untag"`` for a tag added or removed afterwards;
        ``"promote"`` or ``"demote"`` for an alias set or cleared.
    label
        The tag or alias.
    model_name
        Registered model name.
    job_id
        Job identifier, or ``None`` for a single run.
    version
        The version affected, or ``None`` for a demotion, which clears an
        alias wherever it points.
    at
        When the decision was recorded, in UTC.
    """

    kind: EventKind
    label: str
    model_name: str
    job_id: str | None
    version: int | None
    at: datetime

    def describe(self) -> str:
        """
        Return the event as one line for a log or a review.

        Returns
        -------
        str
            For example ``"2026-10-04T09:12:00+00:00 promote production ->
            ridge/north/v3"``.
        """
        scope = "/".join(part for part in (self.model_name, self.job_id) if part)
        target = f"{scope}/v{self.version}" if self.version is not None else scope
        arrow = "->" if self.kind in ("tag", "promote") else "from"
        return f"{self.at.isoformat()} {self.kind} {self.label} {arrow} {target}"


@dataclass(frozen=True, slots=True)
class RegisteredRun:
    """
    A trained run as the registry sees it: the bundle, plus what was decided.

    Parameters
    ----------
    manifest
        The bundle's manifest, exactly as written at training time.
    location
        The bundle's directory, or ``None`` if the catalog entry predates
        locations being recorded.
    added_tags
        Tags added after training, in the order they were added.
    aliases
        Aliases currently pointing at this run, sorted.
    """

    manifest: Manifest
    location: Path | None = None
    added_tags: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()

    @property
    def model_name(self) -> str:
        """The registered model name."""
        return self.manifest.model_name

    @property
    def job_id(self) -> str | None:
        """The job identifier, or ``None`` for a single run."""
        return self.manifest.job_id

    @property
    def version(self) -> int:
        """The version within the model and job."""
        return self.manifest.version

    @property
    def identifier(self) -> str:
        """``model/job/vN``, or ``model/vN`` for a single run."""
        return self.manifest.identifier

    @property
    def created_at(self) -> datetime:
        """When the bundle was written."""
        return self.manifest.created_at

    @property
    def metrics(self) -> dict[str, float]:
        """The headline metrics recorded in the manifest."""
        return dict(self.manifest.metrics)

    @property
    def tags(self) -> tuple[str, ...]:
        """Every tag: those set at training time, then those added since."""
        return tuple(dict.fromkeys((*self.manifest.tags, *self.added_tags)))

    @property
    def directory(self) -> Path:
        """
        The bundle directory, for passing to ``api.evaluate`` or ``api.infer``.

        Returns
        -------
        Path
            The directory.

        Raises
        ------
        BundleError
            If the catalog entry has no location, or the directory is gone.
            Raised here rather than returned as ``None`` so the caller learns
            why at the point of use rather than from a later ``TypeError``.
        """
        if self.location is None:
            raise BundleError(
                f"{self.identifier} was recorded without a location; it predates "
                f"locations being recorded, so open it by path instead"
            )
        if not self.location.is_dir():
            raise BundleError(
                f"{self.identifier} was recorded at {self.location}, which no longer exists"
            )
        return self.location

    def metric(self, name: str) -> float | None:
        """
        Return one headline metric.

        Parameters
        ----------
        name
            The metric name, such as ``"mae"``.

        Returns
        -------
        float or None
            The value, or ``None`` if the run did not record it.
        """
        value = self.manifest.metrics.get(name)
        return float(value) if value is not None else None

    def describe(self) -> str:
        """
        Return the run as one line, for a listing.

        Returns
        -------
        str
            For example ``"ridge/north/v3 [production] tags: sweep, reviewed"``.
        """
        parts = [self.identifier]
        if self.aliases:
            parts.append(f"[{', '.join(self.aliases)}]")
        if self.tags:
            parts.append(f"tags: {', '.join(self.tags)}")
        return " ".join(parts)


class RunRegistry:
    """
    Select trained runs by tag, metric or alias, and record promotions.

    Reads the catalog a training run wrote, so it needs no separate
    registration step: anything trained into ``root`` is already here.

    Parameters
    ----------
    root
        The catalog directory -- the ``catalog_root`` training used, which
        defaults to its ``output_root``.

    Examples
    --------
    ::

        registry = RunRegistry("artifacts/eod")
        sweep = registry.runs(model="ridge", tags=["lr-sweep"])
        best = registry.best("mae", direction="minimise", tags=["lr-sweep"])
        registry.promote(best, "production")
        result = api.evaluate(registry.get("ridge", alias="production").directory)
    """

    def __init__(self, root: Path | str) -> None:
        """
        Open the registry over an existing or new catalog directory.

        Parameters
        ----------
        root
            The catalog directory. Created if absent, as the catalog does.
        """
        self.catalog = JsonlCatalog(Path(root))
        self.root = self.catalog.root
        self.path = self.root / REGISTRY_FILENAME
        self.lock_path = self.root / REGISTRY_LOCK_FILENAME

    def runs(
        self,
        *,
        model: str | None = None,
        job: str | None = None,
        tags: Iterable[str] = (),
    ) -> tuple[RegisteredRun, ...]:
        """
        Return recorded runs, filtered.

        Parameters
        ----------
        model
            Restrict to one model.
        job
            Restrict to one job.
        tags
            Restrict to runs carrying *every* one of these tags, whether set
            at training time or added since. All rather than any, because
            narrowing -- "the sweep, on this data snapshot" -- is the common
            question.

        Returns
        -------
        tuple of RegisteredRun
            Matching runs, ordered by model, job and version.
        """
        required = tuple(tags)
        state = self._replay()
        selected = [
            state.decorate(entry) for entry in self.catalog.records(model_name=model, job_id=job)
        ]
        return tuple(run for run in selected if all(tag in run.tags for tag in required))

    def get(
        self,
        model: str,
        *,
        job: str | None = None,
        version: int | None = None,
        alias: str | None = None,
    ) -> RegisteredRun:
        """
        Return one run of a model: a version, an alias, or the latest.

        Parameters
        ----------
        model
            Registered model name.
        job
            Job identifier, or ``None`` for a single run.
        version
            A version number. Mutually exclusive with ``alias``.
        alias
            An alias such as ``"production"``, or ``"latest"`` for the
            highest version. Mutually exclusive with ``version``. With
            neither, the latest is returned.

        Returns
        -------
        RegisteredRun
            The run.

        Raises
        ------
        SpecError
            If both ``version`` and ``alias`` are given.
        BundleError
            If no such run exists, naming what does exist.
        """
        if version is not None and alias is not None:
            raise SpecError("pass a version or an alias, not both")
        candidates = self._scope(model, job)
        scope = _scope_name(model, job)
        if not candidates:
            raise BundleError(f"nothing is recorded for {scope} in {self.root}")

        if alias is not None and alias not in RESERVED_ALIASES:
            state = self._replay()
            pointed = state.aliases.get((model, job, alias))
            if pointed is None:
                known = sorted(name for (m, j, name) in state.aliases if (m, j) == (model, job))
                raise BundleError(f"{scope} has no alias {alias!r}; aliases set: {known or 'none'}")
            version = pointed

        if version is None:
            return candidates[-1]
        for run in candidates:
            if run.version == version:
                return run
        raise BundleError(
            f"{scope} has no version {version}; versions: {[run.version for run in candidates]}"
        )

    def best(
        self,
        metric: str,
        *,
        direction: Direction,
        model: str | None = None,
        job: str | None = None,
        tags: Iterable[str] = (),
    ) -> RegisteredRun:
        """
        Return the run with the best value of one headline metric.

        Parameters
        ----------
        metric
            The metric name, as recorded in the manifest (``"mae"``, ``"r2"``).
        direction
            ``"minimise"`` or ``"maximise"``. Required, never inferred.
        model, job, tags
            Filters, as for :meth:`runs`.

        Returns
        -------
        RegisteredRun
            The best run. Ties go to the most recent, which is usually the
            one somebody meant.

        Raises
        ------
        BundleError
            If no run matches the filters, or none of those that do recorded
            the metric -- naming the metrics they did record, since a
            misspelt name is the usual cause.
        """
        required = tuple(tags)
        candidates = self.runs(model=model, job=job, tags=required)
        if not candidates:
            raise BundleError(f"no run matches model={model!r}, job={job!r}, tags={list(required)}")
        scored = [(run.metric(metric), run) for run in candidates]
        with_metric = [(value, run) for value, run in scored if value is not None]
        if not with_metric:
            recorded = sorted({name for run in candidates for name in run.metrics})
            raise BundleError(
                f"none of the {len(candidates)} matching run(s) recorded {metric!r}; "
                f"recorded metrics: {recorded}"
            )
        sign = 1.0 if direction == "minimise" else -1.0
        # Sorted on (score, -recency) so the first element is the best, with
        # the newest winning a tie.
        ranked = sorted(
            with_metric,
            key=lambda pair: (sign * pair[0], -pair[1].created_at.timestamp(), -pair[1].version),
        )
        return ranked[0][1]

    def promote(self, run: RegisteredRun, alias: str) -> None:
        """
        Point an alias at a run, moving it from wherever it pointed before.

        Parameters
        ----------
        run
            The run to promote, as returned by this registry.
        alias
            The alias, such as ``"production"``.

        Raises
        ------
        SpecError
            If the alias is reserved, looks like a version, or is not a
            single word.
        BundleError
            If the run is not recorded in this registry's catalog.
        """
        _check_label(alias, kind="alias")
        if alias in RESERVED_ALIASES or _VERSION_PATTERN.match(alias):
            raise SpecError(f"{alias!r} cannot be an alias: it would shadow a version reference")
        self._require_recorded(run)
        self._append("promote", alias, run.model_name, run.job_id, run.version)
        _LOGGER.info("promoted %s to %r", run.identifier, alias)

    def demote(self, model: str, alias: str, *, job: str | None = None) -> None:
        """
        Clear an alias, so it points at nothing.

        Parameters
        ----------
        model
            Registered model name.
        alias
            The alias to clear.
        job
            Job identifier, or ``None`` for a single run.

        Raises
        ------
        BundleError
            If the alias is not currently set -- clearing one that is not
            set is almost always a typo, and silently succeeding would hide
            it.
        """
        if (model, job, alias) not in self._replay().aliases:
            raise BundleError(f"{_scope_name(model, job)} has no alias {alias!r} to clear")
        self._append("demote", alias, model, job, None)
        _LOGGER.info("cleared alias %r on %s", alias, _scope_name(model, job))

    def tag(self, run: RegisteredRun, tag: str) -> None:
        """
        Add a tag to a run after training.

        Parameters
        ----------
        run
            The run to tag.
        tag
            The tag. Adding one the run already carries is a no-op.

        Raises
        ------
        SpecError
            If the tag is not a single word.
        BundleError
            If the run is not recorded in this registry's catalog.
        """
        _check_label(tag, kind="tag")
        # Re-read rather than trusting `run`, which may be stale: another
        # process may have tagged it since it was fetched.
        current = self.get(run.model_name, job=run.job_id, version=run.version)
        if tag in current.tags:
            return
        self._append("tag", tag, run.model_name, run.job_id, run.version)

    def untag(self, run: RegisteredRun, tag: str) -> None:
        """
        Remove a tag that was added after training.

        Parameters
        ----------
        run
            The run.
        tag
            The tag to remove.

        Raises
        ------
        BundleError
            If the tag was set at training time -- it is part of the bundle's
            record and cannot be withdrawn -- or the run does not carry it.
        """
        current = self.get(run.model_name, job=run.job_id, version=run.version)
        if tag in current.manifest.tags:
            raise BundleError(
                f"{run.identifier} was trained with tag {tag!r}; a training-time tag "
                f"is part of the bundle's record and cannot be removed"
            )
        if tag not in current.added_tags:
            raise BundleError(f"{run.identifier} has no tag {tag!r} to remove")
        self._append("untag", tag, run.model_name, run.job_id, run.version)

    def history(
        self, *, model: str | None = None, job: str | None = None
    ) -> tuple[RegistryEvent, ...]:
        """
        Return every recorded decision, oldest first.

        Parameters
        ----------
        model
            Restrict to one model.
        job
            Restrict to one job.

        Returns
        -------
        tuple of RegistryEvent
            The events, in the order they were recorded.
        """
        return tuple(
            event
            for event in self._read_events()
            if (model is None or event.model_name == model) and (job is None or event.job_id == job)
        )

    def _scope(self, model: str, job: str | None) -> list[RegisteredRun]:
        """
        Return every run of one model and job, oldest version first.

        Exact on ``job``, unlike :meth:`runs`: ``None`` here means *the single
        run*, not *any job*, because a version number only means something
        within one job.

        Parameters
        ----------
        model
            Registered model name.
        job
            Job identifier, or ``None`` for a single run.

        Returns
        -------
        list of RegisteredRun
            The runs.
        """
        return [run for run in self.runs(model=model, job=job) if run.job_id == job]

    def _require_recorded(self, run: RegisteredRun) -> None:
        """
        Refuse a run that is not in this registry's catalog.

        Parameters
        ----------
        run
            The run.

        Raises
        ------
        BundleError
            If it is absent -- typically a run taken from another registry,
            whose alias would otherwise point at nothing here.
        """
        self.get(run.model_name, job=run.job_id, version=run.version)

    def _append(
        self,
        kind: EventKind,
        label: str,
        model_name: str,
        job_id: str | None,
        version: int | None,
    ) -> None:
        """
        Append one event under the registry lock.

        Parameters
        ----------
        kind
            What happened.
        label
            The tag or alias.
        model_name
            Registered model name.
        job_id
            Job identifier, or ``None``.
        version
            The version, or ``None`` for a demotion.
        """
        payload = {
            "kind": kind,
            "label": label,
            "model_name": model_name,
            "job_id": job_id,
            "version": version,
            "at": datetime.now(UTC).isoformat(),
        }
        with exclusive_lock(self.lock_path), self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
            # Synced inside the lock, as the catalog does, so the next holder
            # never reads a half-written line.
            handle.flush()
            os.fsync(handle.fileno())

    def _read_events(self) -> list[RegistryEvent]:
        """
        Read every event, skipping lines that cannot be parsed.

        Returns
        -------
        list of RegistryEvent
            The events, in file order. A torn line costs that one decision,
            logged, rather than the whole history.
        """
        if not self.path.is_file():
            return []
        events: list[RegistryEvent] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    events.append(
                        RegistryEvent(
                            kind=payload["kind"],
                            label=str(payload["label"]),
                            model_name=str(payload["model_name"]),
                            job_id=payload.get("job_id"),
                            version=payload.get("version"),
                            at=datetime.fromisoformat(payload["at"]),
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    _LOGGER.warning("skipping unreadable registry line %d in %s", number, self.path)
        return events

    def _replay(self) -> _RegistryState:
        """
        Fold the event log into the current tags and aliases.

        Returns
        -------
        _RegistryState
            Who carries which added tag, and where each alias points.
        """
        state = _RegistryState()
        for event in self._read_events():
            state.apply(event)
        return state


@dataclass
class _RegistryState:
    """
    The registry's current state, as the event log leaves it.

    Parameters
    ----------
    added_tags
        ``(model, job, version)`` to the tags added afterwards, in order.
    aliases
        ``(model, job, alias)`` to the version it points at.
    """

    added_tags: dict[tuple[str, str | None, int], list[str]] = field(default_factory=dict)
    aliases: dict[tuple[str, str | None, str], int] = field(default_factory=dict)

    def apply(self, event: RegistryEvent) -> None:
        """
        Apply one event.

        Parameters
        ----------
        event
            The event, applied in log order.
        """
        if event.kind in ("tag", "untag") and event.version is not None:
            tags = self.added_tags.setdefault((event.model_name, event.job_id, event.version), [])
            if event.kind == "tag" and event.label not in tags:
                tags.append(event.label)
            elif event.kind == "untag" and event.label in tags:
                tags.remove(event.label)
        elif event.kind == "promote" and event.version is not None:
            self.aliases[(event.model_name, event.job_id, event.label)] = event.version
        elif event.kind == "demote":
            self.aliases.pop((event.model_name, event.job_id, event.label), None)

    def decorate(self, entry: CatalogEntry) -> RegisteredRun:
        """
        Combine a catalog entry with the decisions recorded about it.

        Parameters
        ----------
        entry
            The catalog entry.

        Returns
        -------
        RegisteredRun
            The run, with its added tags and current aliases.
        """
        manifest = entry.manifest
        key = (manifest.model_name, manifest.job_id)
        aliases = sorted(
            alias
            for (model, job, alias), version in self.aliases.items()
            if (model, job) == key and version == manifest.version
        )
        return RegisteredRun(
            manifest=manifest,
            location=entry.location,
            added_tags=tuple(self.added_tags.get((*key, manifest.version), ())),
            aliases=tuple(aliases),
        )


def _check_label(value: str, *, kind: str) -> None:
    """
    Refuse a tag or alias that would be awkward to type or ambiguous to read.

    Parameters
    ----------
    value
        The proposed label.
    kind
        ``"tag"`` or ``"alias"``, for the message.

    Raises
    ------
    SpecError
        If it is empty, contains whitespace, or contains ``/`` -- which
        separates the parts of a run identifier.
    """
    if not value or any(character.isspace() for character in value) or "/" in value:
        raise SpecError(f"{kind} {value!r} must be a non-empty word without spaces or '/'")


def _scope_name(model: str, job: str | None) -> str:
    """
    Return ``model/job``, or ``model`` for a single run, for messages.

    Parameters
    ----------
    model
        Registered model name.
    job
        Job identifier, or ``None``.

    Returns
    -------
    str
        The scope.
    """
    return f"{model}/{job}" if job is not None else model
```

---

## 7. `src/rade_qnet/storage/tracker.py`

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

