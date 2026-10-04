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
