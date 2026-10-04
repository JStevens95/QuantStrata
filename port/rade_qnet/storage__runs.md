# `src/rade_qnet/storage/runs`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 46 | 2214 | `48dcf2a1299e07de` |
| 2 | `catalog.py` | 664 | 22127 | `9c922f96d5c8cd98` |
| 3 | `registry.py` | 806 | 26765 | `2a91c5e65bf839b9` |
| 4 | `tracker.py` | 186 | 5622 | `d7c5799fb1f03913` |

---

## 1. `src/rade_qnet/storage/runs/__init__.py`

2214 bytes · SHA-256 `48dcf2a1299e07de`

```python
"""
Every run that happened, which ones are blessed, and which one is going now.

Three questions, three modules, and the reason they are a package rather than
three flat files beside ``bundle.py`` is that they answer questions *about the
collection*, where the rest of ``storage`` answers questions about a single
directory.  A pipeline writes one bundle and never reads this package; a quant
choosing a model for Monday reads only this package and never opens a bundle
by hand.

Why the catalog and the registry are two things
------------------------------------------------
They look like one, and the distinction is load-bearing.  The **catalog** is a
record of fact: this run happened, with this spec, producing this bundle.  It
is append-oriented and nothing ever edits an entry, because a training run
that occurred does not stop having occurred.

The **registry** is a record of judgement: this run is the one serving
production, that one is the champion on the validation set.  Judgements change
-- a model is promoted, demoted, retagged -- and they change without any run
having been performed.

Keeping them apart is what lets a promotion be recorded without rewriting a
bundle, and it is why ``production`` can be repointed at a six-month-old run
in a single append.  Collapsing them would mean mutating the record of fact
every time somebody changed their mind about it.

Modules
-------
``catalog.py``
    The index of bundles, queryable by model, job and tag, with a
    single-writer discipline and an append-oriented log rather than whole-file
    rewrites.  ``JsonlCatalog`` for real runs, ``InMemoryCatalog`` for tests
    and notebooks.  [Phase 1, delivered]
``registry.py``
    ``RunRegistry``: choose a trained run by tag, by best metric or by an
    alias such as ``production``, and record promotions and tags added after
    training as append-only events beside the catalog -- never by rewriting a
    bundle.  [Delivered after Phase 6]
``tracker.py``
    Experiment tracking behind one interface, with a no-op default.  Tracking
    is optional infrastructure; a run must never fail because a tracking
    server is unreachable.  [Phase 1, delivered]
"""

__all__: tuple[str, ...] = ()
```

---

## 2. `src/rade_qnet/storage/runs/catalog.py`

22127 bytes · SHA-256 `9c922f96d5c8cd98`

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
:class:`~rade_qnet.core.lifecycle.context.Catalog` protocol: an alternative
implementation -- including :class:`InMemoryCatalog` below -- substitutes with
no change anywhere else, which is the reason that protocol is declared in
``core`` rather than this class being used directly.

Where each bundle lives
-----------------------
An entry records the bundle's directory alongside its manifest. A reader
selecting a run -- by tag, by alias, by best metric, through
:mod:`rade_qnet.storage.runs.registry` -- needs to open it, and cannot derive the
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

from ...core.contract.bundle import Manifest
from ...core.provenance.logging import get_logger
from ..locking import exclusive_lock

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

    Satisfies :class:`~rade_qnet.core.lifecycle.context.Catalog`.

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

    Satisfies :class:`~rade_qnet.core.lifecycle.context.Catalog`. Used by
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

## 3. `src/rade_qnet/storage/runs/registry.py`

26765 bytes · SHA-256 `2a91c5e65bf839b9`

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

from ...core.lifecycle.errors import BundleError, SpecError
from ...core.provenance.logging import get_logger
from .catalog import CatalogEntry, JsonlCatalog
from ..locking import exclusive_lock

if TYPE_CHECKING:
    from ...core.contract.bundle import Manifest
    from ...core.spec.tune import Direction

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

## 4. `src/rade_qnet/storage/runs/tracker.py`

5622 bytes · SHA-256 `d7c5799fb1f03913`

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

from ...core.provenance.hashing import canonical_json
from ...core.provenance.logging import get_logger

__all__ = ["JsonlTracker", "NullTracker"]

_LOGGER = get_logger(__name__)


class NullTracker:
    """
    A tracker that records nothing.

    Satisfies :class:`~rade_qnet.core.lifecycle.context.Tracker`. The default, so
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

    Satisfies :class:`~rade_qnet.core.lifecycle.context.Tracker`. Useful when a
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
    :class:`~rade_qnet.core.lifecycle.context.RunContext` naturally produces.
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

