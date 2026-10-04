"""
Which data groups exist on disk, and which snapshot of them a run used.

A **data group** is one self-contained training problem: its own directory,
its own input and target columns, its own history. A set of them is what
turns a single run into a job set -- one model trained per group, in
parallel, each writing its own bundle.

This is the general mechanism behind "train this model over every slice of my
data". A slice might be a region, a desk, a product line, a book of
instruments; the framework does not care, and that is the point. Nothing here
knows what a group *means*, only that it has a name, a directory and a set of
column identifiers.

Why a declared manifest rather than a directory listing
-------------------------------------------------------
A group set is described by a JSON file, not by whichever subdirectories
happen to be present. A listing silently changes meaning the moment somebody
leaves a scratch folder behind, and a job set that quietly grew a
forty-first member is a result nobody asked for. The manifest makes the
membership a decision rather than an accident.

The fingerprint
---------------
Every group set read here is fingerprinted, and the fingerprint is tagged
onto every bundle the resulting job set produces. The reason is the question
asked about every result that looks surprising: *was this trained on the data
I think it was?* Six months later the directory has been refreshed, columns
have been added, and a digest captured at the time is the only way to answer.

The digest covers what identifies the snapshot -- the group names, their
column identifiers, and the size and modification time of every file read --
rather than the file contents. Hashing gigabytes on every run to detect a
change that a size and a timestamp already reveal would make the check
expensive enough that somebody eventually turns it off, and a provenance
check nobody runs is worth nothing.

What this module does not do
----------------------------
It does not read the data. A group's history is read by whatever data module
the model declares, through the ordinary source machinery, because the shape
that history needs to be in is the model's business. This module answers the
prior question: which groups exist, what is in them, and which snapshot is
it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.lifecycle.errors import SpecError
from ...core.provenance.hashing import digest_payload
from ...core.provenance.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = ["MANIFEST_FILENAME", "DataGroup", "GroupSet", "read_group_set"]

_LOGGER = get_logger(__name__)

#: The file a group set directory is described by.
MANIFEST_FILENAME = "groups.json"

#: Manifest keys consumed directly by this module. Anything else in an entry
#: is carried through as a free-form attribute, which is what lets a caller
#: annotate a group without this module growing a field for it.
_RESERVED_KEYS = frozenset({"name", "directory", "input_ids", "target_ids"})


@dataclass(frozen=True, slots=True)
class DataGroup:
    """
    One self-contained training problem.

    Parameters
    ----------
    name
        The group's identifier, which becomes a job identifier and therefore
        a directory name and a seed component. Keep it to characters that are
        safe in a path, for that reason.
    directory
        Where this group's data lives.
    input_ids, target_ids
        Identifiers for the group's input and target columns, in column
        order.

        Carried for **provenance and reporting**, not for computation: they
        go into the snapshot fingerprint and into a log line, and that is
        all this module does with them. A model that needs to interpret its
        columns reads them through its own ``data.py``, in whatever
        vocabulary it uses -- which is why these are plain tuples here
        rather than a framework type that would have to name the two sides
        of somebody else's problem.
    attributes
        Free-form annotations from the manifest -- a region, a desk, a
        currency. Carried rather than interpreted, because every caller
        partitions their data differently and a fixed set of fields here
        would be wrong for the second one.
    """

    name: str
    directory: Path
    input_ids: tuple[str, ...]
    target_ids: tuple[str, ...]
    attributes: dict[str, str]

    @property
    def n_columns(self) -> int:
        """
        How many columns the group covers, inputs and targets together.

        Returns
        -------
        int
            The count.
        """
        return len(self.input_ids) + len(self.target_ids)

    def describe(self) -> str:
        """
        Return a one-line description, for a log and a progress line.

        Returns
        -------
        str
            For example ``FX__G10: 24 input, 6 target column(s)``.
        """
        return f"{self.name}: {len(self.input_ids)} input, {len(self.target_ids)} target column(s)"


@dataclass(frozen=True, slots=True)
class GroupSet:
    """
    A set of data groups and the fingerprint of the snapshot they came from.

    Parameters
    ----------
    groups
        The groups, in manifest order. Order is preserved rather than sorted
        so that a job set's directory listing matches the file somebody
        wrote.
    root
        The directory the set was read from.
    fingerprint
        Digest of the snapshot, recorded against every bundle trained from
        it.
    """

    groups: tuple[DataGroup, ...]
    root: Path
    fingerprint: str

    def __len__(self) -> int:
        """
        Return how many groups there are.

        Returns
        -------
        int
            The count.
        """
        return len(self.groups)

    def __iter__(self) -> Iterator[DataGroup]:
        """
        Iterate the groups in manifest order.

        Yields
        ------
        DataGroup
            Each group.
        """
        return iter(self.groups)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Every group name, in manifest order.

        Returns
        -------
        tuple of str
            The names.
        """
        return tuple(group.name for group in self.groups)

    def group(self, name: str) -> DataGroup:
        """
        Return one group by name.

        Parameters
        ----------
        name
            The group's identifier.

        Returns
        -------
        DataGroup
            The group.

        Raises
        ------
        KeyError
            If there is no such group, listing the ones there are. The usual
            cause is a typo in a caller's group filter, and the list is what
            turns that into a one-second fix.
        """
        for group in self.groups:
            if group.name == name:
                return group
        raise KeyError(f"no group named {name!r}; this set has {list(self.names)}")

    def select(self, names: Iterable[str] | None) -> GroupSet:
        """
        Return a set restricted to some groups.

        The fingerprint is **kept**, not recomputed. A subset of a snapshot
        was still trained on that snapshot, and recomputing would make two
        runs over different subsets of identical data look like runs over
        different data.

        Parameters
        ----------
        names
            Which groups to keep, in the order given. ``None`` keeps
            everything, which is the common case and avoids the caller
            branching.

        Returns
        -------
        GroupSet
            The restricted set.

        Raises
        ------
        KeyError
            If any name is not in the set.
        """
        if names is None:
            return self
        return GroupSet(
            groups=tuple(self.group(name) for name in names),
            root=self.root,
            fingerprint=self.fingerprint,
        )

    def describe(self) -> str:
        """
        Return a one-line summary.

        Returns
        -------
        str
            For example ``4 group(s), 138 column(s), snapshot a1b2c3d4``.
        """
        columns = sum(group.n_columns for group in self.groups)
        return f"{len(self.groups)} group(s), {columns} column(s), snapshot {self.fingerprint[:8]}"


def read_group_set(root: Path) -> GroupSet:
    """
    Read a group set from a directory and fingerprint the snapshot.

    Parameters
    ----------
    root
        The directory, containing a manifest named ``groups.json``.

    Returns
    -------
    GroupSet
        The groups and the snapshot digest.

    Raises
    ------
    SpecError
        If the manifest is absent, unreadable, or describes a group whose
        directory does not exist. Checked here, once, rather than left to
        surface inside whichever job happened to be scheduled first: a set
        with a missing directory is knowable before any training starts, and
        discovering it thirty-nine jobs in is the framework's fault.
    """
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SpecError(
            f"no group manifest at {manifest_path}. A group set directory is "
            f"described by a {MANIFEST_FILENAME} listing its groups, rather than "
            f"by whatever subdirectories happen to be present"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SpecError(f"{manifest_path} is not readable JSON: {error}") from error

    groups = tuple(
        _group(entry, root=root, origin=manifest_path) for entry in _entries(payload, manifest_path)
    )
    if not groups:
        raise SpecError(f"{manifest_path} declares no groups; a set needs at least one")

    group_set = GroupSet(
        groups=groups,
        root=root,
        fingerprint=_fingerprint(groups, manifest_path),
    )
    _LOGGER.info("read group set from %s: %s", root, group_set.describe())
    return group_set


def _entries(payload: object, origin: Path) -> Sequence[dict[str, object]]:
    """
    Pull the group list out of a manifest payload.

    Parameters
    ----------
    payload
        The parsed JSON.
    origin
        The file, for error messages.

    Returns
    -------
    sequence of dict
        The group entries.

    Raises
    ------
    SpecError
        If the payload is not a mapping with a list of groups.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise SpecError(
            f"{origin} must be a mapping with a 'groups' list, received {type(payload).__name__}"
        )
    return payload["groups"]


def _group(entry: object, *, root: Path, origin: Path) -> DataGroup:
    """
    Build one group from a manifest entry.

    Parameters
    ----------
    entry
        The entry.
    root
        The set's directory, which relative group paths are resolved
        against.
    origin
        The manifest file, for error messages.

    Returns
    -------
    DataGroup
        The group.

    Raises
    ------
    SpecError
        If the entry is malformed or names a directory that is not there.
    """
    if not isinstance(entry, dict) or "name" not in entry:
        raise SpecError(f"{origin}: each group needs at least a 'name', received {entry!r}")

    name = str(entry["name"])
    directory = root / str(entry.get("directory", name))
    if not directory.is_dir():
        raise SpecError(f"{origin}: group {name!r} points at {directory}, which is not a directory")

    return DataGroup(
        name=name,
        directory=directory,
        input_ids=tuple(str(value) for value in entry.get("input_ids", ())),
        target_ids=tuple(str(value) for value in entry.get("target_ids", ())),
        attributes={
            str(key): str(value) for key, value in entry.items() if key not in _RESERVED_KEYS
        },
    )


def _fingerprint(groups: Sequence[DataGroup], manifest_path: Path) -> str:
    """
    Digest what identifies this snapshot.

    Covers the manifest's own size and modification time, plus each group's
    name, column identifiers and directory contents by size and timestamp.
    Not the data itself: hashing gigabytes on every run to detect a change a
    timestamp already reveals makes the check expensive enough that somebody
    eventually disables it, and a provenance check nobody runs is worth
    nothing.

    Parameters
    ----------
    groups
        The groups.
    manifest_path
        The manifest, which is included in the digest.

    Returns
    -------
    str
        A hex digest.
    """
    payload: dict[str, object] = {"manifest": _stat(manifest_path)}
    for group in groups:
        payload[group.name] = {
            "input_ids": list(group.input_ids),
            "target_ids": list(group.target_ids),
            # Sorted, because a directory listing's order is filesystem
            # specific and would make the same snapshot fingerprint
            # differently on two machines.
            "files": {
                path.name: _stat(path)
                for path in sorted(group.directory.rglob("*"))
                if path.is_file()
            },
        }
    return digest_payload(payload)


def _stat(path: Path) -> dict[str, int]:
    """
    Return the size and modification time a fingerprint uses.

    Parameters
    ----------
    path
        The file.

    Returns
    -------
    dict
        Size in bytes and modification time in nanoseconds.
    """
    info = path.stat()
    return {"size": info.st_size, "mtime_ns": info.st_mtime_ns}
