"""
What a portfolio is, and which snapshot of it a run used.

A replication portfolio is a set of clusters, each of which is a self-
contained replication problem: its own elementary instruments, its own
targets, its own history. The flagship trains one model per cluster, which is
what makes a portfolio a job set rather than a single run.

The fingerprint
---------------
Every portfolio read here is fingerprinted, and the fingerprint goes into the
bundle. The reason is a question that gets asked about every result that
looks surprising: *was this trained on the data I think it was?* Six months
later the directory has been refreshed, the desk has added instruments, and
the only way to answer is a digest captured at the time.

The digest covers what identifies the snapshot -- the cluster names, their
instrument identifiers and the size and modification time of each file read
-- rather than the file contents. Hashing gigabytes of P&L on every run to
detect a change that a size and a timestamp already reveal would make the
check expensive enough that someone would eventually turn it off, and a
provenance check nobody runs is worth nothing.

What this module does not do
----------------------------
It does not read P&L. A cluster's history is read by whatever data module the
model declares, through the ordinary source machinery, because the shape that
history needs to be in is a model's business. This module answers the prior
question: which clusters exist, what is in them, and which snapshot is it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.runtime.errors import SpecError
from ...core.runtime.hashing import digest_payload
from ...core.runtime.logging import get_logger
from .universe import Universe

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

__all__ = ["Cluster", "Portfolio", "read_portfolio"]

_LOGGER = get_logger(__name__)

#: The file a portfolio directory is described by. A declared manifest rather
#: than "whatever subdirectories happen to be here", because a directory
#: listing silently changes meaning when somebody leaves a scratch folder
#: behind -- and a job set that quietly grew a forty-first cluster is a
#: result nobody asked for.
MANIFEST_FILENAME = "universe.json"


@dataclass(frozen=True, slots=True)
class Cluster:
    """
    One self-contained replication problem.

    Parameters
    ----------
    name
        The cluster's identifier, which becomes a job identifier and
        therefore a directory name and a seed. Constrained to characters
        that are safe in a path for that reason.
    directory
        Where this cluster's data lives.
    universe
        Which instruments its columns refer to.
    attributes
        Free-form annotations from the manifest -- asset class, desk,
        currency. Carried rather than interpreted, because every institution
        partitions a book differently and a fixed set of fields here would
        be wrong for the second one.
    """

    name: str
    directory: Path
    universe: Universe
    attributes: dict[str, str]

    @property
    def n_instruments(self) -> int:
        """
        How many instruments the cluster covers, both sides together.

        Returns
        -------
        int
            The count.
        """
        return self.universe.n_elementary + self.universe.n_targets

    def describe(self) -> str:
        """
        Return a one-line description, for a log and a progress line.

        Returns
        -------
        str
            For example ``FX__G10: 24 elementary, 6 target``.
        """
        return (
            f"{self.name}: {self.universe.n_elementary} elementary, "
            f"{self.universe.n_targets} target"
        )


@dataclass(frozen=True, slots=True)
class Portfolio:
    """
    A set of clusters and the fingerprint of the snapshot they came from.

    Parameters
    ----------
    clusters
        The clusters, in manifest order. Order is preserved rather than
        sorted so that a job set's directory listing matches the file
        somebody wrote.
    root
        The directory the portfolio was read from.
    fingerprint
        Digest of the snapshot, recorded against every bundle trained from
        it.
    """

    clusters: tuple[Cluster, ...]
    root: Path
    fingerprint: str

    def __len__(self) -> int:
        """
        Return how many clusters there are.

        Returns
        -------
        int
            The count.
        """
        return len(self.clusters)

    def __iter__(self) -> Iterator[Cluster]:
        """
        Iterate the clusters in manifest order.

        Yields
        ------
        Cluster
            Each cluster.
        """
        return iter(self.clusters)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Every cluster name, in manifest order.

        Returns
        -------
        tuple of str
            The names.
        """
        return tuple(cluster.name for cluster in self.clusters)

    def cluster(self, name: str) -> Cluster:
        """
        Return one cluster by name.

        Parameters
        ----------
        name
            The cluster's identifier.

        Returns
        -------
        Cluster
            The cluster.

        Raises
        ------
        KeyError
            If there is no such cluster, listing the ones there are. The
            usual cause is a typo in a job set's cluster filter, and the
            list is what turns that into a one-second fix.
        """
        for cluster in self.clusters:
            if cluster.name == name:
                return cluster
        raise KeyError(f"no cluster named {name!r}; this portfolio has {list(self.names)}")

    def select(self, names: Iterable[str] | None) -> Portfolio:
        """
        Return a portfolio restricted to some clusters.

        The fingerprint is **kept**, not recomputed. A subset of a snapshot
        was still trained on that snapshot, and recomputing would make two
        runs over different subsets of identical data look like runs over
        different data.

        Parameters
        ----------
        names
            Which clusters to keep, in the order given. ``None`` keeps
            everything, which is the common case and avoids the caller
            branching.

        Returns
        -------
        Portfolio
            The restricted portfolio.

        Raises
        ------
        KeyError
            If any name is not in the portfolio.
        """
        if names is None:
            return self
        return Portfolio(
            clusters=tuple(self.cluster(name) for name in names),
            root=self.root,
            fingerprint=self.fingerprint,
        )

    def describe(self) -> str:
        """
        Return a one-line summary.

        Returns
        -------
        str
            For example ``4 cluster(s), 138 instrument(s), snapshot a1b2c3d4``.
        """
        instruments = sum(cluster.n_instruments for cluster in self.clusters)
        return (
            f"{len(self.clusters)} cluster(s), {instruments} instrument(s), "
            f"snapshot {self.fingerprint[:8]}"
        )


def read_portfolio(root: Path) -> Portfolio:
    """
    Read a portfolio from a directory and fingerprint the snapshot.

    Parameters
    ----------
    root
        The portfolio directory, containing a manifest named
        ``universe.json``.

    Returns
    -------
    Portfolio
        The clusters and the snapshot digest.

    Raises
    ------
    SpecError
        If the manifest is absent, unreadable, or describes a cluster whose
        directory does not exist. Checked here, once, rather than left to
        surface inside whichever job happened to be scheduled first: a
        portfolio with a missing directory is knowable before any training
        starts, and discovering it thirty-nine jobs in is the framework's
        fault.
    """
    manifest_path = root / MANIFEST_FILENAME
    if not manifest_path.exists():
        raise SpecError(
            f"no portfolio manifest at {manifest_path}. A portfolio directory is "
            f"described by a {MANIFEST_FILENAME} listing its clusters, rather than "
            f"by whatever subdirectories happen to be present"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SpecError(f"{manifest_path} is not readable JSON: {error}") from error

    clusters = tuple(
        _cluster(entry, root=root, origin=manifest_path)
        for entry in _entries(payload, manifest_path)
    )
    if not clusters:
        raise SpecError(f"{manifest_path} declares no clusters; a portfolio needs at least one")

    portfolio = Portfolio(
        clusters=clusters,
        root=root,
        fingerprint=_fingerprint(clusters, manifest_path),
    )
    _LOGGER.info("read portfolio from %s: %s", root, portfolio.describe())
    return portfolio


def _entries(payload: object, origin: Path) -> Sequence[dict[str, object]]:
    """
    Pull the cluster list out of a manifest payload.

    Parameters
    ----------
    payload
        The parsed JSON.
    origin
        The file, for error messages.

    Returns
    -------
    sequence of dict
        The cluster entries.

    Raises
    ------
    SpecError
        If the payload is not a mapping with a list of clusters.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("clusters"), list):
        raise SpecError(
            f"{origin} must be a mapping with a 'clusters' list, received {type(payload).__name__}"
        )
    return payload["clusters"]


def _cluster(entry: object, *, root: Path, origin: Path) -> Cluster:
    """
    Build one cluster from a manifest entry.

    Parameters
    ----------
    entry
        The entry.
    root
        The portfolio directory, which relative cluster paths are resolved
        against.
    origin
        The manifest file, for error messages.

    Returns
    -------
    Cluster
        The cluster.

    Raises
    ------
    SpecError
        If the entry is malformed or names a directory that is not there.
    """
    if not isinstance(entry, dict) or "name" not in entry:
        raise SpecError(f"{origin}: each cluster needs at least a 'name', received {entry!r}")

    name = str(entry["name"])
    directory = root / str(entry.get("directory", name))
    if not directory.is_dir():
        raise SpecError(
            f"{origin}: cluster {name!r} points at {directory}, which is not a directory"
        )

    return Cluster(
        name=name,
        directory=directory,
        universe=Universe(
            elementary_ids=tuple(str(value) for value in entry.get("elementary_ids", ())),
            target_ids=tuple(str(value) for value in entry.get("target_ids", ())),
        ),
        attributes={
            str(key): str(value)
            for key, value in entry.items()
            if key not in {"name", "directory", "elementary_ids", "target_ids"}
        },
    )


def _fingerprint(clusters: Sequence[Cluster], manifest_path: Path) -> str:
    """
    Digest what identifies this snapshot.

    Covers the manifest's own size and modification time, plus each
    cluster's name, universe and directory contents by size and timestamp.
    Not the P&L itself: hashing gigabytes on every run to detect a change a
    timestamp already reveals makes the check expensive enough that somebody
    eventually disables it, and a provenance check nobody runs is worth
    nothing.

    Parameters
    ----------
    clusters
        The clusters.
    manifest_path
        The manifest, which is included in the digest.

    Returns
    -------
    str
        A hex digest.
    """
    payload: dict[str, object] = {"manifest": _stat(manifest_path)}
    for cluster in clusters:
        payload[cluster.name] = {
            "elementary_ids": list(cluster.universe.elementary_ids),
            "target_ids": list(cluster.universe.target_ids),
            # Sorted, because a directory listing's order is filesystem
            # specific and would make the same snapshot fingerprint
            # differently on two machines.
            "files": {
                path.name: _stat(path)
                for path in sorted(cluster.directory.rglob("*"))
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
