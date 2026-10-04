# `src/rade_qnet/domains/pnl`

4 file(s). Create the directory, then create each file below with the exact contents of its block.

| # | File | Lines | Bytes | SHA-256 |
| --- | --- | ---: | ---: | --- |
| 1 | `__init__.py` | 52 | 1881 | `991ce35b86efaba6` |
| 2 | `clusters.py` | 187 | 6709 | `5c72e16b36cc41f9` |
| 3 | `portfolio.py` | 431 | 12960 | `a513a7e3381577cf` |
| 4 | `universe.py` | 130 | 4398 | `de16151432225b48` |

---

## 1. `src/rade_qnet/domains/pnl/__init__.py`

1881 bytes · SHA-256 `991ce35b86efaba6`

```python
"""
The P&L replication domain.

Holds everything that is true about the replication problem and nothing that
is true about a particular model. The flagship network reads its data through
these adapters, but so could a ridge regression -- the domain defines the
problem, not the solution.

Where this package sits
-----------------------
``models`` may import ``domains``; ``domains`` may not import ``models``, and
``orchestration`` may import neither. That one-way flow is what keeps a
second model from having to depend on the flagship in order to describe the
same book, and what keeps the job-set runner from becoming a P&L tool.

The practical consequence is that a portfolio is expanded into jobs *here*,
before the runner sees anything. By the time work is dispatched there are
only jobs, which is exactly why the same runner serves any domain.

Modules
-------
``universe.py``
    Which instrument each column refers to. The contract between a matrix
    position and a real position, and the thing that keeps a prediction
    attributable.
``portfolio.py``
    Reads a portfolio's clusters and fingerprints the snapshot, so a bundle
    records which version of the book it was trained on.
``clusters.py``
    Expands a portfolio into the job set a run fans out over, including the
    per-cluster complexity overrides that are the reason it is a set.

Planned
-------
``metrics.py``
    Replication quality as the desk understands it: unexplained P&L, tail
    replication error and hedge-ratio stability. Deferred to Phase 5, which
    is where the evaluation surface they belong to is built.
"""

from .clusters import cluster_overrides, job_set_for
from .portfolio import Cluster, Portfolio, read_portfolio
from .universe import Universe

__all__ = [
    "Cluster",
    "Portfolio",
    "Universe",
    "cluster_overrides",
    "job_set_for",
    "read_portfolio",
]
```

---

## 2. `src/rade_qnet/domains/pnl/clusters.py`

6709 bytes · SHA-256 `5c72e16b36cc41f9`

```python
"""
Turning a portfolio into the jobs a set fans out over.

This is the bridge between "a book of four clusters" and "a job set with four
jobs". It produces a :class:`~rade_qnet.core.spec.jobs.JobSetSpec`: shared
defaults, one job per cluster, each job overriding the directory it reads and
anything the caller wants to vary.

Why expansion happens here and not in the runner
------------------------------------------------
``orchestration`` may not import ``domains`` or ``models``. That is the
layering rule, and it is not bureaucratic: the job-set runner is a general
piece of machinery that knows how to run *jobs*, and the moment it knows what
a cluster is, it becomes a P&L tool that happens to run jobs. Every other
domain would then need its own runner.

So the expansion is pushed up. By the time the runner sees anything, there
are only jobs. A caller writes the job set by hand, or a domain writes it,
or ``api`` does -- and the runner cannot tell the difference, which is
exactly the property that makes it reusable.

Complexity per cluster
----------------------
The reason a portfolio is a job set rather than a loop is that clusters are
not alike. A liquid cluster with abundant history supports a wider, deeper
model than a thin one, and forcing both to the same configuration means
either underfitting the first or overfitting the second.
:func:`job_set_for` therefore takes a per-cluster override hook, which is the
mechanism behind that whole proposition.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ...core.runtime.logging import get_logger
from ...core.spec.jobs import JobSetSpec, parse_job_set_spec
from ...core.spec.merge import deep_merge

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from .portfolio import Cluster, Portfolio

__all__ = ["FINGERPRINT_KEY", "cluster_overrides", "fingerprint_tag", "job_set_for"]

_LOGGER = get_logger(__name__)

#: Prefix the portfolio's snapshot digest is tagged with, so every bundle in
#: the set can be traced back to the data it was trained on.
FINGERPRINT_KEY = "portfolio_fingerprint"


def fingerprint_tag(fingerprint: str) -> str:
    """
    Encode a snapshot digest as a set tag.

    A set's tags are free-form labels rather than a mapping, so a key-value
    fact has to be encoded into one. Done here, in one place, because the
    writer and every future reader have to agree on the encoding and a
    convention applied at two call sites is a convention that will
    eventually be applied at one.

    Parameters
    ----------
    fingerprint
        The portfolio's snapshot digest.

    Returns
    -------
    str
        For example ``portfolio_fingerprint=7f3a9c21...``.
    """
    return f"{FINGERPRINT_KEY}={fingerprint}"


def cluster_overrides(cluster: Cluster) -> dict[str, Any]:
    """
    Return the overrides every cluster job needs, whatever the model.

    Only the source directory. Deliberately minimal: anything else would be
    this module deciding how a model should be configured, which it is in no
    position to know -- the whole point of the framework is that the model
    declares that.

    Parameters
    ----------
    cluster
        The cluster.

    Returns
    -------
    dict
        An override fragment, merged over the set's defaults.
    """
    return {"source": {"params": {"directory": str(cluster.directory)}}}


def job_set_for(
    portfolio: Portfolio,
    *,
    defaults: Mapping[str, Any],
    output_root: Path,
    name: str | None = None,
    placement: Mapping[str, Any] | None = None,
    overrides_for: Callable[[Cluster], Mapping[str, Any]] | None = None,
) -> JobSetSpec:
    """
    Expand a portfolio into a job set, one job per cluster.

    Parameters
    ----------
    portfolio
        The clusters to train over.
    defaults
        The run-specification fragment every job shares: the model, the
        source kind, the training settings. Merged *under* each job's
        overrides, so a per-cluster setting wins and everything it does not
        mention survives.
    output_root
        Where the set writes.
    name
        The set's label, used in its run identifier and so in its directory
        name. Defaults to ``portfolio``.
    placement
        Where jobs run. ``None`` leaves it to the placement policy, which is
        the normal path.
    overrides_for
        Per-cluster overrides, merged over :func:`cluster_overrides`. This is
        the hook that lets a liquid cluster get a wider model than a thin
        one, which is the reason a portfolio is a job set at all.

    Returns
    -------
    JobSetSpec
        A validated job set. Validated here rather than at dispatch, so a
        portfolio that expands into something misconfigured fails at
        expansion -- where the error can say which cluster -- rather than
        after some of its jobs have already trained.
    """
    jobs = []
    for cluster in portfolio:
        overrides: dict[str, Any] = dict(cluster_overrides(cluster))
        if overrides_for is not None:
            # Merged by the job-set loader rather than here, so that a
            # per-cluster fragment follows exactly the same merge rules as
            # one written by hand in a file. Two merge implementations would
            # eventually disagree, and the disagreement would look like a
            # model behaving differently depending on how it was launched.
            overrides = _merged(overrides, overrides_for(cluster))
        jobs.append({"id": cluster.name, "overrides": overrides, "description": cluster.describe()})

    spec = parse_job_set_spec(
        {
            "name": name or "portfolio",
            "output_root": str(output_root),
            "defaults": dict(defaults),
            "jobs": jobs,
            **({"placement": dict(placement)} if placement is not None else {}),
            # Carried as a tag so it reaches every bundle. A result that
            # looks surprising six months from now is asked one question
            # first: was this trained on the data I think it was.
            "tags": [fingerprint_tag(portfolio.fingerprint)],
        }
    )
    _LOGGER.info("expanded %s into %d job(s)", portfolio.describe(), len(spec.jobs))
    return spec


def _merged(base: Mapping[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    """
    Deep-merge two override fragments.

    Parameters
    ----------
    base
        The fragment this module produced.
    extra
        The caller's fragment, which wins where the two overlap.

    Returns
    -------
    dict
        The merged fragment.
    """
    return deep_merge(base, extra)
```

---

## 3. `src/rade_qnet/domains/pnl/portfolio.py`

12960 bytes · SHA-256 `a513a7e3381577cf`

```python
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
```

---

## 4. `src/rade_qnet/domains/pnl/universe.py`

4398 bytes · SHA-256 `de16151432225b48`

```python
"""
Which instruments a column refers to.

A replication problem has two sides. The **elementary** instruments are the
liquid things a desk can actually trade -- the hedging basis. The **target**
instruments are the illiquid or structured positions whose P&L is being
replicated. A model predicts the second from the first, and a universe is the
record of which is which, in which order.

Why this is not a model's concern
---------------------------------
The ordering is the contract between a matrix column and a real instrument. A
prediction is a vector of numbers; without a universe it is a vector of
numbers about nothing. Getting the order wrong does not raise -- it produces
plausible predictions attributed to the wrong instruments, which is the
single most expensive mistake available in this problem and the hardest to
see.

That contract is a property of the *problem*, not of whatever model happens
to be solving it. A ridge regression over the same portfolio has exactly the
same universe. Keeping it in the flagship model's state file would mean any
second model either imported from the flagship -- making a baseline depend on
the thing it is a baseline for -- or defined its own, at which point two
definitions of "column three" exist and nothing checks they agree.

``models`` may import ``domains``; ``domains`` may not import ``models``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Universe"]


@dataclass(frozen=True, slots=True)
class Universe:
    """
    Which instruments the feature and target columns refer to.

    Frozen because a universe is the thing every downstream index is
    interpreted against. A fitted state, a prediction and a report all read
    positions out of it, and a universe that could be reordered after a fit
    would silently reattribute every one of them.

    Parameters
    ----------
    elementary_ids
        The elementary instruments, **in column order** and already reduced
        to the selected basis. Already reduced, because the alternative --
        carrying the full set and a separate index of survivors -- means
        every consumer has to apply the reduction itself, and one that
        forgets produces results that are wrong rather than absent.
    target_ids
        The target instruments, in column order.
    """

    elementary_ids: tuple[str, ...]
    target_ids: tuple[str, ...]

    @property
    def n_elementary(self) -> int:
        """
        How many elementary instruments survived basis selection.

        Returns
        -------
        int
            The count.
        """
        return len(self.elementary_ids)

    @property
    def n_targets(self) -> int:
        """
        How many target instruments are predicted.

        Returns
        -------
        int
            The count.
        """
        return len(self.target_ids)

    @property
    def instrument_ids(self) -> tuple[str, ...]:
        """
        Every instrument, elementary block first.

        The row order of the combined attribute matrix, which is why the
        concatenation happens here rather than at each call site: two places
        choosing the same order by convention is a convention that will
        eventually be broken by someone who did not know it existed.

        Returns
        -------
        tuple of str
            Elementary identifiers followed by target identifiers.
        """
        return self.elementary_ids + self.target_ids

    def position_of(self, instrument_id: str) -> int:
        """
        Return an instrument's row in the combined attribute matrix.

        Parameters
        ----------
        instrument_id
            The identifier.

        Returns
        -------
        int
            Its position.

        Raises
        ------
        KeyError
            If the universe does not contain it. Raised rather than
            returning ``-1``, which indexes the last row perfectly happily
            and would attribute a prediction to whichever instrument
            happened to be there.
        """
        try:
            return self.instrument_ids.index(instrument_id)
        except ValueError:
            raise KeyError(
                f"{instrument_id!r} is not in this universe of "
                f"{self.n_elementary} elementary and {self.n_targets} target instrument(s)"
            ) from None
```

