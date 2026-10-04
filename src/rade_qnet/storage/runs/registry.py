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

from ...core.runtime.errors import BundleError, SpecError
from ...core.runtime.logging import get_logger
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
