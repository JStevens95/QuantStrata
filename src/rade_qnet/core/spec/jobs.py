"""
The specification for running one model across many jobs.

A job set is a list of jobs, each a full independent training run of the same
model, differing in its data slice and -- optionally -- in its architecture
complexity. This module describes that in the same declarative, validated,
hashable style as everything else in the package.

One schema, not two
-------------------
The job set introduces **no new field names**. A job set's ``defaults`` and
each job's overrides are *fragments of a* :data:`~.run.RunSpec`, written with
the same keys a single-run file uses, and :meth:`JobSetSpec.run_spec_for`
merges them and validates the result through the same
:func:`~.run.parse_run_spec` a single run goes through.

That is worth more than it first appears. A separate job-set vocabulary would
need its own translation layer, its own error messages and its own tests, and
would give every field name a second place to drift. Reusing the run schema
also means every rule Phase 1 wrote -- the sequence-versus-split compatibility
check, the hardware combination checks -- applies per job, for free, with no
further code.

Why the overrides stay raw until the merge
------------------------------------------
:attr:`JobSpec.overrides` is an unvalidated mapping, which is unusual in this
package and deliberate. The reasoning is in
:mod:`rade_qnet.core.spec.merge`: a validated fragment cannot distinguish a
field the user set from a field that defaulted, so merging validated specs
silently overwrites shared defaults with per-job defaults.

The openness is bounded. An override is validated the moment it is merged, by
the full run schema with ``extra="forbid"``, so a misspelled key is still a
load-time error -- just one raised against the merged specification, where the
message can name the real field. :meth:`JobSetSpec.validate_jobs` performs
that check for every job up front, so a typo in the fortieth job is reported
before the first one starts training.

Placement is not hardware
-------------------------
:class:`PlacementSpec` answers *where do jobs run*;
:class:`~.hardware.HardwareSpec` answers *how does one job use the machine it
has*. They are kept apart because they are set by different people for
different reasons: "use bfloat16" is a modelling decision made once, and "run
forty of these across four GPUs" is an operational decision made per run.
Conflating them is why "use the GPU" and "run forty in parallel" so often end
up as the same overloaded flag.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..lifecycle.errors import SpecError
from .base import Spec
from .merge import deep_merge
from .run import RunSpec, parse_run_spec

__all__ = [
    "ExecutorName",
    "JobSetSpec",
    "JobSpec",
    "PlacementSpec",
    "dump_job_set_spec",
    "load_job_set_spec",
    "parse_job_set_spec",
]

#: Where jobs run. ``auto`` defers to
#: :func:`rade_qnet.orchestration.compute.placement.choose_placement`.
ExecutorName = Literal["auto", "local", "processes", "gpus"]

#: Worker start methods that exist only on POSIX platforms.
#:
#: ``fork`` is here for completeness even though :class:`PlacementSpec` does
#: not offer it -- the set describes the platforms, not this schema, so it
#: stays correct if the schema ever widens.
_POSIX_ONLY_START_METHODS = frozenset({"fork", "forkserver"})

#: Top-level keys of a job entry that are the job's own metadata rather than
#: part of its run-specification override. Everything else in a job entry is
#: an override, which is what makes the shorthand in :class:`JobSpec` work.
_JOB_METADATA_KEYS = frozenset({"id", "overrides", "description"})

#: Fields of a job-set file that are the set's own, as opposed to the shared
#: run-specification defaults. Used to reject the common mistake of writing a
#: run-spec field at the top level instead of under ``defaults``.
_JOB_SET_KEYS = frozenset({"defaults", "jobs", "placement", "name", "output_root", "tags"})


class JobSpec(Spec):
    """
    One job: an identifier, and what it changes about the shared defaults.

    Two spellings are accepted, because both are natural in YAML::

        - id: EURUSD
          source: {params: {directory: clusters/EURUSD}}
          model: {units: 64}

        - id: EURUSD
          overrides:
            source: {params: {directory: clusters/EURUSD}}

    The first keeps the overrides at the top level, which is how a user would
    write them and how the architecture document presents them. The ``before``
    validator normalises it into the second. Both are supported rather than
    one, because the explicit form is clearer when a job's overrides are
    long, and the shorthand is clearer when they are two lines.

    Parameters
    ----------
    id
        Identifier for this job, unique within the set. It names the job's
        output directory, its catalog entries and its row in the manifest, and
        -- importantly -- it is what the job's seed is derived from, so
        changing it changes the model. See
        :meth:`rade_qnet.core.lifecycle.context.RunContext.for_job`.
    overrides
        A fragment of a run specification, merged over the set's defaults.
        Unvalidated here and validated on merge; see this module's docstring.
    description
        Free text for reports. Carries no behaviour.
    """

    id: str = Field(min_length=1)
    overrides: Mapping[str, Any] = Field(default_factory=dict)
    description: str = ""

    @model_validator(mode="before")
    @classmethod
    def _accept_shorthand(cls, value: object) -> object:
        """
        Fold top-level override keys into ``overrides``.

        Parameters
        ----------
        value
            The raw job entry from the specification file.

        Returns
        -------
        object
            Either the value unchanged, or a normalised mapping.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if both spellings are used at once. That
            combination has no obvious meaning -- it is unclear whether the
            top-level keys should merge into ``overrides`` or replace it --
            and silently picking one would be a configuration mistake that
            trains a model rather than reporting itself.
        """
        if not isinstance(value, Mapping):
            return value

        inline = {key: item for key, item in value.items() if key not in _JOB_METADATA_KEYS}
        if not inline:
            return value

        if "overrides" in value:
            raise SpecError(
                f"job {value.get('id', '<unnamed>')!r} mixes the two override spellings: "
                f"it sets 'overrides' and also names {sorted(inline)} at the top level. "
                f"Use one or the other"
            )
        return {
            **{key: item for key, item in value.items() if key in _JOB_METADATA_KEYS},
            "overrides": inline,
        }


class PlacementSpec(Spec):
    """
    Where the jobs of a set run, and how many at a time.

    Parameters
    ----------
    executor
        ``auto`` lets the placement policy decide from the hardware present
        and the size of the set, which is the right default: a sensible
        choice unaided is worth more than a hand-tuned one that is copied
        between machines and stops being sensible.
    workers
        How many jobs run concurrently. ``None`` lets the policy decide.
        Ignored by the sequential executor, which is always one.
    threads_per_worker
        Intra-op thread budget for each worker, set in the worker before the
        training library is imported. ``None`` lets the policy divide the
        machine's cores between the workers.

        This is the setting that decides whether a parallel run is faster or
        slower than a sequential one. Eight workers each defaulting to every
        core produces eight times the core count in threads, and throughput
        collapses below sequential -- with nothing in any log to say why.
    start_method
        How worker processes are created. ``spawn`` is the default and the
        only one tested: ``fork`` is unsafe in a process that has already
        initialised a threaded numerical library or a GPU context, and the
        resulting hangs are intermittent and extremely hard to attribute.

        ``forkserver`` does not exist on Windows, so it is rejected there at
        validation time rather than raising a bare ``ValueError`` from deep
        inside the executor once the data has already been built.
    memory_per_job_gb
        An estimate of a single job's peak resident memory, used by the
        policy to cap the worker count. ``None`` means do not cap, which is
        correct when jobs are small and wrong when they are not -- an
        over-subscribed machine kills workers with a signal that carries no
        explanation.

        Taken as an input rather than measured: estimating a job's memory
        before running it is not something the framework can do, and sniffing
        total system memory portably is not worth a dependency. A user who
        knows the figure can supply it; one who does not is no worse off.
    """

    executor: ExecutorName = "auto"
    workers: int | None = Field(default=None, ge=1)
    threads_per_worker: int | None = Field(default=None, ge=1)
    start_method: Literal["spawn", "forkserver"] = "spawn"
    memory_per_job_gb: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _reject_a_start_method_this_platform_lacks(self) -> PlacementSpec:
        """
        Refuse a start method the running platform cannot provide.

        ``forkserver`` is POSIX-only. Without this check, a specification
        written on Linux and run on Windows raises ``ValueError: cannot find
        context for 'forkserver'`` from inside
        :class:`~rade_qnet.orchestration.compute.processes.ProcessExecutor` --
        after the data build, with a message that names neither the setting
        nor the file it came from.

        Decided from ``sys.platform`` rather than by asking
        ``multiprocessing.get_all_start_methods()``, which would be the
        authoritative answer. Importing :mod:`multiprocessing` costs about
        28ms and nothing else in ``core`` pulls it in, so asking it here would
        load a process-management library into every program that merely
        parses a specification -- against the property that ``core`` stays
        cheap to import. Windows is the only platform without
        ``forkserver``, so the cheap test and the authoritative one agree.

        Returns
        -------
        PlacementSpec
            Unchanged.

        Raises
        ------
        ValueError
            If this platform does not offer the requested start method. Names
            the alternative, because the useful next action is picking it.
        """
        if sys.platform == "win32" and self.start_method in _POSIX_ONLY_START_METHODS:
            raise ValueError(
                f"start_method {self.start_method!r} is not available on Windows; use 'spawn'"
            )
        return self


class JobSetSpec(Spec):
    """
    Shared defaults, a list of jobs, and where they run.

    Parameters
    ----------
    defaults
        A run-specification fragment applied to every job. Unvalidated on its
        own, because defaults alone are rarely a complete run -- a job set
        whose defaults name no data source is perfectly legitimate, since
        each job supplies its own.
    jobs
        The jobs. At least one, and identifiers must be unique.
    placement
        Where the jobs run.
    name
        A label for the set, used in the manifest and in log lines.
    output_root
        Root for everything the set writes. Each job writes beneath it.
    tags
        Free-form labels recorded against every bundle the set produces, so a
        whole set can be found later by one query.
    """

    defaults: Mapping[str, Any] = Field(default_factory=dict)
    jobs: tuple[JobSpec, ...] = Field(min_length=1)
    placement: PlacementSpec = Field(default_factory=PlacementSpec)
    name: str | None = None
    output_root: Path = Path("artifacts/rade_qnet")
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _reject_duplicate_job_ids(self) -> JobSetSpec:
        """
        Reject a set in which two jobs share an identifier.

        Returns
        -------
        JobSetSpec
            The validated spec.

        Raises
        ------
        ValidationError
            Wrapping a ``SpecError`` if any identifier repeats. Left
            unchecked, the second job would write into the first one's
            directory, overwrite its bundle and replace its row in the
            manifest -- producing a set that reports forty successes and
            holds thirty-nine models.
        """
        counts = Counter(job.id for job in self.jobs)
        duplicated = sorted(name for name, count in counts.items() if count > 1)
        if duplicated:
            raise SpecError(
                f"job identifiers must be unique within a set; repeated: {duplicated}. "
                f"Each identifier names an output directory and derives a seed, so two "
                f"jobs sharing one would overwrite each other"
            )
        return self

    @property
    def job_ids(self) -> tuple[str, ...]:
        """
        Return every job identifier, in the order the set declares them.

        Returns
        -------
        tuple of str
            The identifiers.
        """
        return tuple(job.id for job in self.jobs)

    def run_spec_for(self, job: JobSpec | str) -> RunSpec:
        """
        Merge the defaults with one job's overrides and validate the result.

        This is where a job becomes a real run specification, and it is the
        only place that conversion happens -- so the merge rules, the
        validation and the error message all live in one spot.

        Parameters
        ----------
        job
            A job, or the identifier of one in this set.

        Returns
        -------
        RunSpec
            The validated specification for that job.

        Raises
        ------
        SpecError
            If the identifier is not in this set, or if the merged
            specification is invalid. The message names the job, because a
            validation error against a merged document is otherwise very hard
            to trace back to the override that caused it.
        """
        resolved = self.job(job) if isinstance(job, str) else job
        merged = deep_merge(self.defaults, resolved.overrides)
        # Tags are a property of the set, so they are unioned rather than
        # merged: a job may add its own, and dropping the set's would make a
        # set unfindable by the label it was launched under.
        if self.tags:
            merged["tags"] = tuple(dict.fromkeys((*self.tags, *merged.get("tags", ()))))
        merged.setdefault("output_root", str(self.output_root))
        return parse_run_spec(merged, origin=f"job {resolved.id!r}")

    def job(self, job_id: str) -> JobSpec:
        """
        Return the job with an identifier.

        Parameters
        ----------
        job_id
            The identifier.

        Returns
        -------
        JobSpec
            The job.

        Raises
        ------
        SpecError
            If no job has that identifier.
        """
        for candidate in self.jobs:
            if candidate.id == job_id:
                return candidate
        raise SpecError(f"no job named {job_id!r} in this set; available: {list(self.job_ids)}")

    def validate_jobs(self) -> dict[str, RunSpec]:
        """
        Merge and validate every job, reporting all failures at once.

        Called before a set starts, so a typo in the fortieth job's overrides
        is reported in the first second rather than three hours in. Reporting
        every failure together rather than the first matters just as much: a
        user who mistyped two keys should learn both now, not discover the
        second after fixing the first and waiting again.

        Returns
        -------
        dict
            Job identifier to validated run specification, in declaration
            order.

        Raises
        ------
        SpecError
            If any job fails to validate, with one entry per failing job.
        """
        specs: dict[str, RunSpec] = {}
        failures: list[str] = []
        for job in self.jobs:
            try:
                specs[job.id] = self.run_spec_for(job)
            except SpecError as error:
                failures.append(f"  {job.id}: {error}")

        if failures:
            raise SpecError(
                f"{len(failures)} of {len(self.jobs)} job(s) in this set are invalid:\n"
                + "\n".join(failures)
            )
        return specs


def parse_job_set_spec(payload: Mapping[str, Any], *, origin: str = "<mapping>") -> JobSetSpec:
    """
    Validate a mapping into a job-set specification.

    Accepts one convenience the schema does not: ``model`` at the top level,
    which is folded into ``defaults``. Naming the model once, at the top, is
    how the architecture document presents a job set and how a user would
    write one -- a job set is "this model, over these slices", and burying
    the model inside ``defaults`` reads as though it were an incidental
    default rather than the subject.

    Parameters
    ----------
    payload
        The raw mapping, typically parsed from YAML or JSON.
    origin
        Where the payload came from, for error messages.

    Returns
    -------
    JobSetSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the payload is not a mapping, carries run-specification fields at
        the top level, or fails validation.
    """
    if not isinstance(payload, Mapping):
        raise SpecError(
            f"{origin}: a job set must be a mapping of fields, received {type(payload).__name__}"
        )

    remaining = dict(payload)
    model = remaining.pop("model", None)
    if model is not None:
        defaults = dict(remaining.get("defaults", {}))
        if "model" in defaults:
            raise SpecError(
                f"{origin}: 'model' is set both at the top level and under 'defaults'. "
                f"Use one or the other"
            )
        # Normalised to a mapping rather than stored as the bare string the
        # user wrote. Left as a string it would not deep-merge: a job that
        # overrides `model.params` would be merging a mapping over a string,
        # which replaces rather than recurses, and the model's *name* would
        # vanish. The failure reads "model.name: Field required" from a file
        # whose only mention of a name is at the top level and plainly there.
        defaults["model"] = {"name": model} if isinstance(model, str) else model
        remaining["defaults"] = defaults

    # Caught here rather than by `extra="forbid"` so the message can say where
    # the field belongs. A user writing `training:` at the top level has made
    # a reasonable mistake -- it is where it goes in a single-run file -- and
    # "extra inputs are not permitted" would not tell them what to do.
    misplaced = sorted(set(remaining) - _JOB_SET_KEYS)
    if misplaced:
        raise SpecError(
            f"{origin}: {misplaced} are run-specification fields and belong under "
            f"'defaults' (where they apply to every job) or inside a job (where they "
            f"apply to one). A job set's own fields are {sorted(_JOB_SET_KEYS)}"
        )

    try:
        return JobSetSpec.model_validate(remaining)
    except ValueError as exc:
        raise SpecError(f"{origin}: invalid job set: {exc}") from exc


def load_job_set_spec(path: Path | str) -> JobSetSpec:
    """
    Load and validate a job-set specification from a YAML or JSON file.

    Parameters
    ----------
    path
        Path to a ``.yaml``, ``.yml`` or ``.json`` file.

    Returns
    -------
    JobSetSpec
        The validated specification.

    Raises
    ------
    SpecError
        If the file is missing, unparseable, not a mapping, or invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise SpecError(f"job set specification file not found: {path}")

    try:
        # YAML is a superset of JSON, so one parser handles both suffixes and
        # a mislabelled file still loads.
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise SpecError(f"could not parse {path} as YAML or JSON: {exc}") from exc

    return parse_job_set_spec(payload, origin=str(path))


def dump_job_set_spec(spec: JobSetSpec, path: Path | str) -> Path:
    """
    Write a job-set specification to a YAML or JSON file.

    The output round-trips exactly, which is what lets a set that was
    assembled in Python -- by expanding a portfolio, say -- be saved as the
    record of what was run.

    Parameters
    ----------
    spec
        The specification to write.
    path
        Destination. The suffix selects the format.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    SpecError
        If the suffix is not a recognised format.
    """
    path = Path(path)
    payload = spec.model_dump(mode="json")

    if path.suffix in {".yaml", ".yml"}:
        text = yaml.safe_dump(payload, sort_keys=True, default_flow_style=False)
    elif path.suffix == ".json":
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        raise SpecError(
            f"unsupported specification format {path.suffix!r}; use .yaml, .yml or .json"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
