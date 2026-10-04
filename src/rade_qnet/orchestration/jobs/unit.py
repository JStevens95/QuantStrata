"""
One job: train one model, in whichever process is running this.

:func:`run_job` is the function a job set hands to an executor. It is
module-level, and its payload and its return value are plain data, because
both have to survive a round trip through pickle to reach a worker and come
back.

What crosses the boundary, and what does not
---------------------------------------------
:class:`JobPayload` carries the **ingredients** of a
:class:`~rade_qnet.core.runtime.context.RunContext`, not a context.

A context holds hooks, a catalog and a tracker. Hooks are arbitrary user
objects -- a progress bar bound to a terminal, a client holding a socket, a
closure over a notebook's state -- and none of that need be picklable, nor
would it mean anything in another process if it were. So the payload carries
primitives and the worker builds its own context from them.

That has a second benefit worth more than the first. A worker's context is
*provably derived from the specification*, rather than inherited from
whatever the parent process happened to be holding. Two runs of the same
specification therefore configure their workers identically, which is half
of why placement cannot change results. The other half is the seed, which is
derived from the run seed and the job identifier by hash -- never from a
position, a process identifier or a clock.

What comes back
---------------
:class:`JobOutcome` is metrics and locations, not a model.

Returning the trained model would mean pickling a network and its weights
back to the parent for every job, which is slow, memory-hungry and pointless:
the bundle is already on disk, written by the worker, and the parent wants to
know *where* far more often than it wants the object. A caller who needs the
model loads the bundle, which is the same path they would take tomorrow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.runtime.components import get_model, import_registrations
from ...core.runtime.context import RunContext
from ...core.runtime.errors import SpecError
from ...core.runtime.logging import get_logger
from ...storage.catalog import JsonlCatalog
from ..pipelines.train import TrainPipeline
from ..stages.resolve import pipeline_for

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ...core.capability.definition import PredictorDefinition
    from ...core.spec.run import SupervisedRunSpec

__all__ = ["JobOutcome", "JobPayload", "run_job"]

_LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class JobPayload:
    """
    Everything one job needs, in a form that survives pickling.

    Parameters
    ----------
    job_id
        Identifier for this job within its set. Names the output directory
        and the catalog entry, and -- importantly -- derives the seed, so
        changing it changes the model.
    spec
        The merged, validated run specification for this job. A pydantic
        model, so it pickles by value and arrives in the worker identical to
        the one validated in the parent.
    run_id
        Identifier for the whole set, shared by every job in it.
    spec_digest
        Digest of the job-set specification, recorded against every bundle
        so a set's members can be found together afterwards.
    output_directory
        Where this job writes. Derived in the parent rather than the worker,
        so the layout of a set is decided in one place.
    seed
        The set's base seed. The job's own seed is derived from it and the
        job identifier inside :meth:`RunContext.for_job`.
    catalog_root
        Where bundles are recorded, or ``None`` to skip recording. A path
        rather than a :class:`~rade_qnet.storage.catalog.Catalog`, because the
        catalog holds a lock file handle and a handle is meaningless in
        another process. Each worker opens its own against the same path,
        which is exactly what the single-writer design is built for.
    metadata
        Free-form annotations carried into the run context.
    registration_modules
        Modules the worker must import before it can resolve the names this
        job's specification uses.

        A spawned worker starts with a bare interpreter, so a component
        registered purely as an import side effect is absent there. It
        *appears* to work without this, because spawn re-imports the main
        module and a script that trains a model has usually imported it --
        which means the failure arrives the first time the entry point
        changes, with the symptom "no model named ..." from a specification
        that is correct and that worked yesterday.
    """

    job_id: str
    spec: SupervisedRunSpec
    run_id: str
    spec_digest: str
    output_directory: Path
    seed: int = 0
    catalog_root: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    registration_modules: tuple[str, ...] = ()

    def context(self) -> RunContext:
        """
        Rebuild the run context for this job, in this process.

        Parameters
        ----------
        None

        Returns
        -------
        RunContext
            A context whose seed is derived from the set's seed and this
            job's identifier.
        """
        parent = RunContext(
            run_id=self.run_id,
            spec_digest=self.spec_digest,
            output_directory=self.output_directory.parent.parent,
            seed=self.seed,
            catalog=JsonlCatalog(self.catalog_root) if self.catalog_root is not None else None,
            metadata=self.metadata,
        )
        # Derived rather than constructed directly, so the seed-derivation
        # rule has exactly one implementation and a job run alone reproduces
        # what it would have produced inside its set.
        return parent.for_job(self.job_id, output_directory=self.output_directory)


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """
    What one job produced: metrics and locations, never a model.

    Parameters
    ----------
    job_id
        The job this came from.
    metrics
        Metrics per split, as plain floats. Flattened out of
        :class:`~rade_qnet.core.contract.result.TrainingResult` so the manifest
        can be written without the contract types, and so a job set's summary
        does not depend on a model's own result shape.
    seed
        The seed actually applied, recorded so one job can be reproduced
        without re-deriving it.
    epochs
        How many epochs ran.
    stopped_early
        Whether early stopping ended the run. Recorded alongside the epoch
        count rather than inferred from it: a set where every job stopped at
        epoch three is a finding about the configuration, and one where none
        of them stopped at all means the epoch budget was the binding
        constraint -- two quite different conclusions that the epoch count
        alone cannot distinguish.
    bundle_directory
        Where the bundle was written, or ``None`` if nothing was persisted.
    bundle_version
        The version the catalog assigned, or ``None``.
    model_name
        The registered model name, so a bundle can be reloaded from the
        manifest alone.
    """

    job_id: str
    metrics: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    seed: int = 0
    epochs: int = 0
    stopped_early: bool = False
    bundle_directory: Path | None = None
    bundle_version: int | None = None
    model_name: str = ""

    def metric(self, split: str, name: str) -> float | None:
        """
        Return one metric, or ``None`` if it was not recorded.

        ``None`` rather than an exception, because a job-set summary ranks
        and plots across jobs that need not all have the same metrics -- a
        job that failed before evaluating has none at all, and asking for a
        missing one is a normal occurrence rather than a mistake.

        Parameters
        ----------
        split
            Split name.
        name
            Metric name.

        Returns
        -------
        float or None
            The value, or ``None``.
        """
        return self.metrics.get(split, {}).get(name)


def run_job(payload: JobPayload) -> JobOutcome:
    """
    Train one job and return its metrics and locations.

    Module-level, and deliberately so. Under the spawn start method a bound
    method or a closure cannot cross the process boundary, and the error it
    produces names the pickle protocol rather than the design mistake.

    Parameters
    ----------
    payload
        Everything this job needs.

    Returns
    -------
    JobOutcome
        Metrics and locations.

    Raises
    ------
    SpecError
        If the job's specification is interactive rather than supervised.
        Reinforcement-learning runs arrive in Phase 7 and fan out through
        the same machinery; until then this says so, rather than failing
        several stages later with a message about a missing data source.
    """
    # First, before anything looks a name up. Idempotent, so in a sequential
    # run where the parent already imported everything this is a handful of
    # dictionary lookups.
    import_registrations(payload.registration_modules)

    if payload.spec.task != "supervised":
        raise SpecError(
            f"job {payload.job_id!r} is a {payload.spec.task!r} run; job sets "
            f"currently fan out supervised training only"
        )

    context = payload.context()
    with context.activate():
        _LOGGER.info("starting job %s with seed %d", payload.job_id, context.seed)
        definition = _definition(payload.spec)
        # Resolved rather than fixed, so a model's training override runs
        # in a portfolio exactly as it does in a single run. A job set that
        # quietly dropped a model's overrides would produce bundles that
        # differ from the single-run ones in a way no metric reveals.
        pipeline_cls = pipeline_for(definition, "train", TrainPipeline)
        pipeline = pipeline_cls(context=context, spec=payload.spec, definition=definition)
        result = pipeline.run()

    saved = pipeline.saved
    return JobOutcome(
        job_id=payload.job_id,
        metrics={
            split: dict(evaluation.metrics) for split, evaluation in result.evaluations.items()
        },
        seed=result.seed,
        epochs=len(result.fit.history),
        stopped_early=result.fit.stopped_early,
        bundle_directory=saved.directory if saved is not None else None,
        bundle_version=saved.manifest.version if saved is not None else None,
        model_name=payload.spec.model.name,
    )


def _definition(spec: SupervisedRunSpec) -> PredictorDefinition:
    """
    Resolve the model definition this job trains.

    Resolved in the worker rather than passed in the payload. A definition is
    a class the registry already knows how to find from a name, so sending
    the object would pickle a class reference that the worker then has to
    import anyway -- with the difference that a failure to import it would
    surface as a pickle error rather than as "no model named ...".

    Parameters
    ----------
    spec
        The job's run specification.

    Returns
    -------
    PredictorDefinition
        A fresh definition.
    """
    return get_model(spec.model.name)()
