"""
The front door: a handful of functions that cover most of what anybody wants.

Everything underneath is deliberately composable -- a context, a
specification, a definition, a pipeline, an executor -- because that is what
makes the framework extensible. But composability is a cost at the call site,
and the two most common things anyone does are *train this configuration* and
*train it across every group of my data*. Those should be one line each.

Nothing here adds behaviour. Every function assembles the same pieces a
caller could assemble by hand, which is the property that keeps this module
from becoming a second, divergent way to run a model. If a workflow needs
something these do not offer, the answer is to drop one level down rather
than to grow a seventh keyword argument here.

Where the layers sit
--------------------
This module lives at the top of the package rather than inside
``orchestration``, because it reaches across layers that are not allowed to
see each other. ``orchestration`` may not import ``models``; ``api`` may.
:func:`train_groups` is the clearest case: it resolves the model by name
through the registry, reads a group set and expands it into jobs, then hands
the jobs to a runner that has no idea a group exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from .core.capability.definition import PolicyDefinition, PredictorDefinition
from .core.runtime.components import get_model
from .core.runtime.context import RunContext
from .core.runtime.errors import ComponentError, SpecError
from .core.runtime.hashing import abbreviate_digest, digest_spec
from .core.runtime.logging import get_logger
from .core.spec.data import SourceSpec
from .core.spec.jobs import JobSetSpec, load_job_set_spec, parse_job_set_spec
from .core.spec.run import (
    ReinforcementRunSpec,
    RunSpec,
    SupervisedRunSpec,
    load_run_spec,
    parse_run_spec,
)
from .core.spec.tune import TuneSpec, load_tune_spec, parse_tune_spec
from .orchestration.jobs.fanout import job_set_for_groups
from .orchestration.jobs.groups import read_group_set
from .orchestration.jobs.set import JobSetRunner
from .orchestration.pipelines.evaluate import EvaluatePipeline
from .orchestration.pipelines.infer import InferPipeline
from .orchestration.pipelines.reinforce import ReinforcePipeline
from .orchestration.pipelines.train import TrainPipeline
from .orchestration.pipelines.tune import TunePipeline
from .orchestration.stages.resolve import pipeline_for
from .orchestration.stages.scoring import EVALUATED_SPLITS
from .storage.bundle import BundleError, load_manifest
from .storage.runs.catalog import JsonlCatalog
from .storage.runs.registry import RunRegistry

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .core.contract.result import (
        EvaluationResult,
        Predictions,
        TrainingResult,
        TuningResult,
    )
    from .orchestration.compute.base import Executor
    from .orchestration.jobs.groups import DataGroup
    from .orchestration.jobs.manifest import JobSetManifest

__all__ = ["evaluate", "infer", "registry", "train", "train_groups", "train_jobs", "tune"]

_LOGGER = get_logger(__name__)

#: Where a search writes when the caller names no root. A run specification
#: carries its own ``output_root``; a search has no equivalent field,
#: because a search is not a run.
_DEFAULT_ROOT = Path("artifacts/rade_qnet")


def train(
    configuration: Path | str | Mapping[str, object] | RunSpec,
    *,
    output_root: Path | str | None = None,
    catalog_root: Path | str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> TrainingResult:
    """
    Train one model from a configuration.

    Parameters
    ----------
    configuration
        A path to a YAML file, a mapping, or an already-validated
        specification. All three are accepted because all three occur: a
        file in production, a mapping in a notebook, and a specification in
        a test that built one programmatically.
    output_root
        Where the run writes. Defaults to the specification's own
        ``output_root``.
    catalog_root
        Where the bundle is recorded, or ``None`` to use the output root.
    metadata
        Free-form annotations carried into the run context and recorded
        against the bundle.

    Returns
    -------
    TrainingResult
        Metrics per split, the fit history, and the seed that was applied.
        Metrics are already in the original target units -- the pipeline
        inverts any target scaling before evaluating, so a mean absolute
        error is in currency rather than in standard deviations.

    Raises
    ------
    SpecError
        If the configuration describes a reinforcement-learning run. Routed
        rather than refused -- see the note below -- but a configuration
        whose task is neither is rejected here, in milliseconds, rather than
        failing several stages later with a message about a missing data
        source.

    Notes
    -----
    An interactive configuration is forwarded to
    :class:`~rade_qnet.orchestration.pipelines.reinforce.ReinforcePipeline`
    rather than refused. One entry point for both, because "train the thing
    this file describes" is one intent, and a user who wrote
    ``task: reinforcement`` has already said which kind it is -- asking them
    to call a differently named function as well would be asking them to say
    it twice.
    """
    spec = _run_spec(configuration)
    context = _context(spec, output_root=output_root, catalog_root=catalog_root, metadata=metadata)

    with context.activate():
        definition = get_model(spec.model.name)()

        if isinstance(spec, ReinforcementRunSpec):
            if not isinstance(definition, PolicyDefinition):
                raise SpecError(
                    f"this configuration is a {spec.task!r} run, but the model "
                    f"{spec.model.name!r} is a {type(definition).__name__}, which "
                    f"learns from a fixed dataset. An interactive run needs a "
                    f"model that builds an environment and a policy"
                )
            interactive = pipeline_for(definition, "train", ReinforcePipeline)
            return interactive(context=context, spec=spec, definition=definition).run()

        if not isinstance(definition, PredictorDefinition):
            raise SpecError(
                f"this configuration is a {spec.task!r} run, but the model "
                f"{spec.model.name!r} is a {type(definition).__name__}, which "
                f"learns by interacting with an environment. A supervised run "
                f"needs a model that builds a dataset"
            )
        pipeline = pipeline_for(definition, "train", TrainPipeline)
        return pipeline(context=context, spec=spec, definition=definition).run()


def evaluate(
    bundle: Path | str,
    *,
    source: SourceSpec | None = None,
    splits: tuple[str, ...] = EVALUATED_SPLITS,
    output_root: Path | str | None = None,
    verify: bool = True,
) -> EvaluationResult:
    """
    Score a saved model, on its own data or on newer data.

    Parameters
    ----------
    bundle
        The bundle directory.
    source
        A source specification to score against, or ``None`` for the
        bundle's own -- which is the reproduction case. Supplying a
        different one re-scores the same model on other data, and the
        result records that it did.
    splits
        Which splits to score. A re-score against fresh observations
        usually wants only ``("test",)``, since on new data the training
        split is no longer a meaningful category.
    output_root
        Where the evaluation writes. Defaults to the bundle's parent, so
        an evaluation lands beside the thing it evaluated rather than in
        whatever directory the caller happened to be in.
    verify
        Whether to re-hash the bundle's files against its manifest.

    Returns
    -------
    EvaluationResult
        Metrics per split, in the target's original units, with the
        provenance needed to say what was scored against what --
        including whether the source changed, which is the difference
        between reproducing a number and computing a new one.
    """
    directory = Path(bundle)
    context = _bundle_context(directory, action="evaluate", output_root=output_root)
    with context.activate():
        pipeline = pipeline_for(_definition_in(directory), "eval", EvaluatePipeline)
        return pipeline(
            context=context,
            directory=directory,
            source=source,
            splits=splits,
            verify=verify,
        ).run()


def infer(
    bundle: Path | str,
    *,
    source: SourceSpec | None = None,
    split: str = "test",
    entities: Iterable[str] | None = None,
    output_root: Path | str | None = None,
    verify: bool = True,
) -> Predictions:
    """
    Predict with a saved model.

    Parameters
    ----------
    bundle
        The bundle directory.
    source
        A source specification to predict over, or ``None`` for the
        bundle's own.
    split
        Which split of the source to predict over.
    entities
        Identifiers to predict for, or ``None`` for whatever the source
        holds. Naming entities the model never saw requires it to declare
        :class:`~rade_qnet.core.capability.protocols.Inductive`; otherwise
        the request is refused rather than answered with a default
        embedding.
    output_root
        Where the run writes. Defaults to the bundle's parent.
    verify
        Whether to re-hash the bundle's files against its manifest.

    Returns
    -------
    Predictions
        Values in the target's original units, carrying the bundle
        version, the spec digest, both source fingerprints and the time of
        inference -- because reconciling predictions after the fact is the
        ordinary case, not the exception.
    """
    directory = Path(bundle)
    context = _bundle_context(directory, action="infer", output_root=output_root)
    with context.activate():
        pipeline = pipeline_for(_definition_in(directory), "infer", InferPipeline)
        return pipeline(
            context=context,
            directory=directory,
            source=source,
            split=split,
            entities=None if entities is None else tuple(entities),
            verify=verify,
        ).run()


def tune(
    configuration: Path | str | Mapping[str, object] | TuneSpec,
    *,
    output_root: Path | str | None = None,
    catalog_root: Path | str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> TuningResult:
    """
    Search a space of configurations and return every trial.

    Parameters
    ----------
    configuration
        A path to a YAML file, a mapping, or an already-validated search.
    output_root
        Where the search writes. Each trial gets its own directory
        beneath it, so the winning bundle survives the search and can be
        used rather than retrained.
    catalog_root
        Where bundles are recorded, or ``None`` to use the output root.
    metadata
        Free-form annotations.

    Returns
    -------
    TuningResult
        Every trial, successes and failures alike, with the winner
        identified and the direction and objective split recorded -- all
        three of which a column of objective values cannot be read
        without.
    """
    spec = _tune_spec(configuration)
    context = _search_context(
        spec, output_root=output_root, catalog_root=catalog_root, metadata=metadata
    )
    with context.activate():
        # Resolved from the base specification rather than from a trial's,
        # because every trial in a search is the same model: a search that
        # varied the model would be several searches whose objectives are
        # not comparable, which is a different thing with a different name.
        pipeline = pipeline_for(_definition_named(spec.base), "tune", TunePipeline)
        return pipeline(context=context, spec=spec).run()


def train_jobs(
    configuration: Path | str | Mapping[str, object] | JobSetSpec,
    *,
    executor: Executor | None = None,
    run_id: str | None = None,
    catalog_root: Path | str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> JobSetManifest:
    """
    Train every job in a job set.

    Parameters
    ----------
    configuration
        A path to a job-set YAML file, a mapping, or a validated job set.
    executor
        Where jobs run. ``None`` asks the placement policy, which is the
        normal path and what ``placement.executor: auto`` means.
    run_id
        Identifier for the set. Defaults to the set's name and its
        specification digest, so re-running the same file extends the same
        directory rather than scattering partial sets beside it.
    catalog_root
        Where bundles are recorded, or ``None`` for the set's
        ``output_root`` -- the same catalog a single run under that root
        uses, so :func:`registry` sees every run together.
    metadata
        Free-form annotations carried into every job's run context.

    Returns
    -------
    JobSetManifest
        One record per job, in declaration order, with ``succeeded`` and
        ``failed`` partitioning them. A job that failed is a record, not an
        exception: thirty-nine models that trained are thirty-nine results,
        and discarding them over one typo in the fortieth would be the
        wrong trade.
    """
    spec = configuration if isinstance(configuration, JobSetSpec) else _job_set_spec(configuration)
    return JobSetRunner(
        spec,
        run_id=run_id,
        executor=executor,
        catalog_root=catalog_root,
        metadata=metadata,
    ).run()


def train_groups(
    root: Path | str,
    *,
    defaults: Mapping[str, object],
    output_root: Path | str,
    groups: Iterable[str] | None = None,
    name: str | None = None,
    placement: Mapping[str, object] | None = None,
    overrides_for: Callable[[DataGroup], Mapping[str, object]] | None = None,
    executor: Executor | None = None,
    catalog_root: Path | str | None = None,
) -> JobSetManifest:
    """
    Train one model per data group across a group set.

    The three steps a caller would otherwise write out: read the manifest,
    expand it into jobs, run them. Worth a function because the middle step
    is the one that is easy to get subtly wrong -- forgetting to carry the
    snapshot fingerprint, or merging the per-group overrides in the wrong
    order so that each job silently reads the whole dataset.

    Parameters
    ----------
    root
        The group set directory, containing a group manifest.
    defaults
        The run-specification fragment every job shares.
    output_root
        Where the set writes.
    groups
        Which groups to train, or ``None`` for all of them.
    name
        The set's label.
    placement
        Where jobs run, or ``None`` to leave it to the placement policy.
    overrides_for
        Per-group overrides. The hook that lets a data-rich group get a
        wider model than a thin one, which is the reason a group set is a
        job set rather than a loop.
    executor
        An executor to use instead of the one the policy would choose.
    catalog_root
        Where bundles are recorded, or ``None`` for ``output_root``.

    Returns
    -------
    JobSetManifest
        One record per group.
    """
    group_set = read_group_set(Path(root)).select(groups)
    spec = job_set_for_groups(
        group_set,
        defaults=defaults,
        output_root=Path(output_root),
        name=name,
        placement=placement,
        overrides_for=overrides_for,
    )
    return train_jobs(spec, executor=executor, catalog_root=catalog_root)


def registry(root: Path | str | None = None) -> RunRegistry:
    """
    Open the registry of trained runs under a root.

    The way back from "I trained a lot of things" to "this one": list runs
    by tag, pick the best by a metric, promote one to an alias, and pass its
    directory to :func:`evaluate` or :func:`infer`.

    Parameters
    ----------
    root
        The catalog directory -- the ``output_root`` (or ``catalog_root``)
        training used. ``None`` for the default search root.

    Returns
    -------
    RunRegistry
        The registry.

    Examples
    --------
    ::

        runs = api.registry("artifacts/eod")
        best = runs.best("mae", direction="minimise", tags=["lr-sweep"])
        runs.promote(best, "production")
        api.infer(runs.get(best.model_name, alias="production").directory, source=...)
    """
    return RunRegistry(_directory(root) or _DEFAULT_ROOT)


def _directory(value: Path | str | None) -> Path | None:
    """
    Accept a directory as either a path or a string.

    Every other path-shaped argument on this module already takes both --
    ``evaluate`` and ``infer`` accept a bundle as a string, and a
    specification can be a string. An output root that accepted only
    ``Path`` was an inconsistency rather than a rule, and the way it failed
    made it worse: the string travelled several frames inwards before
    ``root / run_id`` raised ``TypeError: unsupported operand type(s) for
    /``, which names neither the argument nor the caller's mistake.

    Parameters
    ----------
    value
        A directory, or ``None`` to leave the default in place.

    Returns
    -------
    pathlib.Path or None
        The directory, or ``None`` unchanged.
    """
    return None if value is None else Path(value)


def _run_spec(configuration: Path | str | Mapping[str, object] | RunSpec) -> RunSpec:
    """
    Accept a specification in any of the forms callers have one in.

    Parameters
    ----------
    configuration
        A path, a mapping, or a validated specification.

    Returns
    -------
    RunSpec
        The validated specification.
    """
    # Against the concrete members rather than `RunSpec`, which is an
    # annotated discriminated union and cannot be used in an instance check
    # at all -- it raises `TypeError` rather than returning `False`, so the
    # mistake is not one a passing test would reveal by accident.
    if isinstance(configuration, SupervisedRunSpec | ReinforcementRunSpec):
        return configuration
    if isinstance(configuration, Mapping):
        return parse_run_spec(configuration)
    return load_run_spec(configuration)


def _job_set_spec(configuration: Path | str | Mapping[str, object]) -> JobSetSpec:
    """
    Accept a job set as a path or a mapping.

    Parameters
    ----------
    configuration
        A path or a mapping.

    Returns
    -------
    JobSetSpec
        The validated job set.
    """
    if isinstance(configuration, Mapping):
        return parse_job_set_spec(configuration)
    return load_job_set_spec(configuration)


def _context(
    spec: SupervisedRunSpec,
    *,
    output_root: Path | str | None,
    catalog_root: Path | str | None,
    metadata: Mapping[str, str] | None,
) -> RunContext:
    """
    Build the run context for a single training run.

    The run identifier is derived from the specification digest, matching
    what a job set does for the same reason: re-running one configuration
    should extend its directory rather than leave a trail of near-identical
    ones nobody can tell apart.

    Parameters
    ----------
    spec
        The validated specification.
    output_root
        Where the run writes, or ``None`` to use the specification's.
    catalog_root
        Where the bundle is recorded, or ``None`` to use the output root.
    metadata
        Free-form annotations.

    Returns
    -------
    RunContext
        The context.
    """
    digest = digest_spec(spec)
    run_id = f"{spec.model.name}-{abbreviate_digest(digest)}"
    root = _directory(output_root) or spec.output_root
    directory = root / run_id

    return RunContext(
        run_id=run_id,
        spec_digest=digest,
        output_directory=directory,
        seed=spec.seed,
        catalog=JsonlCatalog(_directory(catalog_root) or root),
        metadata=dict(metadata or {}),
    )


def _tune_spec(
    configuration: Path | str | Mapping[str, object] | TuneSpec,
) -> TuneSpec:
    """
    Accept a search as a path, a mapping or an already-validated object.

    Parameters
    ----------
    configuration
        One of the three forms.

    Returns
    -------
    TuneSpec
        The validated search.
    """
    if isinstance(configuration, TuneSpec):
        return configuration
    if isinstance(configuration, Mapping):
        return parse_tune_spec(configuration)
    return load_tune_spec(Path(configuration))


def _search_context(
    spec: TuneSpec,
    *,
    output_root: Path | str | None,
    catalog_root: Path | str | None,
    metadata: Mapping[str, str] | None,
) -> RunContext:
    """
    Build the context a search runs under.

    Parameters
    ----------
    spec
        The validated search.
    output_root
        Where the search writes, or ``None`` for a default beneath the
        framework's artifact root.
    catalog_root
        Where bundles are recorded, or ``None`` to use the output root.
    metadata
        Free-form annotations.

    Returns
    -------
    RunContext
        The context. Its identifier is derived from the search digest for
        the same reason a run's is: re-running one search should extend
        its directory rather than leave a trail of near-identical ones.
    """
    digest = digest_spec(spec)
    label = spec.name or "search"
    run_id = f"{label}-{abbreviate_digest(digest)}"
    root = _directory(output_root) or _DEFAULT_ROOT

    return RunContext(
        run_id=run_id,
        spec_digest=digest,
        output_directory=root / run_id,
        seed=spec.seed,
        catalog=JsonlCatalog(_directory(catalog_root) or root),
        metadata=dict(metadata or {}),
    )


def _bundle_context(directory: Path, *, action: str, output_root: Path | str | None) -> RunContext:
    """
    Build the context an evaluation or inference run works under.

    Derived from the bundle rather than from a specification, because
    neither action has one until the bundle has been opened -- and the
    context is needed in order to open it.

    No catalog. Neither action writes a bundle, so there is nothing to
    record, and handing them one would invite a future version to register
    an evaluation as though it were a model.

    Parameters
    ----------
    directory
        The bundle being used.
    action
        ``"evaluate"`` or ``"infer"``, used in the run identifier so the
        two leave distinguishable directories behind.
    output_root
        Where to write, or ``None`` to write beside the bundle.

    Returns
    -------
    RunContext
        The context.
    """
    # Beside the bundle by default, so an evaluation lands next to the thing
    # it evaluated rather than in whatever directory the caller was in.
    root = _directory(output_root) or directory.parent
    run_id = f"{action}-{directory.name}"

    return RunContext(
        run_id=run_id,
        spec_digest="",
        output_directory=root / run_id,
        metadata={"bundle": str(directory), "action": action},
    )


def _definition_in(directory: Path) -> object | None:
    """
    Resolve the framework definition of the model a bundle holds.

    Read from the manifest rather than taken from the caller, because the
    caller of ``evaluate`` or ``infer`` has a directory and nothing else --
    which is the point of a bundle. The manifest is a small JSON file, so
    this costs one read before the pipeline opens the bundle properly.

    Parameters
    ----------
    directory
        The bundle directory.

    Returns
    -------
    object or None
        The definition, or ``None`` if the model is not registered in this
        process. ``None`` rather than an error: the pipeline raises a far
        better message about the missing model a moment later, naming what
        *is* registered, and pre-empting it here would replace a good
        diagnostic with a worse one.
    """
    try:
        return get_model(load_manifest(directory).model_name)()
    except (ComponentError, BundleError, OSError):
        return None


def _definition_named(base: Mapping[str, object]) -> object | None:
    """
    Resolve the framework definition named by a base specification.

    Parameters
    ----------
    base
        The unvalidated run specification a search varies.

    Returns
    -------
    object or None
        The definition, or ``None`` if the specification names no model or
        names one this process has not registered. ``None`` rather than an
        error, because the pipeline's own ``resolve`` stage reports a
        missing model far better than a lookup in a helper can.
    """
    reference = base.get("model")
    name = reference if isinstance(reference, str) else None
    if isinstance(reference, Mapping):
        name = reference.get("name")
    if not isinstance(name, str):
        return None
    try:
        return get_model(name)()
    except ComponentError:
        return None
